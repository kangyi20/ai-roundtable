"""AI 讨论室 —— Starlette 后端 + 静态前端。

复用 engine/memory/providers/scoring/storage/config 全部逻辑，只在外面包一层 HTTP API 与 SSE 推送。

运行：python server.py
"""

import asyncio
import json
import queue
import threading
import time
import tomllib
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

import engine
import prompts
from config import estimate_cost_cny
from profile_store import ProfileStore
from providers import MockProvider, OpenAICompatibleProvider
from storage import DebateStore

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"

STORE = DebateStore()
PROFILE = ProfileStore()  # 用户画像库（跨会话；本地文件，绝不外传）

# 进行中的讨论（内存态） + 自动讨论控制 + SSE 队列
ACTIVE: dict[str, object] = {}
_AUTO_FLAGS: dict[str, bool] = {}
_QUEUES: dict[str, queue.Queue] = {}
_EXPORTS: dict[str, dict[str, Path]] = {}
_DRAFTS: dict[str, dict] = {}  # 提炼草稿：draft_id -> {user_context(L2), user_context_soft(L3)}
_DRAFT_LIMIT = 20
_LOCK = threading.Lock()


def _load_secrets() -> dict:
    """读取 .streamlit/secrets.toml；不存在则返回空。"""
    if SECRETS_PATH.exists():
        with open(SECRETS_PATH, "rb") as f:
            return tomllib.load(f)
    return {}


SECRETS = _load_secrets()


def _get_state(debate_id: str):
    if debate_id in ACTIVE:
        return ACTIVE[debate_id]
    s = STORE.load_latest(debate_id)
    if s is not None:
        ACTIVE[debate_id] = s
    return s


def _persist(state) -> None:
    STORE.append_snapshot(state.debate_id, state)


def _remember_draft(payload: dict) -> str:
    """把提炼得到的 L2 处境暂存在服务端，避免下发给浏览器。"""
    did = "d_" + uuid4().hex[:8]
    _DRAFTS[did] = payload
    if len(_DRAFTS) > _DRAFT_LIMIT:
        for key in list(_DRAFTS.keys())[: len(_DRAFTS) - _DRAFT_LIMIT]:
            _DRAFTS.pop(key, None)
    return did


def build_provider(secrets: dict, role: str, side: str, state, temperature=None, mock_mode=None):
    """根据角色与当前 state 构造 Provider；MOCK 模式用 MockProvider。

    mock_mode 显式传入时优先——用于"讨论还没创建"的场景（如议题提炼），
    那时没有 state 可用来判断是否 mock。
    """
    if mock_mode is None:
        mock_mode = state.config.mock_mode if state is not None else True
    if mock_mode:
        return MockProvider(role=role, side=side)

    if role == "speaker":
        lookup = side
    elif role in ("judge", "memory"):
        lookup = "JUDGE_AI"
    else:
        lookup = role
    section = secrets.get(lookup, {})
    temp = temperature if temperature is not None else section.get("temperature")
    return OpenAICompatibleProvider(
        base_url=section.get("base_url", ""),
        api_key=section.get("api_key", ""),
        model=section.get("model", ""),
        temperature=temp,
        timeout_s=float(section.get("timeout_s", 60)),
        max_retries=int(section.get("max_retries", 1)),
        extra_body=section.get("extra_body") or {},
        debug_raw_response=state.config.debug_raw_response if state is not None else False,
    )


def snapshot(state) -> dict:
    """把 DebateState 转成可 JSON 序列化的前端快照（不含 reasoning）。"""
    cfg = state.config
    cost = estimate_cost_cny(
        state.totals.get("usage_in", 0),
        state.totals.get("usage_out", 0),
        cfg.price_in_per_1m,
        cfg.price_out_per_1m,
    )
    return {
        "debate_id": state.debate_id,
        "state": state.state,
        "round_no": state.round_no,
        "max_rounds": cfg.max_rounds,
        "topic": state.topic,
        "pro_side": state.pro_side,
        "con_side": state.con_side,
        # 注意：只暴露 L3 体贴版；state.user_context 是注入版（含画像信息），绝不下发
        "user_context_soft": state.user_context_soft,
        "stance_a": state.stance_a or state.pro_side,
        "stance_b": state.stance_b or state.con_side,
        "converged": state.converged,
        "word_limit": cfg.word_limit,
        "mock_mode": cfg.mock_mode,
        "early_stop": cfg.early_stop,
        "messages": [m.model_dump(exclude={"reasoning"}) for m in state.messages],
        "conclusion": state.conclusion.model_dump() if state.conclusion else None,
        "conclusion_error": state.conclusion_error,
        "qa": state.qa,
        "score": state.score.model_dump() if state.score else None,
        "totals": state.totals,
        "cost_cny": cost,
    }


def _defaults() -> dict:
    from config import resolve_config

    return resolve_config(ui_config={}, secrets=SECRETS)


# ---------------------------------------------------------------------------
# HTTP 端点
# ---------------------------------------------------------------------------

async def index(request: Request):
    html = (STATIC_DIR / "index.html").read_bytes()
    return Response(html, media_type="text/html", headers={"Cache-Control": "no-cache"})


_MIME = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


async def static(request: Request):
    rel = request.path_params.get("path", "")
    fp = (STATIC_DIR / rel).resolve()
    if not str(fp).startswith(str(STATIC_DIR.resolve())) or not fp.is_file():
        return Response("not found", status_code=404)
    return Response(
        fp.read_bytes(),
        media_type=_MIME.get(fp.suffix.lower(), "application/octet-stream"),
        headers={"Cache-Control": "no-cache"},
    )


async def meta(request: Request):
    models = {
        "A": SECRETS.get("A", {}).get("model", "未配置"),
        "B": SECRETS.get("B", {}).get("model", "未配置"),
        "JUDGE_AI": SECRETS.get("JUDGE_AI", {}).get("model", "未配置"),
    }
    debates = []
    for did in STORE.list_debate_ids():
        s = STORE.load_latest(did)
        if s:
            debates.append(
                {
                    "debate_id": did,
                    "topic": s.topic[:20],
                    "state": s.state,
                    "round_no": s.round_no,
                }
            )
    return JSONResponse({"models": models, "defaults": _defaults(), "debates": debates})


# ---- 隐藏入口：画像查看 / 清空（页面上不提供任何按钮，只能手输地址）----

async def get_profile(request: Request):
    return JSONResponse(PROFILE.load())


async def clear_profile(request: Request):
    PROFILE.clear()
    return JSONResponse({"ok": True, "stats": PROFILE.stats()})


async def extract(request: Request):
    """裁判从用户的一段困惑中提炼议题、双方立场与提问者处境。

    发生在"讨论还没创建"的时候，所以没有 state 可判断 mock 模式，
    由请求体显式传入或回落到默认配置。
    """
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "请先描述你的困惑"}, status_code=400)
    mock_mode = body.get("mock_mode")
    if mock_mode is None:
        mock_mode = _defaults().get("mock_mode", True)

    def _do():
        provider = build_provider(
            SECRETS, "judge", "JUDGE_AI", None, temperature=0.3, mock_mode=mock_mode
        )
        prompt = prompts.build_topic_extraction_prompt(text, PROFILE.brief_for_judge())
        messages = [
            {"role": "system", "content": "你是善于倾听的讨论主持人，只输出合法 JSON。"},
            {"role": "user", "content": prompt},
        ]
        # 结构化抽取任务：限时出 JSON（不吃"深度任务不限 token"那条规则）
        data, _lat, _uin, _uout = provider.chat_json(
            messages, max_tokens=2048, temperature=0.3, schema_hint="extract"
        )
        if not isinstance(data, dict) or not (data.get("topic") or data.get("pro_side")):
            raise ValueError("裁判没能提炼出有效内容，请重试或手动填写")

        # 更新长期画像：new / reinforce / contradict → 权重增减，必要时删除
        updates = data.get("belief_updates")
        if isinstance(updates, list) and updates:
            try:
                PROFILE.apply_updates(updates)
            except Exception:  # noqa: BLE001
                pass  # 画像更新失败不应阻断提炼本身

        # L2（注入版）留在服务端草稿缓存里，不下发给浏览器
        draft_id = _remember_draft(
            {
                "user_context": str(data.get("user_context") or "").strip(),
                "user_context_soft": str(data.get("user_context_soft") or "").strip(),
            }
        )
        return {
            "draft_id": draft_id,
            "topic": str(data.get("topic") or "").strip(),
            "pro_side": str(data.get("pro_side") or "").strip(),
            "con_side": str(data.get("con_side") or "").strip(),
            "note": str(data.get("note") or "").strip(),
            "user_context_soft": str(data.get("user_context_soft") or "").strip(),
        }

    try:
        return JSONResponse(await asyncio.to_thread(_do))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"提炼失败：{e}"}, status_code=400)


async def start(request: Request):
    body = await request.json()
    topic = (body.get("topic") or "").strip() or "人工智能利大于弊还是弊大于利"
    pro_side = (body.get("pro_side") or "").strip() or "人工智能利大于弊"
    con_side = (body.get("con_side") or "").strip() or "人工智能弊大于利"
    # 处境上下文：L2 从服务端草稿缓存取（不下发浏览器）+ 画像里权重最高的几条
    draft = _DRAFTS.get(str(body.get("draft_id") or "").strip()) or {}
    user_context = str(draft.get("user_context") or "").strip()
    user_context_soft = (
        str(body.get("user_context_soft") or "").strip()
        or str(draft.get("user_context_soft") or "").strip()
    )
    profile_text = PROFILE.injection_text()
    merged_context = "\n".join(p for p in (profile_text, user_context) if p)

    config = dict(body.get("config") or {})
    # 兼容旧版：顶层基本键回退
    for k in ("word_limit", "max_rounds", "early_stop", "mock_mode"):
        if k not in config and k in body:
            config[k] = body[k]
    allowed = {
        "word_limit", "max_rounds", "input_budget", "output_max_tokens",
        "reasoning_budget", "reasoning_budget_open_close", "hard_cap",
        "finalizer_on_empty", "debug_raw_response", "weight_judge", "weight_ai",
        "strict_evidence", "mock_mode", "forfeit_policy", "overrun_mode", "early_stop",
    }
    ui_config = {k: v for k, v in config.items() if k in allowed}
    debate_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]

    def _do():
        with _LOCK:
            state = engine.create_initial_state(
                debate_id=debate_id,
                topic=topic,
                pro_side=pro_side,
                con_side=con_side,
                ui_config=ui_config,
                secrets=SECRETS,
                user_context=merged_context,
                user_context_soft=user_context_soft,
            )
            state.state = "A_SPEAKING"
            ACTIVE[debate_id] = state
            _persist(state)
            return snapshot(state)

    snap = await asyncio.to_thread(_do)
    return JSONResponse(snap)


async def get_state(request: Request):
    debate_id = request.path_params["debate_id"]

    def _do():
        with _LOCK:
            s = _get_state(debate_id)
            if s is None:
                return None
            return snapshot(s)

    snap = await asyncio.to_thread(_do)
    if snap is None:
        return JSONResponse({"error": "讨论不存在"}, status_code=404)
    return JSONResponse(snap)


async def _run_mutation(debate_id: str, fn):
    def _do():
        with _LOCK:
            state = _get_state(debate_id)
            if state is None:
                raise ValueError("讨论不存在")
            state = fn(state)
            ACTIVE[debate_id] = state
            _persist(state)
            return snapshot(state)

    return await asyncio.to_thread(_do)


async def speak(request: Request):
    debate_id = request.path_params["debate_id"]
    side = request.path_params["side"].upper()

    def _fn(state):
        provider = build_provider(SECRETS, "speaker", side, state)
        _, state = engine.speak(state, side, provider)
        return state

    try:
        return JSONResponse(await _run_mutation(debate_id, _fn))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


async def confirm(request: Request):
    debate_id = request.path_params["debate_id"]
    side = request.path_params["side"].upper()

    def _fn(state):
        provider = build_provider(SECRETS, "memory", side, state, temperature=0.2)
        if side == "A":
            return engine.confirm_a(state, provider)
        return engine.confirm_b(state, provider)

    try:
        return JSONResponse(await _run_mutation(debate_id, _fn))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


async def reject(request: Request):
    debate_id = request.path_params["debate_id"]
    side = request.path_params["side"].upper()

    def _fn(state):
        return engine.reject_a(state) if side == "A" else engine.reject_b(state)

    try:
        return JSONResponse(await _run_mutation(debate_id, _fn))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


async def add_note(request: Request):
    debate_id = request.path_params["debate_id"]
    body = await request.json()
    text = (body.get("text") or "").strip()
    is_public = bool(body.get("is_public", True))
    if not text:
        return JSONResponse({"error": "备注不能为空"}, status_code=400)

    def _fn(state):
        return engine.add_referee_note(state, debate_id, text, is_public)

    try:
        return JSONResponse(await _run_mutation(debate_id, _fn))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


async def summarize(request: Request):
    debate_id = request.path_params["debate_id"]

    def _fn(state):
        provider = build_provider(SECRETS, "judge", "JUDGE_AI", state, temperature=0.3)
        return engine.summarize(state, provider)

    try:
        return JSONResponse(await _run_mutation(debate_id, _fn))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


async def score(request: Request):
    debate_id = request.path_params["debate_id"]

    def _fn(state):
        provider = build_provider(SECRETS, "judge", "JUDGE_AI", state, temperature=0.0)
        return engine.score(state, provider)

    try:
        return JSONResponse(await _run_mutation(debate_id, _fn))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


async def ask(request: Request):
    """讨论结束后向裁判追问（深度任务，不限制 token）。"""
    debate_id = request.path_params["debate_id"]
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "问题不能为空"}, status_code=400)

    def _fn(state):
        provider = build_provider(SECRETS, "judge", "JUDGE_AI", state, temperature=0.4)
        return engine.ask_judge(state, provider, question)

    try:
        return JSONResponse(await _run_mutation(debate_id, _fn))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


_EXPORT_DOWNLOAD = {
    "transcript": "transcript.jsonl",
    "score": "score.json",
    "config": "config_snapshot.json",
    "conclusion": "conclusion.json",
}
_EXPORT_MIME = {
    "transcript": "application/jsonlines+json",
    "score": "application/json",
    "config": "application/json",
    "conclusion": "application/json",
}


async def export(request: Request):
    debate_id = request.path_params["debate_id"]

    def _do():
        with _LOCK:
            state = _get_state(debate_id)
            if state is None:
                raise ValueError("讨论不存在")
            export_dir, transcript, score_path, config_path = STORE.export(debate_id, state)
            _EXPORTS[debate_id] = {
                "transcript": transcript,
                "score": score_path,
                "config": config_path,
                "conclusion": export_dir / "conclusion.json",
            }
            return {"ok": True, "kinds": list(_EXPORT_DOWNLOAD.keys())}

    try:
        return JSONResponse(await asyncio.to_thread(_do))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


async def export_file(request: Request):
    debate_id = request.path_params["debate_id"]
    kind = request.path_params["kind"]

    def _do():
        entry = _EXPORTS.get(debate_id)
        if not entry or kind not in entry:
            raise ValueError("请先点击导出")
        path = entry[kind]
        if not path.exists():
            raise ValueError("文件不存在")
        return path.read_bytes()

    try:
        data = await asyncio.to_thread(_do)
        filename = _EXPORT_DOWNLOAD.get(kind, f"{kind}.json")
        return Response(
            data,
            media_type=_EXPORT_MIME.get(kind, "application/octet-stream"),
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# 自动讨论 + SSE
# ---------------------------------------------------------------------------

def _get_queue(debate_id: str) -> queue.Queue:
    if debate_id not in _QUEUES:
        _QUEUES[debate_id] = queue.Queue()
    return _QUEUES[debate_id]


def _auto_loop(debate_id: str):
    q = _get_queue(debate_id)
    while _AUTO_FLAGS.get(debate_id):
        try:
            with _LOCK:
                state = _get_state(debate_id)
                if state is None:
                    _AUTO_FLAGS[debate_id] = False
                    q.put({"type": "done"})
                    break

                s = state.state
                if s == "DONE":
                    _AUTO_FLAGS[debate_id] = False
                    q.put({"type": "done"})
                    break
                if s == "ROUND_END":
                    q.put({"type": "thinking", "label": "正在生成结论…"})
                    provider = build_provider(SECRETS, "judge", "JUDGE_AI", state, temperature=0.3)
                    state = engine.summarize(state, provider)
                    ACTIVE[debate_id] = state
                    _persist(state)
                    q.put({"type": "state", "data": snapshot(state)})
                    _AUTO_FLAGS[debate_id] = False
                    q.put({"type": "done"})
                    break
                if s == "A_WAIT_CONFIRM":
                    provider = build_provider(SECRETS, "memory", "A", state, temperature=0.2)
                    state = engine.confirm_a(state, provider)
                    ACTIVE[debate_id] = state
                    _persist(state)
                    q.put({"type": "state", "data": snapshot(state)})
                    continue
                if s == "B_WAIT_CONFIRM":
                    provider = build_provider(SECRETS, "memory", "B", state, temperature=0.2)
                    state = engine.confirm_b(state, provider)
                    ACTIVE[debate_id] = state
                    _persist(state)
                    q.put({"type": "state", "data": snapshot(state)})
                    continue
                if engine.can("speak_a", state):
                    q.put({"type": "thinking", "label": "A 正在思考…"})
                    provider = build_provider(SECRETS, "speaker", "A", state)
                    _, state = engine.speak(state, "A", provider)
                    ACTIVE[debate_id] = state
                    _persist(state)
                    q.put({"type": "state", "data": snapshot(state)})
                    continue
                if engine.can("speak_b", state):
                    q.put({"type": "thinking", "label": "B 正在思考…"})
                    provider = build_provider(SECRETS, "speaker", "B", state)
                    _, state = engine.speak(state, "B", provider)
                    ACTIVE[debate_id] = state
                    _persist(state)
                    q.put({"type": "state", "data": snapshot(state)})
                    continue

                _AUTO_FLAGS[debate_id] = False
                q.put({"type": "done"})
                break
        except Exception as e:  # noqa: BLE001
            _AUTO_FLAGS[debate_id] = False
            q.put({"type": "error", "message": str(e)})
            q.put({"type": "done"})
            break
        time.sleep(0.3)


async def auto_start(request: Request):
    debate_id = request.path_params["debate_id"]
    with _LOCK:
        if _get_state(debate_id) is None:
            return JSONResponse({"error": "讨论不存在"}, status_code=404)
    _AUTO_FLAGS[debate_id] = True
    threading.Thread(target=_auto_loop, args=(debate_id,), daemon=True).start()
    return JSONResponse({"ok": True})


async def auto_stop(request: Request):
    debate_id = request.path_params["debate_id"]
    _AUTO_FLAGS[debate_id] = False
    return JSONResponse({"ok": True})


async def events(request: Request):
    debate_id = request.path_params["debate_id"]
    q = _get_queue(debate_id)

    async def gen():
        with _LOCK:
            state = _get_state(debate_id)
            if state is not None:
                yield f"data: {json.dumps({'type': 'state', 'data': snapshot(state)}, ensure_ascii=False)}\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                ev = q.get_nowait()
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            except queue.Empty:
                pass
            await asyncio.sleep(0.25)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


routes = [
    Route("/", index),
    Route("/static/{path:path}", static),
    Route("/api/meta", meta),
    Route("/api/profile", get_profile),
    Route("/api/profile/clear", clear_profile, methods=["POST"]),
    Route("/api/extract", extract, methods=["POST"]),
    Route("/api/start", start, methods=["POST"]),
    Route("/api/state/{debate_id}", get_state),
    Route("/api/{debate_id}/speak/{side}", speak, methods=["POST"]),
    Route("/api/{debate_id}/confirm/{side}", confirm, methods=["POST"]),
    Route("/api/{debate_id}/reject/{side}", reject, methods=["POST"]),
    Route("/api/{debate_id}/note", add_note, methods=["POST"]),
    Route("/api/{debate_id}/summarize", summarize, methods=["POST"]),
    Route("/api/{debate_id}/score", score, methods=["POST"]),
    Route("/api/{debate_id}/ask", ask, methods=["POST"]),
    Route("/api/{debate_id}/export", export, methods=["POST"]),
    Route("/api/{debate_id}/export/{kind}", export_file),
    Route("/api/{debate_id}/auto/start", auto_start, methods=["POST"]),
    Route("/api/{debate_id}/auto/stop", auto_stop, methods=["POST"]),
    Route("/api/{debate_id}/events", events),
]

app = Starlette(routes=routes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
