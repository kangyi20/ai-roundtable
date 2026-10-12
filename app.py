"""Streamlit 前端：三栏布局、自动/手动讨论、聊天气泡、结论面板、评估面板、导出。"""

import html
import time
import uuid
from datetime import datetime

import streamlit as st

import engine
from config import DEFAULTS, estimate_cost_cny, resolve_config
from memory import DebateState, compute_overrun
from providers import MockProvider, OpenAICompatibleProvider
from storage import DebateStore

STORE = DebateStore()
DIMENSIONS = ("logic", "evidence", "rebuttal", "clarity")

# 内部值 -> 中文展示映射（不修改状态机/配置的内部英文值）
_STATUS_LABEL = {
    "draft": "草稿",
    "confirmed": "已确认",
    "rejected": "已驳回",
    "rejected_twice": "二次驳回",
    "note": "备注",
    "dodge": "放弃",
}
_DIM_LABEL = {
    "logic": "逻辑",
    "evidence": "论据",
    "rebuttal": "回应互动",
    "clarity": "清晰度",
}
_OVERRUN_LABEL = {
    "penalty": "扣分",
    "truncate": "截断",
    "reject": "驳回",
}
_FORFEIT_LABEL = {
    "skip": "跳过",
    "forced_confirm": "强制确认占位",
}
_SPEAKER_LABEL = {
    "A": "A 方",
    "B": "B 方",
    "REFEREE": "主持人",
    "SYSTEM": "系统",
}
_STATE_LABEL = {
    "IDLE": "待开始",
    "A_SPEAKING": "A 方思考中",
    "A_WAIT_CONFIRM": "A 方待确认",
    "B_SPEAKING": "B 方思考中",
    "B_WAIT_CONFIRM": "B 方待确认",
    "ROUND_END": "讨论结束，可生成结论",
    "SUMMARIZING": "正在生成结论",
    "SCORING": "正在评分",
    "DONE": "已完成",
}
_LENGTH_MODES = {
    "简洁 (~200字)": 200,
    "均衡 (~300字)": 300,
    "深入 (~400字)": 400,
}


def _init_session() -> None:
    """初始化 session_state；若 debate_id 存在则从 JSONL 恢复。"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.debate_id = None
        st.session_state.state = None
        st.session_state.speaking_lock = None
        st.session_state.error = None
        st.session_state.auto_running = False
        st.session_state.auto_steps = 0
        st.session_state.auto_failures = 0

    # 优先从持久化恢复（支持刷新后重建）
    if st.session_state.debate_id and (
        not st.session_state.state or st.session_state.state is None
    ):
        loaded = STORE.load_latest(st.session_state.debate_id)
        if loaded:
            st.session_state.state = loaded


def _persist() -> None:
    if st.session_state.debate_id and st.session_state.state:
        STORE.append_snapshot(st.session_state.debate_id, st.session_state.state)


def _provider_for(secrets: dict, role: str, side: str, temperature: float | None = None):
    """根据当前状态构造 Provider；MOCK_MODE 下使用 MockProvider。

    role: speaker | judge | memory; memory 实际映射到 JUDGE_AI 低温输出。
    """
    state: DebateState | None = st.session_state.state
    mock_mode = state.config.mock_mode if state else True
    if mock_mode:
        return MockProvider(role=role, side=side)

    # speaker 按 A/B 取对应配置；judge/memory 复用 JUDGE_AI
    if role == "speaker":
        lookup_role = side
    elif role in ("judge", "memory"):
        lookup_role = "JUDGE_AI"
    else:
        lookup_role = role
    section = secrets.get(lookup_role, {})
    temp = temperature if temperature is not None else section.get("temperature")
    extra_body = section.get("extra_body") or {}
    debug_raw = False
    if state and state.config:
        debug_raw = state.config.debug_raw_response
    return OpenAICompatibleProvider(
        base_url=section.get("base_url", ""),
        api_key=section.get("api_key", ""),
        model=section.get("model", ""),
        temperature=temp,
        timeout_s=float(section.get("timeout_s", 60)),
        max_retries=int(section.get("max_retries", 1)),
        extra_body=extra_body if isinstance(extra_body, dict) else {},
        debug_raw_response=debug_raw,
    )


def _resolved_defaults(secrets: dict) -> dict:
    """返回仅解析 secrets/环境变量后的默认配置，用于 UI 初始值。"""
    return resolve_config(ui_config={}, secrets=secrets)


def _current_word_limit() -> int:
    """根据长度档位计算本场每轮字数。"""
    mode = st.session_state.get("cfg_length_mode", "均衡 (~300字)")
    if mode in _LENGTH_MODES:
        return _LENGTH_MODES[mode]
    return int(st.session_state.get("cfg_custom_word_limit", 400))


def main():
    st.set_page_config(page_title="AI 讨论室", page_icon="💬", layout="wide")
    _init_session()

    # 全局样式 + 右栏粘性定位
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
            position: sticky;
            top: 1rem;
            align-self: flex-start;
            max-height: calc(100vh - 2rem);
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        secrets = dict(st.secrets) if st.secrets else {}
    except st.errors.StreamlitSecretNotFoundError:
        secrets = {}
    state: DebateState | None = st.session_state.state

    if st.session_state.error:
        st.error(st.session_state.error)
        st.session_state.error = None

    _render_header()

    left, mid, right = st.columns([1, 2.3, 1])
    with left:
        _render_sidebar(secrets, state)
    with mid:
        _render_chat(state)
    with right:
        _render_controls(secrets, state)

    _maybe_auto_advance(secrets)


def _render_header():
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#2563eb,#7c3aed);padding:18px 22px;border-radius:14px;color:#fff;margin-bottom:10px;">
            <div style="font-size:1.5em;font-weight:800;">💬 AI 讨论室</div>
            <div style="opacity:.92;font-size:.95em;margin-top:2px;">两个 AI 就一个话题自然讨论，最后自动汇总结论。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(secrets: dict, state: DebateState | None):
    st.header("配置")
    defaults = _resolved_defaults(secrets)
    is_idle = state is None or state.state == "IDLE"

    # 话题与视角
    topic_default = state.topic if state else defaults.get("topic", "人工智能利大于弊还是弊大于利")
    pro_default = state.pro_side if state else defaults.get("pro_side", "人工智能利大于弊")
    con_default = state.con_side if state else defaults.get("con_side", "人工智能弊大于利")

    topic = st.text_input("话题", value=topic_default, disabled=not is_idle, key="cfg_topic")
    pro_side = st.text_input("A 方视角", value=pro_default, disabled=not is_idle, key="cfg_pro_side")
    con_side = st.text_input("B 方视角", value=con_default, disabled=not is_idle, key="cfg_con_side")

    # 主要数值配置
    max_rounds = st.slider(
        "讨论轮数（上限）",
        1,
        20,
        value=state.config.max_rounds if state else defaults.get("max_rounds", DEFAULTS["max_rounds"]),
        disabled=not is_idle,
        key="cfg_max_rounds",
        help="达到该轮数后自动结束并生成结论；配合下方『内容收敛自动结束』，讨论可在更早收敛时提前收尾。",
    )
    current_word = state.config.word_limit if state else defaults.get("word_limit", DEFAULTS["word_limit"])
    if current_word <= 200:
        default_idx = 0
    elif current_word <= 300:
        default_idx = 1
    elif current_word <= 400:
        default_idx = 2
    else:
        default_idx = 3
    length_mode = st.selectbox(
        "讨论长度",
        list(_LENGTH_MODES.keys()) + ["自定义"],
        index=default_idx,
        key="cfg_length_mode",
        help="控制每轮字数上限：越短越省 token，也越不容易灌水。",
    )
    if length_mode == "自定义":
        st.number_input(
            "自定义每轮字数",
            100,
            1000,
            value=max(current_word, 400),
            key="cfg_custom_word_limit",
        )
    word_limit = _current_word_limit()

    input_budget = st.slider(
        "输入预算 (tokens)",
        500,
        10000,
        value=state.config.input_budget if state else defaults.get("input_budget", DEFAULTS["input_budget"]),
        disabled=False,
        key="cfg_input_budget",
    )
    output_max_tokens = st.number_input(
        "输出 token 上限（0=自动）",
        0,
        4096,
        value=state.config.output_max_tokens if state else defaults.get("output_max_tokens", DEFAULTS["output_max_tokens"]),
        disabled=False,
        key="cfg_output_max_tokens",
    )

    with st.expander("本场预算与定稿配置", expanded=False):
        reasoning_budget = st.number_input(
            "中盘思考预算",
            0,
            10000,
            value=state.config.reasoning_budget if state else defaults.get("reasoning_budget", DEFAULTS["reasoning_budget"]),
            disabled=False,
            key="cfg_reasoning_budget",
        )
        reasoning_budget_open_close = st.number_input(
            "开篇/收尾思考预算",
            0,
            10000,
            value=state.config.reasoning_budget_open_close if state else defaults.get("reasoning_budget_open_close", DEFAULTS["reasoning_budget_open_close"]),
            disabled=False,
            key="cfg_reasoning_budget_open_close",
        )
        hard_cap = st.number_input(
            "输出 token 硬上限",
            512,
            16000,
            value=state.config.hard_cap if state else defaults.get("hard_cap", DEFAULTS["hard_cap"]),
            disabled=False,
            key="cfg_hard_cap",
        )
        finalizer_on_empty = st.toggle(
            "空/脏稿启用 finalizer 定稿",
            value=state.config.finalizer_on_empty if state else defaults.get("finalizer_on_empty", DEFAULTS["finalizer_on_empty"]),
            disabled=False,
            key="cfg_finalizer_on_empty",
        )
        debug_raw_response = st.toggle(
            "记录脱敏原始响应（provider_debug.log）",
            value=state.config.debug_raw_response if state else defaults.get("debug_raw_response", DEFAULTS["debug_raw_response"]),
            disabled=False,
            key="cfg_debug_raw_response",
        )

    # 评分权重
    weight_judge = st.slider(
        "人工评估权重",
        0.0,
        1.0,
        value=state.config.weight_judge if state else defaults.get("weight_judge", DEFAULTS["weight_judge"]),
        key="cfg_weight_judge",
    )
    weight_ai = st.slider(
        "AI 评估权重",
        0.0,
        1.0,
        value=state.config.weight_ai if state else defaults.get("weight_ai", DEFAULTS["weight_ai"]),
        key="cfg_weight_ai",
    )

    # 开关与策略
    mock_mode = st.toggle(
        "Mock 模式",
        value=state.config.mock_mode if state else defaults.get("mock_mode", DEFAULTS["mock_mode"]),
        disabled=not is_idle,
        key="cfg_mock_mode",
    )
    early_stop = st.toggle(
        "内容收敛自动结束",
        value=state.config.early_stop if state else defaults.get("early_stop", DEFAULTS["early_stop"]),
        disabled=not is_idle,
        key="cfg_early_stop",
        help="连续两轮没有新观点/新问题就自动结束并生成结论，省 token 也更自然。",
    )
    strict_evidence = st.toggle(
        "严格证据",
        value=state.config.strict_evidence if state else defaults.get("strict_evidence", DEFAULTS["strict_evidence"]),
        disabled=not is_idle,
        key="cfg_strict_evidence",
    )
    forfeit_policy = st.selectbox(
        "弃权策略",
        ["skip", "forced_confirm"],
        index=0 if (state.config.forfeit_policy if state else defaults.get("forfeit_policy", DEFAULTS["forfeit_policy"])) == "skip" else 1,
        disabled=not is_idle,
        key="cfg_forfeit_policy",
        help="强制确认占位当前等同跳过，不会绕过确认，仅为未来扩展占位。",
        format_func=lambda x: _FORFEIT_LABEL.get(x, x),
    )
    overrun_mode = st.selectbox(
        "超字处理",
        ["penalty", "truncate", "reject"],
        index={"penalty": 0, "truncate": 1, "reject": 2}.get(
            state.config.overrun_mode if state else defaults.get("overrun_mode", DEFAULTS["overrun_mode"]),
            0,
        ),
        disabled=False,
        key="cfg_overrun_mode",
        format_func=lambda x: _OVERRUN_LABEL.get(x, x),
    )

    # 进行中时允许应用部分配置变更
    if state and not is_idle:
        if st.button("应用后续配置变更", key="btn_apply_config"):
            changed = {}
            if word_limit != state.config.word_limit:
                changed["word_limit"] = word_limit
                state.config.word_limit = word_limit
            if input_budget != state.config.input_budget:
                changed["input_budget"] = input_budget
                state.config.input_budget = input_budget
            if output_max_tokens != state.config.output_max_tokens:
                changed["output_max_tokens"] = output_max_tokens
                state.config.output_max_tokens = output_max_tokens
            if reasoning_budget != state.config.reasoning_budget:
                changed["reasoning_budget"] = reasoning_budget
                state.config.reasoning_budget = reasoning_budget
            if reasoning_budget_open_close != state.config.reasoning_budget_open_close:
                changed["reasoning_budget_open_close"] = reasoning_budget_open_close
                state.config.reasoning_budget_open_close = reasoning_budget_open_close
            if hard_cap != state.config.hard_cap:
                changed["hard_cap"] = hard_cap
                state.config.hard_cap = hard_cap
            if finalizer_on_empty != state.config.finalizer_on_empty:
                changed["finalizer_on_empty"] = finalizer_on_empty
                state.config.finalizer_on_empty = finalizer_on_empty
            if debug_raw_response != state.config.debug_raw_response:
                changed["debug_raw_response"] = debug_raw_response
                state.config.debug_raw_response = debug_raw_response
            if overrun_mode != state.config.overrun_mode:
                changed["overrun_mode"] = overrun_mode
                state.config.overrun_mode = overrun_mode  # type: ignore[assignment]
            if weight_judge != state.config.weight_judge:
                changed["weight_judge"] = weight_judge
                state.config.weight_judge = weight_judge
            if weight_ai != state.config.weight_ai:
                changed["weight_ai"] = weight_ai
                state.config.weight_ai = weight_ai
            if changed:
                engine.config_change_note(state, st.session_state.debate_id, changed)
                _persist()
                st.rerun()

    st.markdown("---")
    st.markdown("**模型信息（不显示密钥）**")
    for role in ("A", "B", "JUDGE_AI"):
        section = secrets.get(role, {})
        st.markdown(f"- `{role}`: {section.get('model', '未配置')}")

    st.markdown("---")
    st.markdown("**累计用量**")
    if state:
        st.text(f"输入 token: {state.totals.get('usage_in', 0)}")
        st.text(f"输出 token: {state.totals.get('usage_out', 0)}")
        st.text(
            f"估算成本: ¥{estimate_cost_cny(state.totals.get('usage_in', 0), state.totals.get('usage_out', 0), state.config.price_in_per_1m, state.config.price_out_per_1m):.4f}"
        )
    else:
        st.text("输入 token: 0")
        st.text("输出 token: 0")
        st.text("估算成本: $0.00000")

    # 加载已有讨论
    st.markdown("---")
    existing = STORE.list_debate_ids()
    if existing:
        options: dict[str, str] = {"": ""}
        for did in existing:
            info = STORE.load_latest(did)
            if info:
                topic_short = info.topic[:18] + "..." if len(info.topic) > 18 else info.topic
                label = f"{did} | {topic_short} | 第{info.round_no}轮 | {info.state}"
            else:
                label = did
            options[label] = did

        selected_label = st.selectbox("加载已有讨论", list(options.keys()), key="load_debate")
        selected = options.get(selected_label, "")
        if selected and st.button("加载", key="btn_load"):
            loaded = STORE.load_latest(selected)
            if loaded:
                st.session_state.debate_id = selected
                st.session_state.state = loaded
                st.session_state.auto_running = False
                st.rerun()


def _render_chat(state: DebateState | None):
    st.header("讨论记录")
    if not state:
        st.info("请在左侧配置话题，然后点“开始讨论（手动）”或“自动讨论”。")
        return

    _render_status_bar(state)
    _render_stances(state)
    _render_usage(state)
    if state.conclusion:
        _render_conclusion(state)
    for msg in state.messages:
        _render_message(msg)


def _render_status_bar(state: DebateState):
    state_label = _STATE_LABEL.get(state.state, state.state)
    auto = st.session_state.get("auto_running", False)
    badge = "🟢 自动推进中" if auto else "⚪ 手动"
    st.markdown(
        f'<div style="padding:8px 12px;border:1px solid #e5e7eb;border-radius:8px;background:#f9fafb;margin-bottom:10px;font-size:.9em;">'
        f'第 {state.round_no} / {state.config.max_rounds} 轮 · 状态：{state_label} · {badge}</div>',
        unsafe_allow_html=True,
    )


def _render_stances(state: DebateState):
    """展示双方当前立场；立场可能随讨论漂移。"""
    sa = state.stance_a or state.pro_side
    sb = state.stance_b or state.con_side
    converged_html = ""
    if state.converged:
        converged_html = '<div style="color:#16a34a;font-weight:700;margin-top:4px;">✅ 双方已达成一致</div>'
    st.markdown(
        f'<div style="padding:8px 12px;border:1px solid #dbeafe;border-radius:8px;background:#eff6ff;margin-bottom:10px;font-size:.88em;line-height:1.6;">'
        f'<div><strong>A 当前立场</strong>：{html.escape(sa)}</div>'
        f'<div><strong>B 当前立场</strong>：{html.escape(sb)}</div>'
        f'{converged_html}'
        f'<div style="opacity:.65;margin-top:2px;">立场可随讨论漂移，一方可能被另一方说服。</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_usage(state: DebateState):
    """实时展示累计 token 用量与估算成本。"""
    t = state.totals
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("输入 token", f"{t.get('usage_in', 0):,}")
    c2.metric("输出 token", f"{t.get('usage_out', 0):,}")
    c3.metric(
        "估算成本",
        f"¥{estimate_cost_cny(t.get('usage_in', 0), t.get('usage_out', 0), state.config.price_in_per_1m, state.config.price_out_per_1m):.4f}",
    )
    c4.metric("讨论轮次", f"{state.round_no}/{state.config.max_rounds}")


def _render_conclusion(state: DebateState):
    c = state.conclusion
    st.markdown("## 🎯 讨论结论")
    if c.aligned:
        st.markdown(
            '<div style="background:#f0fdf4;border:1px solid #bbf7d0;color:#16a34a;border-radius:8px;padding:8px 12px;margin:4px 0 10px;font-weight:700;">🤝 双方最终达成一致</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#fef2f2;border:1px solid #fecaca;color:#dc2626;border-radius:8px;padding:8px 12px;margin:4px 0 10px;font-weight:700;">🔀 双方仍存在分歧</div>',
            unsafe_allow_html=True,
        )

    def _card(icon: str, title: str, items):
        if not items:
            return
        lis = "".join(f"<li>{html.escape(str(x))}</li>" for x in items)
        st.markdown(
            f'<div style="border-left:5px solid #2563eb;padding:10px 14px;border-radius:6px;background:#f9fafb;margin:8px 0;">'
            f'<div style="font-weight:700;margin-bottom:4px;">{icon} {title}</div>'
            f'<ul style="margin:0;padding-left:18px;">{lis}</ul></div>',
            unsafe_allow_html=True,
        )

    _card("✅", "双方共识", c.consensus)
    _card("⚡", "主要分歧", c.disagreements)
    _card("❓", "待解决问题", c.open_questions)
    if c.takeaway:
        st.markdown(
            f'<div style="background:#eef2ff;border-radius:8px;padding:12px 14px;margin:8px 0;font-weight:600;">'
            f'💡 一句话结论：{html.escape(c.takeaway)}</div>',
            unsafe_allow_html=True,
        )


def _render_message(msg):
    color_map = {
        "A": "#2563eb",
        "B": "#dc2626",
        "REFEREE": "#16a34a",
        "SYSTEM": "#6b7280",
    }
    avatar_map = {"A": "A", "B": "B", "REFEREE": "主", "SYSTEM": "系"}
    color = color_map.get(msg.speaker, "#6b7280")
    name = _SPEAKER_LABEL.get(msg.speaker, msg.speaker)
    avatar = avatar_map.get(msg.speaker, "?")
    status = _STATUS_LABEL.get(msg.status, msg.status)

    meta_bits = [f"第{msg.round_no}轮", status]
    if msg.model:
        meta_bits.append(f"模型:{msg.model}")
    if msg.latency_ms:
        meta_bits.append(f"{msg.latency_ms}ms")
    if msg.usage_in or msg.usage_out:
        meta_bits.append(f"token:{msg.usage_in}/{msg.usage_out}")
    if msg.finish_reason:
        meta_bits.append(f"完成:{msg.finish_reason}")
    meta = " · ".join(meta_bits)
    content_html = html.escape(msg.content).replace("\n", "<br>")

    if msg.status == "draft":
        st.markdown(
            f'<div style="display:flex;gap:10px;border:2px dashed {color};padding:12px;border-radius:10px;margin:8px 0;background:#fff;">'
            f'<div style="min-width:34px;height:34px;border-radius:50%;background:{color};color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;">{avatar}</div>'
            f'<div style="flex:1;"><div style="color:{color};font-weight:700;">{name}</div>'
            f'<div style="color:#6b7280;font-size:.78em;margin-bottom:6px;">{meta} · <strong>未确认，不进入公共记录</strong></div>'
            f'<div style="line-height:1.7;">{content_html}</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="display:flex;gap:10px;border-left:5px solid {color};padding:12px 14px;border-radius:8px;margin:8px 0;background:#f9fafb;">'
            f'<div style="min-width:34px;height:34px;border-radius:50%;background:{color};color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;">{avatar}</div>'
            f'<div style="flex:1;"><div style="color:{color};font-weight:700;">{name}</div>'
            f'<div style="color:#6b7280;font-size:.78em;margin-bottom:6px;">{meta}</div>'
            f'<div style="line-height:1.7;">{content_html}</div></div></div>',
            unsafe_allow_html=True,
        )


def _render_controls(secrets: dict, state: DebateState | None):
    st.header("控制")
    auto = st.session_state.get("auto_running", False)

    if not state or state.state == "IDLE":
        c1, c2 = st.columns(2)
        with c1:
            if st.button("开始讨论（手动）", key="btn_start"):
                _start(secrets, False)
        with c2:
            if st.button("自动讨论", key="btn_auto_start"):
                _start(secrets, True)
        return

    # 自动/暂停
    if auto:
        if st.button("⏸ 暂停自动讨论", key="btn_pause_auto"):
            st.session_state.auto_running = False
            st.rerun()
        st.caption("自动推进中：A/B 轮流发言并自动确认，可随时暂停。")
    else:
        if state.state not in ("DONE",):
            if st.button("▶ 自动讨论", key="btn_resume_auto"):
                st.session_state.auto_running = True
                st.rerun()

    st.markdown("---")

    # 手动操作（自动模式下隐藏，避免冲突）
    if not auto:
        if engine.can("speak_a", state):
            if st.button("让 A 发言", key="btn_speak_a"):
                _handle_speak(secrets, "A")
        if engine.can("speak_b", state):
            if st.button("让 B 发言", key="btn_speak_b"):
                _handle_speak(secrets, "B")

        if engine.can("confirm_a", state):
            draft = _find_draft(state, "A")
            if draft:
                overrun = compute_overrun(state, draft.content)
                if overrun > 0:
                    mode_label = _OVERRUN_LABEL.get(state.config.overrun_mode, state.config.overrun_mode)
                    st.warning(f"⚠️ A 草稿超 {overrun} 字（模式={mode_label}）")
            if st.button("确认 A", key="btn_confirm_a"):
                _handle_confirm(secrets, "A")
        if engine.can("reject_a", state):
            if st.button("驳回 A", key="btn_reject_a"):
                _handle_reject("A")

        if engine.can("confirm_b", state):
            draft = _find_draft(state, "B")
            if draft:
                overrun = compute_overrun(state, draft.content)
                if overrun > 0:
                    mode_label = _OVERRUN_LABEL.get(state.config.overrun_mode, state.config.overrun_mode)
                    st.warning(f"⚠️ B 草稿超 {overrun} 字（模式={mode_label}）")
            if st.button("确认 B", key="btn_confirm_b"):
                _handle_confirm(secrets, "B")
        if engine.can("reject_b", state):
            if st.button("驳回 B", key="btn_reject_b"):
                _handle_reject("B")

    # 主持人备注（自动/手动都可用，用于中途引导方向）
    if engine.can("add_note", state):
        st.markdown("---")
        note_text = st.text_input("主持人备注", key="input_note")
        is_public = st.checkbox("公开给下一位讨论者", value=True, key="chk_public_note")
        if st.button("插入主持人备注", key="btn_note"):
            _handle_add_note(note_text, is_public)

    # 结论
    if engine.can("summarize", state):
        st.markdown("---")
        st.subheader("🎯 结论")
        if state.conclusion_error:
            st.warning(state.conclusion_error)
        if st.button("生成结论", key="btn_summarize"):
            _handle_summarize(secrets)

    # 可选质量评估
    if state.state in ("ROUND_END", "DONE"):
        st.markdown("---")
        with st.expander("可选：质量评估（评分）", expanded=False):
            _render_scoring(secrets, state)

    # 导出
    st.markdown("---")
    if st.button("导出", key="btn_export"):
        _handle_export()

    # 新建讨论
    st.markdown("---")
    if st.button("🔄 新建讨论", key="btn_new_discussion"):
        _new_discussion()


def _render_scoring(secrets: dict, state: DebateState):
    st.subheader("质量评估（可选）")
    if state.score:
        st.json(state.score.model_dump())
        return

    locked = bool(state.judge_scores.get("A") and state.judge_scores.get("B"))
    if locked:
        st.info("评估已锁定，可点击“终局评分”。")
    for side in ("A", "B"):
        st.markdown(f"**{side} 方**")
        cols = st.columns(2)
        for idx, dim in enumerate(DIMENSIONS):
            with cols[idx % 2]:
                default = state.judge_scores.get(side, {}).get(dim, 5)
                val = st.slider(
                    f"{side} 方-{_DIM_LABEL.get(dim, dim)}",
                    0,
                    10,
                    value=default,
                    disabled=locked,
                    key=f"judge_{side}_{dim}",
                )
                state.judge_scores.setdefault(side, {})[dim] = val
    if not locked and st.button("确认评估", key="btn_lock_judge"):
        for side in ("A", "B"):
            for dim in DIMENSIONS:
                if dim not in state.judge_scores.get(side, {}):
                    st.session_state.error = f"{side} 方 {_DIM_LABEL.get(dim, dim)} 未打分"
                    st.rerun()
                    return
        _persist()
        st.success("评估已锁定")
        st.rerun()
    if engine.can("score", state):
        st.markdown("---")
        if not locked:
            st.info("请先完成评估，再点击终局评分。")
        if st.button("终局评分", key="btn_score", disabled=not locked):
            _handle_score(secrets)


def _find_draft(state: DebateState, side: str):
    for m in reversed(state.messages):
        if m.speaker == side and m.status == "draft":
            return m
    return None


def _collect_ui_config(topic: str, pro_side: str, con_side: str) -> dict:
    return {
        "topic": topic,
        "pro_side": pro_side,
        "con_side": con_side,
        "max_rounds": st.session_state.cfg_max_rounds,
        "word_limit": _current_word_limit(),
        "input_budget": st.session_state.cfg_input_budget,
        "output_max_tokens": st.session_state.cfg_output_max_tokens,
        "reasoning_budget": st.session_state.cfg_reasoning_budget,
        "reasoning_budget_open_close": st.session_state.cfg_reasoning_budget_open_close,
        "hard_cap": st.session_state.cfg_hard_cap,
        "finalizer_on_empty": st.session_state.cfg_finalizer_on_empty,
        "debug_raw_response": st.session_state.cfg_debug_raw_response,
        "weight_judge": st.session_state.cfg_weight_judge,
        "weight_ai": st.session_state.cfg_weight_ai,
        "mock_mode": st.session_state.cfg_mock_mode,
        "strict_evidence": st.session_state.cfg_strict_evidence,
        "forfeit_policy": st.session_state.cfg_forfeit_policy,
        "overrun_mode": st.session_state.cfg_overrun_mode,
        "early_stop": st.session_state.cfg_early_stop,
    }


def _start(secrets: dict, auto: bool):
    topic = st.session_state.cfg_topic
    pro_side = st.session_state.cfg_pro_side
    con_side = st.session_state.cfg_con_side
    ui_config = _collect_ui_config(topic, pro_side, con_side)

    debate_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    state = engine.create_initial_state(
        debate_id=debate_id,
        topic=topic,
        pro_side=pro_side,
        con_side=con_side,
        ui_config=ui_config,
        secrets=secrets,
    )
    state.state = "A_SPEAKING"

    st.session_state.debate_id = debate_id
    st.session_state.state = state
    st.session_state.auto_running = auto
    st.session_state.auto_steps = 0
    st.session_state.auto_failures = 0
    _persist()
    st.rerun()


def _auto_failure() -> None:
    """自动模式下记录一次失败；连续 3 次则暂停，避免空烧 token。"""
    st.session_state.auto_failures = st.session_state.get("auto_failures", 0) + 1
    if st.session_state.get("auto_running") and st.session_state.auto_failures >= 3:
        st.session_state.auto_running = False
        if st.session_state.error:
            st.session_state.error += "（连续失败，已自动暂停）"
        else:
            st.session_state.error = "连续失败，已自动暂停。"


def _new_discussion() -> None:
    """清空本场状态，回到新建讨论的初始界面。"""
    st.session_state.debate_id = None
    st.session_state.state = None
    st.session_state.auto_running = False
    st.session_state.auto_steps = 0
    st.session_state.auto_failures = 0
    st.rerun()


def _handle_speak(secrets: dict, side: str):
    if st.session_state.speaking_lock:
        return
    st.session_state.speaking_lock = side
    state = st.session_state.state
    provider = _provider_for(secrets, "speaker", side)
    ok = False
    try:
        msg, state = engine.speak(state, side, provider)
        st.session_state.state = state
        _persist()
        ok = True
    except engine.IllegalTransition as e:
        st.session_state.error = str(e)
    except Exception as e:
        st.session_state.error = f"{side} 发言失败：{e}"
    finally:
        st.session_state.speaking_lock = None
    if ok:
        st.session_state.auto_failures = 0
    else:
        _auto_failure()
    st.rerun()


def _handle_confirm(secrets: dict, side: str):
    state = st.session_state.state
    provider = _provider_for(secrets, "memory", side, temperature=0.2)
    ok = False
    try:
        if side == "A":
            state = engine.confirm_a(state, provider)
        else:
            state = engine.confirm_b(state, provider)
        st.session_state.state = state
        _persist()
        ok = True
    except engine.IllegalTransition as e:
        st.session_state.error = str(e)
    except Exception as e:
        st.session_state.error = f"确认 {side} 失败：{e}"
    if ok:
        st.session_state.auto_failures = 0
    else:
        _auto_failure()
    st.rerun()


def _handle_reject(side: str):
    state = st.session_state.state
    try:
        if side == "A":
            state = engine.reject_a(state)
        else:
            state = engine.reject_b(state)
        st.session_state.state = state
        _persist()
    except engine.IllegalTransition as e:
        st.session_state.error = str(e)
    except Exception as e:
        st.session_state.error = f"驳回 {side} 失败：{e}"
    st.rerun()


def _handle_add_note(note_text: str, is_public: bool):
    if not note_text.strip():
        return
    state = st.session_state.state
    try:
        state = engine.add_referee_note(
            state, st.session_state.debate_id, note_text.strip(), is_public
        )
        st.session_state.state = state
        _persist()
    except engine.IllegalTransition as e:
        st.session_state.error = str(e)
    except Exception as e:
        st.session_state.error = f"插入备注失败：{e}"
    st.rerun()


def _handle_summarize(secrets: dict):
    state = st.session_state.state
    provider = _provider_for(secrets, "judge", "JUDGE_AI", temperature=0.3)
    try:
        state = engine.summarize(state, provider)
        st.session_state.state = state
        _persist()
        if state.conclusion is None:
            st.session_state.error = state.conclusion_error or "结论生成失败，可重试。"
            _auto_failure()
        else:
            st.session_state.auto_failures = 0
            st.session_state.auto_running = False
    except engine.IllegalTransition as e:
        st.session_state.error = str(e)
        _auto_failure()
    except Exception as e:
        st.session_state.error = f"生成结论失败：{e}"
        _auto_failure()
    st.rerun()


def _handle_score(secrets: dict):
    state = st.session_state.state
    provider = _provider_for(secrets, "judge", "JUDGE_AI", temperature=0.0)
    try:
        state = engine.score(state, provider)
        st.session_state.state = state
        _persist()
    except engine.IllegalTransition as e:
        st.session_state.error = str(e)
    except Exception as e:
        st.session_state.error = f"评分失败：{e}"
    st.rerun()


def _maybe_auto_advance(secrets: dict):
    """自动讨论：每次页面重跑只推进一步，然后 rerun 继续，直到结束/暂停。"""
    if not st.session_state.get("auto_running"):
        return
    state = st.session_state.state
    if not state:
        st.session_state.auto_running = False
        return

    st.session_state.auto_steps = st.session_state.get("auto_steps", 0) + 1
    if st.session_state.auto_steps > 400:
        st.session_state.auto_running = False
        st.session_state.error = "自动讨论步数过多，已自动停止。"
        return

    time.sleep(0.35)

    if state.state == "ROUND_END":
        _handle_summarize(secrets)
        return
    if state.state == "DONE":
        st.session_state.auto_running = False
        return
    if state.state == "A_WAIT_CONFIRM":
        _handle_confirm(secrets, "A")
        return
    if state.state == "B_WAIT_CONFIRM":
        _handle_confirm(secrets, "B")
        return
    if engine.can("speak_a", state):
        _handle_speak(secrets, "A")
        return
    if engine.can("speak_b", state):
        _handle_speak(secrets, "B")
        return
    st.session_state.auto_running = False


def _handle_export():
    state = st.session_state.state
    if not state:
        return
    export_dir, transcript_path, score_path, config_path = STORE.export(
        st.session_state.debate_id, state
    )
    st.success(f"已导出至 {export_dir}")
    conclusion_path = export_dir / "conclusion.json"
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with open(transcript_path, "rb") as f:
            st.download_button(
                "下载讨论记录",
                f,
                file_name="transcript.jsonl",
                mime="application/jsonlines+json",
                key="dl_transcript",
            )
    with col2:
        with open(score_path, "rb") as f:
            st.download_button(
                "下载结果",
                f,
                file_name="score.json",
                mime="application/json",
                key="dl_score",
            )
    with col3:
        with open(config_path, "rb") as f:
            st.download_button(
                "下载配置快照",
                f,
                file_name="config_snapshot.json",
                mime="application/json",
                key="dl_config",
            )
    with col4:
        with open(conclusion_path, "rb") as f:
            st.download_button(
                "下载结论",
                f,
                file_name="conclusion.json",
                mime="application/json",
                key="dl_conclusion",
            )


if __name__ == "__main__":
    main()
