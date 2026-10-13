"""数据模型、提示词组装、记忆更新与 token 估算。"""

import json
import re
import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

import prompts
from config import json_output_max_tokens


class Message(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    debate_id: str = ""
    round_no: int = 1
    speaker: Literal["A", "B", "REFEREE", "SYSTEM"] = "A"
    side: Literal["pro", "con", "none"] = "none"
    status: Literal["draft", "confirmed", "rejected", "rejected_twice", "note", "dodge"] = "draft"
    content: str = ""
    reasoning: str = ""  # 思考过程，不落公共 transcript，仅用于 finalizer/调试
    judge_note: str = ""
    model: str = ""
    temperature: float = 0.0
    latency_ms: int = 0
    usage_in: int = 0
    usage_out: int = 0
    finish_reason: str = ""  # 模型 finish_reason，用于审计
    reasoning_tokens: int = 0  # 模型 reasoning tokens（若有）
    finalizer_triggered: bool = False  # 是否触发 finalizer
    dirty_reason: str = ""  # dirty 原因，用于审计
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    owner: Literal["A", "B", "none"] = "none"
    round_created: int = 1
    text: str = ""
    status: Literal["open", "contested", "conceded", "rejected", "closed"] = "open"
    evidence_state: Literal["none", "weak", "sourced"] = "none"
    owed_by: Literal["A", "B", "none"] = "none"
    due: Literal["next", "none"] = "none"
    attacks: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    quote_ptr: Optional[str] = None
    summary: Optional[str] = None


class Obligation(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    owner: Literal["A", "B", "none"] = "none"  # 谁需要回应
    text: str = ""
    round_created: int = 1
    status: Literal["open", "settled", "dodged"] = "open"
    source_claim_id: Optional[str] = None


class DebateConfig(BaseModel):
    word_limit: int
    max_rounds: int
    input_budget: int
    output_max_tokens: int
    final_token_ratio: float = 3.5
    reasoning_budget: int = 2048
    reasoning_budget_open_close: int = 4096
    final_token_floor: int = 1024
    hard_cap: int = 8192
    finalizer_on_empty: bool = True
    keep_reasoning_private: bool = True
    reasoning_heuristic: bool = True
    debug_raw_response: bool = False
    weight_judge: float
    weight_ai: float
    strict_evidence: bool = False
    mock_mode: bool = True
    forfeit_policy: Literal["skip", "forced_confirm"] = "skip"
    overrun_mode: Literal["truncate", "penalty", "reject"] = "penalty"
    early_stop: bool = True
    price_in_per_1m: float = 2.0
    price_out_per_1m: float = 8.0


class ScoreResult(BaseModel):
    A: dict[str, int] = Field(default_factory=dict)
    B: dict[str, int] = Field(default_factory=dict)
    final: dict[str, float] = Field(default_factory=dict)
    winner: Literal["A", "B", "tie"] = "tie"
    reason: str = ""
    judge_ai_invalid: bool = False


class Conclusion(BaseModel):
    """讨论结束时由第三 AI 生成的结构化结论。"""

    consensus: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    takeaway: str = ""
    aligned: bool = False  # 双方最终是否达成一致


class DebateState(BaseModel):
    state: Literal[
        "IDLE",
        "A_SPEAKING",
        "A_WAIT_CONFIRM",
        "B_SPEAKING",
        "B_WAIT_CONFIRM",
        "ROUND_END",
        "SUMMARIZING",
        "SCORING",
        "DONE",
    ] = "IDLE"
    debate_id: str = ""
    round_no: int = 1
    topic: str = ""
    pro_side: str = ""
    con_side: str = ""
    user_context: str = ""  # 注入讨论用的处境说明（L2，含画像信息，**不对外暴露**）
    user_context_soft: str = ""  # 给提问者本人看的体贴版（L3，可对外展示）
    qa: list[dict[str, Any]] = Field(default_factory=list)  # 讨论后向裁判追问的多轮记录
    config: DebateConfig
    messages: list[Message] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)
    summary: str = ""
    totals: dict[str, Any] = Field(
        default_factory=lambda: {"usage_in": 0, "usage_out": 0, "cost_cny": 0.0}
    )
    score: Optional[ScoreResult] = None
    conclusion: Optional[Conclusion] = None
    conclusion_error: str = ""
    no_progress_rounds: int = 0
    stance_a: str = ""  # A 方当前立场（初始=pro_side，可被说服后漂移）
    stance_b: str = ""  # B 方当前立场（初始=con_side，可被说服后漂移）
    converged: bool = False  # 双方是否已就核心问题达成一致
    judge_scores: dict[str, dict[str, int]] = Field(default_factory=lambda: {"A": {}, "B": {}})
    rewrite_counts: dict[str, int] = Field(default_factory=lambda: {"A": 0, "B": 0})


def estimate_tokens(text: str) -> int:
    """优先 tiktoken，否则按中文字符粗略估算。"""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text) // 1.6)


def _public_confirmed_messages(state: DebateState) -> list[Message]:
    """仅返回已确认或系统/裁判备注的公共记录。"""
    return [m for m in state.messages if m.status in ("confirmed", "note", "dodge")]


def _last_opponent_confirmed(state: DebateState, side: str) -> tuple[str, str]:
    """返回（对方最近一条 confirmed 全文，裁判备注）。"""
    opponent = "B" if side == "A" else "A"
    confirmed = [m for m in state.messages if m.speaker == opponent and m.status == "confirmed"]
    last = confirmed[-1].content if confirmed else "（暂无对方已确认发言）"
    # 最近一条公开裁判备注
    notes = [
        m.judge_note
        for m in state.messages
        if m.speaker == "REFEREE" and m.status == "note" and m.judge_note
    ]
    note = notes[-1] if notes else ""
    return last, note


def _format_claims(state: DebateState) -> str:
    """open<=6，closed 摘要<=3；单条文本截断至 40 字，控制 prompt 增长。"""
    open_claims = [c for c in state.claims if c.status == "open"][:6]
    closed_claims = [c for c in state.claims if c.status == "closed"][:3]
    lines = []
    if open_claims:
        lines.append("【双方观点】")
        for c in open_claims:
            text = c.text[:40] + "..." if len(c.text) > 40 else c.text
            lines.append(f"- [{c.owner}] {text}")
    if closed_claims:
        lines.append("【已讨论清楚的观点】")
        for c in closed_claims:
            summary = (c.summary or c.text)[:25]
            lines.append(f"- [{c.owner}] {summary}")
    return "\n".join(lines) if lines else "（暂无）"


def _format_obligations(state: DebateState) -> str:
    """待回应事项只展示最近 4 条；单条文本截断至 35 字。"""
    open_obligations = [o for o in state.obligations if o.status == "open"][-4:]
    if not open_obligations:
        return "（无待回应事项）"
    return "\n".join(
        f"- [{o.owner}] {o.text[:35]}{'...' if len(o.text) > 35 else ''}"
        for o in open_obligations
    )


def _format_contradictions(state: DebateState) -> str:
    if not state.claims:
        return "（暂无）"
    pairs = []
    for c in state.claims:
        if c.contradicts:
            for target_id in c.contradicts:
                pairs.append(f"- [{c.owner}] {c.text[:50]} ↔ {target_id}")
    return "\n".join(pairs) if pairs else "（暂无）"


def _detect_quote_need(text: str) -> bool:
    """检测是否提到旧论。"""
    keywords = [r"之前", r"刚才", r"第\s*\d+\s*轮", r"承认", r"说过", r"此前"]
    return any(re.search(kw, text) for kw in keywords)


def _fetch_quotes(state: DebateState, side: str, max_quotes: int = 3, max_len: int = 120) -> str:
    """从已确认旧论中取最多3条短片段。"""
    opponent = "B" if side == "A" else "A"
    old = [
        m
        for m in state.messages
        if m.speaker == opponent and m.status == "confirmed"
    ]
    quotes = []
    for m in old[-5:]:  # 只看最近5条，避免太远
        for sent in re.split(r"[。！？；]", m.content):
            if 20 <= len(sent) <= max_len:
                quotes.append(sent)
                if len(quotes) >= max_quotes:
                    break
        if len(quotes) >= max_quotes:
            break
    if not quotes:
        return ""
    return "\n".join(f"- {q}" for q in quotes[:max_quotes])


def build_prompt_for(state: DebateState, side: str) -> list[dict[str, str]]:
    """为指定讨论者构造 API messages；超预算时自动降级。"""
    opponent_last, judge_note = _last_opponent_confirmed(state, side)
    claims_text = _format_claims(state)
    obligations_text = _format_obligations(state)
    contradictions_text = _format_contradictions(state)

    # 按需引用
    quotes_text = ""
    if obligations_text != "（无待回应事项）" or _detect_quote_need(opponent_last):
        quotes_text = _fetch_quotes(state, side)

    system_prompt = prompts.build_debater_system_prompt(state, side)
    user_prompt = prompts.build_debater_user_prompt(
        state,
        side,
        opponent_last,
        judge_note,
        claims_text,
        obligations_text,
        contradictions_text,
        quotes_text,
        state.summary or "（暂无摘要）",
    )

    full_text = system_prompt + user_prompt
    tokens = estimate_tokens(full_text)

    # 超预算降级
    if tokens > state.config.input_budget:
        open_claims = [c for c in state.claims if c.status == "open"][:5]
        claims_text = "\n".join(
            f"- [{c.owner}] {c.text}" for c in open_claims
        ) or "（暂无）"
        user_prompt = prompts.build_debater_user_prompt(
            state,
            side,
            opponent_last,
            judge_note,
            claims_text,
            obligations_text,
            "（已省略）",
            "",
            "（已省略）",
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def update_after_confirmed(
    state: DebateState, message: Message, provider
) -> DebateState:
    """裁判确认后，调用记忆写入器更新 claims/obligations/summary。"""
    prompt = prompts.build_memory_writer_prompt(state, message)
    messages = [
        {"role": "system", "content": "你是中立的讨论记录员，只输出合法 JSON。"},
        {"role": "user", "content": prompt},
    ]
    max_tokens = json_output_max_tokens(state.config)
    try:
        delta, latency, usage_in, usage_out = provider.chat_json(
            messages, max_tokens=max_tokens, temperature=0.2, schema_hint="memory"
        )
    except Exception as e:
        delta = {}
        latency = 0
        usage_in = estimate_tokens(prompt)
        usage_out = 0
        state.messages.append(
            Message(
                debate_id=message.debate_id,
                round_no=state.round_no,
                speaker="SYSTEM",
                side="none",
                status="note",
                content=f"记忆写入器调用失败：{e}",
                latency_ms=latency,
                usage_in=usage_in,
                usage_out=usage_out,
            )
        )
        return state

    state.totals["usage_in"] += usage_in
    state.totals["usage_out"] += usage_out

    try:
        _apply_memory_delta(state, message, delta)
    except Exception as e:
        state.messages.append(
            Message(
                debate_id=message.debate_id,
                round_no=state.round_no,
                speaker="SYSTEM",
                side="none",
                status="note",
                content=f"记忆解析失败，已跳过：{e}",
            )
        )
    return state


def _apply_memory_delta(state: DebateState, message: Message, delta: dict[str, Any]) -> None:
    """把模型返回的 delta 应用到状态。"""
    speaker = message.speaker
    # 新主张
    for item in delta.get("new_claims", []):
        claim = Claim(
            owner=speaker,
            round_created=state.round_no,
            text=item.get("text", ""),
            evidence_state=item.get("evidence_state", "none") or "none",
            quote_ptr=item.get("quote_ptr"),
        )
        state.claims.append(claim)

    # 新义务
    for item in delta.get("new_obligations", []):
        ob = Obligation(
            owner=item.get("owed_by", "B" if speaker == "A" else "A"),
            text=item.get("text", ""),
            round_created=state.round_no,
        )
        state.obligations.append(ob)

    # 攻击：尝试匹配 target_id
    for item in delta.get("attacks", []):
        target_id = item.get("target_id", "")
        text = item.get("text", "")
        for c in state.claims:
            if c.id == target_id or c.text[:30] in text:
                c.attacks.append(text)
                c.status = "contested"
                # 被攻击方产生义务
                owed = "B" if c.owner == "A" else "A"
                state.obligations.append(
                    Obligation(
                        owner=owed,
                        text=f"回应对方的不同意见：{text[:60]}",
                        round_created=state.round_no,
                        source_claim_id=c.id,
                    )
                )

    # 承认
    for item in delta.get("concessions", []):
        target_id = item.get("target_id", "")
        for c in state.claims:
            if c.id == target_id:
                c.status = "conceded"

    # 矛盾
    for item in delta.get("contradictions", []):
        text = item.get("text", "")
        ca_id = item.get("claim_a_id", "")
        cb_id = item.get("claim_b_id", "")
        for c in state.claims:
            if c.id == ca_id:
                c.contradicts.append(cb_id or text)

    # 关闭主张
    for item in delta.get("close_claims", []):
        cid = item.get("claim_id", "")
        reason = item.get("reason", "")
        for c in state.claims:
            if c.id == cid:
                c.status = "closed"
                c.summary = reason or c.text[:40]

    # 立场转变 / 被说服
    stance_change = delta.get("stance_change")
    if isinstance(stance_change, str) and stance_change.strip():
        new_stance = stance_change.strip()
        if speaker == "A":
            old = state.stance_a or state.pro_side
            state.stance_a = new_stance
        else:
            old = state.stance_b or state.con_side
            state.stance_b = new_stance
        state.messages.append(
            Message(
                debate_id=message.debate_id,
                round_no=state.round_no,
                speaker="SYSTEM",
                side="none",
                status="note",
                content=f"{speaker} 方立场发生转变：由「{old}」转向「{new_stance}」。",
            )
        )

    # 达成一致
    if delta.get("converged"):
        state.converged = True
        state.messages.append(
            Message(
                debate_id=message.debate_id,
                round_no=state.round_no,
                speaker="SYSTEM",
                side="none",
                status="note",
                content="双方已就核心问题达成一致。",
            )
        )

    # 更新摘要
    open_count = len([c for c in state.claims if c.status == "open"])
    contested_count = len([c for c in state.claims if c.status == "contested"])
    state.summary = (
        f"第{state.round_no}轮结束后："
        f"{speaker}发言；"
        f"双方观点{open_count}条，存在不同意见{contested_count}条，"
        f"待回应事项{len([o for o in state.obligations if o.status == 'open'])}条。"
    )

    # 自动结算本轮产生的义务：如果同轮已经回应则标 settled
    for ob in state.obligations:
        if ob.round_created == state.round_no and ob.owner == speaker:
            ob.status = "settled"


def compute_overrun(state: DebateState, content: str) -> int:
    """计算超字数；中文字符按字计数。"""
    # 简单按字符数（不含标点和空格可进一步细化，但 MVP 用字符数）
    return max(0, len(content) - state.config.word_limit)
