"""状态机与核心流程：can()/transition()/speak()/confirm()/reject()/score()。"""

import copy
import uuid
from typing import Any

import prompts
from config import estimate_cost_cny, json_output_max_tokens, resolve_config
from memory import (
    Conclusion,
    DebateConfig,
    DebateState,
    Message,
    build_prompt_for,
    compute_overrun,
    update_after_confirmed,
)
from providers import NormalizedMessage, ProviderEmptyContent
from scoring import score_debate


class IllegalTransition(Exception):
    """非法状态转换。"""

    pass


def create_initial_state(
    debate_id: str,
    topic: str,
    pro_side: str,
    con_side: str,
    ui_config: dict[str, Any] | None = None,
    secrets: dict[str, Any] | None = None,
    user_context: str = "",
    user_context_soft: str = "",
) -> DebateState:
    """用三层配置构造初始状态。"""
    cfg = resolve_config(ui_config, secrets)
    config = DebateConfig(**cfg)
    state = DebateState(
        debate_id=debate_id,
        topic=topic,
        pro_side=pro_side,
        con_side=con_side,
        config=config,
    )
    state.stance_a = pro_side
    state.stance_b = con_side
    state.user_context = (user_context or "").strip()
    state.user_context_soft = (user_context_soft or "").strip()
    # 初始系统提示作为第 0 轮记录，方便恢复时了解配置
    state.messages.append(
        Message(
            debate_id=debate_id,
            round_no=0,
            speaker="SYSTEM",
            side="none",
            status="note",
            content=f"讨论初始化：{topic} | A 方视角：{pro_side} | B 方视角：{con_side} | "
            f"word_limit={config.word_limit}, max_rounds={config.max_rounds}, "
            f"mock_mode={config.mock_mode}",
        )
    )
    return state


def _count_drafts(state: DebateState, side: str) -> int:
    """统计本回合某方已生成的草稿数（含 draft/confirmed/rejected/rejected_twice）。"""
    return len(
        [
            m
            for m in state.messages
            if m.speaker == side
            and m.round_no == state.round_no
            and m.status in ("draft", "confirmed", "rejected", "rejected_twice")
        ]
    )


def _find_current_draft(state: DebateState, side: str) -> Message | None:
    """找到当前待确认的草稿。"""
    for m in reversed(state.messages):
        if m.speaker == side and m.status == "draft":
            return m
    return None


def _count_rejected(state: DebateState, side: str) -> int:
    """统计本回合某方已被驳回的草稿数。"""
    return len(
        [
            m
            for m in state.messages
            if m.speaker == side
            and m.round_no == state.round_no
            and m.status == "rejected"
        ]
    )


def _phase_for_round(round_no: int, max_rounds: int) -> str:
    """按回合判定阶段：open / mid / close。单轮讨论优先 close。"""
    if max_rounds == 1:
        return "close"
    if round_no == 1:
        return "open"
    if round_no == max_rounds:
        return "close"
    return "mid"


def _should_technical_retry(dirty_reason: str) -> bool:
    """只有技术截断/空稿才触发自动技术重试。"""
    if not dirty_reason:
        return False
    if dirty_reason in ("empty", "stripped_empty", "length"):
        return True
    if dirty_reason.startswith("finalizer_failed:"):
        sub = dirty_reason.split(":", 1)[1]
        return sub in ("empty", "stripped_empty", "length")
    return False


def _round_had_progress(state: DebateState) -> bool:
    """判断当前回合是否产生了新的开放观点或待回应事项。"""
    r = state.round_no
    for c in state.claims:
        if c.round_created == r and c.status in ("open", "contested"):
            return True
    for o in state.obligations:
        if o.round_created == r and o.status == "open":
            return True
    return False


def _round_end_or_advance(state: DebateState) -> None:
    """B 方回合结束后：判定收敛早停或进入下一轮。"""
    if _round_had_progress(state):
        state.no_progress_rounds = 0
    else:
        state.no_progress_rounds += 1

    if state.converged:
        state.state = "ROUND_END"
        state.messages.append(
            Message(
                debate_id=state.debate_id,
                round_no=state.round_no,
                speaker="SYSTEM",
                side="none",
                status="note",
                content="双方已就核心问题达成一致，讨论结束，可生成结论。",
            )
        )
    elif (
        state.config.early_stop
        and state.no_progress_rounds >= 2
        and state.round_no < state.config.max_rounds
    ):
        state.state = "ROUND_END"
        state.messages.append(
            Message(
                debate_id=state.debate_id,
                round_no=state.round_no,
                speaker="SYSTEM",
                side="none",
                status="note",
                content="双方已连续两轮未提出新观点或新问题，讨论趋于收敛，提前进入结论阶段。",
            )
        )
    elif state.round_no >= state.config.max_rounds:
        state.state = "ROUND_END"
    else:
        state.state = "A_SPEAKING"
        state.round_no += 1


def can(action: str, state: DebateState) -> bool:
    """判断在指定状态下能否执行 action。"""
    s = state.state
    if action == "start":
        return s == "IDLE"
    if action == "speak_a":
        return s == "A_SPEAKING" and _count_drafts(state, "A") < 2
    if action == "confirm_a":
        return s == "A_WAIT_CONFIRM" and _find_current_draft(state, "A") is not None
    if action == "reject_a":
        return s == "A_WAIT_CONFIRM" and _find_current_draft(state, "A") is not None
    if action == "speak_b":
        return s == "B_SPEAKING" and _count_drafts(state, "B") < 2
    if action == "confirm_b":
        return s == "B_WAIT_CONFIRM" and _find_current_draft(state, "B") is not None
    if action == "reject_b":
        return s == "B_WAIT_CONFIRM" and _find_current_draft(state, "B") is not None
    if action == "add_note":
        return s not in ("IDLE", "DONE")
    if action == "summarize":
        return s in ("ROUND_END", "DONE") and state.conclusion is None
    if action == "score":
        return s in ("ROUND_END", "DONE") and state.score is None
    if action == "export":
        return True
    return False


def transition(state: DebateState, action: str, **kwargs: Any) -> DebateState:
    """执行状态转换；不合法的转换抛 IllegalTransition。"""
    if not can(action, state):
        raise IllegalTransition(f"不能在状态 {state.state} 执行 {action}")

    if action == "start":
        state.state = "A_SPEAKING"
    elif action in ("speak_a", "speak_b"):
        # speak() 函数负责生成草稿并设置 WAIT_CONFIRM
        pass
    elif action == "confirm_a":
        state.state = "B_SPEAKING"
    elif action == "confirm_b":
        if state.round_no >= state.config.max_rounds:
            state.state = "ROUND_END"
        else:
            state.state = "A_SPEAKING"
            state.round_no += 1
    elif action in ("reject_a", "reject_b"):
        # 实际处理在 reject_side() 中；这里只做兜底
        pass
    elif action == "add_note":
        pass
    elif action == "summarize":
        state.state = "SUMMARIZING"
    elif action == "score":
        state.state = "SCORING"
    elif action == "export":
        pass
    return state


def speak(state: DebateState, side: str, provider) -> tuple[Message, DebateState]:
    """调用模型生成草稿，返回 draft Message 与新状态。

    处理流程：
    1. 按回合计算 phase，按 word_limit/phase/config 计算足量 max_tokens；
    2. provider 内部剥离 <think> 等标签，提取 reasoning 到独立字段；
    3. 若 dirty 属技术截断/空稿，自动技术重试 1 次（reasoning_budget +512，不超 hard_cap）；
    4. 仍 dirty 则自动标 rejected，由 engine 按二次驳回规则处理；
    5. 干净内容才作为 draft 进入 WAIT_CONFIRM。
    """
    if side not in ("A", "B"):
        raise ValueError("side 必须是 A 或 B")
    action = f"speak_{side.lower()}"
    if not can(action, state):
        raise IllegalTransition(f"不能在状态 {state.state} 让 {side} 发言")

    # 先进入 SPEAKING（幂等，若已经是则不变）
    state.state = f"{side}_SPEAKING"

    debate_id = ""
    for m in state.messages:
        if m.debate_id:
            debate_id = m.debate_id
            break

    messages = build_prompt_for(state, side)
    phase = _phase_for_round(state.round_no, state.config.max_rounds)

    norm: NormalizedMessage
    latency = 0
    usage_in = 0
    usage_out = 0

    try:
        norm, latency, usage_in, usage_out = provider.chat_text(
            messages,
            word_limit=state.config.word_limit,
            phase=phase,
            config=state.config,
        )
    except Exception as e:
        norm = NormalizedMessage(
            content=f"【调用失败】{e}",
            dirty_reason="exception",
        )
        latency = 0
        usage_in = 0
        usage_out = 0

    # 技术重试：只针对空稿/标签剥空/length 截断；不计入裁判驳回
    if _should_technical_retry(norm.dirty_reason):
        retry_config = copy.copy(state.config)
        retry_config.reasoning_budget = min(
            retry_config.reasoning_budget + 512, retry_config.hard_cap
        )
        retry_config.reasoning_budget_open_close = min(
            retry_config.reasoning_budget_open_close + 512, retry_config.hard_cap
        )
        try:
            norm2, lat2, in2, out2 = provider.chat_text(
                messages,
                word_limit=state.config.word_limit,
                phase=phase,
                config=retry_config,
            )
            latency += lat2
            usage_in += in2
            usage_out += out2
            if not norm2.dirty_reason:
                norm = norm2
        except Exception:
            pass

    msg = Message(
        debate_id=debate_id,
        round_no=state.round_no,
        speaker=side,
        side="pro" if side == "A" else "con",
        status="draft",
        content=norm.content,
        reasoning=norm.reasoning or "",
        model=getattr(provider, "model", "unknown"),
        temperature=getattr(provider, "temperature", 0.0),
        latency_ms=latency,
        usage_in=usage_in,
        usage_out=usage_out,
        finish_reason=norm.finish_reason or "",
        reasoning_tokens=norm.reasoning_tokens or 0,
        finalizer_triggered=norm.finalizer_triggered,
        dirty_reason=norm.dirty_reason,
    )
    state.messages.append(msg)
    state.totals["usage_in"] += usage_in
    state.totals["usage_out"] += usage_out
    _update_cost(state)

    if norm.dirty_reason:
        # 自动驳回 dirty 草稿，走现有二次驳回 -> DODGE 规则
        msg.status = "rejected"
        state = _auto_reject(state, side, msg)
    else:
        if side == "A":
            state.state = "A_WAIT_CONFIRM"
        else:
            state.state = "B_WAIT_CONFIRM"
    return msg, state


def _auto_reject(state: DebateState, side: str, draft: Message) -> DebateState:
    """模型输出 dirty 时自动驳回；第二次自动驳回触发 DODGE。

    技术重试不计入裁判驳回，但这里的自动标 rejected 会计入本回合失败次数。
    """
    prior_rejected = _count_rejected(state, side)
    if prior_rejected >= 1:
        # 本轮第二次失败：DODGE
        draft.status = "rejected_twice"
        state.messages.append(
            Message(
                debate_id=draft.debate_id,
                round_no=state.round_no,
                speaker="SYSTEM",
                side="none",
                status="dodge",
                content=f"{side} 方两次草稿未通过，本轮视为放弃/未有效回应，记 DODGE。",
            )
        )
        if side == "A":
            state.state = "B_SPEAKING"
        else:
            _round_end_or_advance(state)
    else:
        draft.status = "rejected"
        state.state = f"{side}_SPEAKING"
    return state


def confirm_a(state: DebateState, provider) -> DebateState:
    """确认 A 草稿，更新记忆，进入 B_SPEAKING。"""
    if not can("confirm_a", state):
        raise IllegalTransition("当前不能确认 A")
    draft = _find_current_draft(state, "A")
    if not draft:
        raise IllegalTransition("没有 A 的待确认草稿")
    draft.status = "confirmed"
    state = update_after_confirmed(state, draft, provider)
    state.state = "B_SPEAKING"
    return state


def confirm_b(state: DebateState, provider) -> DebateState:
    """确认 B 草稿，更新记忆，进入下一回合或 ROUND_END。"""
    if not can("confirm_b", state):
        raise IllegalTransition("当前不能确认 B")
    draft = _find_current_draft(state, "B")
    if not draft:
        raise IllegalTransition("没有 B 的待确认草稿")
    draft.status = "confirmed"
    state = update_after_confirmed(state, draft, provider)
    _round_end_or_advance(state)
    return state


def reject_a(state: DebateState) -> DebateState:
    """驳回 A 草稿；第二次驳回触发 DODGE。"""
    if not can("reject_a", state):
        raise IllegalTransition("当前不能驳回 A")
    return _reject_side(state, "A")


def reject_b(state: DebateState) -> DebateState:
    """驳回 B 草稿；第二次驳回触发 DODGE。"""
    if not can("reject_b", state):
        raise IllegalTransition("当前不能驳回 B")
    return _reject_side(state, "B")


def _reject_side(state: DebateState, side: str) -> DebateState:
    draft = _find_current_draft(state, side)
    if not draft:
        raise IllegalTransition(f"没有 {side} 的待确认草稿")

    debate_id = ""
    for m in state.messages:
        if m.debate_id:
            debate_id = m.debate_id
            break

    if _count_rejected(state, side) >= 1:
        # 第二次驳回：DODGE
        draft.status = "rejected_twice"
        state.messages.append(
            Message(
                debate_id=debate_id,
                round_no=state.round_no,
                speaker="SYSTEM",
                side="none",
                status="dodge",
                content=f"{side} 方两次草稿未通过，本轮视为放弃/未有效回应，记 DODGE。",
            )
        )
        # 按配置处理；forced_confirm 当前等同 skip，不会绕过裁判确认，仅为未来扩展占位
        if state.config.forfeit_policy == "forced_confirm":
            pass
        if side == "A":
            state.state = "B_SPEAKING"
        else:
            _round_end_or_advance(state)
    else:
        # 第一次驳回：允许重写
        draft.status = "rejected"
        state.state = f"{side}_SPEAKING"
    return state


def add_referee_note(
    state: DebateState,
    debate_id: str,
    note_text: str,
    is_public: bool = True,
) -> DebateState:
    """插入裁判备注。"""
    if not can("add_note", state):
        raise IllegalTransition(f"不能在状态 {state.state} 插入备注")
    state.messages.append(
        Message(
            debate_id=debate_id,
            round_no=state.round_no,
            speaker="REFEREE",
            side="none",
            status="note",
            content=note_text,
            judge_note=note_text if is_public else "",
        )
    )
    return state


def score(state: DebateState, judge_provider) -> DebateState:
    """调用 JUDGE_AI 评分，合并人类裁判分，进入 DONE。"""
    if not can("score", state):
        raise IllegalTransition("当前不能评分")
    state.state = "SCORING"
    score_result, state = score_debate(state, judge_provider)
    state.score = score_result
    state.state = "DONE"
    _update_cost(state)
    return state


def summarize(state: DebateState, provider, max_retries: int = 2) -> DebateState:
    """调用第三 AI 生成讨论结论；失败重试，全部失败则回到 ROUND_END 供重试。"""
    if not can("summarize", state):
        raise IllegalTransition("当前不能生成结论")
    state.state = "SUMMARIZING"

    prompt_text = prompts.build_conclusion_prompt(state)
    messages = [
        {"role": "system", "content": "你是中立的讨论主持人，只输出合法 JSON。"},
        {"role": "user", "content": prompt_text},
    ]
    max_tokens = json_output_max_tokens(state.config)

    delta = None
    usage_in = 0
    usage_out = 0
    last_err = ""
    for _attempt in range(max_retries + 1):
        try:
            d, latency, uin, uout = provider.chat_json(
                messages, max_tokens=max_tokens, temperature=0.3, schema_hint="conclusion"
            )
            usage_in += uin
            usage_out += uout
            if isinstance(d, dict) and (
                d.get("takeaway") or d.get("consensus") or d.get("disagreements") or d.get("open_questions")
            ):
                delta = d
                break
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            continue

    state.totals["usage_in"] += usage_in
    state.totals["usage_out"] += usage_out

    if delta is None:
        state.state = "ROUND_END"
        state.conclusion_error = (
            f"结论生成失败（已重试 {max_retries} 次）：{last_err or '返回内容为空'}"
        )
        state.messages.append(
            Message(
                debate_id=state.debate_id,
                round_no=state.round_no,
                speaker="SYSTEM",
                side="none",
                status="note",
                content=state.conclusion_error,
            )
        )
        _update_cost(state)
        return state

    state.conclusion = Conclusion(
        consensus=[str(x) for x in (delta.get("consensus") or [])],
        disagreements=[str(x) for x in (delta.get("disagreements") or [])],
        open_questions=[str(x) for x in (delta.get("open_questions") or [])],
        takeaway=str(delta.get("takeaway") or ""),
        aligned=bool(delta.get("aligned")),
    )
    state.conclusion_error = ""
    state.state = "DONE"
    _update_cost(state)
    return state


def _update_cost(state: DebateState) -> None:
    """粗略估算成本（人民币）。"""
    # 仅作展示用；单价可在 secrets 的 price_in_per_1m / price_out_per_1m 覆盖
    state.totals["cost_cny"] = estimate_cost_cny(
        state.totals["usage_in"],
        state.totals["usage_out"],
        state.config.price_in_per_1m,
        state.config.price_out_per_1m,
    )


def config_change_note(state: DebateState, debate_id: str, changed: dict[str, Any]) -> DebateState:
    """本场配置被修改时记录 SYSTEM note。"""
    if not changed:
        return state
    parts = [f"{k}={v}" for k, v in changed.items()]
    state.messages.append(
        Message(
            debate_id=debate_id,
            round_no=state.round_no,
            speaker="SYSTEM",
            side="none",
            status="note",
            content=f"从第 {state.round_no} 轮起，配置变更：{', '.join(parts)}",
        )
    )
    return state
