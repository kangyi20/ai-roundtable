"""JUDGE_AI 评分、JSON 提取、重试、与人类裁判分合并。"""

from config import json_output_max_tokens
from memory import DebateState, ScoreResult
import prompts


DIMENSIONS = ("logic", "evidence", "rebuttal", "clarity")


def _clamp(n: int, lo: int = 0, hi: int = 10) -> int:
    return max(lo, min(hi, n))


def _avg(scores: dict[str, int]) -> float:
    vals = [scores.get(d, 0) for d in DIMENSIONS]
    return sum(vals) / len(vals) if vals else 0.0


def _validate_score(raw: dict) -> dict | None:
    """验证并清洗 AI 返回的评分字典。"""
    if not raw:
        return None
    cleaned: dict = {"A": {}, "B": {}, "winner": "tie", "reason": ""}
    for side in ("A", "B"):
        side_raw = raw.get(side)
        if not isinstance(side_raw, dict):
            return None
        for dim in DIMENSIONS:
            try:
                cleaned[side][dim] = _clamp(int(side_raw.get(dim, 0)))
            except (TypeError, ValueError):
                return None
    winner = raw.get("winner", "tie")
    if winner not in ("A", "B", "tie"):
        winner = "tie"
    cleaned["winner"] = winner
    reason = raw.get("reason", "")
    if isinstance(reason, str):
        cleaned["reason"] = reason[:120]
    return cleaned


def score_debate(state: DebateState, judge_provider, max_retries: int = 2) -> tuple[ScoreResult, DebateState]:
    """调用 JUDGE_AI 评分，失败重试，最终与人类裁判分合并。"""
    prompt = prompts.build_judge_prompt(state)
    messages = [
        {"role": "system", "content": "你是公正的讨论评估员，只输出合法 JSON。"},
        {"role": "user", "content": prompt},
    ]
    max_tokens = json_output_max_tokens(state.config)

    ai_raw: dict | None = None
    judge_ai_invalid = False
    usage_in = 0
    usage_out = 0

    for attempt in range(max_retries + 1):
        try:
            parsed, latency, u_in, u_out = judge_provider.chat_json(
                messages, max_tokens=max_tokens, temperature=0.0, schema_hint="score"
            )
            usage_in += u_in
            usage_out += u_out
            cleaned = _validate_score(parsed)
            if cleaned:
                ai_raw = cleaned
                break
        except Exception:
            # 继续重试
            continue

    if ai_raw is None:
        judge_ai_invalid = True
        ai_raw = {
            "A": {d: 5 for d in DIMENSIONS},
            "B": {d: 5 for d in DIMENSIONS},
            "winner": "tie",
            "reason": "AI 评分返回格式无效，已使用默认中分。",
        }

    state.totals["usage_in"] += usage_in
    state.totals["usage_out"] += usage_out

    # 合并人类裁判分
    ai_a_avg = _avg(ai_raw["A"])
    ai_b_avg = _avg(ai_raw["B"])
    judge_a_avg = _avg(state.judge_scores.get("A", {}))
    judge_b_avg = _avg(state.judge_scores.get("B", {}))

    # 若人类裁判未打分，则只使用 AI 分
    if not state.judge_scores.get("A") and not state.judge_scores.get("B"):
        final_a = ai_a_avg
        final_b = ai_b_avg
    else:
        wj = state.config.weight_judge
        wa = state.config.weight_ai
        final_a = wj * judge_a_avg + wa * ai_a_avg
        final_b = wj * judge_b_avg + wa * ai_b_avg

    if abs(final_a - final_b) < 0.2:
        winner = "tie"
    else:
        winner = "A" if final_a > final_b else "B"

    score_result = ScoreResult(
        A=ai_raw["A"],
        B=ai_raw["B"],
        final={"A": final_a, "B": final_b},
        winner=winner,
        reason=ai_raw.get("reason", ""),
        judge_ai_invalid=judge_ai_invalid,
    )

    return score_result, state
