"""配置解析与默认值。"""

import os
from typing import Any

DEFAULTS: dict[str, Any] = {
    "word_limit": 300,
    "max_rounds": 6,
    "input_budget": 3500,
    "output_max_tokens": 0,  # 0 表示自动推导
    "final_token_ratio": 3.5,
    "reasoning_budget": 2048,
    "reasoning_budget_open_close": 4096,
    "final_token_floor": 1024,
    "hard_cap": 8192,
    "finalizer_on_empty": True,
    "keep_reasoning_private": True,
    "reasoning_heuristic": True,
    "debug_raw_response": False,
    "weight_judge": 0.6,
    "weight_ai": 0.4,
    "strict_evidence": False,
    "mock_mode": True,
    "forfeit_policy": "skip",  # skip | forced_confirm（forced_confirm 当前等同 skip，仅为未来扩展占位）
    "overrun_mode": "penalty",  # truncate | penalty | reject
    "early_stop": True,  # 连续无新内容时自动收敛，提前进入结论阶段
    "price_in_per_1m": 2.0,  # 输入价格（¥/百万 token，估算用，可在 secrets 覆盖）
    "price_out_per_1m": 8.0,  # 输出价格（¥/百万 token，估算用，可在 secrets 覆盖）
}

# 环境变量名 -> 配置键映射
_ENV_MAP: dict[str, str] = {
    "AI_DEBATE_WORD_LIMIT": "word_limit",
    "AI_DEBATE_MAX_ROUNDS": "max_rounds",
    "AI_DEBATE_INPUT_BUDGET": "input_budget",
    "AI_DEBATE_OUTPUT_MAX_TOKENS": "output_max_tokens",
    "AI_DEBATE_FINAL_TOKEN_RATIO": "final_token_ratio",
    "AI_DEBATE_REASONING_BUDGET": "reasoning_budget",
    "AI_DEBATE_REASONING_BUDGET_OPEN_CLOSE": "reasoning_budget_open_close",
    "AI_DEBATE_FINAL_TOKEN_FLOOR": "final_token_floor",
    "AI_DEBATE_HARD_CAP": "hard_cap",
    "AI_DEBATE_FINALIZER_ON_EMPTY": "finalizer_on_empty",
    "AI_DEBATE_KEEP_REASONING_PRIVATE": "keep_reasoning_private",
    "AI_DEBATE_REASONING_HEURISTIC": "reasoning_heuristic",
    "AI_DEBATE_DEBUG_RAW_RESPONSE": "debug_raw_response",
    "AI_DEBATE_WEIGHT_JUDGE": "weight_judge",
    "AI_DEBATE_WEIGHT_AI": "weight_ai",
    "AI_DEBATE_MOCK_MODE": "mock_mode",
    "AI_DEBATE_STRICT_EVIDENCE": "strict_evidence",
    "AI_DEBATE_FORFEIT_POLICY": "forfeit_policy",
    "AI_DEBATE_OVERRUN_MODE": "overrun_mode",
    "AI_DEBATE_EARLY_STOP": "early_stop",
    "AI_DEBATE_PRICE_IN_PER_1M": "price_in_per_1m",
    "AI_DEBATE_PRICE_OUT_PER_1M": "price_out_per_1m",
}


def resolve_config(
    ui_config: dict[str, Any] | None = None,
    secrets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """默认优先级：UI 本场 > 环境变量 > st.secrets[defaults] > 代码 DEFAULTS。

    设置 AI_DEBATE_LOCK_CONFIG=true 后，优先级变为：
    环境变量 > UI 本场 > st.secrets[defaults] > 代码 DEFAULTS。
    用于部署/CI 场景强制锁定配置。
    """
    ui_config = ui_config or {}
    secrets_defaults = (secrets or {}).get("defaults", {})
    lock_config = os.environ.get("AI_DEBATE_LOCK_CONFIG", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    result = dict(DEFAULTS)
    # secrets 层
    for key, value in secrets_defaults.items():
        if value is None:
            continue
        if key in DEFAULTS:
            result[key] = _coerce(key, value)

    if lock_config:
        # 锁定模式：环境变量压过 UI
        _apply_ui(result, ui_config)
        _apply_env(result)
    else:
        # 默认模式：UI 临时本场配置优先
        _apply_env(result)
        _apply_ui(result, ui_config)
    return result


def _apply_env(result: dict[str, Any]) -> None:
    for env_key, cfg_key in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is not None:
            result[cfg_key] = _coerce(cfg_key, val)


def _apply_ui(result: dict[str, Any], ui_config: dict[str, Any]) -> None:
    for key, value in ui_config.items():
        if value is None:
            continue
        if key in DEFAULTS:
            result[key] = _coerce(key, value)


def _coerce(key: str, value: Any) -> Any:
    """把字符串/UI 输入转成正确类型。"""
    if key in ("weight_judge", "weight_ai", "final_token_ratio", "price_in_per_1m", "price_out_per_1m"):
        return float(value)
    if key in (
        "word_limit",
        "max_rounds",
        "input_budget",
        "output_max_tokens",
        "reasoning_budget",
        "reasoning_budget_open_close",
        "final_token_floor",
        "hard_cap",
        "reasoning_tokens",
    ):
        return int(value)
    if key in (
        "mock_mode",
        "strict_evidence",
        "finalizer_on_empty",
        "keep_reasoning_private",
        "reasoning_heuristic",
        "debug_raw_response",
        "early_stop",
    ):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    return value


def json_output_max_tokens(config) -> int:
    """JSON 结构化调用（评分/记忆/结论）的输出预算。

    推理模型（如 deepseek 系列）会把相当多 token 花在思考上，若预算太小会导致
    content 为空；这里至少给 8192，并尊重更大的 hard_cap。
    output_max_tokens > 0 时作为显式覆盖。
    """
    if config.output_max_tokens > 0:
        return config.output_max_tokens
    return max(config.hard_cap, 8192)


def estimate_cost_cny(usage_in: int, usage_out: int, price_in: float, price_out: float) -> float:
    """按人民币单价估算本次用量成本。"""
    return usage_in / 1e6 * price_in + usage_out / 1e6 * price_out
