"""模型 Provider 抽象：OpenAI 兼容接口与 Mock 模式。"""

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


class ProviderEmptyContent(Exception):
    """模型返回空/脏内容且无法通过 finalizer/重试修复。"""

    def __init__(self, reason: str = ""):
        self.reason = reason
        super().__init__(reason)


@dataclass
class NormalizedMessage:
    """从各厂商 API 统一提取后的消息。

    reasoning 永远私有，不进公共 transcript、确认、ledger 或评分。
    """

    content: str = ""
    reasoning: str | None = None
    raw: dict[str, Any] | None = None
    finish_reason: str | None = None
    reasoning_tokens: int | None = None
    finalizer_triggered: bool = False  # 是否触发 finalizer 二次定稿
    dirty_reason: str = ""  # 空表示干净；否则记录 dirty 原因


_THINKING_TAG_RE = re.compile(
    r"<(think|thinking|analysis|reasoning)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

# 启发式 dirty 标记：出现这些表达且缺少明确立场表达，则判为 dirty
_HEURISTIC_DIRTY_MARKERS = [
    r"我需要先分析",
    r"我先分析",
    r"让我想想",
    r"字数检查",
    r"字数统计",
    r"Word count check",
    r"Character count",
    r"分析当前状态",
    r"制定回应策略",
]

_STANCE_MARKERS = [
    r"我认为",
    r"我觉得",
    r"我的看法",
    r"在我看来",
    r"我的观点",
]


def _strip_thinking_tags(text: str) -> str:
    """仅删除标准 <think|thinking|analysis|reasoning>...</...> 标签。"""
    if not text:
        return text
    return _THINKING_TAG_RE.sub("", text).strip()


def _is_dirty(content: str, finish_reason: str | None, use_heuristic: bool) -> str:
    """返回 dirty 原因字符串；空字符串表示干净。"""
    if not content:
        return "empty"

    stripped = _strip_thinking_tags(content)
    if not stripped:
        return "stripped_empty"

    if finish_reason == "length":
        return "length"

    if use_heuristic:
        has_dirty_marker = any(re.search(p, content) for p in _HEURISTIC_DIRTY_MARKERS)
        has_stance = any(re.search(p, content) for p in _STANCE_MARKERS)
        if has_dirty_marker and not has_stance:
            return "heuristic"

    return ""


def _request_max_tokens(word_limit: int, phase: str, config) -> int:
    """计算单次请求用的 max_tokens（含 reasoning + final content 预算）。"""
    ratio = config.final_token_ratio
    budget = (
        config.reasoning_budget_open_close
        if phase in ("open", "close")
        else config.reasoning_budget
    )
    estimated = round(word_limit * ratio) + budget + 256
    return max(config.final_token_floor, min(config.hard_cap, estimated))


def _final_reserve(word_limit: int, config) -> int:
    """finalizer 二次定稿的输出 token 预算。"""
    estimated = round(word_limit * config.final_token_ratio) + 256
    return max(512, min(config.hard_cap, estimated))


def _redacted_raw(msg) -> dict[str, Any]:
    """把 OpenAI SDK 消息对象脱敏后转成 dict。"""
    try:
        data = msg.model_dump() if hasattr(msg, "model_dump") else vars(msg)
    except Exception:
        data = {"repr": str(msg)}
    return data


def _log_debug(debug_raw_response: bool, model: str, raw: dict[str, Any]) -> None:
    if not debug_raw_response:
        return
    try:
        with open("provider_debug.log", "a", encoding="utf-8") as f:
            line = json.dumps(
                {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model, "raw": raw},
                ensure_ascii=False,
                default=str,
            )
            f.write(line + "\n")
    except Exception:
        pass


def _extract_json(text: str) -> dict[str, Any]:
    """从模型原始文本中尽量提取 JSON 对象。

    兼容：纯 JSON、```json 代码块、夹在散文/思考里的 JSON。
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        break
    return {}


class Provider(ABC):
    """通用 Provider 接口。"""

    @abstractmethod
    def chat_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> tuple[NormalizedMessage, int, int, int]:
        """返回 (NormalizedMessage, latency_ms, usage_in, usage_out)。"""
        ...

    @abstractmethod
    def chat_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        schema_hint: str | None = None,
    ) -> tuple[dict[str, Any], int, int, int]:
        """返回 (parsed_json, latency_ms, usage_in, usage_out)。"""
        ...


class OpenAICompatibleProvider(Provider):
    """OpenAI 兼容接口；支持 temperature/timeout/max_retries/extra_body 覆盖。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 1,
        extra_body: dict[str, Any] | None = None,
        debug_raw_response: bool = False,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature if temperature is not None else 0.7
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.debug_raw_response = debug_raw_response
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=max_retries,
        )

    def _call_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
    ) -> tuple[NormalizedMessage, int, int, int]:
        temp = temperature if temperature is not None else self.temperature
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self.extra_body:
            kwargs.update(self.extra_body)

        start = time.time()
        resp = self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.time() - start) * 1000)

        choice = resp.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason or ""

        content = (msg.content or "").strip()

        # 标准 thinking 标签只在独立 reasoning 中保留；content 必须干净
        content = _strip_thinking_tags(content)

        # 提取 reasoning 到独立字段，永远禁止回填 content
        reasoning_parts: list[str] = []
        for field in ("reasoning_content", "reasoning", "thought", "thoughts"):
            val = getattr(msg, field, None)
            if val:
                reasoning_parts.append(str(val).strip())
        reasoning = "\n".join(reasoning_parts) if reasoning_parts else None

        # 尝试从 usage 读 reasoning tokens
        reasoning_tokens = None
        if resp.usage:
            details = getattr(resp.usage, "completion_tokens_details", None)
            if details:
                reasoning_tokens = getattr(details, "reasoning_tokens", None)
            # 部分厂商把 reasoning tokens 放在 response.usage.reasoning_tokens
            if reasoning_tokens is None:
                reasoning_tokens = getattr(resp.usage, "reasoning_tokens", None)

        usage_in = resp.usage.prompt_tokens if resp.usage else 0
        usage_out = resp.usage.completion_tokens if resp.usage else 0

        raw = _redacted_raw(msg)
        _log_debug(self.debug_raw_response, self.model, raw)

        return NormalizedMessage(
            content=content,
            reasoning=reasoning,
            raw=raw,
            finish_reason=finish_reason,
            reasoning_tokens=reasoning_tokens,
        ), latency_ms, usage_in, usage_out

    def _run_finalizer(
        self,
        brief: str,
        reasoning_excerpt: str,
        word_limit: int,
        temperature: float | None,
        config,
    ) -> tuple[NormalizedMessage, int, int, int]:
        """调用 finalizer 把私有 reasoning 草稿压缩成正式发言。"""
        import prompts

        finalizer_messages = prompts.build_finalizer_prompt(
            brief, reasoning_excerpt, word_limit
        )
        max_tokens = _final_reserve(word_limit, config)
        return self._call_chat(finalizer_messages, max_tokens, temperature)

    def chat_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        word_limit: int = 350,
        phase: str = "mid",
        config=None,
        **kwargs: Any,
    ) -> tuple[NormalizedMessage, int, int, int]:
        """生成发言；必要时调用 finalizer 二次定稿。"""
        if config is None:
            # 兜底：没有 config 时按默认值计算
            from dataclasses import dataclass as _dc

            @_dc
            class _DefaultConfig:
                final_token_ratio: float = 3.5
                reasoning_budget: int = 2048
                reasoning_budget_open_close: int = 4096
                final_token_floor: int = 1024
                hard_cap: int = 8192
                finalizer_on_empty: bool = True
                keep_reasoning_private: bool = True
                reasoning_heuristic: bool = True
                debug_raw_response: bool = False

            config = _DefaultConfig()

        # 计算足量预算
        request_max = _request_max_tokens(word_limit, phase, config)
        norm, latency, usage_in, usage_out = self._call_chat(
            messages, request_max, temperature
        )

        # 判定 dirty
        dirty = _is_dirty(
            norm.content, norm.finish_reason, config.reasoning_heuristic
        )
        norm.dirty_reason = dirty

        if dirty and config.finalizer_on_empty:
            # finalizer 只使用 reasoning 摘要，不进入公共记录
            excerpt = (norm.reasoning or norm.content)[:1200]
            brief = messages[-1]["content"][:500] if messages else ""
            final_norm, f_latency, f_in, f_out = self._run_finalizer(
                brief, excerpt, word_limit, temperature, config
            )
            latency += f_latency
            usage_in += f_in
            usage_out += f_out
            final_norm.finalizer_triggered = True

            # finalizer 输出仍需 strip 与 dirty 检查
            final_dirty = _is_dirty(
                final_norm.content,
                final_norm.finish_reason,
                config.reasoning_heuristic,
            )
            if not final_dirty:
                # 成功：保留原始 reasoning 用于调试，但 content 已是 final
                final_norm.reasoning = norm.reasoning
                final_norm.dirty_reason = ""
                return final_norm, latency, usage_in, usage_out
            else:
                # finalizer 失败：把原始 reasoning 带回去，engine 做技术重试/驳回
                norm.dirty_reason = f"finalizer_failed:{final_dirty}"
                norm.finalizer_triggered = True
                return norm, latency, usage_in, usage_out

        return norm, latency, usage_in, usage_out

    def chat_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        schema_hint: str | None = None,
    ) -> tuple[dict[str, Any], int, int, int]:
        temp = temperature if temperature is not None else self.temperature
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self.extra_body:
            kwargs.update(self.extra_body)
        start = time.time()
        resp = self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.time() - start) * 1000)
        msg = resp.choices[0].message

        data = msg.model_dump() if hasattr(msg, "model_dump") else vars(msg)
        content = str(data.get("content") or "").strip()
        if not content:
            # 推理模型可能把最终答案放在 reasoning_content 等字段
            for field in ("reasoning_content", "reasoning", "thought", "thoughts"):
                val = data.get(field)
                if val:
                    content = str(val).strip()
                    if content:
                        break

        usage_in = resp.usage.prompt_tokens if resp.usage else 0
        usage_out = resp.usage.completion_tokens if resp.usage else 0
        _log_debug(self.debug_raw_response, self.model, data)
        parsed = _extract_json(content)
        return parsed, latency_ms, usage_in, usage_out


class MockProvider(Provider):
    """MOCK_MODE 用：不调用真实 API，返回可预测假输出。"""

    def __init__(
        self,
        role: str = "speaker",
        side: str = "",
        empty_content: bool = False,
        thinking_content: str = "",
        finish_reason: str = "stop",
    ):
        self.role = role  # speaker | memory | judge
        self.side = side  # A | B | JUDGE_AI
        self.empty_content = empty_content
        self.thinking_content = thinking_content
        self.finish_reason = finish_reason

    def chat_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> tuple[NormalizedMessage, int, int, int]:
        content = ""
        if not self.empty_content:
            content = self._mock_speaker_text(messages)
        usage_in = len(str(messages)) // 2
        usage_out = len(content) // 2 if content else 64
        dirty_reason = ""
        if not content:
            dirty_reason = "empty"
        elif self.finish_reason == "length":
            dirty_reason = "length"
        norm = NormalizedMessage(
            content=content,
            reasoning=self.thinking_content or None,
            finish_reason=self.finish_reason,
            reasoning_tokens=len(self.thinking_content) // 2 if self.thinking_content else None,
            dirty_reason=dirty_reason,
        )
        return norm, 120, usage_in, usage_out

    def chat_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        schema_hint: str | None = None,
    ) -> tuple[dict[str, Any], int, int, int]:
        if schema_hint == "score":
            data = self._mock_score()
        elif schema_hint == "conclusion":
            data = self._mock_conclusion()
        elif schema_hint == "extract":
            data = self._mock_extract(messages)
        else:
            data = self._mock_memory_delta(messages)
        usage_in = len(str(messages)) // 2
        usage_out = len(json.dumps(data, ensure_ascii=False)) // 2
        return data, 80, usage_in, usage_out

    def _mock_speaker_text(self, messages: list[dict[str, str]]) -> str:
        side_label = self.side or "某方"
        return (
            f"【MOCK {side_label} 模拟发言】"
            f"我觉得这个问题还可以换个角度再看：对方刚才提到的点确实有道理，"
            f"不过我们可能还忽略了一个更重要的因素。"
            f"如果顺着这个思路继续想，也许能找到一个两边都能接受的中间地带。"
        )

    def _mock_memory_delta(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        side = self.side or "A"
        return {
            "new_claims": [
                {
                    "text": f"{side} 提出的一个模拟观点",
                    "evidence_state": "weak",
                    "quote_ptr": None,
                }
            ],
            "attacks": [],
            "concessions": [],
            "new_obligations": [
                {
                    "text": "请继续回应上面这个观点",
                    "owed_by": "B" if side == "A" else "A",
                }
            ],
            "contradictions": [],
            "close_claims": [],
            "stance_change": None,
            "converged": False,
        }

    def _mock_score(self) -> dict[str, Any]:
        return {
            "A": {"logic": 7, "evidence": 6, "rebuttal": 7, "clarity": 7},
            "B": {"logic": 6, "evidence": 7, "rebuttal": 6, "clarity": 6},
            "winner": "A",
            "reason": "MOCK 评估：A 方的讨论质量略好。",
        }

    def _mock_conclusion(self) -> dict[str, Any]:
        return {
            "consensus": ["双方都认为需要更谨慎、更平衡地看待这个问题"],
            "disagreements": ["在优先级的判断上仍有不同看法"],
            "open_questions": ["缺少一个可量化的判断标准"],
            "takeaway": "总体上，双方倾向于一个更平衡、留有校验余地的结论。",
            "aligned": False,
        }

    def _mock_extract(self, messages: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """MOCK：模拟裁判的提炼结果（含画像增量）。

        若提示里已经带上画像条目（形如 id=b_xxxx），则返回 reinforce，
        以便离线测试也能覆盖"权重累加"这条逻辑。
        """
        text = json.dumps(messages or [], ensure_ascii=False)
        ids = re.findall(r"id=(b_[0-9a-f]+)", text)
        if ids:
            updates: list[dict[str, Any]] = [
                {
                    "action": "reinforce",
                    "id": ids[0],
                    "injected_as": "他做决定时负担感较重",
                    "practical": "不要轻飘飘地建议花钱，要正视他的取舍成本",
                    "evidence": "（MOCK）用户再次提到类似的取舍困境",
                }
            ]
        else:
            updates = [
                {
                    "action": "new",
                    "category": "经济状况",
                    "belief": "（MOCK）手头并不宽裕，这笔钱对他有分量",
                    "injected_as": "他做决定时负担感较重",
                    "practical": "不要轻飘飘地建议花钱，要正视他的取舍成本",
                    "evidence": "（MOCK）用户提到手里有几千块钱并在两个用途间犹豫",
                }
            ]
        return {
            "topic": "（MOCK）这笔钱该投在长期收益还是当下体验上",
            "pro_side": "（MOCK）投在能带来长期收益的方向",
            "con_side": "（MOCK）投在当下最想体验、以后未必有机会的事",
            "note": "（MOCK）分歧在于如何看待短期体验与长期积累的取舍",
            "user_context": "（MOCK）他做决定时负担感较重，讨论请贴近他的取舍成本，不要轻飘飘地建议花钱",
            "user_context_soft": "（MOCK）我理解你在认真权衡这笔钱该怎么用",
            "belief_updates": updates,
        }
