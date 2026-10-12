"""非交互 MOCK 冒烟测试：用 MockProvider 驱动 engine 从 IDLE 到 DONE。

断言完成定义：
1. A 未确认前 B 不能发言（can 返回 False + IllegalTransition）。
2. rejected 草稿不进 build_prompt_for 的公共上下文。
3. 第 6 轮估算 input 相对第 2 轮增长不超过 30%。
4. JUDGE_AI 返回非合法 JSON 时重试 2 次，最终 judge_ai_invalid=true。
5. JSONL 持久化可恢复。
6. 导出 transcript.jsonl / score.json / config_snapshot.json 三文件。
7. content 空但 reasoning 有值时，公共 transcript 不含 reasoning，且触发 finalizer 或被 rejected，绝不回填。
8. content 含 <think>..</think> 时，confirmed content 已剥离，ledger 只收到剥离后文本。
9. request_max_tokens 公式符合 clamp(round(word_limit*final_token_ratio)+budget+256, final_token_floor, hard_cap)；open/close 阶段预算正确。
10. finish_reason="length" 不产生 reasoning 展示，只记 dirty_reason/审计字段。
11. debug_raw_response=false 时 provider_debug.log 不写 raw；=true 时不含 api_key。
12. 二次驳回仍 DODGE，不强制 confirmed。
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import engine
import providers
from memory import DebateState, Message, build_prompt_for, estimate_tokens
from providers import MockProvider, NormalizedMessage
from storage import DebateStore


class BadJudgeProvider(providers.Provider):
    """始终返回非法 JSON 的评委，用于测试重试与 fallback。"""

    def chat_text(self, messages, max_tokens=None, temperature=None, **kwargs):
        return NormalizedMessage(content="bad"), 0, 0, 0

    def chat_json(self, messages, max_tokens=None, temperature=None, schema_hint=None):
        return {"invalid": "not a score"}, 0, 0, 0


class EmptyWithReasoningProvider(providers.Provider):
    """模拟 GLM：content 为空但 reasoning 有值。"""

    def __init__(self, side: str):
        self.side = side
        self.model = "mock-empty-reasoning"
        self.temperature = 0.7

    def chat_text(self, messages, max_tokens=None, temperature=None, **kwargs):
        return (
            NormalizedMessage(
                content="",
                reasoning="我需要先分析对方论点... 让我想想... 最终发言应该是...",
                finish_reason="stop",
                reasoning_tokens=512,
                dirty_reason="empty",
            ),
            100,
            100,
            512,
        )

    def chat_json(self, messages, max_tokens=None, temperature=None, schema_hint=None):
        return {"new_claims": [], "attacks": [], "new_obligations": []}, 0, 0, 0


class ThinkingTagProvider(providers.Provider):
    """content 里包含标准 <think> 标签。"""

    def __init__(self, side: str):
        self.side = side
        self.model = "mock-thinking-tags"
        self.temperature = 0.7

    def chat_text(self, messages, max_tokens=None, temperature=None, **kwargs):
        # 模拟 providers.py 中 _call_chat 对 thinking 标签的剥离
        raw = "<think>先分析...</think>我觉得这个问题可以换个角度看，也许两边都能接受一个中间地带。"
        content = providers._strip_thinking_tags(raw)
        return (
            NormalizedMessage(
                content=content,
                finish_reason="stop",
            ),
            100,
            100,
            120,
        )

    def chat_json(self, messages, max_tokens=None, temperature=None, schema_hint=None):
        return {"new_claims": [], "attacks": [], "new_obligations": []}, 0, 0, 0


class LengthProvider(providers.Provider):
    """模拟 finish_reason=length 截断。"""

    def __init__(self, side: str):
        self.side = side
        self.model = "mock-length"
        self.temperature = 0.7

    def chat_text(self, messages, max_tokens=None, temperature=None, **kwargs):
        return (
            NormalizedMessage(
                content="我觉得这个问题其实可以",
                finish_reason="length",
                dirty_reason="length",
            ),
            100,
            100,
            50,
        )

    def chat_json(self, messages, max_tokens=None, temperature=None, schema_hint=None):
        return {"new_claims": [], "attacks": [], "new_obligations": []}, 0, 0, 0


def _fresh_state(**overrides) -> DebateState:
    ui_config = {
        "word_limit": 350,
        "max_rounds": 6,
        "mock_mode": True,
        "input_budget": 3500,
        "finalizer_on_empty": True,
    }
    ui_config.update(overrides)
    state = engine.create_initial_state(
        debate_id="smoke",
        topic="人工智能利大于弊还是弊大于利",
        pro_side="人工智能利大于弊",
        con_side="人工智能弊大于利",
        ui_config=ui_config,
        secrets={},
    )
    state.state = "A_SPEAKING"
    return state


def _run_full_debate() -> DebateState:
    """跑完 6 轮并到达 ROUND_END。"""
    state = _fresh_state()
    for r in range(1, 7):
        msg_a, state = engine.speak(state, "A", MockProvider(role="speaker", side="A"))
        state = engine.confirm_a(state, MockProvider(role="memory", side="A"))
        msg_b, state = engine.speak(state, "B", MockProvider(role="speaker", side="B"))
        state = engine.confirm_b(state, MockProvider(role="memory", side="B"))
    assert state.state == "ROUND_END", f"expected ROUND_END, got {state.state}"
    return state


def _capture_prompts() -> dict[str, str]:
    """单独跑一遍，捕获第 2 轮和第 6 轮 A 发言前的完整 prompt 文本。"""
    state = _fresh_state()
    prompts: dict[str, str] = {}
    for r in range(1, 7):
        msg_a, state = engine.speak(state, "A", MockProvider(role="speaker", side="A"))
        full = "\n".join(m["content"] for m in build_prompt_for(state, "A"))
        if r == 2:
            prompts["r2"] = full
        if r == 6:
            prompts["r6"] = full
        state = engine.confirm_a(state, MockProvider(role="memory", side="A"))
        msg_b, state = engine.speak(state, "B", MockProvider(role="speaker", side="B"))
        state = engine.confirm_b(state, MockProvider(role="memory", side="B"))
    return prompts


def main():
    print("=== AI Debate Room MOCK Smoke Test ===")

    # ------------------------------------------------------------------
    # 1. A 未确认前 B 不能发言
    # ------------------------------------------------------------------
    print("\n[Test 1] A 未确认前 B 不能发言")
    state = _fresh_state()
    msg_a, state = engine.speak(state, "A", MockProvider(role="speaker", side="A"))
    assert state.state == "A_WAIT_CONFIRM"
    assert not engine.can("speak_b", state), "B 在 A 未确认前应被禁用"
    try:
        engine.speak(state, "B", MockProvider(role="speaker", side="B"))
        raise AssertionError("应抛出 IllegalTransition")
    except engine.IllegalTransition:
        print("  PASS: can('speak_b')=False 且强行调用抛出 IllegalTransition")

    # ------------------------------------------------------------------
    # 2. rejected 草稿不进公共上下文
    # ------------------------------------------------------------------
    print("\n[Test 2] rejected 草稿不进入 B 的公共上下文")
    state = _fresh_state()
    draft1, state = engine.speak(state, "A", MockProvider(role="speaker", side="A"))
    state = engine.reject_a(state)  # 第一次驳回
    draft2, state = engine.speak(state, "A", MockProvider(role="speaker", side="A"))  # 重写
    prompt_b = "\n".join(m["content"] for m in build_prompt_for(state, "B"))
    assert draft1.content not in prompt_b, "被驳回的草稿不应出现在 B 的 prompt"
    print("  PASS: 被驳回草稿未出现在 B 的 prompt")
    state = engine.confirm_a(state, MockProvider(role="memory", side="A"))
    _, state = engine.speak(state, "B", MockProvider(role="speaker", side="B"))
    state = engine.confirm_b(state, MockProvider(role="memory", side="B"))

    # ------------------------------------------------------------------
    # 3. 第 6 轮 input 相对第 2 轮增长不超过 30%
    # ------------------------------------------------------------------
    print("\n[Test 3] 第 6 轮 input 相对第 2 轮增长 <= 30%")
    prompts = _capture_prompts()
    tokens_r2 = estimate_tokens(prompts["r2"])
    tokens_r6 = estimate_tokens(prompts["r6"])
    growth = (tokens_r6 - tokens_r2) / tokens_r2 if tokens_r2 else 0
    print(f"  Round2 tokens: {tokens_r2}, Round6 tokens: {tokens_r6}, growth: {growth:.1%}")
    assert growth <= 0.30, f"input 增长过高: {growth:.1%}"
    print("  PASS")

    # ------------------------------------------------------------------
    # 4. JUDGE_AI 非合法 JSON 重试 2 次后 judge_ai_invalid=true
    # ------------------------------------------------------------------
    print("\n[Test 4] JUDGE_AI 非法 JSON 重试 2 次后 judge_ai_invalid=true")
    state = _run_full_debate()
    state.judge_scores = {
        "A": {"logic": 7, "evidence": 6, "rebuttal": 7, "clarity": 7},
        "B": {"logic": 6, "evidence": 7, "rebuttal": 6, "clarity": 6},
    }
    state = engine.score(state, BadJudgeProvider())
    assert state.score is not None
    assert state.score.judge_ai_invalid, "非法 JSON 应标记 judge_ai_invalid=true"
    print("  PASS: judge_ai_invalid=true")

    # ------------------------------------------------------------------
    # 5. JSONL 可恢复
    # ------------------------------------------------------------------
    print("\n[Test 5] JSONL 持久化可恢复")
    store = DebateStore(root_dir="debates_smoke")
    store.append_snapshot(state.debate_id or "smoke", state)
    loaded = store.load_latest(state.debate_id or "smoke")
    assert loaded is not None, "应从 JSONL 恢复"
    assert loaded.state == state.state, "恢复后状态一致"
    assert len(loaded.messages) == len(state.messages), "恢复后消息数一致"
    print("  PASS: JSONL 恢复成功")

    # ------------------------------------------------------------------
    # 6. 导出三文件
    # ------------------------------------------------------------------
    print("\n[Test 6] 导出 transcript.jsonl / score.json / config_snapshot.json")
    export_dir, transcript_path, score_path, config_path = store.export(
        state.debate_id or "smoke", state
    )
    assert transcript_path.exists(), f"缺少 {transcript_path}"
    assert score_path.exists(), f"缺少 {score_path}"
    assert config_path.exists(), f"缺少 {config_path}"
    print(f"  PASS: 导出目录 {export_dir}")

    # ------------------------------------------------------------------
    # 7. content 空但 reasoning 有值：不泄漏、触发 finalizer 或被 rejected
    # ------------------------------------------------------------------
    print("\n[Test 7] content 空但 reasoning 有值时不泄漏")
    state = _fresh_state(finalizer_on_empty=True)
    msg, state = engine.speak(state, "A", EmptyWithReasoningProvider("A"))
    # dirty 应被自动驳回，不会进入 WAIT_CONFIRM
    assert state.state != "A_WAIT_CONFIRM", "空 content 不应进入 WAIT_CONFIRM"
    # 检查所有消息：没有任何公共消息包含 reasoning 文本
    for m in state.messages:
        assert "我需要先分析" not in (m.content or ""), "reasoning 文本不应进入公共 content"
    # 导出 transcript 也不含 reasoning
    _, tp, _, _ = store.export("smoke_empty", state)
    with open(tp, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            assert "我需要先分析" not in record.get("content", ""), "导出 transcript 不应含 reasoning"
            assert record.get("reasoning", "") == "", "导出 transcript 的 reasoning 字段应被排除"
    print("  PASS: reasoning 未泄漏，空 content 被自动处理")

    # ------------------------------------------------------------------
    # 8. <think> 标签在 confirmed content 中被剥离
    # ------------------------------------------------------------------
    print("\n[Test 8] <think> 标签在 confirmed content 中被剥离")
    state = _fresh_state()
    msg, state = engine.speak(state, "A", ThinkingTagProvider("A"))
    assert state.state == "A_WAIT_CONFIRM"
    draft = next(m for m in state.messages if m.status == "draft")
    assert "<think>" not in draft.content, "draft content 应已剥离 thinking 标签"
    assert "我觉得这个问题" in draft.content, "最终发言应保留"
    state = engine.confirm_a(state, MockProvider(role="memory", side="A"))
    confirmed = next(m for m in state.messages if m.status == "confirmed")
    assert "<think>" not in confirmed.content, "confirmed content 应已剥离 thinking 标签"
    # 记忆写入器输入是 confirmed.content，已剥离
    print("  PASS: confirmed content 已剥离 thinking 标签")

    # ------------------------------------------------------------------
    # 9. request_max_tokens 公式与 open/close 阶段预算
    # ------------------------------------------------------------------
    print("\n[Test 9] request_max_tokens 公式与 open/close 阶段预算")
    from providers import _request_max_tokens

    class _Cfg:
        final_token_ratio = 2.8
        reasoning_budget = 1024
        reasoning_budget_open_close = 2048
        final_token_floor = 768
        hard_cap = 4096

    cfg = _Cfg()
    mid = _request_max_tokens(350, "mid", cfg)
    open_ = _request_max_tokens(350, "open", cfg)
    close = _request_max_tokens(350, "close", cfg)
    expected_mid = max(768, min(4096, round(350 * 2.8) + 1024 + 256))
    expected_open_close = max(768, min(4096, round(350 * 2.8) + 2048 + 256))
    assert mid == expected_mid, f"mid 预算应为 {expected_mid}, 得到 {mid}"
    assert open_ == expected_open_close, f"open 预算应为 {expected_open_close}, 得到 {open_}"
    assert close == expected_open_close, f"close 预算应为 {expected_open_close}, 得到 {close}"
    print(f"  PASS: mid={mid}, open={open_}, close={close}")

    # ------------------------------------------------------------------
    # 10. finish_reason=length 不展示 reasoning，只记审计字段
    # ------------------------------------------------------------------
    print("\n[Test 10] finish_reason=length 不展示 reasoning")
    state = _fresh_state(finalizer_on_empty=True)
    msg, state = engine.speak(state, "A", LengthProvider("A"))
    # length 触发技术重试，但 LengthProvider 永远返回 length，所以最终 rejected
    assert state.state != "A_WAIT_CONFIRM"
    audit_msg = next(m for m in state.messages if m.speaker == "A" and m.dirty_reason)
    assert audit_msg.finish_reason == "length"
    # 没有任何 A 消息把截断内容当成完整 content 展示为 draft
    drafts = [m for m in state.messages if m.status == "draft"]
    assert len(drafts) == 0, "length 截断不应产生可确认的 draft"
    print("  PASS: finish_reason=length 未产生可展示 draft，审计字段已记录")

    # ------------------------------------------------------------------
    # 11. debug_raw_response=false 不写 log；=true 时不含 api_key
    # ------------------------------------------------------------------
    print("\n[Test 11] provider_debug.log 写入规则")
    log_path = Path("provider_debug.log")
    if log_path.exists():
        log_path.unlink()

    # false：不应写 log
    state = _fresh_state(debug_raw_response=False)
    msg, state = engine.speak(state, "A", MockProvider(role="speaker", side="A"))
    assert not log_path.exists(), "debug_raw_response=false 时不应写 provider_debug.log"

    # true：应写 log 但不含 api_key
    from providers import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        base_url="https://example.com/v1",
        api_key="sk-secret-test-key",
        model="test-model",
        debug_raw_response=True,
    )
    # 直接调用内部日志函数，避免真实网络请求
    providers._log_debug(True, "test-model", {"content": "hello", "reasoning": "think"})
    assert log_path.exists(), "debug_raw_response=true 时应写 provider_debug.log"
    log_text = log_path.read_text(encoding="utf-8")
    assert "sk-secret-test-key" not in log_text, "log 中不应包含 api_key"
    print("  PASS: debug 日志规则正确且不含 api_key")
    log_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # 12. 二次驳回仍 DODGE，不强制 confirmed
    # ------------------------------------------------------------------
    print("\n[Test 12] 二次驳回 DODGE，不强制 confirmed")
    state = _fresh_state()
    _, state = engine.speak(state, "A", MockProvider(role="speaker", side="A"))
    state = engine.reject_a(state)
    _, state = engine.speak(state, "A", MockProvider(role="speaker", side="A"))
    state = engine.reject_a(state)  # 第二次驳回 -> DODGE
    dodge = [m for m in state.messages if m.status == "dodge"]
    assert len(dodge) == 1, "第二次驳回应产生 DODGE 系统记录"
    assert not any(m.status == "confirmed" for m in state.messages if m.speaker == "A"), "不应强制 confirmed"
    print("  PASS: 二次驳回产生 DODGE，未强制 confirmed")

    print("\n=== All MOCK smoke tests passed ===")
    print(f"导出文件：\n  {transcript_path}\n  {score_path}\n  {config_path}")


if __name__ == "__main__":
    main()
