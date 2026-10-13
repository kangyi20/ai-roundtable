"""引擎扩展功能回归测试：结论生成（成功/失败重试）与收敛早停。

运行：python scripts/test_engine_extras.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import engine
import prompts
from providers import MockProvider, _extract_json


class NoProgressMemory(MockProvider):
    """记忆写入器始终返回空 delta，用于触发收敛早停。"""

    def chat_json(self, messages, max_tokens=None, temperature=None, schema_hint=None):
        if schema_hint == "score":
            return self._mock_score(), 0, 0, 0
        return {
            "new_claims": [],
            "attacks": [],
            "concessions": [],
            "new_obligations": [],
            "contradictions": [],
            "close_claims": [],
        }, 0, 0, 0


class BadConclusionProvider(MockProvider):
    """结论生成器始终返回空 dict，触发失败重试。"""

    def chat_json(self, messages, max_tokens=None, temperature=None, schema_hint=None):
        return {}, 0, 0, 0


class StanceChangeMemory(MockProvider):
    """记忆写入器：B 方被说服、立场转变并标记达成一致。"""

    def chat_json(self, messages, max_tokens=None, temperature=None, schema_hint=None):
        if schema_hint == "score":
            return self._mock_score(), 0, 0, 0
        return {
            "new_claims": [],
            "attacks": [],
            "concessions": [],
            "new_obligations": [],
            "contradictions": [],
            "close_claims": [],
            "stance_change": "我认同 A 方的观点",
            "converged": True,
        }, 0, 0, 0


def _fresh(**kw):
    ui = {"word_limit": 300, "max_rounds": 6, "mock_mode": True, "early_stop": True}
    ui.update(kw)
    s = engine.create_initial_state(
        debate_id="test",
        topic="话题",
        pro_side="A 视角",
        con_side="B 视角",
        ui_config=ui,
        secrets={},
    )
    s.state = "A_SPEAKING"
    return s


def test_summarize_success():
    s = _fresh()
    s.state = "ROUND_END"
    s = engine.summarize(s, MockProvider(role="judge", side="JUDGE_AI"))
    assert s.conclusion is not None and s.conclusion.takeaway
    assert s.state == "DONE"
    assert s.conclusion_error == ""


def test_summarize_failure_retry():
    s = _fresh()
    s.state = "ROUND_END"
    s = engine.summarize(s, BadConclusionProvider(role="judge"))
    assert s.conclusion is None
    assert s.conclusion_error
    assert s.state == "ROUND_END"  # 回到 ROUND_END，可重试
    assert engine.can("summarize", s)  # 仍可再次生成结论


def test_extract_json():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"b": 2}\n```') == {"b": 2}
    assert _extract_json('好的，结论如下：{"c": 3}') == {"c": 3}
    assert _extract_json('{"d": {"e": 4}}') == {"d": {"e": 4}}  # 嵌套
    assert _extract_json("") == {}
    assert _extract_json("这里没有 json") == {}


def test_stance_change_and_converge():
    s = _fresh()
    _, s = engine.speak(s, "A", MockProvider(role="speaker", side="A"))
    s = engine.confirm_a(s, MockProvider(role="memory", side="A"))
    _, s = engine.speak(s, "B", MockProvider(role="speaker", side="B"))
    s = engine.confirm_b(s, StanceChangeMemory(role="memory", side="B"))
    assert s.converged is True
    assert s.stance_b != s.con_side  # B 立场已改变
    assert "A 方的观点" in s.stance_b
    assert s.state == "ROUND_END"
    assert s.round_no < 6  # 达成一致后提前结束


def test_early_stop():
    s = _fresh()
    for _r in range(1, 7):
        _, s = engine.speak(s, "A", MockProvider(role="speaker", side="A"))
        s = engine.confirm_a(s, NoProgressMemory(role="memory", side="A"))
        _, s = engine.speak(s, "B", MockProvider(role="speaker", side="B"))
        s = engine.confirm_b(s, NoProgressMemory(role="memory", side="B"))
        if s.state == "ROUND_END":
            break
    assert s.state == "ROUND_END"
    assert s.round_no < 6


def test_ask_judge_requires_done():
    """讨论没结束之前不能向裁判追问。"""
    s = _fresh()
    assert not engine.can("ask", s)
    try:
        engine.ask_judge(s, MockProvider(role="judge", side="JUDGE_AI"), "我该怎么选？")
        raise AssertionError("未结束时追问应当抛错")
    except engine.IllegalTransition:
        pass


def test_ask_judge_multiturn():
    """DONE 之后可多轮追问，问答成对记录，用量计入统计。"""
    s = _fresh()
    s.state = "ROUND_END"
    s = engine.summarize(s, MockProvider(role="judge", side="JUDGE_AI"))
    assert s.state == "DONE"
    assert engine.can("ask", s)

    judge = MockProvider(role="judge", side="JUDGE_AI")
    s = engine.ask_judge(s, judge, "如果换成我，你会建议我怎么选？")
    assert len(s.qa) == 2, "一轮问答应产生 user + judge 两条"
    assert s.qa[0]["role"] == "user" and s.qa[1]["role"] == "judge"
    assert s.qa[1]["content"].strip()
    assert s.totals["usage_out"] > 0, "追问用量应计入统计"

    s = engine.ask_judge(s, judge, "那哪一边更着急？")
    assert len(s.qa) == 4, "应支持多轮追问"

    # 空问题应被拒绝，且不写入记录
    try:
        engine.ask_judge(s, judge, "   ")
        raise AssertionError("空问题应当抛错")
    except ValueError:
        pass
    assert len(s.qa) == 4


def test_qa_prompt_excludes_L1():
    """追问提示词里只能出现 L2 处境，绝不能出现 L1 之外的注入内容。"""
    s = _fresh()
    s.user_context = "他做决定时负担感较重"
    s.state = "ROUND_END"
    s = engine.summarize(s, MockProvider(role="judge", side="JUDGE_AI"))
    msgs = prompts.build_judge_qa_messages(s, "我该怎么选？", history=s.qa)
    blob = " ".join(m["content"] for m in msgs)
    assert "他做决定时负担感较重" in blob, "L2 处境应注入"
    assert "绝不能提及或暗示" in blob, "提示词必须带禁止泄露的约束"
    assert msgs[-1]["content"] == "我该怎么选？", "最后一条应是当前问题"


def test_ask_judge_empty_answer_rolls_back():
    """裁判返回空内容时应报错，且不留下半截问答。"""
    s = _fresh()
    s.state = "ROUND_END"
    s = engine.summarize(s, MockProvider(role="judge", side="JUDGE_AI"))
    empty_judge = MockProvider(role="judge", side="JUDGE_AI", empty_content=True)
    try:
        engine.ask_judge(s, empty_judge, "问题")
        raise AssertionError("空回答应当抛错")
    except ValueError:
        pass
    assert s.qa == [], "失败时不应留下半截记录"


if __name__ == "__main__":
    test_summarize_success()
    test_summarize_failure_retry()
    test_early_stop()
    test_extract_json()
    test_stance_change_and_converge()
    test_ask_judge_requires_done()
    test_ask_judge_multiturn()
    test_qa_prompt_excludes_L1()
    test_ask_judge_empty_answer_rolls_back()
    print("ALL ENGINE EXTRA TESTS PASSED")
