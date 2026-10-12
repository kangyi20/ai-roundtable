"""引擎扩展功能回归测试：结论生成（成功/失败重试）与收敛早停。

运行：python scripts/test_engine_extras.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import engine
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


if __name__ == "__main__":
    test_summarize_success()
    test_summarize_failure_retry()
    test_early_stop()
    test_extract_json()
    test_stance_change_and_converge()
    print("ALL ENGINE EXTRA TESTS PASSED")
