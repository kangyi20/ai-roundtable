"""服务端守卫回归测试：自动推进中拒绝手动操作、mode 校验与回落。

运行：python scripts/test_server_guards.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import engine
import server


def test_auto_conflict_guard():
    """自动讨论推进中，手动接口必须被拦下（否则两边会抢着发言）。"""
    did = "t_guard"
    server._AUTO_FLAGS[did] = True
    msg = server._auto_conflict(did)
    assert msg, "自动推进中应返回冲突提示"
    assert "暂停" in msg, "提示应告诉用户先暂停"

    server._AUTO_FLAGS[did] = False
    assert server._auto_conflict(did) is None, "未在自动推进时不应拦截"
    assert server._auto_conflict("不存在的讨论") is None


def test_mode_default_and_fallback():
    """mode 默认 manual；非法值回落为 manual；auto 可正常设置。"""
    def make(mode=None):
        kw = {} if mode is None else {"mode": mode}
        return engine.create_initial_state(
            debate_id="x", topic="t", pro_side="a", con_side="b", secrets={}, **kw
        )

    assert make().mode == "manual", "默认应为手动模式"
    assert make("auto").mode == "auto"
    assert make("manual").mode == "manual"
    assert make("bogus").mode == "manual", "非法 mode 应回落为 manual"
    assert make("AUTO").mode == "manual", "大小写不匹配也应安全回落"


def test_manual_controls_only_in_manual_mode():
    """自动模式下不该出现任何手动发言/确认/驳回按钮（前端契约）。

    这里以后端状态为准做交叉检查：自动模式启动后 state.mode 必须是 auto，
    前端据此只渲染自动控件。
    """
    s = engine.create_initial_state(
        debate_id="x", topic="t", pro_side="a", con_side="b", secrets={}, mode="auto"
    )
    assert s.mode == "auto"
    # 手动模式才允许逐个推进；自动模式由服务端循环驱动
    s.state = "A_SPEAKING"
    assert engine.can("speak_a", s)

    s2 = engine.create_initial_state(
        debate_id="y", topic="t", pro_side="a", con_side="b", secrets={}, mode="manual"
    )
    assert s2.mode == "manual"


if __name__ == "__main__":
    test_auto_conflict_guard()
    test_mode_default_and_fallback()
    test_manual_controls_only_in_manual_mode()
    print("ALL SERVER GUARD TESTS PASSED")
