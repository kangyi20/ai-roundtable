"""画像库回归测试：权重增减、自动删除、条数上限、L1/L2 隔离。

运行：python scripts/test_profile_store.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from profile_store import MAX_BELIEFS, ProfileStore

TEST_PATH = "profile/_unittest_profile.json"


def test_weights():
    st = ProfileStore(TEST_PATH)
    st.clear()

    # 新建 → weight 1.0
    data = st.apply_updates(
        [
            {
                "action": "new",
                "category": "经济状况",
                "belief": "L1-直白推断",
                "injected_as": "L2-委婉说法",
                "practical": "L2-行为指导",
                "evidence": "依据",
            }
        ]
    )
    assert len(data["beliefs"]) == 1
    b = data["beliefs"][0]
    assert b["weight"] == 1.0 and b["hits"] == 1
    bid = b["id"]

    # 印证 → 权重 +1
    data = st.apply_updates([{"action": "reinforce", "id": bid}])
    assert data["beliefs"][0]["weight"] == 2.0
    assert data["beliefs"][0]["hits"] == 2

    # 推翻两次 → 权重归零 → 删除并留痕
    st.apply_updates([{"action": "contradict", "id": bid}])
    data = st.apply_updates([{"action": "contradict", "id": bid}])
    assert data["beliefs"] == [], "权重耗尽应删除该条"
    assert len(data["deleted"]) == 1, "删除应留痕"
    assert data["deleted"][0].get("deleted_reason")

    # 未知 id 的 reinforce/contradict 不应崩、也不应新增
    data = st.apply_updates([{"action": "reinforce", "id": "b_notexist"}])
    assert data["beliefs"] == []


def test_l1_never_injected():
    st = ProfileStore(TEST_PATH)
    st.clear()
    st.apply_updates(
        [{"action": "new", "belief": "L1-绝不能外传", "injected_as": "L2-可以给AI看", "practical": "P"}]
    )
    text = st.injection_text()
    assert "L2-可以给AI看" in text, "L2 应出现在注入文本里"
    assert "L1-绝不能外传" not in text, "注入文本绝不能含 L1"

    # 给裁判看的 brief 则可以含 L1（裁判是唯一能看到直白版的角色）
    brief = st.brief_for_judge()
    assert "L1-绝不能外传" in brief


def test_cap():
    st = ProfileStore(TEST_PATH)
    st.clear()
    for i in range(MAX_BELIEFS + 5):
        st.apply_updates([{"action": "new", "belief": f"b{i}", "injected_as": f"inj{i}"}])
    data = st.load()
    assert len(data["beliefs"]) == MAX_BELIEFS, "应受条数上限约束"
    assert len(data["deleted"]) >= 5, "被淘汰的条目应留痕"


if __name__ == "__main__":
    try:
        test_weights()
        test_l1_never_injected()
        test_cap()
        print("ALL PROFILE TESTS PASSED")
    finally:
        Path(TEST_PATH).unlink(missing_ok=True)
