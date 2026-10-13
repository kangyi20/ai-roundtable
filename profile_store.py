"""用户画像库：跨会话累积「对提问者的推断」，带权重增减与自动删除。

分层存储（见 需求-v2.md 任务 5）：
- belief      : L1 直白推断 —— **只存库，绝不注入辩论 AI**
- injected_as / practical : L2 委婉版 —— 注入辩论 AI 的提示词

权重规则：
    首次推断出 → 1.0；再次印证 → +1（上限 5）；被推翻 → -1；<= 0 → 删除
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

MAX_BELIEFS = 50          # 条数上限，超出淘汰权重最低的
MAX_WEIGHT = 5.0          # 单条权重上限
TOP_N_FOR_INJECT = 8      # 注入辩论 AI 时只取权重最高的前 N 条
# 测试时用 AI_DEBATE_PROFILE_PATH 指到别处，避免动到用户真实画像
DEFAULT_PATH = os.environ.get("AI_DEBATE_PROFILE_PATH") or "profile/user_profile.json"


def _empty() -> dict[str, Any]:
    return {"version": 1, "updated_at": "", "beliefs": [], "deleted": []}


class ProfileStore:
    """本地 JSON 画像库；写入用临时文件 + 原子替换，避免写坏。"""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = Path(path)

    # ---------------- 读写 ----------------

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return _empty()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return _empty()
            data.setdefault("version", 1)
            data.setdefault("updated_at", "")
            if not isinstance(data.get("beliefs"), list):
                data["beliefs"] = []
            if not isinstance(data.get("deleted"), list):
                data["deleted"] = []
            return data
        except Exception:
            # 文件损坏时不拖垮主流程，退回空画像
            return _empty()

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def clear(self) -> None:
        self.save(_empty())

    # ---------------- 权重更新 ----------------

    def apply_updates(self, updates: list[dict[str, Any]]) -> dict[str, Any]:
        """应用裁判给出的画像增量（new / reinforce / contradict），落盘并返回新画像。"""
        data = self.load()
        beliefs: list[dict[str, Any]] = data["beliefs"]
        by_id = {b.get("id"): b for b in beliefs}
        now = datetime.now().isoformat(timespec="seconds")

        for up in updates or []:
            if not isinstance(up, dict):
                continue
            action = str(up.get("action") or "").strip().lower()
            bid = str(up.get("id") or "").strip()

            if action == "new":
                text = str(up.get("belief") or "").strip()
                if not text:
                    continue
                evidence = str(up.get("evidence") or "").strip()
                beliefs.append(
                    {
                        "id": "b_" + uuid4().hex[:6],
                        "category": str(up.get("category") or "其他").strip() or "其他",
                        "belief": text,
                        "injected_as": str(up.get("injected_as") or "").strip(),
                        "practical": str(up.get("practical") or "").strip(),
                        "weight": 1.0,
                        "hits": 1,
                        "misses": 0,
                        "evidence": [evidence] if evidence else [],
                        "created_at": now,
                        "updated_at": now,
                    }
                )
            elif action in ("reinforce", "contradict") and bid in by_id:
                b = by_id[bid]
                if action == "reinforce":
                    b["weight"] = min(MAX_WEIGHT, float(b.get("weight", 1.0)) + 1)
                    b["hits"] = int(b.get("hits", 0)) + 1
                    evidence = str(up.get("evidence") or "").strip()
                    if evidence:
                        ev = b.setdefault("evidence", [])
                        ev.append(evidence)
                        b["evidence"] = ev[-5:]
                else:
                    b["weight"] = float(b.get("weight", 1.0)) - 1
                    b["misses"] = int(b.get("misses", 0)) + 1
                b["updated_at"] = now
                # 允许裁判顺手修正措辞（只在提供了非空值时覆盖）
                for key in ("belief", "injected_as", "practical"):
                    val = str(up.get(key) or "").strip()
                    if val:
                        b[key] = val

        # 删除权重耗尽的条目
        alive: list[dict[str, Any]] = []
        gone: list[dict[str, Any]] = []
        for b in beliefs:
            if float(b.get("weight", 0)) <= 0:
                b["deleted_reason"] = "权重耗尽（被后续信息反复推翻）"
                gone.append(b)
            else:
                alive.append(b)

        # 条数上限：淘汰权重最低的
        if len(alive) > MAX_BELIEFS:
            alive.sort(key=lambda x: float(x.get("weight", 0)), reverse=True)
            for b in alive[MAX_BELIEFS:]:
                b["deleted_reason"] = "超过条数上限，权重最低被淘汰"
                gone.append(b)
            alive = alive[:MAX_BELIEFS]

        data["beliefs"] = alive
        data["deleted"] = ((data.get("deleted") or [])[-200:]) + gone
        self.save(data)
        return data

    # ---------------- 供注入 / 供裁判 ----------------

    def injection_text(self, top_n: int = TOP_N_FOR_INJECT) -> str:
        """生成注入辩论 AI 的 L2 文本（按权重取前 N 条）。L1 绝不出现。"""
        beliefs = sorted(
            self.load().get("beliefs", []),
            key=lambda x: float(x.get("weight", 0)),
            reverse=True,
        )[:top_n]
        lines = []
        for b in beliefs:
            parts = [str(p).strip() for p in (b.get("injected_as"), b.get("practical")) if p]
            if parts:
                lines.append("- " + "；".join(parts))
        return "\n".join(lines)

    def brief_for_judge(self) -> str:
        """给裁判看的已有画像（含 id 与 L1 直白版），用于判断印证/推翻。"""
        beliefs = self.load().get("beliefs", [])
        if not beliefs:
            return "（暂无）"
        return "\n".join(
            f"- id={b.get('id')} [{b.get('category')}] {b.get('belief')}"
            f"（权重 {b.get('weight')}；印证 {b.get('hits')} 次 / 推翻 {b.get('misses')} 次）"
            for b in beliefs
        )

    def stats(self) -> dict[str, Any]:
        data = self.load()
        return {
            "beliefs": len(data.get("beliefs", [])),
            "deleted": len(data.get("deleted", [])),
            "updated_at": data.get("updated_at", ""),
        }
