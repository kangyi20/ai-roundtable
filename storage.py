"""持久化：JSONL 追加、恢复、导出。"""

import json
from datetime import datetime
from pathlib import Path

from memory import DebateState


class DebateStore:
    """所有辩论状态以 snapshot 行追加到 debates/{debate_id}.jsonl；恢复时读取最后一行。"""

    def __init__(self, root_dir: str = "debates"):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def debate_path(self, debate_id: str) -> Path:
        return self.root / f"{debate_id}.jsonl"

    def append_snapshot(self, debate_id: str, state: DebateState) -> None:
        path = self.debate_path(debate_id)
        # reasoning 正文只在内存/调试日志中使用，不落公共 JSONL
        payload = state.model_dump(exclude={"messages": {"__all__": {"reasoning"}}})
        with open(path, "a", encoding="utf-8") as f:
            line = json.dumps(
                {
                    "type": "snapshot",
                    "timestamp": datetime.now().isoformat(),
                    "payload": payload,
                },
                ensure_ascii=False,
            )
            f.write(line + "\n")

    def load_latest(self, debate_id: str) -> DebateState | None:
        path = self.debate_path(debate_id)
        if not path.exists():
            return None
        last = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
        if last and last.get("type") == "snapshot":
            return DebateState(**last["payload"])
        return None

    def list_debate_ids(self) -> list[str]:
        """返回所有已持久化的 debate_id。"""
        ids = []
        for p in self.root.glob("*.jsonl"):
            ids.append(p.stem)
        return sorted(ids)

    def export(self, debate_id: str, state: DebateState) -> tuple[Path, Path, Path, Path]:
        """写出 transcript.jsonl、score.json、config_snapshot.json。

        每次导出放到 debates/{debate_id}/exports/{timestamp}/ 子目录，避免覆盖历史导出。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.root / debate_id / "exports" / timestamp
        export_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = export_dir / "transcript.jsonl"
        with open(transcript_path, "w", encoding="utf-8") as f:
            for msg in state.messages:
                # 公共 transcript 不含 reasoning 正文
                f.write(json.dumps(msg.model_dump(exclude={"reasoning"}), ensure_ascii=False) + "\n")

        score_path = export_dir / "score.json"
        if state.score:
            score_data = state.score.model_dump()
        else:
            score_data = {"status": "not_scored"}
        score_data["config_snapshot"] = state.config.model_dump()
        score_data["judge_scores"] = state.judge_scores
        score_data["conclusion"] = state.conclusion.model_dump() if state.conclusion else None
        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(score_data, f, ensure_ascii=False, indent=2)

        config_path = export_dir / "config_snapshot.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(state.config.model_dump(), f, ensure_ascii=False, indent=2)

        conclusion_path = export_dir / "conclusion.json"
        if state.conclusion:
            conclusion_data = state.conclusion.model_dump()
        else:
            conclusion_data = {"status": "no_conclusion"}
        with open(conclusion_path, "w", encoding="utf-8") as f:
            json.dump(conclusion_data, f, ensure_ascii=False, indent=2)

        return export_dir, transcript_path, score_path, config_path
