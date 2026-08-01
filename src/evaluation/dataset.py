import json
from pathlib import Path

from src.evaluation.models import EvalExample, EvalTurn, QuestionType


def load_eval_dataset(path: str | Path) -> list[EvalExample]:
    path = Path(path)
    examples = []
    with path.open("r" , encoding="utf-8") as f:
        for line_num , line in enumerate(f , start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            try:
                raw["question_type"] = QuestionType(raw["question_type"])
                raw["follow_ups"] = [EvalTurn(**t) for t in raw.get("follow_ups", [])]
                examples.append(EvalExample(**raw))
            except (KeyError , ValueError) as e:
                raise ValueError(f"Error parsing line / Invalid format at line {line_num} in {path}: {e}")
        return examples
    