"""Receiver-only independent-process checkpoint inference; reads no donor."""

import json
import sys
from pathlib import Path

from learner import TinyLearner, sha256


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: predictor.py RECEIVER_CHECKPOINT.json < QUERIES.json")
    checkpoint = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    model = TinyLearner.from_checkpoint(checkpoint)
    queries = json.load(sys.stdin)
    if not isinstance(queries, list) or any(not isinstance(x, list) or len(x) != 2 for x in queries):
        raise ValueError("queries must be [subject, relation] pairs")
    print(json.dumps({"checkpoint_sha256": sha256(checkpoint),
                      "predictions": [model.predict(*pair) for pair in queries]}, sort_keys=True))


if __name__ == "__main__":
    main()
