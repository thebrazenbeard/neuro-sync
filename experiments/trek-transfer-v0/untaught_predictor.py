"""Fresh-process recipient-only evaluator for the exploratory untaught split.

Only recipient checkpoint bytes and ordered query symbols cross this process
boundary. This is a mechanical check, not independent authenticated custody.
"""

import json
from pathlib import Path
import sys

from learner import TinyLearner, sha256


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: untaught_predictor.py RECIPIENT.json < QUERIES.json")
    checkpoint = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
    model = TinyLearner.from_checkpoint(checkpoint)
    queries = json.load(sys.stdin)
    if (type(queries) is not list or
            any(type(query) is not list or len(query) != 2 or
                any(type(value) is not str for value in query)
                for query in queries)):
        raise ValueError("queries must be ordered subject/relation pairs")
    print(json.dumps({
        "checkpoint_sha256": sha256(checkpoint),
        "query_sha256": sha256(queries),
        "predictions": [model.predict(*query) for query in queries],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
