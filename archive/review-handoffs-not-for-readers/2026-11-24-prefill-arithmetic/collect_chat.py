"""Copy only evidence from the synthetic requests made by this study."""
import hashlib
import json
from pathlib import Path

from shared import HERE, adopt, read


def main():
    secret = adopt.KEY.read_bytes().strip()
    assert len(secret) > 12
    for label, _ in read(HERE / "CHAT_DESIGN.json")["periods"]:
        src = HERE / "runs" / ("chat-" + label)
        assert read(src / "STOP.json")["child_return"] == 0
        rows = [json.loads(s) for s in (src / "results.jsonl").read_text().splitlines()]
        for row in rows:
            case = row["case"]
            assert case in {"warmup", *read(HERE / "CHAT_DESIGN.json")["prompts"]}
            to = HERE / "raw-chat" / label / case
            to.mkdir(parents=True, exist_ok=True)
            for name in ("request.effective.json", "summary.json", "events.jsonl", "content.txt", "response_body.bin"):
                data = (src / case / name).read_bytes()
                assert secret not in data
                if name == "response_body.bin":
                    assert hashlib.sha256(data).hexdigest() == row["response_sha256"]
                if (to / name).exists():
                    assert (to / name).read_bytes() == data
                else:
                    (to / name).write_bytes(data)
    print("Copied 36 synthetic streams and receipts")


if __name__ == "__main__":
    main()
