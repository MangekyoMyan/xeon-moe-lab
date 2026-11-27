"""Tokenize objective answers after the service window, without running inference."""
import http.client
import json
from pathlib import Path
from analyze import R, CHATS, read, sha


def main():
    assert read(R/"MEASURED.json")["complete"]
    assert not any(read(p)["restore_required"] for p in R.glob("WINDOW_SUFFIX-*.json"))
    cfg_path = R.parent/"deepseek41-bringup-20261101/server-config.json"
    assert sha(cfg_path) == read(R/"DESIGN.json")["runtime_config"]
    cfg = read(cfg_path)
    auth = {"Authorization":"Bearer "+Path(cfg["api_key_file"]).read_text().strip(),"Content-Type":"application/json"}
    f = read(R/"FIXTURES.json")
    vocab = {}
    def tokenize(s):
        if s in vocab:
            return vocab[s]
        con = http.client.HTTPConnection(cfg["host"],cfg["port"],timeout=20)
        try:
            con.request("POST","/tokenize",json.dumps({"content":s,"add_special":False,"parse_special":True}),auth)
            res = con.getresponse()
            assert res.status == 200
            vocab[s] = json.loads(res.read())["tokens"]
        finally:
            con.close()
        return vocab[s]
    cases = {}
    for label in CHATS:
        case = f["edit"] if label.endswith("-edit") else f["cases"][label.split("-")[0]]
        answer = case["expected"]
        spellings = [json.dumps(answer),json.dumps(answer,separators=(",",":"))] if isinstance(answer,list) else [answer]
        cases[label] = {text:tokenize(text) for text in spellings}
    state = {"method":"restored production /tokenize, add_special=false, parse_special=true; synthetic answer strings only",
             "expected_spellings":cases}
    (R/"TARGET_TOKENS.json").write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
