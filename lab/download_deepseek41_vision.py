#!/usr/bin/env python3
"""Download only the matching OrcaRouter F16 vision projector; never start/stop inference."""
import argparse
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "models/deepseek41-vision"
REPO = "orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF"
REVISION = "62168fc534d42dbe85843ef6b71bfb6384e32ba5"
NAME = "mmproj-DeepSeek-V4.1-Flash-1017-abliterated-F16.gguf"


def verify(path, size, digest):
    if path.stat().st_size != size:
        raise RuntimeError("容量が一致しません。途中ファイルを保持します。")
    print("SHA256検証中…", flush=True)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise RuntimeError("GGUFではありません。ファイルを保持します。")
        stream.seek(0)
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    if h.hexdigest() != digest:
        raise RuntimeError("SHA256不一致です。自動削除・上書きはしません。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="追加ファイル1個だけを取得・再開・検証")
    args = parser.parse_args()
    print(f"対象: {NAME}\n保存先: {DEST}\n約971 MB。本体モデル・API設定は変更しません。")
    if not args.download:
        print("取得する場合は --download を指定してください。通信はまだ行っていません。")
        return
    if not shutil.which("curl"):
        raise RuntimeError("curl がありません。")
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token = os.environ.get("HF_TOKEN") or getpass.getpass("Hugging FaceのReadトークン（非表示入力）: ").strip()
        if not token or any(c in token for c in "\r\n"):
            raise RuntimeError("有効なReadトークンを入力してください。")
        # Inspect only the root file list at an immutable revision, not any model weights.
        req = urllib.request.Request(
            f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=false&limit=100",
            headers={"Authorization": "Bearer " + token},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            listing = json.load(response)
        entry = next((x for x in listing if x.get("path") == NAME), None)
        if entry is None:
            raise RuntimeError("固定リビジョンのファイル一覧に対象がありません。")
        size = entry["size"]
        digest = entry.get("lfs", {}).get("oid", "")
        if not isinstance(size, int) or not 0 < size < 2_000_000_000 or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError("配布元の容量・SHA256を確認できません。取得を開始しません。")
        final = DEST / NAME
        partial = DEST / (NAME + ".part")
        if final.exists():
            verify(final, size, digest)
        else:
            have = partial.stat().st_size if partial.exists() else 0
            if have > size:
                raise RuntimeError("途中ファイルが予定容量を超えています。保持して停止します。")
            if shutil.disk_usage(DEST).free < size - have + 100 * 1024 * 1024:
                raise RuntimeError("空き容量が不足しています。")
            if have < size:
                print(f"取得: {size / 1e6:.1f} MB（取得済み {have / 1e6:.1f} MB）", flush=True)
                result = subprocess.run([
                    "curl", "--fail", "--location", "--continue-at", "-",
                    "--header", "@-", "--connect-timeout", "30", "--retry", "3",
                    "--speed-limit", "1024", "--speed-time", "180",
                    "--proto", "=https", "--proto-redir", "=https",
                    "--max-filesize", str(size), "--progress-bar", "--output", str(partial),
                    f"https://huggingface.co/{REPO}/resolve/{REVISION}/{NAME}?download=true",
                ], input="Authorization: Bearer " + token + "\n", text=True, check=False)
                if result.returncode:
                    raise RuntimeError(f"転送未完了 (curl={result.returncode})。同じコマンドで再開できます。401/403の場合は利用条件への同意・Read権限を確認してください。")
            verify(partial, size, digest)
            partial.rename(final)
        (DEST / "download-verified.json").write_text(json.dumps({
            "repository": REPO, "revision": REVISION, "file": NAME,
            "size": size, "sha256": digest,
        }, indent=2) + "\n")
        print(f"完了・SHA256 OK: {final}\nモデルの停止・起動・設定変更は行っていません。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断しました。同じコマンドで再開できます。", file=sys.stderr)
        sys.exit(130)
    except (OSError, RuntimeError, ValueError, KeyError, EOFError) as error:
        print(f"エラー: {error}", file=sys.stderr)
        sys.exit(1)
