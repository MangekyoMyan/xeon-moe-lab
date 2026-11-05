#!/usr/bin/env python3
"""Pinned, resumable GLM-4.5-Air Q8 download. No inference is started."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from xeon_download_progress import write_progress

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "models/ggml-org--GLM-4.5-Air-GGUF"
REPO = "ggml-org/GLM-4.5-Air-GGUF"
REVISION = "6d3b6301efec560414534beafe6a5cb2bf5940fa"
FILES = (
    {
        "name": "GLM-4.5-Air-Q8_0-00001-of-00002.gguf",
        "size": 9308768,
        "sha256": "f1451f049d87363a5aa99bee10d7db42f024f78e9c9fb1f35914f31a7fecb1ef",
    },
    {
        "name": "GLM-4.5-Air-Q8_0-00002-of-00002.gguf",
        "size": 113602942912,
        "sha256": "312133fe4c7db6a5b339fd2a0e90db23d47099ad7ec47f01f806abe3bece2690",
    },
)


PROGRESS_OWNED = False


def report(phase, item=None, done=0):
    if PROGRESS_OWNED:
        write_progress(DEST, REPO, REVISION, phase, item, done)


def preflight():
    if not shutil.which("curl"):
        raise RuntimeError("curl が見つかりません。")
    remaining = 0
    for item in FILES:
        final = DEST / item["name"]
        partial = DEST / (item["name"] + ".part")
        if final.exists():
            if final.stat().st_size != item["size"]:
                raise RuntimeError(f"既存ファイルの容量が異なります。自動上書きしません: {final}")
            continue
        size = partial.stat().st_size if partial.exists() else 0
        if size > item["size"]:
            raise RuntimeError(f"途中ファイルが予定容量を超えています: {partial}")
        remaining += item["size"] - size
    existing_parent = DEST
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    free = shutil.disk_usage(existing_parent).free
    print(f"モデル: {REPO}, Q8_0")
    print(f"固定リビジョン: {REVISION}")
    print(f"合計: {sum(x['size'] for x in FILES) / 1e9:.2f} GB "
          f"({sum(x['size'] for x in FILES) / 2**30:.2f} GiB)")
    print(f"保存先: {DEST}")
    print(f"残り転送量: {remaining / 1e9:.2f} GB / ディスク空き: {free / 2**30:.1f} GiB",
          flush=True)
    if free < remaining + 5 * 2**30:
        raise RuntimeError("必要容量 + 5 GiB の空きを確保できません。")


def check_sha256(path, item):
    if path.stat().st_size != item["size"]:
        raise RuntimeError(f"容量不一致: {path}")
    print(f"\nSHA256検証中: {path.name}\n"
          "HDDから全体を読むため時間がかかります。", flush=True)
    digest = hashlib.sha256()
    done = 0
    last = time.monotonic()
    last_status = last
    report('verifying', item, 0)
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
            done += len(block)
            if time.monotonic() - last_status >= 1:
                report('verifying', item, done)
                last_status = time.monotonic()
            if time.monotonic() - last >= 30:
                print(f"  検証 {done / item['size']:.1%}", flush=True)
                last = time.monotonic()
    if digest.hexdigest() != item["sha256"]:
        raise RuntimeError(
            f"SHA256不一致: {path}\n"
            "ファイルは調査用に残しました。自動的には削除しません。")
    print("SHA256 OK", flush=True)
    report('file_verified', item, done)


def fetch(item):
    final = DEST / item["name"]
    partial = DEST / (item["name"] + ".part")
    if final.exists():
        check_sha256(final, item)
        return
    for attempt in range(1, 13):
        if partial.exists() and partial.stat().st_size == item["size"]:
            break
        print(f"\n取得: {item['name']}（試行 {attempt}/12）", flush=True)
        report('downloading', item, partial.stat().st_size if partial.exists() else 0)
        # Refresh redirect URLs on every retry; curl -C - resumes the current file.
        url = (f"https://huggingface.co/{REPO}/resolve/{REVISION}/{item['name']}"
               f"?download=true&resume_at={time.time_ns()}")
        result = subprocess.run([
            "curl", "--fail", "--location", "--continue-at", "-",
            "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "180",
            "--max-filesize", str(item["size"]),
            "--proto", "=https", "--proto-redir", "=https",
            "--progress-bar", "--output", str(partial), url,
        ], check=False)
        size = partial.stat().st_size if partial.exists() else 0
        if size > item["size"]:
            raise RuntimeError(f"予定容量を超えるファイルを検出しました: {partial}")
        if size == item["size"]:
            break
        print(f"転送未完了 (curl={result.returncode}, {size}/{item['size']} bytes)。"
              "途中データを保持します。", flush=True)
        if attempt < 12:
            time.sleep(10)
    else:
        raise RuntimeError("転送を中断しました。同じコマンドで続きから再開できます。")
    check_sha256(partial, item)
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    partial.rename(final)


def main():
    global PROGRESS_OWNED
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="容量確認のみ。通信しません。")
    mode.add_argument("--download", action="store_true", help="取得・再開・SHA256検証を実行。")
    mode.add_argument("--verify-only", action="store_true", help="取得済み全ファイルを再検証。")
    args = parser.parse_args()
    preflight()
    if args.check:
        print("事前確認 OK。本体のダウンロードはまだ開始していません。")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("同じモデルのダウンロードまたは検証が実行中です。")
        PROGRESS_OWNED = True
        report('starting')
        # Repeat under lock in case another invocation changed the partial files.
        preflight()
        for item in FILES:
            if args.verify_only:
                check_sha256(DEST / item["name"], item)
            else:
                fetch(item)
        manifest = dict(repository=REPO, revision=REVISION, files=FILES,
                        verified_at_unix=time.time(), entrypoint=FILES[0]["name"])
        pending = DEST / "download-verified.json.tmp"
        pending.write_text(json.dumps(manifest, indent=2) + "\n")
        pending.replace(DEST / "download-verified.json")
        report('complete')
    print("\n完了: 2ファイルとも取得・SHA256検証に成功しました。\n"
          "モデルの結合は不要です。ベンチマークは自動実行しません。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        report('interrupted')
        print("\n中断しました。同じコマンドで再開できます。", file=sys.stderr)
        sys.exit(130)
    except (OSError, RuntimeError) as error:
        report('error')
        print(f"\nエラー: {error}", file=sys.stderr)
        sys.exit(1)
