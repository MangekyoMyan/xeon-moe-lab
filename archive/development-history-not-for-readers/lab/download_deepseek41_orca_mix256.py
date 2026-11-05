#!/usr/bin/env python3
"""Pinned, resumable OrcaRouter DeepSeek4.1 Mix-256G download. No inference is started."""
import argparse
import getpass
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
DEST = ROOT / "models/orcarouter--DeepSeek-V4.1-Flash-1017-abliterated-GGUF"
REPO = "orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF"
REVISION = '349470a09c1fca35862b0796e726b9fe00a947df'
FILES = ({'name': 'Mix-256G/DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-00001-of-00007.gguf', 'size': 709084736, 'sha256': '2cb066de483877f75b3dc79936b1fccf34aeecc60461701e5c39c98d7f922e03'}, {'name': 'Mix-256G/DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-00002-of-00007.gguf', 'size': 110595290592, 'sha256': '342b6ee30a33f02fcef1d261ec7e700907f8b114370ebe90c228339f686a8c1b'}, {'name': 'Mix-256G/DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-00003-of-00007.gguf', 'size': 46798562592, 'sha256': 'b2583b9da6c3598f43d22714f3216709ab3160c618ad94e36d491a3bb20b5d2b'}, {'name': 'Mix-256G/DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-00004-of-00007.gguf', 'size': 47706985792, 'sha256': 'de9c5f8f3462c7df3db7bb2b79301eeab9d7874f581a9fa574ae4fffe127c602'}, {'name': 'Mix-256G/DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-00005-of-00007.gguf', 'size': 46565998336, 'sha256': '1202e0ae311beb38de1b0899ab3cbba485385299a62e8e136e01fa922f0aeb13'}, {'name': 'Mix-256G/DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-00006-of-00007.gguf', 'size': 47857178816, 'sha256': 'dc7ce9ee6910e8f46a201e54c96e47b6b3bc9380c88b6187be3c860a0d0b1074'}, {'name': 'Mix-256G/DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-00007-of-00007.gguf', 'size': 46145195616, 'sha256': 'ed00667671a8ca5c48da45e022b6482fc2d88a21848a98d26fba1d1f2848b429'})
TOKEN = None
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
    print(f"モデル: {REPO}, Mix-256G")
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
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        check_sha256(final, item)
        return
    global TOKEN
    if TOKEN is None:
        TOKEN = os.environ.get("HF_TOKEN") or getpass.getpass("Hugging FaceのReadトークン（非表示入力）: ").strip()
        if not TOKEN or any(c in TOKEN for c in "\r\n"):
            raise RuntimeError("有効なトークンを入力してください。")
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
            "--header", "@-",
            "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "180",
            "--max-filesize", str(item["size"]),
            "--proto", "=https", "--proto-redir", "=https",
            "--progress-bar", "--output", str(partial), url,
        ], input="Authorization: Bearer " + TOKEN + "\n", text=True, check=False)
        if result.returncode == 22:
            raise RuntimeError("HTTPエラーです。401/403なら、モデルページの利用条件への同意とトークンのRead権限を確認してください。途中ファイルは保持しています。")
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
    print("\n完了: Mix-256Gの全7ファイルの取得・SHA256検証に成功しました。\n"
          "モデルの結合は不要です。ベンチマークは自動実行しません。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        report('interrupted')
        print("\n中断しました。同じコマンドで再開できます。", file=sys.stderr)
        sys.exit(130)
    except (OSError, RuntimeError, EOFError) as error:
        report('error')
        print(f"\nエラー: {error}", file=sys.stderr)
        sys.exit(1)
