#!/usr/bin/env python3
"""Summarize matched MXFP4 measurements and preserve their scope."""
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import statistics
import zipfile

from run_mxfp4_study import LAB, OUT, BIN, TYPES, audit, sha
from summarize_quantize_tradeoff import matrix_summary, counters


def stock(folder):
    kind = operation = None
    rows = []
    row = None
    for line in (folder / "stdout.txt").read_text().splitlines():
        if line and not line.startswith(" "):
            kind = line
        elif line.startswith("  ") and not line.startswith("    "):
            operation = line.strip()
        elif " values (" in line:
            row = {"type": kind, "operation": operation, "values": int(line.strip().split()[0])}
            rows.append(row)
        elif row is not None and ":" in line:
            key, value = line.strip().split(":", 1)
            row[key.strip()] = float(value.strip().split()[0])
    assert len(rows) == 2 * len(TYPES)
    assert {r["type"] for r in rows} == set(TYPES)
    return rows


def main():
    with (LAB / "results/glm-air-q8-20261020/.experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert json.loads((OUT / "complete.json").read_text())["complete"]
        matrices = {p.name: matrix_summary(p) for p in sorted(OUT.glob("matvec-*/"))}
        for group in matrices.values():
            for item in group.values():
                meta = item["metadata"]
                times = [s["seconds_per_matvec"] for s in item["samples"]]
                item["effective_GOPS"] = 2 * meta["k"] * meta["m"] / item["seconds_per_matvec"] / 1e9
                item["effective_TOPS"] = item["effective_GOPS"] / 1000
                item["min_max_seconds_per_matvec"] = [min(times), max(times)]
                item["relative_MAD"] = statistics.median(abs(t - statistics.median(times)) for t in times) / statistics.median(times)
        stocks = {p.name: stock(p) for p in sorted(OUT.glob("quant-perf-*/"))}
        stock_ratios = {}
        for regime in ("cache", "stream"):
            ops = {}
            for op in ("vec_dot_q", "dequantize_row_q"):
                ratios = collections.defaultdict(list)
                for rep in range(2):
                    rows = [r for r in stocks[f"quant-perf-{regime}-{rep}"] if r["operation"] == op]
                    base = next(r["avg cycles/32 vals"] for r in rows if r["type"] == "q8_0")
                    for row in rows:
                        ratios[row["type"]].append(base / row["avg cycles/32 vals"])
                ops[op] = {k: statistics.median(v) for k, v in ratios.items()}
            stock_ratios[regime] = ops
        profiles = {k: counters(OUT / ("counters-" + k)) for k in ("q8_0", "q4_K", "mxfp4")}
        telemetry = []
        for p in sorted(OUT.glob("*/run.json")):
            row = json.loads(p.read_text())
            assert row["exit_code"] == 0
            observations = row["observations"]
            temperatures = [v for r in observations for v in r.get("cpu_temperatures_C", {}).values()]
            telemetry.append({"label": p.parent.name, "max_cpu_C": max(temperatures) if temperatures else None,
                              "max_sampled_RSS_MiB": max((r["rss_kib"] for r in observations), default=0) / 1024})
        after = audit()
        before = json.loads((OUT / "before.json").read_text())
        active = []
        for p in Path("/proc").iterdir():
            if p.name.isdigit():
                try:
                    comm = (p / "comm").read_text().strip()
                    if comm.startswith(("llama", "quantize-matvec", "test-quantize")) or comm == "perf":
                        active.append({"pid": int(p.name), "comm": comm})
                except OSError:
                    pass
        assert after == before and not active
        audit_report = {"runtime_and_settings_unchanged": after == before,
                        "active_benchmark_processes": active, **after}
        (OUT / "after-audit.json").write_text(json.dumps(audit_report, indent=2) + "\n")
        report = {"complete": True, "client_date": "2026-10-22", "timezone": "Asia/Tokyo",
                  "matrix_comparisons": matrices, "stock_test_results": stocks,
                  "stock_ratios_vs_q8": stock_ratios, "counter_results": profiles,
                  "telemetry": telemetry, "audit": audit_report,
                  "full_model_tested": False, "model_downloaded": False,
                  "limitations": [
                      "These are CPU kernel measurements, not full-model tokens/s or quality comparisons.",
                      "Synthetic deterministic uniform inputs and 256 distinct row patterns; full matrix storage is physical and not shared/deduplicated.",
                      "MXFP4 activation operand is Q8_0; not native FP4-by-FP4 tensor-core arithmetic.",
                      "Cache timing includes L1/L2/cache and instruction costs, not only arithmetic.",
                      "Direct matvec excludes per-layer synchronization, routing, attention, and graph scheduling.",
                      "Stock test reports TSC ticks as cycles and logical GiB/s as GB/s; neither is a hardware-counter DRAM bandwidth.",
                      "Stock type order is enum order regardless of --type argument order. Matvec format order alternates forward/reverse over six rounds.",
                      "Stock large test streams both long input vectors. Matvec reuses one short activation vector.",
                      "Hardware counters are separate 15-second diagnostic windows. IMC is system-wide; event fractions overlap.",
                      "One matrix row pattern or one model architecture does not predict all MXFP4 models.",
                      "All formats use their own quantized approximation; errors are checked against dequantized operands, not an equal-accuracy model baseline."
                  ]}
        (OUT / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        cache = matrices["matvec-k4096-cache-t24"]
        stream = matrices["matvec-k4096-stream-t24"]
        mx = stream["mxfp4"]
        mx_cache = cache["mxfp4"]["speed_ratio_vs_q8"]
        mx_q4 = stream["q4_K"]["seconds_per_matvec"] / mx["seconds_per_matvec"]
        text = f"""MXFP4のCPU処理負担とメモリ削減効果（2026-10-22、日本時間）

実測結果
このXeonの現在のAVX実装では、MXFP4の内積を正常に実行できた。
24スレッド、小配列の内積速度はQ8_0比 {mx_cache:.3f}倍。
同じ要素数の大きな行列ではQ8_0比 {mx['speed_ratio_vs_q8']:.3f}倍、Q4_K比 {mx_q4:.3f}倍。
処理時間と速度は逆数。速度比1.00未満は遅く、1.00超は速い。
これはカーネルの速度比較であり、モデル全体の生成速度ではない。

主比較: K4096、24スレッド、6反復の中央値
キャッシュ条件 M24、大容量条件 M65536。同じ重み要素数で比較。
形式       重みQ8比    キャッシュ速度Q8比   大容量速度Q8比   大容量ms    実効GOPS
"""
        for k in TYPES:
            row = stream[k]
            text += f"{k:<10} {row['weight_size_ratio_vs_q8']:>7.3f}       {cache[k]['speed_ratio_vs_q8']:>7.3f}           {row['speed_ratio_vs_q8']:>7.3f}       {row['seconds_per_matvec']*1000:>7.3f}      {row['effective_GOPS']:>7.1f}\n"
        text += "\nスレッド数別: K4096のMXFP4速度比（各条件のQ8_0=1.00）\n"
        for t in (1, 12, 24):
            c = matrices[f"matvec-k4096-cache-t{t}"]["mxfp4"]["speed_ratio_vs_q8"]
            s = matrices[f"matvec-k4096-stream-t{t}"]["mxfp4"]["speed_ratio_vs_q8"]
            text += f"{t:2d}スレッド: 小配列 {c:.3f}倍 / 大容量 {s:.3f}倍\n"
        text += "\n行列形状を変えた追加測定（24スレッド）\n"
        for k in (1408, 2880, 16384):
            row = matrices[f"matvec-k{k}-stream-t24"]["mxfp4"]
            text += f"K{k}, M{row['metadata']['m']}: MXFP4はQ8_0比 {row['speed_ratio_vs_q8']:.3f}倍、実効 {row['effective_GOPS']:.1f} GOPS\n"
        text += "\n実DRAMカウンター（K4096、大容量、24スレッド、15秒）\n形式       読出GB/s     IPC    L2未完了読出しを伴う停止/cycles\n"
        for k, row in profiles.items():
            text += f"{k:<10} {row['dram_read_GB_s']:>7.2f}      {row['ipc']:.3f}            {row['stall_l2_pending_fraction']*100:>6.2f}%\n"
        text += "\ntest-quantize-perf（CPU0/node0、各条件2回）\n形式       小配列内積速度Q8比  大配列内積速度Q8比  小配列FP32展開速度Q8比\n"
        for k in TYPES:
            text += f"{k:<10} {stock_ratios['cache']['vec_dot_q'][k]:>7.3f}              {stock_ratios['stream']['vec_dot_q'][k]:>7.3f}               {stock_ratios['cache']['dequantize_row_q'][k]:>7.3f}\n"
        text += """
解釈と測定範囲
MXFP4は32要素を17バイト（4.25 bit/weight）で保存。Q8_0は34バイト、Q4_K/Q4_0/IQ4_NLは平均18バイト相当。
計算側の負担と、重みを半分にする読み出し削減が同時に働く。
FP32へ全展開するdequantize_row_qは、内積を直接計算するvec_dot_qとは別の処理。
MXFP4のCPU内積はFP32行列全体を作らず、4bit値をSIMDの表引きで展開し、Q8_0の活性値との積を計算する。
従って、FP32全展開の遅さをそのままLLM推論の遅さと解釈しない。
キャッシュ条件は演算・展開の負担を見やすくする条件であり、純粋な計算能力の上限測定ではない。
既存の通常ランタイム、CPUクロック設定、NUMA balancing、THPを変更していない。GPUは使用していない。
カーネル性能を問う既存テストが使えたため、今回の検証にモデルの追加ダウンロードは不要だった。

再現条件
Xeon E5-2630L v2 x2、12物理/24論理CPU。AVX/SSSE3/F16C、AVX2/FMAなし。
既存GLM用libggml-cpu.soへリンクし、比較形式間で同一ライブラリを使用。
test-quantize-perfは前回のRHS形式修正版をそのままコピー。MXFP4の右辺はQ8_0、Q4_Kの右辺はQ8_K。
小配列4096要素 x 500,000回、大配列33,554,432要素 x 20回。各演算の初回5回はウォームアップ。
行列テストは実際に全行の物理メモリを確保。型ごとに同じF32入力を量子化し、準備時間を計測から除外。
CPU0-23にワーカー固定、メモリnode0/1へinterleave。重みの割当のみMADV_NOHUGEPAGE。
行列形式の測定順を正順/逆順で交互にし、6反復の中央値を採用。
K1408/K2880は256で割り切れないためQ4_Kを除外。他の4形式は測定。
カウンター計測は行列の準備・ウォームアップ終了後の別走行。主比較の速度はカウンター観察なしの走行。
DRAMカウンターはホスト全体の値。停止率は他の指標と重複するため足し合わせない。
内積の正確性は量子化した両入力を独立にF32へ展開し、倍精度の参照内積と照合。
これは実装の計算検証であり、元のモデルと同じ回答品質を保証するテストではない。
実効GOPS = 2*K*M / 1回の実測秒数 / 1e9。1 TOPS = 1000 GOPS。
有効な内積の乗算・加算のみを2演算として数え、展開命令数は演算数に加えず、その実行時間は含める。

対象外
モデル全体のtokens/s、入力処理全体、注意機構、MoEルーティング、レイヤー間同期、回答品質は未測定。
同じ名前のMXFP4でも、CPU実装、モデル形状、重みの使い回し、同時処理数が変われば結果も変わる。
今回の速度倍率をそのままGLMやGPT-OSS等のモデル全体へ適用しない。

参照実装
同梱runtime-mxfp4-kernel.txtは実際に測定したローカルソースの該当部分。
上流の参照先: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/x86/quants.c
主測定とRHS修正のソース、ビルドコマンド、ハッシュ、全ログ、CPU/NUMA構成を保存。
"""
        max_temp = max([r["max_cpu_C"] or 0 for r in telemetry] + [r["maximum_cpu_temperature_C"] for r in profiles.values()])
        text += f"\n検証: 全実行終了コード0、計算参照照合成功、診断中のディスク読出/major/minor fault/熱スロットル増加0。最大観測CPU温度 {max_temp:.1f} C。\n"
        text += "既存ランタイム・OS設定のハッシュ/値は前後一致。測定プロセスの残存なし。\n"
        (OUT / "REPORT_JA.txt").write_text(text)
        (OUT / "WORK_STATE.txt").write_text("Completed CPU-only MXFP4 kernel study, counters, and report. No runtime or OS changes. No active benchmarks.\n")
        prefs_path = LAB / "experiment-preferences.json"
        prefs = json.loads(prefs_path.read_text())
        prefs["before_mxfp4_study"] = prefs["current_work"]
        prefs["current_authorization"] = "User requested MXFP4 CPU cost testing, using a test if available or a lightweight model otherwise. Completed CPU-only kernel tests."
        prefs["current_work"] = (f"Completed MXFP4 study using existing AVX kernels. K4096/t24 cache speed {mx_cache:.3f}x Q8; large matrix {mx['speed_ratio_vs_q8']:.3f}x Q8 and {mx_q4:.3f}x Q4_K. Full-model tok/s and quality not tested. No model download, inference code/OS changes, or GPU use. All benchmarks stopped.")
        prefs["mxfp4_study_report"] = str(OUT / "REPORT_JA.txt")
        prefs_path.write_text(json.dumps(prefs, ensure_ascii=False, indent=2) + "\n")
        account = pwd.getpwnam("codex")
        paths = [OUT, *OUT.rglob("*"), BIN, *BIN.rglob("*")]
        for p in paths:
            if not p.is_symlink(): os.chown(p, account.pw_uid, account.pw_gid)
        package = LAB.parent / "xeon-mxfp4-study-20261022.zip"
        entries = [(p, "results/" + str(p.relative_to(OUT))) for p in OUT.rglob("*") if p.is_file()]
        entries += [(p, "code/" + p.name) for p in BIN.iterdir() if p.is_file()]
        helpers = ["run_mxfp4_study.py", "observe_mxfp4_study.py", "summarize_mxfp4_study.py",
                   "run_quantize_tradeoff.py", "observe_quantize_tradeoff.py", "summarize_quantize_tradeoff.py",
                   "bench_deepseek_light.py", "glm_q8_experiment.py", "observe_glm_counters_light.py"]
        entries += [(LAB / name, "helpers/" + name) for name in helpers]
        manifest = {name: sha(p) for p, name in entries}
        with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as z:
            for p, name in entries: z.write(p, name)
            z.writestr("SHA256.json", json.dumps(manifest, indent=2) + "\n")
        with zipfile.ZipFile(package) as z:
            assert z.testzip() is None
            assert all(hashlib.sha256(z.read(name)).hexdigest() == value for name, value in manifest.items())
        os.chown(package, account.pw_uid, account.pw_gid)
        print(json.dumps({"package": str(package), "bytes": package.stat().st_size, "sha256": sha(package),
            "max_cpu_C": max_temp, "MXFP4_cache_vs_Q8": mx_cache,
            "MXFP4_stream_vs_Q8": mx["speed_ratio_vs_q8"], "MXFP4_stream_vs_Q4_K": mx_q4}, indent=2))


if __name__ == "__main__":
    main()
