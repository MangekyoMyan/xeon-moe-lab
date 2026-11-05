#!/usr/bin/env python3
"""Summarize quantization costs, keeping payload rates distinct from DRAM counters."""
import collections
import csv
import json
from pathlib import Path
import statistics

from run_quantize_tradeoff import LAB, OUT


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def matrix_summary(folder):
    rows = json_lines(folder / "stdout.txt")
    prepared = {r["prepared"]: r for r in rows if "prepared" in r}
    result = {}
    for kind, metadata in prepared.items():
        samples = [r for r in rows if r.get("type") == kind]
        assert len(samples) == 6
        result[kind] = {"metadata": metadata,
                        "seconds_per_matvec": statistics.median(r["seconds_per_matvec"] for r in samples),
                        "Gweights_s": statistics.median(r["Gweights_s"] for r in samples),
                        "logical_weight_GB_s": statistics.median(r["weight_GB_s"] for r in samples),
                        "samples": samples}
    for kind, item in result.items():
        item["speed_ratio_vs_q8"] = result["q8_0"]["seconds_per_matvec"] / item["seconds_per_matvec"]
        item["weight_size_ratio_vs_q8"] = item["metadata"]["bytes"] / result["q8_0"]["metadata"]["bytes"]
    return result


def stock_summary(folder):
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
    assert len(rows) == 16
    return rows


def counters(folder):
    report = json.loads((folder / "observations.json").read_text())
    assert report["valid"]

    def rows(name):
        return [r for r in csv.reader((folder / name).read_text().splitlines()) if r and not r[0].startswith("#")]

    core = rows("core.csv")
    assert all(float(r[-3]) >= 99.9 and "<not" not in r[0] for r in core)
    events = {",".join(r[2:-4]): float(r[0]) for r in core}
    imc = collections.defaultdict(lambda: [0, 0])
    for r in rows("imc.csv"):
        assert float(r[-3]) >= 99.9 and "<not" not in r[2]
        key = r[1] + "/" + ",".join(r[4:-4])
        imc[key][0] += float(r[2])
        imc[key][1] += float(r[-4]) / 1e9
    rates = {k: count * 64 / seconds / 1e9 for k, (count, seconds) in imc.items()}
    telemetry = report["telemetry"]

    def read_bytes(item):
        return int(next(line.split(":")[1] for line in item["io"].splitlines() if line.startswith("read_bytes:")))

    data = {
        "valid": True, "all_events_running_percent_at_least": 99.9,
        "seconds": events["duration_time"] / 1e9,
        "dram_read_GB_s": sum(v for k, v in rates.items() if "umask=0x03" in k),
        "dram_write_GB_s": sum(v for k, v in rates.items() if "umask=0x0c" in k),
        "ipc": events["instructions:u"] / events["cycles:u"],
        "stall_l2_pending_fraction": events["cycle_activity.stalls_l2_pending:u"] / events["cycles:u"],
        "stall_total_fraction": events["cycle_activity.stalls_total:u"] / events["cycles:u"],
        "major_faults": events["major-faults"], "minor_faults": events["minor-faults"],
        "cpu_migrations": events["cpu-migrations"],
        "window_disk_read_bytes": read_bytes(telemetry[-1]) - read_bytes(telemetry[0]),
        "maximum_cpu_temperature_C": max(v for row in telemetry for v in row["cpu_temperatures_mC"].values()) / 1000,
        "throttle_counter_delta_sum": sum(telemetry[-1]["throttle_counts"][k] - v
                                           for k, v in telemetry[0]["throttle_counts"].items()),
        "raw_core_events": events, "channel_rates_GB_s": rates,
    }
    assert data["window_disk_read_bytes"] == 0 and data["major_faults"] == 0 and data["minor_faults"] == 0
    assert data["throttle_counter_delta_sum"] == 0
    return data


def main():
    assert json.loads((OUT / "complete.json").read_text())["complete"]
    matrices = {f.name: matrix_summary(f) for f in sorted(OUT.glob("matvec-k*")) if f.is_dir()}
    stock = {f.name: stock_summary(f) for f in sorted(OUT.glob("quant-perf-*")) if f.is_dir()}
    stock_normalized = {}
    for regime, names in (("cache_long", ["quant-perf-cache-long-0", "quant-perf-cache-long-1"]),
                          ("stream", ["quant-perf-stream-0", "quant-perf-stream-1"])):
        ratios = collections.defaultdict(list)
        for name in names:
            rows = [r for r in stock[name] if r["operation"] == "vec_dot_q"]
            baseline = next(r["avg cycles/32 vals"] for r in rows if r["type"] == "q8_0")
            for r in rows:
                ratios[r["type"]].append(baseline / r["avg cycles/32 vals"])
        stock_normalized[regime] = {kind: statistics.median(values) for kind, values in ratios.items()}
    profiles = {kind: counters(OUT / ("counters-" + kind)) for kind in ("q8_0", "q6_K", "q4_K", "q3_K")}
    cache = matrices["matvec-k4096-cache-t24"]
    stream = matrices["matvec-k4096-stream-t24"]
    kinds = ["q8_0", "q6_K", "q5_K", "q4_K", "q3_K", "q2_K", "q4_0", "iq4_xs"]
    table = []
    for kind in kinds:
        table.append({"type": kind, "weight_size_ratio_vs_q8": stream[kind]["weight_size_ratio_vs_q8"],
                      "cache_speed_ratio_vs_q8": cache[kind]["speed_ratio_vs_q8"],
                      "stream_speed_ratio_vs_q8": stream[kind]["speed_ratio_vs_q8"],
                      "stream_matvec_ms": stream[kind]["seconds_per_matvec"] * 1000})
    telemetry = []
    for path in sorted(OUT.glob("*/run.json")):
        if path.parent.name == "validation-first-attempt": continue
        run = json.loads(path.read_text()); assert run["exit_code"] == 0
        temps = [t for row in run["observations"] for t in row.get("cpu_temperatures_C", {}).values()]
        telemetry.append({"name": path.parent.name, "max_sampled_cpu_temperature_C": max(temps) if temps else None,
                          "peak_sampled_RSS_MiB": max(r["rss_kib"] for r in run["observations"]) / 1024})
    data = {"complete": True, "boot_id": json.loads((OUT / "before.json").read_text())["settings"]["/proc/sys/kernel/random/boot_id"],
            "matrix_comparisons": matrices, "stock_test_results": stock, "stock_normalized": stock_normalized,
            "counter_results": profiles,
            "main_table": table, "telemetry": telemetry,
            "source": str(LAB / "review-glm-q8-columns4-20261020/source"),
            "library": str(LAB / "review-glm-q8-columns4-20261020/build/bin/libggml-cpu.so"),
            "full_model_quantization_or_tok_s_tested": False,
            "limitations": ["Synthetic data, same weights count, not equal model quality.",
                "Stock test uses one CPU and two large vectors; supplemental matvec reuses one activation vector.",
                "Stock GB/s label uses GiB and a logical numerator, not measured DRAM traffic.",
                "Stock cycles are RDTSCP ticks, not frequency-normalized executed core cycles.",
                "Cache tests are not pure arithmetic: L1/L2 accesses and instruction decode are included.",
                "Direct kernels exclude graph dispatch, routing, attention, synchronization per layer, and repacking.",
                "K-type tensor formats are not complete mixed-quantization GGUF presets such as Q4_K_M.",
                "15-second diagnostic counters overlap; percentages are not additive wall-time components.",
                "Earlier GLM measurement was a different workload and boot; 32.8 GB/s was not an absolute hardware ceiling."]}
    (OUT / "REPORT.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    lines = ["形式       Q8比の重み容量  キャッシュ内速度比  大容量読み出し速度比  大容量1回(ms)"]
    for row in table:
        lines.append(f"{row['type']:<10} {row['weight_size_ratio_vs_q8']*100:>7.2f}%          {row['cache_speed_ratio_vs_q8']:>5.2f}x              {row['stream_speed_ratio_vs_q8']:>5.2f}x           {row['stream_matvec_ms']:>7.3f}")
    counter_lines = ["形式     DRAM read GB/s   IPC    L2未完了読出しを伴う停止/cycles"]
    for kind, item in profiles.items():
        counter_lines.append(f"{kind:<8} {item['dram_read_GB_s']:>8.2f}        {item['ipc']:.3f}          {item['stall_l2_pending_fraction']*100:>6.2f}%")
    max_temp = max(r['max_sampled_cpu_temperature_C'] or 0 for r in telemetry)
    stock_lines = ["形式       小配列内積速度比   大配列内積速度比（1スレッド、両入力を読む）"]
    for kind in kinds:
        stock_lines.append(f"{kind:<10} {stock_normalized['cache_long'][kind]:>7.2f}x           {stock_normalized['stream'][kind]:>7.2f}x")
    report = """Q8から低ビット化した場合のCPU負担とメモリ効果（2026-10-21、日本時間）

結論
このAVX Xeonでは、低ビット化の計算負担と読み出し量削減の両方が観測された。
24スレッド・大容量の行列では、Q6_K/Q5_K/Q4_K/Q3_K/Q2_Kは全てQ8_0より速かった。
Q6_Kはキャッシュ内ではQ8より遅いが、大容量では約1.33倍速い。
Q4_Kはキャッシュ内でも約1.07倍、大容量では約1.93倍速い。
Q3_KはQ4_Kより小さいのに遅い。Q3_Kでは計算側の負担が強く出ている。
従って、古いCPUだからQ8を下げると一律に遅くなる、という結果ではない。
数値はカーネル単体の比較。実GLMの生成tokens/sの改善率ではない。

主比較（各条件のQ8_0を1.00、同じ要素数、速度比は大きい方が速い）
""" + "\n".join(lines) + """

測定条件
Xeon E5-2630L v2 x2、12物理/24論理CPU、DDR3-1600 32 GB x4。
AVX/F16C/SSSE3、AVX2/FMAなし。GPUなし。
既存GLM用buildの同一libggml-cpu.soへリンク。通常ランタイムを変更していない。
直前GLM検証からは再起動されており、今回のboot IDを各ログへ保存した。

1. test-quantize-perf
元ソースのQ8を実行したうえで、測定用コピーの内積入力形式を修正した。
CPU0へ固定、メモリnode0。Q8/Q6_K/Q5_K/Q4_K/Q3_K/Q2_K/Q4_0/IQ4_XS。
キャッシュ内: 4096要素、初回10,000反復 x2、補測500,000反復 x2。
大容量: 33,554,432要素、20反復 x2。各演算のウォームアップは元ソース通り5回。
vec_dot_qとdequantize_row_qを測定。量子化ファイルの作成時間を推論時間に混ぜていない。
短い測定にクロック立ち上がり・タイマー粒度の影響が見えたため、補測を追加した。
以下は実際のtest-quantize-perf修正版のvec_dot_q結果。各実行内のQ8に対する比の中央値。
小配列側には500,000反復の補測を使う。小配列でも演算呼び出し・計時の負担を含む。
""" + "\n".join(stock_lines) + """

ログのcyclesはRDTSCPの基準クロック刻み。CPU実行コアの実サイクル値とは区別する。
ログのGB/sはコード上1024^3で割ったGiB/s。論理的な処理量の指標であり、DRAM実測値ではない。
特に内積では両オペランドを読むのに、表示のquantized throughputは片側のサイズだけを使う。

付属テストの修正点
手元の元ソースはvec_dot_qの右入力も左入力の量子化形式で作っていた。
Q4_Kの内積カーネルはQ4_K x Q8_Kなので、そのままでは入力形式が一致しない。
右入力はvec_dot_typeのfrom_floatで作るよう、独立コピーを修正した。
Q8_0 x Q8_0は元のまま形式が合うため、元版Q8の測定も残した。
実行ライブラリ・モデルファイルは変更していない。差分はtest-quantize-perf-rhs.patch。
これは手元の版についての確認であり、すべての配布版に同じ問題があるという主張ではない。

2. 補足の行列ベクトル内積
標準テストは1スレッドであり、大容量時には右入力も長くなる。
生成に近づけるため、実際のggml vec_dotを使い、短い右入力を全行で使い回す測定を追加した。
K=4096、キャッシュ側M=24、大容量側M=65536。
大容量は268,435,456個の重み。Q8は272 MiB、Q4_Kは144 MiB、Q2_Kでも84 MiB。
最小形式でも各ソケットのLLCに収まらないサイズを選んだ。
キャッシュ条件は24スレッドで各ワーカー1行、全形式で重みと右入力が小さい。
NUMA interleave=0,1、明示したCPUへ固定、匿名メモリは4 KiBページ。
1/12/24スレッドを別測定。24スレッドはSMTを含み、24物理コアではない。
各形式を6ラウンド、順序を正順/逆順で交替。予備測定で反復数を調整し、各試行の実時間を保存。
代表値は6回の中央値。完全な各試行・揺らぎはREPORT.jsonにある。
同じ固定乱数の浮動小数点配列を各形式へ変換した。
大容量配列は256種類の行の繰り返しだが、各行に別の物理領域を確保して全体を読む。
実モデルのexpert選択、attention、グラフ実行、層間同期、重みrepackは含めていない。

正確性の確認
全対象形式で、量子化済みの左右入力を独立にfloatへ戻し、double参照内積と比較した。
Q8_K/Q8_1は内部形式のため、公開traitsに展開関数がない部分を明示して処理した。
初回の補助プログラムはQ8_1参照展開が未対応で停止した。その記録は除外して保存した。
修正後の検証と全測定は成功。各形式24または256行の誤差と有限値を検査。
低ビット形式どうしで出力が一致することは求めていない。量子化による誤差は存在する。
記録したweight_rmseは合成データの誤差であり、LLMの回答品質評価ではない。

実メモリカウンタ（大容量K4096/M65536、24スレッド）
""" + "\n".join(counter_lines) + f"""

通常の速度比較とは別に、各形式20秒程度のループの内部で15秒窓を採取した。
データ生成・量子化・初期ページ割り当てが終わってから開始。
全イベントの稼働率99.9%以上、測定窓内のHDD読み出し・major/minor faultsは全て0。
測定窓内のthermal throttle増分0。通常テストで観測したCPU温度最大{max_temp:.0f} C。
IMCはシステム全体の読み出し。CPUイベントは対象プロセスへ限定。
停止割合は重なる指標で、壁時計時間の独立した内訳ではない。

解釈
Q8/Q6_K/Q4_Kは大容量内積で約47～48 GB/sを読み、Q6/Q4は重みが小さい分早く終わる。
Q3_Kは約31 GB/sで、L2未完了読み出しを伴う停止割合も約1%まで低下する。
キャッシュ側でもQ3は遅いため、この形式では展開・積和を処理するCPU側の制約が強い。
Q2_Kはさらに小さく、大容量では今回の主比較で最速だが、回答品質を根拠に推奨したものではない。
Q4_KはQ3_Kより大きくても速い。ビット数だけでCPU負担や速度を並べられない。
floatへ全面展開するdequantize_row_qの遅さだけから、内積の遅さを判断してはいけない。
実際のvec_dotは圧縮データを処理しながら積和を計算するので、全面展開とは違う処理である。

前回GLM実モデルの約32.8 GB/sとの関係
今回は連続した大行列を読むため、約47～48 GB/sまで到達した。
前回の32.8 GB/sは、そのモデル・アクセス順・同期を含めた実効値であり、ハードの絶対上限ではない。
別ワークロード・別起動での比較なので、今回サーバーの性能自体が向上したという意味でもない。
メモリ待ちが大きいことと、物理帯域の上限を完全に使い切ることは別の状態である。

GLMへ適用する際の形状制約
現在のGLMにはK=1408と10944の行列があり、256要素ブロックのK量子化で割り切れない。
手元の量子化コードでは、Q6_KはQ8_0、Q5_KはQ5_1、Q4_KはQ5_0、Q3/Q2_KはQ4_0へfallbackする。
IQ4_XSはIQ4_NLへfallbackする。
そのためK=1408ではQ8_0/Q5_0/Q5_1/Q4_0/IQ4_NLも別途比較した。
またQ4_K_Mなどのモデル名はテンソルごとの混合設定であり、単一Q4_Kカーネルと同じではない。
カーネル速度比をそのまま実GLMのtokens/sへ掛けることはできない。

今回の到達点
test-quantize-perfの実行、入力形式の修正、1/12/24スレッド比較、実DRAM/CPUカウンタの切り分けを完了。
大きなモデルの追加ダウンロード・再量子化は行っていない。CPUやOSのクロック設定も変更していない。
Q8からQ6_K/Q4_Kへ下げる速度上の根拠は得られた。実モデルの速度・品質は別途同一モデルで測る必要がある。
全ログ・ソース・パッチはこの結果フォルダおよびレビュー用ZIPに保存した。
"""
    (OUT / "REPORT_JA.txt").write_text(report)
    print(json.dumps({"main_table": table, "counter_results": {k: {name: d[name] for name in
        ("dram_read_GB_s", "ipc", "stall_l2_pending_fraction")} for k, d in profiles.items()}}, indent=2))


if __name__ == "__main__":
    main()
