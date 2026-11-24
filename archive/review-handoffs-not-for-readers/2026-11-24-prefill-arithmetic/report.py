"""Write a numerical report with the live configuration and both independent correctness standards."""
import csv
import json
import math
from pathlib import Path
import statistics

R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def loss(a):return math.sqrt(statistics.mean(x*x for x in a))
def num(x):return f"{x:.6g}"


def main():
    practical=read(R/"CHAT_RESULT.json");ca=read(R/"PRECISION_CAUSE.json");gate=read(R/"CAUSAL_RESULT.json")
    final=read(R/"FINAL_STATE.json");pieces=read(R/"TOKEN_LABELS.json");worker=read(R/"WORKER_RESULT.json");pcs=read(R/"PC_RESULT.json")
    group=read(R/"risk/both.json")["grouped"]
    arms=["64-64","64-128","128-64","128-128"]
    bymode={"64-128":"MX_only","128-64":"general_only","128-128":"both"}
    tab=[];summary={}
    for a in arms:
        g=read(R/"runs"/("precision-"+a)/"GREEDY.json")
        pre=[(v["clock_ns"][0][1]-v["clock_ns"][0][0])/1e9 for v in g.values() if 64<=len(v["prefix_ids"])<128]
        metrics={"seconds":statistics.mean(pre),"fixed_changes":0,"chat_changes":0,"TV":0.}
        if a in bymode:
            q=read(R/"risk"/(bymode[a]+".json"))["grouped"]
            metrics.update(fixed_changes=q["original96"]["changes"],chat_changes=q["affected_chats"]["changes"],TV=q["original96"]["TV"]["mean"])
        tab.append(f'| {a.replace("-"," / ")} | {metrics["seconds"]:.2f} 秒 | {metrics["fixed_changes"]}/130 | {metrics["chat_changes"]}/145 | {metrics["TV"]:.4f} |')
        summary[a]=metrics
    causal_tab=[]
    mn={"control":"128 標準相当の対照","gpu-gates":"第0層 gate・up を64の実測出力に置換","residual":"第0層 gate・up のCPU内積を残差補正","gpu-layer-output":"第0層の最終出力を64の実測値に置換"}
    for m in mn:
        v=gate[m]
        causal_tab.append(f'| {mn[m]} | {v["total_top_changes_to_64"]}/130 | {len(v["original_switches_recovered"])} | {len(v["additional_switches"])} | {v["mean_TV_to_64"]:.5f} | {v["mean_target_NLL_delta_to_64"]:+.5f} |')
    assert worker["fully_covered_routed_projections"]==1776 and pcs["spin_by_location"]["outside_all_logged_cpu_graph_work"]==1728
    pool={}
    for key in ("cpu","cuda","cpu_corrected","mode_estimate"):
        pool[key]=loss([v["arithmetic_sample"]["all"]["vs_GGUF_F64"][key] for v in ca.values()])
    mimic=loss([v["arithmetic_sample"]["all"]["vs_observed_CUDA"]["mode_estimate"] for v in ca.values()])
    diff_dot=loss([v["arithmetic_sample"]["all"]["vs_observed_CUDA"]["cpu"] for v in ca.values()])
    faithful=loss([v["arithmetic_sample"]["all"]["observed_CPU_vs_Q8K_F64"] for v in ca.values()])
    changes=group["original96"]["changed_positions"];case_table=[]
    for z in changes:
        def piece(x):
            return json.dumps(pieces[str(x)],ensure_ascii=False).replace("|","\\\\|").replace(chr(96),"&#96;")
        case_table.append(f'| {z["case"]} 行{z["row"]} | {piece(z["A_top"])}（{z["A_top"]}） | {piece(z["B_top"])}（{z["B_top"]}） | {z["Btop_rank_in_A"]}位 → 1位 | {100*z["A_p_Btop"]:.2f}% → {100*z["B_p_Btop"]:.2f}% |')
    assert all(read(R/"runs/precision-64-64/GREEDY.json")[k]["reference_text"]==v["example_64"] for k,v in practical["per_case"].items())
    matching=[]
    for n in ("JA96","PY96"):
        vals=list(csv.DictReader((R/"risk/both"/(n+".tsv")).open(),delimiter="\t"))
        matching.append([sum(v[k]==v["target"] for v in vals) for k in ("A_top","B_top")])
    assert matching==[[32,32],[37,37]]
    text=f"""# 入力のGPU振り分けでDeepSeekの予測が変わる場所

2026年11月24日。E5-2697 v2 ×2、24物理コア、512 GiB、PG500-216、150 W、ECC有効。モデルはOrcaRouter DeepSeek-V4.1-Flash-1017-abliterated Mix-256G。通常APIは入力24・生成24、しきい値64 / 64で稼働している。以下は同じ実機・重み・KV設定での比較である。

## 最初の差は第0層のexpert内積

同じ96トークンの日本語とPythonについて、両設定で第0層のAttention出力、FFNの正規化後の入力、routerのスコア、選ばれた6個のexpert IDがすべてビット一致した。**同一の活性値と同一のIQ2_S重みを使ったgate・up射影で差が出た。** gateだけで各課題1,327,104要素を直接比較した。内部の値を取り出したすべての試験で、入力直後の全語彙129,280スコアが、その設定の観測なしの結果とバイト一致した。

64 / 64ではここをVoltaで実行し、128 / 128ではCPUで実行する。128側の作業バッファ名 CUDA_Host はGPUからも参照できるホストメモリを指す。CPUで元のQ8_K入力を使い、同じGGUFの重みを展開して倍精度で積和し直した結果は、実際のCPU出力とRMSE {num(faithful)}で一致した。

差は後続のrouterにも入った。expert集合が初めて入れ替わったのは両課題とも第1層。40層 × 96位置で、集合そのものが変わったものは日本語 {ca["JA96"]["routed_token_layers_changed"]:,}/3,840、Python {ca["PY96"]["routed_token_layers_changed"]:,}/3,840。並び順だけの変化はこの数に含めていない。

## 内積の精度を同じ基準で比較

第0層のgateとupそれぞれで、6トークン位置 × 6選択expert × 32出力行 × 2課題、計4,608出力を抽出した。参照値は実際のGGUF重みを展開し、量子化前のFP32活性値との内積をFP64で合計した値。元の学習時の非量子化モデルはこの基準に使っていない。

| 第0層の射影 | GGUF重みを使ったFP64基準に対するRMSE |
|---|---:|
| 64設定でV100が実際に返した値 | {num(pool["cuda"])} |
| 128設定でXeonが実際に返した値 | {num(pool["cpu"])} |
| 入力丸めの残差に対し、Xeonの同じ内積をもう一度計算して加算 | {num(pool["cpu_corrected"])} |

CPUはこの形状で256要素単位のQ8_Kに活性値を量子化する。丸めで失った差分を同じ方法で量子化してもう一度積和すると、標本のRMSEは約 **{pool["cpu"]/pool["cpu_corrected"]:.0f} 分の1**になった。対象射影には2回目の重み読み出しと内積が加わる。

V100のコードは、このIQ2_S・96入力の条件で選択トークンをexpert別にまとめる。そのexpertの入力が8以下の場合に使う32要素単位の量子化と半精度スケール、9～63の場合の32要素単位の量子化と単精度スケール、64以上の場合の半精度オペランドの積和を、それぞれホスト側で再現した。実際のV100出力とのRMSEはCPUの通常値で {num(diff_dot)}、経路別の再計算で {num(mimic)}。切替境界は使用中のCUDAソースから取得した。ここで記録したのは実入出力の一致度で、CUDA命令単位のトレースは取っていない。

## 問題が表に出た頻度

| 比較対象 | 観測 |
|---|---|
| 以前と同じ96入力の日本語とPython、同じ後続トークンを順番に与えた130位置 | 第一候補は3/130位置で変化。元の文章の次トークンと一致した位置は64・128とも69/130 |
| 新しい73～102入力の実用6課題、実際の貪欲生成の共通履歴 | 全語彙の値には差があり、第一候補は145位置すべてで一致。実際の本文も6件すべて一致 |
| 対照の28・173入力の会話、および160入力の固定文 | 計160位置で全語彙スコアまでバイト一致 |
| 予約変更、在庫、Python、SQL、業務メール、単位変換ほか計8課題 | OpenAI互換APIを64・128・128・64の順で独立に起動。全32要求で採点基準を通過し、同一課題の本文は4回すべて一致 |

Pythonの式は1,365種類のリスト、SQLは同点や負数を含む5つのSQLiteテーブルで検算した。日本語メールは宛先・新期限・提出先・丁寧さ・文数を照合し、本文にも余分な変更理由がないことを確認した。8課題は用途を選んだ固定例である。実利用全般での誤答率を推定するには、独立な問題がさらに必要になる。

| 固定履歴の位置 | 64での第一候補 | 128での第一候補 | 128の候補の順位 | 同候補の確率 |
|---|---|---|---:|---:|
{chr(10).join(case_table)}

確率は全語彙の生スコアに温度1のsoftmaxを適用したもの。実用の生成はtemperature 0。この3位置のいずれでも、元の文章に書かれた次トークンは両方の第一候補とは異なった。

予測分布全体の総変動距離（TV、0～1）は、以前の130位置で平均 {group["original96"]["TV"]["mean"]:.5f}、p95 {group["original96"]["TV"]["p95"]:.5f}。新しい会話の対象145位置では平均 {group["affected_chats"]["TV"]["mean"]:.5f}、p95 {group["affected_chats"]["TV"]["p95"]:.5f} だった。TVは同じ履歴である候補集合の確率が最大どれだけ動くかを表す。特定の文章が誤答になる確率を求めるには、別途回答の正誤を判定する必要がある。

## 2つのしきい値を独立に変えた結果

左がCUDAの一般的なしきい値、右がMXFP4のrouted積に加わるしきい値。下表の時間は固定ドライバから対象6課題の入力が終わるまで。各設定1起動なので、実際のAPIでの配信時間は別に測った。

| 一般 / MXFP4 | 対象6課題の平均入力時間 | 固定文の第一候補変化 | 実用6課題の第一候補変化 | 固定文の平均TV |
|---|---:|---:|---:|---:|
{chr(10).join(tab)}

一般128の2設定は全435行の全語彙値がバイト一致。MXFP4側だけ128にした場合は別の2位置（日本語44、Python23）で第一候補が変わった。この中間設定は平均の差を小さくするが、元の3位置の単純な部分集合にはならない。

通常サーバーと同じバイナリで測った、送信から最初の本文までの時間は、対象6課題で **7.57→4.35秒**。各設定2起動、全要求でサーバーが報告した再利用トークン数は0だった。

## 第0層だけを変えた因果試験

128 / 128で動く隔離プロセスで、表に示す箇所だけを変えて残りを通常経路で通し、元の96入力の直後と同じ続き64トークンの全語彙を保存した。GPUの値を入れる行では、同じ入力で先に記録した本物の64設定の値を用い、書き戻しと読戻しのバイト一致を確認した。CPU残差の行では実際に24物理コアで追加の内積を計算した。対照は両課題とも65行すべてで通常128とバイト一致した。

| 128上の比較 | 64との第一候補差 | 元の3差のうち戻った数 | 新しく変わった数 | 平均TV | 参照次トークンの平均NLL差（対64） |
|---|---:|---:|---:|---:|---:|
{chr(10).join(causal_tab)}

CPU残差補正は元の3位置すべてを64の第一候補に戻した。同時に日本語34、Python19・23の3位置で別の第一候補を選び、差の総数は3/130にとどまった。平均TVは0.03104から0.03359、参照次トークンの平均NLL差は+0.00330から−0.00264に動いた。内部射影の精度、元のGPU版との一致、与えた参照トークンへの尤度は異なるものを測っている。

残差補正の2射影に追加した診断処理は日本語0.211秒、Python0.192秒。ここには演算のほか読戻しや検算も入る。第0層だけの修正を通常APIへ適用する根拠は得られなかった。次の精度対策を評価するなら、独立な問題に対する正答率・コードのテスト通過を主指標にし、モデル全体で測る。今回の通常APIにはしきい値64を採用している。

## 生成側の同期を調べた結果

16生成にわたり、37層 × gate・up・downの計1,776演算を24ワーカーすべてで追跡した。gate・upは各expertの出力を全員96行ずつ、downは23人214行、1人198行で担当した。最新到着時刻と平均到着時刻の差を111演算で合計すると、準備側は1トークン当たり中央値0.38 ms、演算終了側は3.01 msだった。これらは早く終わったワーカーの平均待ちに対応する。

別に同時計測できた14生成のCPU実行中6,380標本では、2,071標本がOpenMPの待ち合わせでスピンしていた。そのうち1,728標本（83.4%）は、どのCPUグラフのワーカーも計算区間に入っていない時間にあった。24人の担当行数の不均衡だけでは以前の待ち標本を説明できない。時系列を採った呼出しでも、固定130行の全語彙出力は通常実装とバイト一致した。

## 実機の終了状態

通常設定のSHA-256は `{final["ordinary_profile_sha256"]}`。終了時のモデルPIDは {final["ordinary_pid"]}、通常APIはHTTP正常、監視2サービスはactive。全{len(final["diagnostic_processes"])}診断プロセスが正常に終了し、照合した112個のECC・熱抑制カウンターの変化は0。監視標本の最高はCPU {final["peak_C"]["cpu_max_C"]:.0f}℃、GPU {final["peak_C"]["gpu_C"]:.0f}℃、HBM {final["peak_C"]["hbm_C"]:.0f}℃、DIMM {final["peak_DIMM_C"]:.0f}℃。ファン出力の最高は{final["max_fan_pct"]:.0f}%だった。
"""
    (R/"RESULTS_JA.md").write_text(text)
    (R/"EVIDENCE.json").write_text(json.dumps({"factorial":summary,"heldout_API":practical["mean_first_text_ms_affected"],
        "local_RMSE_to_GGUF_F64":pool,"GPU_mode_surrogate_RMSE":mimic,"original_gold_top1_hits":matching,
        "total_rows_per_policy":435,"non_EOG_and_EOG_are_both_scored":True,"ordinary_profile":final["ordinary_profile_sha256"]},indent=2)+"\n")


if __name__=="__main__":main()
