"""Render a concise Japanese report from completed measurements, not estimates."""
from pathlib import Path
import csv,hashlib,json,os,time
R=Path(__file__).resolve().parent
def load(name):return json.loads((R/name).read_text())
def main():
    final=load('FINAL_ABBA.json');status=load('FINAL_STATUS.json');selection=load('FINAL_SELECTION.json');wait=load('cpu-wait/RESULTS.json')
    assert final['comparison_accepted'] and status['passed']
    control=final['aggregate']['control'];candidate=final['aggregate']['candidate'];pp=(1-candidate['mean_ttft_s']/control['mean_ttft_s'])*100
    rows=[]
    for row in final['rows']:rows.append(f"  {row['name']}: {2/sum(1/x for x in row['control_tok_s']):.3f} → {2/sum(1/x for x in row['candidate_tok_s']):.3f} tok/s ({row['gain_percent']:+.2f}%)")
    table=[]
    for row in wait['summary']:table.append(f"  {row['profile']:24s} {row['aggregate_tok_s']:6.3f} tok/s / CPU {row['mean_cpu_seconds_per_step']:.3f} s/step / IMC読込 {row['mean_IMC_read_GB_s']:.2f} GB/s")
    native=load('native-pool/LIVE_ABBA.json') if (R/'native-pool/LIVE_ABBA.json').exists() else None
    locality=load('cpu-coordination-mx/main-last/LIVE_ABBA.json') if (R/'cpu-coordination-mx/main-last/LIVE_ABBA.json').exists() else None
    locality_adopted=selection['candidate_profile']=='main-last'
    locality_adopt_note='  4. OpenMPのworker配置を回転し、主スレッドをGPU側のsocketへ配置。算術と同期方式は維持。\n' if locality_adopted else ''
    long=json.loads(Path(status['long_cache_result']).read_text());cold,continuation=long['requests'];vision=json.loads(Path(status['vision_result']).read_text())
    text=f'''2026-11-12 追加最適化・16時頃の第1区切り

この記録は16時頃に検証した構成の保存用。ユーザーの依頼により、20時頃まで探索を継続する。
以下の「最終」はこの区切り内での選択を指す。後続の実験・採否とは分離して扱う。

今回の結論
  生成速度: {control['decode_tok_s']:.3f} → {candidate['decode_tok_s']:.3f} tok/s、{final['aggregate_gain_percent']:+.2f}%。
  入力処理時間: {control['mean_ttft_s']:.3f} → {candidate['mean_ttft_s']:.3f} 秒/要求、{pp:.2f}%短縮。
  選択した構成: {selection['candidate_profile']}。通常起動用の最終profile: profiles/final.json。
  APIは稼働中、コンテキスト設定は131072、GPU150W/ECC有効、ファン上限70%の方針を維持。

比較条件
  今回開始時のoptimization-20261112/prefill64を基準とした直接ABBA。
  基準には以前のCPU cache120GiB・GPU offload閾値64が既に含まれる。
  訓練と以前の候補選定に使っていない8入力を、最終選択前に固定。各設定2回、各入力128生成token。
  全文および非streamで取得した全128token IDsが、すべて一致した。
  生成速度は総decode steps/総predicted_ms。1要求のstepsはserver定義どおりn_gen-1=127。
  入力時間はserver prompt_msで、クライアントのstream TTFT実測ではない。
  品質ベンチマークではなく、同じ仕事の速度と出力一致の検証。個別改善率の足し算・乗算はしない。

最終比較の内訳
{chr(10).join(rows)}

効果が確認できた変更
  1. CPU RAMの展開済み重みをexpert内行列単位で選択。
     同じ118.65234375GiBで19,200個を固定頻度planへ配置。GPU動的cacheではない。
     訓練8入力/2048生成tokenで配置を決定。最初の別8入力で+9.52%（各設定1回）。
     最新の総合効果は上記の直接ABBAを採用する。元GGUFと量子化形式は維持。
  2. MXFP4の入力補正とscaleを再利用。
     単体kernelでは約15–24%、実モデルABBAでは+3.19%。両者は別の指標。
     計算順を保持し、-128や非有限scale等は従来経路へ戻す。
  3. CUDA expert ID分類を安定したcount/prefix/scatterへ変更。
     演算fixture88条件・3variantで出力byte hash一致。
     実モデルの128/1024/4096入力で、入力時間4.43–5.61%短縮。
     GPU kernel自体や量子化精度を変える変更ではない。
{locality_adopt_note}

CPU数・待機方式
{chr(10).join(table)}
  このscreenはcount/codeの2入力。default12だけ前後2回、ほかは各1回。
  全回答一致。decodeで実際にCPU時間を使ったworkerのaffinityも設定と一致。
  6物理coreは両socket3coreずつ。6/12ともuser cycles/reference比は約1.0833で、clock差はほぼない。
  6→12はこの実モデルで約1.57倍。以前の単体kernel約1.9倍と混同しない。
  ただしchunk分割も変わるため、純粋なcore数だけの比較ではない。CPU交換後は再計測する。
  PASSIVEはCPU時間を約20%減らす一方、生成を約8%遅くした。IPC上昇だけで高速化とは判断できない。
  CPU時間は論理CPU時間で、物理core数や消費電力ではない。IMCはsystem全体の実測。
  IMCは各run内を時間加重し、上表はrun平均。理論帯域との単純比較だけで律速を断定しない。

追加比較と採否
  - loop unroll縮小: 遅くなったため不採用。
  - 重さも加味したCPU cache plan: frequency比+0.316%の単回差。優位未確認。
  - NUMA行配置: ABBA+0.70%。効果が小さく構成依存が強いため標準はinterleave。
  - IQ3_XXS入力補正の再利用: 同じ新framework内で+0.51%。旧MX版を超える根拠が不足。
  - active counter初期化/最終barrier削減/MX worker maskの組合せ: ABBA -0.79%、不採用。
  - batch1536: 長文入力では速いが、日本語の回答とtoken確率に変化。実験profileのみ。標準1024。
  - 24論理threads、spin10000、PASSIVE: 平均で標準12物理threads/defaultを上回らず。
  - native pthread graph/poll0: {('decode '+format(native['aggregate_gain_percent'],'+.2f')+'%、実測stream TTFT '+format((1-native['aggregate']['candidate']['mean_ttft_s']/native['aggregate']['control']['mean_ttft_s'])*100,'.2f')+'%短縮（ABBA）。標準には不採用。初期化にはOpenMPを保持。') if native else '詳細はnative-pool/を参照。'}
  - masterを含むworker配置回転（{'採用' if locality_adopted else '不採用'}）: {('生成 '+format(locality['decode_gain_percent'],'+.2f')+'%、128入力時間 '+format(locality['prefill128_time_reduction_percent'],'.2f')+'%短縮、1024入力 '+format(locality['prefill1024_time_reduction_percent'],'.2f')+'%短縮。同一binary0/1比較。利益全部をGPU localityだけには帰属しない。') if locality else '未測定。採用結果に含めない。'}

機能・cache確認
  同じ最終APIプロセスで画像→長文→継続を確認。
  既存の新聞画像: {vision['answer'].strip()}。観測VRAM最大 {vision['observed_gpu_peak_MiB']:.0f} MiB。
  {long['context']}入力token: 入力処理 {cold['response']['timings']['prompt_ms']/1000:.3f}秒。
  返却された全token IDsを連結して続行: cache_n={continuation['response']['timings']['cache_n']}、新規prompt_n={continuation['response']['timings']['prompt_n']}。
  継続の入力処理 {continuation['response']['timings']['prompt_ms']/1000:.3f}秒、要求全体 {continuation['wall_seconds']:.3f}秒。
  128k確保は128k全域の速度・品質を実測した意味ではない。
  最後のOpenAI chat APIの簡単な確認も成功（2+3→{status['chat_smoke']['answer']}）。
  API PID {status['pid']}、config SHA256 {status['profile_sha256']}。
  稼働libraryとplanのSHA256を最終確認。xeon-watchは稼働を確認。

実行環境
  CPU: {', '.join(status['cpu_models'])} ×2（12物理core/24logical）。
  Board: {status['board'].get('board_vendor','')} {status['board'].get('board_name','')}。
  RAM: 公称512GiB。最終MemTotal {status['memory']['MemTotal_KiB']/1024/1024:.2f}GiB、MemAvailable {status['memory']['MemAvailable_KiB']/1024/1024:.2f}GiB。
  GPU: PG500-216 / Volta sm70 / 約32GB、150W、ECC有効。PCIe 0000:84:00.0、NUMA node1。
  Kernel: {status['uname'][2]}。
  Context131072 / f16 KV / batch1024 / GPU resident expert layers37–39。

使い方
  既存のAPI URL: {status['API_url']}
  Model alias: {status['model_alias']}
  次回の通常起動: sudo python3 /home/codex/projects/llm-lab/deepseek41-bringup-20261101/start.py
  今回開始時へ戻す: sudo python3 {R}/switch_and_wait.py baseline
  API再起動が必要になる。元モデル・以前の凍結記録・公開repositoryは変更していない。

証拠
  FINAL_ABBA.json / FINAL_STATUS.json / FINAL_PROFILE_MANIFEST.json
  cpu-wait/RESULTS.json / mx-prepack/LIVE_ABBA.json / cuda-sort/LIVE_ABBA.json
  native-pool/ / cpu-coordination-mx/ / iq-rhs/ / row-local/ / execution/
  kernelのbit一致、sanitizer、実回答一致は異なる範囲の検証として保存。
  旧SSE試験のtoken hashは観測stream IDのhashで、UTF-8処理により全raw IDsではない場合がある。
  最終fresh比較と長文継続は非streamの全IDsを使い、この制約を避けた。
'''
    path=R/'RESULTS_JA.txt';path.write_text(text);path.chmod(0o600)
    if os.geteuid()==0:os.chown(path,R.stat().st_uid,R.stat().st_gid)
    with (R/'CPU_WAIT_RESULTS.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(wait['summary'][0]));writer.writeheader();writer.writerows(wait['summary'])
    print(str(path),flush=True)
if __name__=='__main__':main()
