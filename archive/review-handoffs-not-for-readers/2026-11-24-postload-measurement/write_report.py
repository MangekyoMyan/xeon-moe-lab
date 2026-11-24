"""Format the verified host comparisons and original-profile restoration."""
import json
from pathlib import Path

R=Path(__file__).resolve().parent
CASES={"text_cache_budget_ja":"日本語272入力","code_stable_grouping_python":"Python 240入力"}
LABEL={"OLD_P12":"交換前12/12","NEW_P12":"交換後12/12","NEW_D24_B12":"交換後24/12"}


def read(p):
    return json.loads(p.read_bytes())


def doc():
    f=read(R/"FIXED_COMPARISON.json");host=read(R/"HOST_AND_THERMALS.json");s=read(R/"NORMAL_STREAM_COMPARISON.json")
    pilot_path=R/"PRACTICAL_COMPARISON.json"
    pilot=read(pilot_path) if pilot_path.exists() else None
    text=["CPU交換後のDeepSeek実モデル比較　2026年11月24日",
          "", "固定入力で比較した生成速度", "",
          "条件の数字は生成／入力の物理コア数を表す。12は各ソケットから6コアずつ、24は各ソケットの12コアを使用。各組は別プロセスで2回計時し、2回目は課題順を逆にした。各欄は固定64トークン×2回、計128回の実測。T1/sはトークン数を合計処理時間で割った。", "",
          "条件            課題              T1/s     平均ms    中央ms    p95 ms    最大ms   入力処理s    2回のT1/s"]
    for label,cohort in f["conditions"].items():
        for key,short in CASES.items():
            v=cohort["cases"][key];pair=", ".join(f"{x:.3f}" for x in v["per_process_T1_s"])
            text.append(f"{LABEL[label]:12}  {short:13}  {v['T1_s']:7.3f}  {v['mean_ms']:8.3f}  {v['median_ms']:8.3f}  {v['p95_ms']:8.3f}  {v['max_ms']:8.3f}  {v['prefill_mean_s']:8.3f}   {pair}")
    text+=["","旧CPUはE5-2630L v2 ×2、新CPUはE5-2697 v2 ×2。新旧の12/12は同じ左右6＋6コアで比較した。24/12では生成とプロセス全体の許可範囲を24物理に広げ、入力コア、ドライバのprojector引数、packing引数は12に保った。"]
    for a,b in (("OLD_P12","NEW_P12"),("NEW_P12","NEW_D24_B12"),("OLD_P12","NEW_D24_B12")):
        change=f["throughput_ratios"][b+"_over_"+a]
        text.append(f"{LABEL[a]} → {LABEL[b]} の生成速度　"+ "、".join(f"{CASES[k]} {(change[k]-1)*100:+.1f}%" for k in CASES))
    text+=["","出力の照合","","各新条件で、日本語・Pythonの固定prefixと後続64入力を交換前と同じ位置に与えた。品質試験の65行×2課題×2条件、計260行（各行129,280個のF32）はすべて有限で、交換前の参照とバイト単位で一致。最大絶対差、NMSE、argmaxの変化はいずれも0。4計時プロセスの各課題で保存した最終行、計8行も同条件の品質試験と一致。対象はこの2課題の保存出力である。",
           "", "通常APIの日本語1要求", "",
           "各APIに同一バイトの依頼を1回送信。seed 19、temperature 0。交換前の通常APIの入力24はSMT込みの24論理CPU、交換後の入力24は24物理コア。本文の受信時刻はクライアント側、下表の生成速度はサーバー報告値である。",
           "条件                入力/キャッシュ/出力    初回本文s    入力処理s    サーバー生成tok/s    完了(EOF)s"]
    ordinary=[("交換前 12/24",s["old_normal_request"]),("交換後 通常12/24",s["new_normal_request"])]
    if pilot:
        ordinary.append(("交換後 候補24/24",pilot["candidate_24_24"]))
    for name,v in ordinary:
        u=v["reported_usage"]; t=v["server_timings"]; tok=f'{u["prompt_tokens"]}/{t["cache_n"]}/{u["completion_tokens"]}'
        text.append(f"{name:17}  {tok:17}  {v['client_first_content_ms']/1000:8.3f}  {t['prompt_ms']/1000:10.3f}   {t['predicted_per_second']:11.3f}     {v['eof_ms']/1000:10.3f}")
    text.append("交換前と交換後の通常設定で、出力本文のSHA-256は一致。いずれも生成59トークン、本文イベント55個。本文イベント間隔（中央値／p95）は次のとおり。")
    for label,name in (("OLD","交換前通常"),("NEW","交換後通常"),("PILOT","交換後候補")):
        p=R/("SSE_REPLAY_"+label+".json")
        if p.is_file():
            e=read(p);g=e["receive_event_gaps_ms"]["content"]
            text.append(f"{name}　本文{e['content_events']}イベント、{g['median_ms']:.2f} / {g['p95_ms']:.2f} ms、0 ms間隔{g['zero_gap_count']}件")
    s0=s["pinned_thread_runtime_ns_by_socket"]["0"]/1e9;s1=s["pinned_thread_runtime_ns_by_socket"]["1"]/1e9
    total=s["matched_thread_runtime_ns"]/1e9
    text+=["",f"新CPUの通常12/24では、本文の最初と最後の受信時刻の内側{s['coverage_elapsed_s']:.2f}秒で、照合できたスレッドのCPU時間は計{total:.2f} CPU秒。同じCPUへの固定を両端で確認した累積はS0が{s0:.2f} CPU秒、S1が{s1:.2f} CPU秒。0～11の12物理コアに活動を確認した。各時点のスレッド数をコアの利用数に読み替えず、schedstatの増分を使った。",
           "新CPUの通常API計測には、CPU 47に固定した読取り観測を追加した。各スレッドの許可CPUと実行時間を約0.35秒周期で取得。旧CPUの通常API記録にはこの観測がなく、各設定1要求で再現性や一般的な会話速度は評価できない。"]
    if pilot:
        same="一致" if pilot["same_response_text"] else "相違"
        p=pilot["candidate_scheduling"]
        text+=["",f"24/24候補の実用確認では通常APIと同じサーバーバイナリ・モデル・認証・キャッシュ設定を使い、生成指定だけを12→24として同じ要求を再生した。テスト専用の待受は127.0.0.1、通常APIは同じ実機の10.10.10.2で、どちらもloopbackを通る。画像projector初期化もこのサーバーでは生成指定に従って24になる。テキスト応答は通常設定と{same}。サーバー報告生成速度は{(pilot['server_decode_tps_ratio']-1)*100:+.1f}%、初回本文は{pilot['first_content_ms_difference']:+.1f} ms。",
               f"同じスレッド観測では候補の本文受信中{p['coverage_elapsed_s']:.2f}秒で、両端の固定とCPU時間増分を確認した量はS0が{p['pinned_thread_runtime_ns_by_socket']['0']/1e9:.2f} CPU秒、S1が{p['pinned_thread_runtime_ns_by_socket']['1']/1e9:.2f} CPU秒。観測器のCPU 47は候補が使用するCPU 23のSMT兄弟で、観測コストを0とは仮定していない。"]
    text+=["","物理コアへの分担とクロック","","日本語の品質試験中に79 Hzの数値カーネルPCを抽出した。下表のコア数は実行を観測できた範囲。クロックは同じ品質プロセスの日本語入力＋生成をまたぐ、対象CPU全体のAPERF/MPERF/TSCからの推定。主計時プロセスへのperf取り付けはない。"]
    for name,co in host["worker_evidence"].items():
        p=co["strict_numeric_pc"]["decode"];freq=co["weighted_CPU_wide_busy_MHz"]
        cpus=sorted(int(c) for c in p["tids_by_os_cpu"])
        cnt=p["tids_per_call"]
        text.append(f"{LABEL[name]}　CPU番号{','.join(map(str,cpus))}、全体で{p['tids_total']} TID。64回の各T1で拾えたTIDは{min(cnt)}～{max(cnt)}。入力中の対象数値PCは{co['strict_numeric_pc']['prefill']['sampled_pc_count']}件。クロック {freq/1000:.3f} GHz。")
    text+=["","ロード、温度、保護","","512 GiB DDR3、PG500-216 32 GB、ECC有効、GPU上限150 W、RAM予備64 GiB、context 131072・F16 KVを維持。CPU向けの登録は80テンソル・19,200射影・127,401,984,000 B。重み7分割・projectorは既存の取得時記録と現サイズを照合した。今回、重み全体のハッシュ再計算は行っていない。",
           "プロセス                    text+ctx s   projector s   warmup s   unload s      読込B  major faults"]
    for label,co in f["conditions"].items():
        if label=="OLD_P12":continue
        for j in co["runs"]:
            text.append(f"{j['name']:25} {j['text_context_load_s']:9.3f}  {j['projector_load_s']:10.3f} {j['warmup_s']:10.3f} {j['unload_s']:9.3f} {j['load_read_bytes']:10d} {j['load_major_faults']:7d}")
    w=host["all_stages"]
    text+=["",f"6プロセスの既存保護ログで採れた{w['sample_count']}点の最高値はCPUパッケージS0 {w['package0_C']['max']:.0f}℃、S1 {w['package1_C']['max']:.0f}℃、GPU {w['gpu_C']['max']:.0f}℃、HBM {w['hbm_C']['max']:.0f}℃。観測された最小MemAvailableは{w['available_GiB']['min']:.1f} GiB。既存の常駐監視ではDIMM最高{host['watch_max_dimm_C']:.0f}℃。ECC・熱抑制のカウンター変化、監視アラーム、センサー異常のあるレコードは{host['watch_warnings_or_errors']}件。各条件と課題区間のサンプル数もHOST_AND_THERMALS.jsonに保存した。"]
    final=read(R/"FINAL_API.json")
    if pilot:
        a=pilot["normal_12_24"];b=pilot["candidate_24_24"]
        text[2:2]=[f"今回のテキスト推論の常用候補は生成24・入力24。通常サーバーの同じ日本語要求で、生成は{a['server_timings']['predicted_per_second']:.2f}→{b['server_timings']['predicted_per_second']:.2f} tok/s、最初の本文は{a['client_first_content_ms']/1000:.2f}→{b['client_first_content_ms']/1000:.2f}秒だった。両方とも入力29・キャッシュ0・生成59で、同じ本文を返した。",""]
    text+=["","通常APIと次の判断","",
        f"測定前のモデルPID {read(R/'production-before-live.json')['pid']}、最終確認のモデルPID {final['pid']}。元の12/24設定で /health はok。実際にマップされたCPU・CUDA・base・llamaのデバイス・inode・ハッシュ、GPU/ECC、両監視サービスを再確認した。",
           "固定課題では生成を24コアにすると2回とも速くなり、p95も両課題で短縮した。入力12でのprefill時間は日本語が12.05→12.13秒、Pythonが12.78→12.70秒。速度改善は主に生成中に現れた。残る時間のうち、DRAM待ち・CPU/GPU間の分担・同期がそれぞれ何割を占めるかは今回未計測。",
        "", "再計算用データ","","FIXED_COMPARISON.json と COMPARE_NEW_P12/COMPARE_NEW_D24_B12.json は別の実装で作った。trial/runs は新しい6プロセスのSPEC/RUN/LAUNCH・F32・呼出時刻・保護ログ。品質時の数値PCはアドレスを除いたTSVとMSRの記録を収めた。perf.dataとELFは実機に保存し、元ファイルのハッシュを記録した。",
        "通常APIのstreamには人工の固定依頼への応答本文・受信イベント・受信時刻、referenceには交換前の同条件データがある。展開したディレクトリで python3 -B replay_bundle.py . を実行すると、ネット接続・モデル・管理者権限なしで照合できる。"]
    if pilot:
        where=text.index("再計算用データ")
        text[where:where]=["元の待受アドレスを維持し、生成の -t だけを12から24へ変えた候補ファイルを実機のprofiles/normal-24-24.jsonに用意した。元の設定と候補のハッシュ、変更するJSON項目はPROFILE_CANDIDATE.jsonにある。画像入力・長い会話・同時接続での候補の性能は未測定。",""]
    (R/"SUMMARY_JA.txt").write_text("\n".join(text)+"\n")


if __name__=="__main__":
    doc()
