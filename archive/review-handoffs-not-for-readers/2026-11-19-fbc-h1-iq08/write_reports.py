from pathlib import Path
import json,statistics
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text())
def main():
    f=read('fbc/ANALYSIS.json');h=read('h1/ANALYSIS.json');c=read('cpu08/ANALYSIS.json');restore=read('RESTORED_CPU08.json')
    assert restore['normal_API_restored']
    fr=[]
    for x in f['rows']:
        fr.append(f"{x['case_id']}: F {x['arms']['F']['mean_ms']:.6f} / B {x['arms']['B']['mean_ms']:.6f} / C {x['arms']['C']['mean_ms']:.6f} ms/T1")
        for k,v in x['contrasts'].items():fr.append(f"  {k}: 平均{v['average_ms']:+.6f}ms、対応2組 = "+', '.join(f"{a['difference_ms']:+.6f}" for a in v['pairs']))
        fr.append(f"  F後−前 = {x['F_last_minus_first_ms']:+.6f}ms")
    hr=[]
    for x in h['primary_summary']:
        cp=x['plans']['H1_COMPACT'];nc=x['plans']['H1_NONCOMPACT'];ratio=statistics.mean(nc['process_C_minus_P_ms'])/statistics.mean(cp['process_C_minus_P_ms'])
        hr.append(f"cell{x['cell']}: compact C−P {cp['process_C_minus_P_ms']}ms、noncompact C−P {nc['process_C_minus_P_ms']}ms。短縮量の比 {ratio:.5f}。")
        hr.append(f"  S−P: compact {cp['process_S_minus_P_ms']}ms、noncompact {nc['process_S_minus_P_ms']}ms。")
    cr=[]
    for x in c['rows']:
        cr.append(f"cell{x['cell']}: B/A時間比中央値{x['median_ratio']:.9f}、時間短縮{100*(1-x['median_ratio']):+.4f}%、差中央値{x['median_saved_ms']:+.6f}ms、短縮{x['faster_pairs']}/4組、範囲{x['range']}。")
        for p in x['pairs']:cr.append(f"  A{p['control_process']}/B{p['candidate_process']} position{p['position']} A先行={p['control_first']}: 比{p['ratio']:.9f}, 差{p['saved_ms']:+.6f}ms")
    text='''追加分担の費用・GPUへ渡すexpertの選別・08 CPU候補の実機検証
2026-11-19 JST / be05-followup-20261119
AIレビュー用の詳細記録。通常利用向けの起動手順ではありません。

結論
予定のF/B/C一巡、既存H1の2主要セル、08単独12物理スレッドの一巡を実行した。計23の試験processが正常終了。
追加経路のC−Bは両入力で正。GPU領域を保持しただけのB−Fは入力/組によって増減し、保持だけを主要因とする根拠はない。
同じ1expertをGPUへ渡すH1では、CPU非compactのexpertを渡した場合の局所短縮量が、CPU compactのexpertを渡した場合の約2.46/2.49倍だった。選別に時間価値を含める根拠になる。
対象85/100は訓練時2回/1回と低頻度なので、この診断planを通常bankへ採用していない。
08は数値と実経路の確認に通過し、主要セルの時間比中央値は約0.957/0.978。ただし実行位置依存と一部逆転が残り、事前の順序安定条件を満たすには不十分として保留。全モデル08試験・通常採用へは進めていない。
通常APIは元設定・元CPU/llama/CUDA/baseで復帰した。04/09/03の追加試験、24Tによる救済、guard緩和、downloadは行っていない。

受領資料と担当
xeon_moe_be05_next_steps_2026-11-18.zip の975ファイルのmanifestを照合し、内包された前回726ファイルも前回原本とbyte一致。
添付の次案を、人間から以前に得ていた最適化・実機操作の許可内で具体化した。添付の記載自体を新たな包括承認とはしていない。
作業途中の人間からの希望に従い、以後のsubagentを6.2 Astraへ切り替えた。gpt-6.2-astra/maxが実装・計測設計と最終生データを独立レビュー。重要編集・build・実機操作・採否は主担当が担当。
その希望より前に完了したintakeのTerraレビューは、担当を偽らず残している。過去の全レビューを6.2が行ったとは書かない。

F/B/C：同じ新しい診断libllamaによる追加経路の費用
F: bankなし、元main graph。
B: bank/private backend/pool/pinned領域/warmupを保持、元main graph。Bはcontext構築時の不変フラグであり、set_mode('B')ではない。
C: 同じ保持、custom23層、6 expert全てCPU。実行histはH0のみ。
CPU602a…、CUDA45f6…、base2e13…を固定し、全armを新libllama f7fdfd5838f64e30a07059e7f010eba7a21b3a61b643c798c604324111df39deで実行。旧F時刻へ新B/Cだけを継ぎ足していない。
旧libllamaの無変更relinkは元aff89…とbyte一致。新しい変更はBのflagとquality限定metadata観測で、base/CPU/CUDAは変更していない。
訓練JA prefix272/Python240、凍結prefix/参照IDを直接使用。JA先頭64warm、fullJA+4T1、同期/KV clear、対象prefix。準備は計時へ入れず、timed64の先頭も保持。
quality3processの65行×2例はfiniteかつF=B=C全byte一致。Bはready/payloadがありcustom build/call/histが0。Cは準備後deltaでplanの23層だけ各64call、executed H0。
configured CPU budget120GiBに対してactual compactは127,401,984,000B。original35 native475,453,440B、GPU weight buffer475,457,024B。これらを総VRAM使用量やpeak余裕と呼ばない。
fullGPU layers37–39、ctx131072、output reservation1、batch/ubatch1024、decode12/batch24、vision weightsロード＋dummy shape warmupを維持。実画像推論はしていない。

graph観測
元graphをallocation前、配置/optimizer後を別々に保存。元tensor IDを引き継ぎ、CUDA optimizerのnode順変更を取り違えない。op/subtype、shape/stride、依存関係、view、非pointer op paramsを保存。
eval callback/route recorderは使わず、graphを切断していない。qualityだけ既存GGML_SCHED_DEBUG=2とmetadata観測を使用し、timingは両方無効。
F/Bのallocation前DAGは両ケースで完全一致。postのnode順/tensor記録も一致。全arm/caseで76split。
再確認ではscheduler copy object147個も全arm共通。rewritten consumer edgeはF/B458、C389。これはsoftwareの対象object/依存数であり、物理DMA回数や実転送byte/時間ではない。private CPUの追加copyはこの外側台帳に含まれない。
JAのmain scheduler GPU bufferは全arm4,896,076,160B。host564,898,144BはFでCPU欄、B/CでXEON_HOT_FFN欄に計上された。tensorのCPU/CUDA_Host配置が移ったという意味ではなく、allocatorの同型buffer共有の計上先を区別する。物理NUMA配置は今回新計測していない。
pre node数F/B4866、C4797。減った69nodeを69回の転送削減とは読まない。

F/B/Cの時間
fresh順F/B/C/C/B/F、後半のcase順反転。2case×64×6=768 calls。時計はdecode開始→logits取得でのoutput-ready。各arm/caseは2processだけ。
'''+ '\n'.join(fr)+'''
B−Fは保持・初期化・backend存在・配置を合わせた差。C−Bはcustom dispatch/private CPU/copy/外側配置同期を合わせた差。
C−Bの4対応差は全て正だが、値の振れがあり純copy時間と同定していない。B−Fは一貫した遅延ではない。元のC−F約2msを普遍の定数にしない。
quality observerの有無を跨ぐendpoint整合も、timing各block末尾のlogits全rowとqualityの最終rowで確認した。
一巡で切り分けを閉じ、追加のF/B/C反復はせずH1へ進んだ。

H1：同じtop6からGPUへ渡す相手を変更
H1_COMPACT/H1_NONCOMPACTは各35expert、同じ475,453,440 native B、層別個数・型・初期化shapeが同じでmembershipが異なる。
cell1 layer7の331対85、cell4 layer35の78対100が主比較。runtimeの各projectionで前者compact/後者noncompactを確認した。
元の診断libllama83bf…＋CPU602a/CUDA45/base2e13で4process固定。新FBC libllamaを混ぜず、callerだけがbankを作る。二重bankを作る条件にはしていない。
COMPACT/NONCOMPACT/NONCOMPACT/COMPACTの順。全8cellを保持、cell3はH3対H2なので全8cellを同じ仕事量として平均しない。
O=C=capture、各partition内S=P、cold slots一致、実bank upload全native bytesのreadbackを確認。partition間PはGPU/CPU担当が違い、bit一致は要求しない。
sourceで数える5 GPU API操作はhardware command数ではない。全1536定常観測と128 first-useを保存、warmup4周/OCSP12balanced blocksの定義を維持。
境界は凍結host x/IDs ready→canonical host6expert出力。main schedulerやtoken全体ではない。
'''+ '\n'.join(hr)+'''
上記の各数値は、process内12blockの対応C−P/S−P中央値。約2.5倍は2processのC−P中央値を平均した短縮量の比であり、推論全体が2.5倍になる意味ではない。
PのjoinはCPU完了後の残待ち、SのjoinはCPU実行前の待ち。GPU全所要時間と同一視しない。S−PはこのH1条件では小さい/揺れており、重なりだけの利益を誇張しない。
訓練5例11,840routeの331/85出現102/2、78/100は128/1。頻度とjoint top6を合わせて配置を選ぶ必要がある。
CPUで重いものを渡す機序を支持したが、今回のdiagnostic membershipをそのまま採用しない。頻出147/270の新captureは次の候補として残し、今回追加していない。

08：現行CPUを対照とした単独12物理スレッドscreen
現行33a9…を再現するcontrolと、08-only a2211850ee0d34ebbe431ae520b9b451a727f5c7bb12caa865de6c1323f54f77を別source/buildで作った。
変更したTUはggml-cpu.cとiq3-cache.cppの2個。無変更control両objectおよびCPU ELFは元とbyte一致。header/include優先順位、全link入力、FP flagsを保存。Q8_K wrapperは両版refへのjmpで04なし。
保存実入力cell1/5/6、元GGUF read-only mmap、384expert形状、projection別compactを維持。subset ownerであり通常120GiB全cacheの圧力や長寿命動作を再現した試験ではない。
外側CPU0..11、owner packing12、graph12、THP無効、node0/1 interleave、IQ_ROW_LOCAL=0。24Tへ広げていない。
quality両processだけ同じinterposerを使い、第3既存warmupの実行経路を確認。timingはLD_PRELOAD空、counter symbol不在を検査。
candidateの各IQセルはprepare94blocks（call数96、うち空18）、folded dot行数cell1=48,640/cell5=58,368。型別行数と元関数のDSOも保存。MXは0。selector回数とdot行数を混同しない。
候補qualityで実packed weightの先頭2blockとcrafted Q8（-128を含むblock＋含まないblock）を使い、legacy compact/foldedを4丸めで直接比較。6projection×4=24比較がbit一致。これは同candidate内の従来kernelとの比較で、数学的な真値やモデル全体の4丸め保証ではない。
quality/timingの30保存出力、計921,600 F32はfiniteかつ元captureとbyte一致。warmup末尾/各sample末尾を確認しており、全timed graphの出力を個別コピーしたものではない。
IQ graphのplan要求work_sizeは6,354,240→6,375,840B、増加21,600B。gate/up/downのfolded領域累計37,600Bを同時peakとは呼ばない。MXは70,536Bで同じ。backendの保持capacityやRSSとは区別する。
時計はggml_backend_graph_computeで、内部plan作成・量子化・準備・lookup・barrier・dot・activationを含む。routing weight乗算/merge、attention、GPU/main scheduler、API/token生成は含まない。
8fresh process、AB/BA/BA/ABの4対応組、cell順1/6/5と5/6/1を交差。3warmup、5sample×8call。計960 graph。Aも新測定し旧04の対照時間は流用していない。
'''+ '\n'.join(cr)+'''
cell1は全4組で短縮するが、先頭では7.5/19.0%、後段では約1%と大きさが違う。cell5は1組で1.03626に逆転し、arm/positionに対する方向の安定性が不足。
MXはfoldを使わないが共通planner/callerを変更しているため、完全に無関係なnoiseとも扱わず、比の差引き補正もしない。
中央値の改善を隠さず残したうえで、予定した一巡で保留。08モデル全体試験/通常採用へ進めず、09追加・24T探索・再試行で救済していない。

復帰と回復記録
FBC品質processは全て正常だったが、初回offline gateがdebugログの不正UTF8断片で停止した。元bytesを保持しASCII markerをbytesで読むparserへ修正。品質は再実行せず、未実施6timingだけを同じcaller/engineで再開。
その際も通常APIの復帰を行い、RESTORED_FBCとRESTORED_FBC_TIMINGを別に保存。続くRESTORED_H1、RESTORED_CPU08も元config/4DSOへ戻ったことを確認。
'''+f"最終PID {restore['pid']}、health ok、監視2service active、receipt時alert/errorなし。設定hash {restore['config_sha256']}。\n"+'''
GPU150W/ECC、Standard fan<=70%、RAM reserve64GiBの既存保護を維持。重いbuildと性能計時は直列化した。通常API設定・モデル重み・既存DSOは不変、候補は未採用。

証拠の場所
fbc/QUALITY_GATE.json、ANALYSIS.json、全runs、fbc-engine/、build_fbc_engine.py、prepare_fbc_caller.py、run_fbc.py。
h1/ANALYSIS.json、H1_*.plan、H1_EXPECTED.json、全runs、run_h1.py。BANK_DESIGN/capacity.planは旧original35の履歴資料で、active planはLAUNCHとplan_membershipを読む。
cpu08/BUILD.json、PROBE_BUILD.json、QUALITY_GATE.json、ANALYSIS.json、SCREEN_DECISION.json、全runsとsource。原本/改版とcounter有無を区別。
astra62-review/FINAL_RESULTS_REVIEW_JA.txtに独立監査。全23manifest/419内部manifest項目、実byte、全時刻、冷側150slotなどを照合。
incoming/は最新975ファイルの原本。04と09の保留、未実行A/A/03等の過去案を現在のpending taskへ読み替えない。
replay_shared.pyは公開コピーだけで数値/計時/出力を再検査する。モデル・ELF・保存コマンドを実行しない。外部capture参照のpathだけ一時環境で対応付ける。
'''
    (R/'RESULTS_JA.txt').write_text(text)
if __name__=='__main__':main()
