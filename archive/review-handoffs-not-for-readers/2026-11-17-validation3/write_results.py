"""Assemble measured values and explicitly bounded interpretation for review."""
from pathlib import Path
import json
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text())
def main():
    t=read('TIMING_ANALYSIS.json');q=read('QUALITY_ANALYSIS.json');extra=read('python-repeat/ANALYSIS.json');restore=read('RESTORED_RUNTIME.json');answers=read('ANSWER_GRADING.json');hits=read('HIT_ANALYSIS.json')
    assert restore['normal_API_restored'] and all(x['semantic_requirements_observed_met'] for x in answers['responses'])
    rows=[]
    for c in t['cases']:
        f,p=c['summary']['F'],c['summary']['P']
        rows.append(f"{c['case_id']}: F {f['T1_s']:.6f} → P {p['T1_s']:.6f} T1/s; 速度比 {(c['speed_ratio']-1)*100:+.4f}%; 1call短縮 {c['saved_ms']:.6f}ms。F前後差 {c['F_before_after_difference_ms']:+.6f}ms。")
        rows.append('  block平均ms F/P/P/F = '+', '.join(f"{b['mean_ms']:.6f}" for b in c['blocks'])+'; paired速度比 '+', '.join(f"{p['speed_ratio']:.6f}" for p in c['paired']))
    quality=[]
    for c in q['cases']:
        z=c['comparison']['P_vs_F']
        quality.append(f"{c['case_id']}: S=P全65行byte一致。F/P top1差 {len(z['top1_changed_rows'])}/65行、固定target平均NLL差(P−F) {z['mean_reference_token_NLL_delta']:+.9f}、maxabs {z['max_abs']:.6f}。")
    ar=[]
    for c in answers['responses']:
        ar.append(f"{c['case_id']} {c['mode']}: {c['generated_count']}生成ID, {c['stop_reason']}; 問題の要件は保存内容内で確認。応答完結={c['response_complete']}。")
    hr=[]
    for c in hits['cases']:
        b=c['blocks'][0];hr.append(f"{c['case_id']}: hot選択 {b['hot_slots']}/14208={b['all37_CPU_layer_slot_fraction']*100:.4f}%、追加経路23層のH0率 {b['selected23_H0_fraction']*100:.4f}%。P2反復の層別hist一致={c['P_repeats_same_histogram']}。")
    eF,eP=extra['summary']['F'],extra['summary']['P']
    text='''固定35expertのGPU追加分担：未使用validation3例・実回答の実機評価
2026-11-17 / validation3-20261117

今回変えたものは測定callerだけ。前回の35expert配置、CPU/llama/CUDA/baseの実験DSO、GPU全層37–39、CPU compact120GiBは固定した。
Fは同じ統合実験libllamaで追加hot分担を無効、SはGPU提出後すぐjoin、PはGPU提出→CPU残仕事→GPU join。
Fは通常productionバイナリを新しく測り直したものではない。通常APIへの候補採用はしていない。

既存validationから日本語/英語の予定課題とPythonリングキューを評価した。3変種だが独立課題は2family。
固定35案の選定に使ったtraining入力とは異なる。保存履歴の範囲で未使用を照合済みだが、未知の手動操作まで否定しない。
今回の結果から同じ3例向けに配置を再選定していない。残りvalidation2とholdout5は未使用。

速度（teacher-forced64T1、全件保持、API/sampling/UIは計時外）
'''+ '\n'.join(rows)+f'''

初回はF/P/P/Fの4fresh process。前半JA/EN/Python、後半逆順、計768call。各mode各case2blocksのみ。
processごとの共通準備はJA先頭64token warm、JA full236prefix+4reference T1、同期、KV clear、対象prefix再入力。
64 timed callsの先頭も保持。quality保存・logits全走査・route collector・callbackはtiming区間に入れない。
速度は総call/総時間。1024個の独立promptや精密な母集団推定ではない。プロセスのロード/guard時間とprefill/firstT1/call時刻はrawを参照し、純ロード時間を捏造しない。

Pythonのみ探索的追試（初回と混ぜない）
初回の短縮1.053316msはF前後差1.799031msより小さいため、決定を保存して1回だけ追加F/P/P/Fを実施。
F {eF['T1_s']:.6f} → P {eP['T1_s']:.6f} T1/s、速度比 {(extra['speed_ratio']-1)*100:+.4f}%、短縮 {extra['saved_ms']:.6f}ms。
F前後差 {extra['F_before_after_difference_ms']:+.6f}ms。block平均ms F/P/P/F = {', '.join(f"{b['mean_ms']:.6f}" for b in extra['blocks'])}。
paired速度比 = {', '.join(f"{p['speed_ratio']:.6f}" for p in extra['paired'])}。
同一callerバイナリ/DSO/IDs/共通準備。Python-onlyの順序で初回multi-caseと異なるため、独立の事前登録holdoutでも初回の置換でもない。
256追加callすべて保持。良い結果になるまで繰り返す試験ではなく、追加はこの1回で終了。

数値の確認（固定reference軌道）
'''+ '\n'.join(quality)+'''
全75,628,800保存logitはfinite。F/Pはprefix直後row0同一、T1のrow1..64で異なる。
Python row31ではF top7854、P top1351（参照target1351）へ変化。参照に近づいた1行を一般品質改善としない。
平均NLL差はこの固定参照文に対する記述であり、本番品質合格閾値にはしていない。
JA/Pythonはposition256を跨ぎ、custom node buildが23→46。ENは23。bank/Stateは維持し再初期化していない。
S=P一致はこの範囲で並列追加による数値差を検出しなかった意味。CPU/GPU算術全体の完全一致や全入力の競合不在の証明ではない。

実回答（referenceを入力しないgreedy、上限1024全生成ID、EOG有効）
'''+ '\n'.join(ar)+'''
予定問題4本とも10:45終了の実行可能表と最短性の説明がある。JA Fは最後の重複説明途中で打ち切り。必要な表と証明はそれ以前にあるが、完結した応答とは呼ばない。
JA F/Pとも中国語回答だった。設問に明示的な日本語指定はないが、日本語利用の問題として記録。Pだけの退行ではなく、通常Open WebUIを今回測ったものでもない。
Python両コードを親が全文確認し、モデル本体から切り離した非root・CPU8秒/メモリ256MiB上限のPythonで未修正classを検査。
capacity1/2/3/1024、無効capacity、0/負値/重複/int64両端、16,000ランダム操作と決定的なfill/drain/設問例を通過。head/count、固定配列、要素をずらさない構造と説明も目視確認。
成功pushの返り値や循環方向など、設問にない条件は追加しない。採点器は一般的なPython sandboxではなく、手動確認済みのコードだけを対象にした。
6本ともF/Pの全文は同一ではない。最初の生成ID差はJA ordinal45、EN151、Python67（0始まり）。JA/ENは補足表現、Pythonは初期化順序から分岐。
今回の6回答ではPだけの明確な課題不正解は観測しなかった。一般品質同等性は未証明。
visibleはtoken_to_piece(special=false)の連結。journalと保存bytesの整合を確認したもので、独立detokenizeでもWebUIレンダリング試験でもない。
回答長が違うので、完了までの時間をそのまま速度の因果比較にしない。最終IDは生成して保存するが再decodeしないため、T1回数は生成ID数−1。

固定配置の活用状況（初回Pの実行hist）
'''+ '\n'.join(hr)+'''
14208は37CPU層×64T1×top6。元のGPU全層37–39は分母に含めない。追加経路は23層だけで、14CPU層は空bankとしてbypass。
約453.43MiBの追加bankの利用率であり、GPU全使用量が小さいという意味ではない。元GPU3層の約15.79GiB native weightsも保持。
Hは1呼び出しで何個追加GPUへ行ったかの数で、expert ID、個別saved time、F/Pの最初のroute差は分からない。
日本語・英語・Pythonで速度差とhit率の大小が並ぶが、3例から相関の一般法則や因果寄与の全量を証明しない。

実機/復帰/採用判断
dual Xeon E5-2630L v2、12物理/24論理、512GB RAM、PG500-216 Volta32GB。CPU threads12、batch threads24、ctx131072、f16KV、ubatch1024/output cap1。
GPU150W/ECC、Standard fan≤70%、RAM reserve64GiB、既存supervisorを維持。温度の手動常時pollはせず常駐監視を利用。
vision weightsは実際にロードし、warmupはdummy shapeのgraph予約。実画像の推論は今回行っていない。M3の空き容量は終了時点の保持allocationであり、瞬間peak余裕とは呼ばない。
'''+f"通常APIは元設定・元4DSOへ復帰、PID {restore['pid']}、health ok。config SHA {restore['config_sha256']}。監視service active、復帰receipt時alert/errorなし。\n"+'''
実験の全14processは正常終了。通常設定、モデル、既存ライブラリは変更していない。callerのみbuild。原35、GPU37–39、既存guardを維持。
次は配置の時間価値とH0/経路費用を検討する価値がある。ただし今回の検証結果を同じvalidationへ当て直して優秀に見せる運用にはしない。
8層案、B dormant control、origin修正、大きいbank、ubatch変更、通常API採用は今回未実施。初回と追試から有利なrunだけ選ばない。

原本の読み順
caller/FROZEN_PLAN.json、caller/DESIGN.json、CALLER_CHANGES.diff、caller/BUILD.json、実行runのLAUNCH_MANIFEST/PROVIDER_EXPECTED/RUN/RESULTS_SHA256。
QUALITY_ANALYSIS.json/QUALITY_GATE.jsonと9本のquality-*.f32、TIMING_ANALYSIS.jsonと全時刻列、python-repeat/DECISION.json/ANALYSIS.jsonと追加4run。
answers/の全文・未修正Python、ANSWER_VERIFICATION/ANSWER_GRADING/ANSWER_DIVERGENCE、各runのsample/decode journalとall/visible UTF8原本。
HIT_ANALYSIS.json、独立Terra review各txt、RESTORED_RUNTIME.json、STATE.json。過去失敗資料は前回archiveに保持。
今回のZIP内提案はincoming/へ保存し、実測・人間の指示と区別した。人間の新しい運用条件変更は今回ない。
公開snapshotのhealth.jsonはguard-health.jsonへファイル名だけ変更。秘密configは除外し、offline replayでは非秘密LAUNCHのcommand/environmentから一時ディレクトリ内で同じJSON bytesを復元する。
replay_shared.pyはraw数値/計時/journalを再検査する。モデル、保存された起動コマンド、生成Pythonは実行しない。親による意味採点を自動再現したとは主張しない。
'''
    (R/'RESULTS_JA.txt').write_text(text)
if __name__=='__main__':main()
