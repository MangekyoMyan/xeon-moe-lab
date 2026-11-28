"""Build the Japanese report from all paired raw receipts."""
from pathlib import Path
import statistics as st
from analyze import read,lines
R=Path(__file__).resolve().parent
MODES=("off","fixed","auto")
LABEL={"linear":"戻らない会話","preparation":"計測履歴をつくる通常操作","late":"Bより後を編集","early":"Bより前を編集","main_s":"本編3場面の合計","total_s":"準備・枝切替を含むセッション全体"}
def table(h,body):
    line=lambda cells:"| "+" | ".join(str(x) for x in cells)+" |"
    return "\n".join([line(h),line(["---"]*len(h)),*(line(row) for row in body)])
def numbers(xs,d=2):return "／".join(f"{x:.{d}f}" for x in xs)
def mb(x):return x/2**20
def main():
    summary=read(R/"SUMMARY.json");m=summary["modes"];pairs=summary["paired"]
    result=read(R/"RESULT.json");extra=read(R/"EXTRA.json");numeric=read(R/"NUMERIC.json");service=read(R/"SERVICE_RESULT.json")
    manifest=read(R/"CODE.json");auto=m["auto"];lines1=[];cases=[];cost=[];metadata=[];models=[]
    for phase in ("linear","late","early","main_s","preparation","total_s"):
        g=pairs[phase]
        lines1.append([LABEL[phase],*[f'{g["medians"][mode]:.2f}' for mode in MODES],
                       f'{g["paired_medians"]["fixed"]:+.2f}',f'{g["paired_medians"]["auto"]:+.2f}'])
    for phase in ("linear","late","early"):
        for mode in MODES:
            z=m[mode]["tasks"][phase]
            cases.append([LABEL[phase],mode,"／".join(map(str,z["optional_n"])),f'{st.median(z["optional_ms"])/1000:.3f}',
                str(int(st.median(z["last_cached"]))),str(int(st.median(z["last_new"]))),f'{st.median(z["last_first_ms"])/1000:.3f}',
                f'{st.median(z["last_done_ms"])/1000:.3f}',"／".join(dict.fromkeys(z["answers"]))])
    for mode in MODES:
        z=m[mode];c=z["median_session_counts"]
        cost.append([mode,int(c["optional_writes"]),int(c["mandatory_writes"]),int(c["noop"]),int(c["evictions"]),
                     f'{mb(c["optional_bytes"]+c["mandatory_bytes"]):.1f}',f'{mb(z["image_bytes"]):.1f}',
                     f'{int(z["initial_snapshots"])}→{int(z["snapshots"])}'])
        metadata.append([mode,*[f'{mb(z["metadata_before"][k]):.3f}→{mb(z["metadata_after"][k]):.3f}' for k in ("","-wal","-shm")],
                         int(c["file_fsync"]),int(c["directory_fsync"]),int(c["sql_commit"]),int(c["sql_autocommit"])])
    for tag,z in summary["estimates_by_tag"].items():
        def num(k):return "未観測" if z[k] is None else f'{z[k]:,.0f}'
        models.append([tag,z["evaluations"],z["selected"],num("replay"),num("save"),num("restore_anchor"),num("restore_candidate"),num("gain"),
                       num("observed_prompt_P_ms"),num("observed_optional_S_ms"),num("observed_R_A_ms"),num("observed_R_B_ms")])
    ext=[]
    for phase,e in extra.items():
        ext.append([LABEL[phase],len(e["times_s"]["off"]),*[f"{st.median(e['times_s'][k]):.2f}" for k in MODES],
                    f'{e["paired_medians"]["fixed"]:+.2f}',f'{e["paired_medians"]["auto"]:+.2f}'])
    extra_text=(table(["本編","組数","off","fixed","auto","fixedのペア秒差","autoのペア秒差"],ext) if ext else "追加条件に該当した場面はなかった。")
    late={k:m[k]["tasks"]["late"] for k in MODES};early={k:m[k]["tasks"]["early"] for k in MODES}
    out=auto["outcomes_all"];basis=auto["decision_bases_all"];blank=m["off"]["initial_image_bytes"]
    r=f"""# 会話の自動チェックポイントをどこに置くと得か

共通履歴16,388トークンの会話で、途中のBを過ぎてから最後の質問を編集した。手動運用の `off` では編集後に{int(st.median(late["off"]["last_new"])):,}トークンを再処理し、`fixed` と `auto` ではそれぞれ{int(st.median(late["fixed"]["last_new"]))}、{int(st.median(late["auto"]["last_new"]))}トークンで済んだ。3ターン合計の中央値は **{pairs["late"]["medians"]["off"]:.2f}秒 → {pairs["late"]["medians"]["fixed"]:.2f}秒／{pairs["late"]["medians"]["auto"]:.2f}秒**。Bより前を編集すると、どの方式でも{int(st.median(early["off"]["last_new"]))}トークンを再処理した。[実装・起動例](../../../current/experiments/state-branches/README.md)・[固定した入力](fixtures/workload.json)

## 配置と耐久化

既定は `off`。独立CLIを `sudo ./branch open --checkpoint-mode fixed --checkpoint-interval-tokens 1024`、または `--checkpoint-mode auto` で起動できる。間隔は正の整数。fixedは実トークンprefixが一致し、カタログ上residentでサイズ・所在が合う最も近い祖先Aから、処理済み位置Bまでの距離がNに達した最初のターン末尾で保存する。長いターンはNを超過し得る。実復元では従来どおり画像のSHA・処理済みIDを確認する。

autoは「Bへ一度戻る」仮定で `P(A,B) + R(A) − R(B) > 1.5 × S(B)` を使う。Pは検証済みの通常generate／replayで受け取った `prompt_ms` を実処理数 `prompt_n` で重み付けした単価から求める。現構成の128/128のしきい値では、実際の演算の入力幅が128以上でオフロード候補になる。推定は要求全体の実処理数を128未満／以上の経験区分に分ける。複数ubatchにまたがる要求の内訳を1本の時計から推定していない。

Sは追加のmutatingの永続化から、native保存・境界確認・SHA・file/directory fsync・予算処理・snapshotとreadyの同時commit・通常の後処理まで。成功した実際の任意保存だけで推定する。Rは復帰が必要な実操作の選択開始から再入力を始められるまで。編集時には復元先の利用時刻更新も含め、checkout／recoverとは別の系列にする。後続の再入力はPにだけ計上する。

各系列で直近8件をRAMに保持し、候補長の0.5～2倍に3件以上必要とする。S、R(A)、R(B)は処理済み位置から別々に中央値を求める。空起点の消去の観測不足時だけR(A)に下限0を許す。不足時は固定Nを使用する。これらの定数は暫定で、managerの再起動で統計をリセットする。[判断式](../../../current/experiments/state-branches/checkpoint_policy.py)

turnの本文・tokens・pending・世代を先に確定し、同じ単一ライターで任意保存を最大1回判断する。保存前にmutatingを確定し、新しいsnapshotのhead/base・revision・readyを一つのトランザクションで公開する。生成世代は1ターンに1回だけ増える。観測や見送りには追加のSQLite確定や画像SHAは生じなかった。必須のdirty checkout保存と既存のpinは継続する。[manager](../../../current/experiments/state-branches/manager.py)・[画像と祖先の扱い](../../../current/experiments/state-branches/storage.py)

## 実機と操作列

Xeon E5-2697 v2 ×2、512 GiB、PG500-216、150 W、ECC有効、ファン上限70%。入力／生成各24物理スレッド、両NUMAノード、131,072 context、f16 KV、batch/ubatch 1,024、一般／MXしきい値128/128。テキスト1スロット、HDD上のext4、SQLite WAL・synchronous=FULL、画像予算1 GiB。基準 `0d79bf458c4e16fa28e0e251bccd0e848b966c1c` に対して起動時の実コードを照合した。実装コミットは `{manifest["code_commit"]}`。変更対象はcheckpoint_policy、manager、storage、native、branchctl、x9ram_pilotと回帰試験・README。通常範囲を外れる浮動小数点集計の保護を後から加え、配布版の判断を保存済み原票に再適用して一致を確認した。[設定と対応順](DESIGN.json)・[開始時の照合](BASE.json)

各組は同じ二つの完全native画像（255／16,388トークン）と同一カタログを使い、managerとRAM統計を作り直した。ネイティブプロセスは本編全9セッションで1回ロードし、温まったファイルキャッシュを利用した。初期の採取項目参照・空のSQLite WAL処理・採取スクリプトのimportを直した3回は、いずれも計時対象の最初のturn前に終了し、通常APIへ復帰した。数値照合8ケース取得後に故障試験側のカタログ読み取り用importを修正した際も、取得済みの行を保持して未実行部分だけを再開した。本編の実行順はoff→fixed→auto、fixed→auto→off、auto→off→fixedの3組。端末上のPython起動と最初のモデルロードは計時外。クライアントからUnixソケットでticketを取り、最初の空でない本文、最後のreceiptを単調時計で採取した。

1. 短い親から進む2ターンを、途中に戻らず実行した。この区間の前には学習用の準備を置かなかった。
2. 長い親から三つの作業票を扱う通常のcheckout、turn、markを全方式で同じ順に実行した。この準備費を別掲し、セッション総額にも含めた。本編の途中に強制markは入れていない。
3. 長い親の別々の枝で、B、次の質問、編集した入力の順に送った。後の編集ではBの全処理済み列が共通。前の編集ではBの最初の登録番号から異なる。全方式で同一の事前固定messagesと正規IDを使用し、各turnで実テンプレート出力のSHAを照合した。[全操作・応答・時計](evidence/sessions.jsonl)

## 3組の操作列（秒）

{table(["区間","off中央値","fixed中央値","auto中央値","off−fixed ペア中央値","off−auto ペア中央値"],lines1)}

各区間の開始から最終receiptまでの壁時計。本編合計は3場面の和。セッション全体には準備、別の本編へ入るcheckout、区間間の採取も含む。SやRをもう一度加算していない。ペア差は正なら短縮。各回の値と差はすべて公開した。[原票の再集計](RESULT.json)

{table(["本編","モード","任意保存回数（各回）","任意保存秒・中央値","最後のcache_n","最後のprompt_n","最後の初回本文 秒","最後の完了 秒","回答"],cases)}

各回の保存・失敗を含むrawタイミングを維持した。保存点が将来役立つ確率や、Bの先まで変更したときの最適な分割はこの式の対象外である。今回の `fixed` と `auto` は、Bより後を編集した場面で再入力を大きく減らした。戻らない区間では任意保存そのものが追加費用であり、Bより前の編集ではBを飛ばして古い祖先へ戻った。

ばらつきの追加判定では、実際の画像書込みまたは再処理量が変わり、3組の総時間差に正負の両方があった本編だけを対象とした。下表は対象本編に2組を追加した場合の全組。各方式ごとに最後の対象本編まで先行操作を同じ順に再現した。[追加判定](SUPPLEMENT.json)・[追加ペア集計](EXTRA.json)

{extra_text}

## autoは何にいくら払ったか

全3セッションで実測式の評価{basis.get("cost_model",0)}回、採用{out.get("cost_model_selected",0)}回、見送り{out.get("cost_model_declined",0)}回、cold_fallback {basis.get("cold_fallback",0)}回。すべての判断・サンプル数・実行条件IDと直近の復帰経路を保存した。下表は同じ操作タグの中央値（ms）。Bが未保存のケースではそのSの実測は得られない。

{table(["操作","評価","採用","予測P","予測S","予測R(A)","予測R(B)","予測gain","対応する実prefill","実任意保存S","同じ画像・経路の実R(A)","同じ画像・経路の実R(B)"],models)}

Pの実測欄は、実際の処理長と見積もる距離、起点、終端が一致し、そのturnの処理済み境界が入力末尾だった場合のみ。その観測自体も判断時の履歴に入り得る。同じ近傍サンプルがAとBに該当してR推定値が一致する場合がある。同じ経路・同じ保存点IDで実際に復元したRだけを原票の `real_R_A/B` に照合し、別経路や未実行のRは補完しなかった。autoの判断時間の合計は1セッション中央値{auto["median_decision_total_ms"]:.2f} ms、RAM観測更新は{auto["median_stats_ms"]:.3f} ms。[予測と対応する実測](RESULT.json)

実測式は early.B で固定Nより短い768トークンの地点を3組とも選んだ。実保存費は中央値{summary["estimates_by_tag"]["early.B"]["observed_optional_S_ms"]/1000:.2f}秒。直後の26トークンでは3組とも保存を見送った。実際の編集はBより前だったので、この768トークンの保存点による再入力短縮は0だった。今回のようにBより後へ戻る用途では、より少ない保存回数で同程度の短縮を得たfixedが実用上の第一候補になる。

## 容量と同期（準備を含む1セッション中央値）

開始時は{m["off"]["initial_snapshots"]}保存点、resident画像{mb(blank):.1f} MiB。累積書込みは成功したnative保存の実バイト数（同一SHAへの再保存も計上）、residentは重複を除いた終了時点。必須系統には明示mark・checkoutが含まれる。

{table(["モード","任意書込み回数","必須系統書込み回数","既存head再利用","退避画像","累積書込み MiB","最終resident MiB","保存点数"],cost)}

{table(["モード","DB MiB 開始→終了","WAL MiB","SHM MiB","file fsync","dir fsync","明示COMMIT","単独自動commit"],metadata)}

メタデータはmanagerを閉じる前のDB・WAL・SHMの実ファイルサイズ。画像の上限はメタデータを制限しない。SQLiteのCOMMIT回数には物理ディスク内部のflush回数を割り当てていない。[容量・採用数・時間](SUMMARY.json)

## 数値、故障、再開

同じ任意保存画像とsuffixから、独立版のoff/fixedおよび変更前native HTTP版でID・処理位置・CPU/CUDA配置・全129,280スコアを比較した。全{numeric["rows"]}行のうち{numeric["equal"]}行がバイト一致、比較は{numeric["scores"]:,}スコア。これは同じ完全画像を復元したときの照合であり、異なるprefill分割の無停止計算には外挿しない。[位置・SHA・最大差](NUMERIC.json)

回帰試験では間隔境界、重み付き単価、欠測・外れ値、同一コスト時の見送り、未pin退避と予算抑止、コミット前の失敗、公開後のcleanup、境界不一致、manager中断を検査した。実機ではnativeが完了した保存の応答を試験アダプターで失わせた。回答済みturnは確定のまま `unknown` を返し、同じbackendは隔離された。実際に `./branch open --checkpoint-mode auto --checkpoint-interval-tokens 8` で新epochを起動すると、記録したIDを{numeric["recovered_tokens"]}トークン再入力し、生成世代・本文・pendingを復元した。続きの問いへの答えは `{numeric["restored_answer"]}`。この小さいNは短い端末試験で保存を起こすための値である。[故障注入](evidence/faults.json)・[実機の結果不明と確定済みturn](evidence/uncertain.json)・[実ランチャーからの復帰](evidence/entry.json)

通常APIは24/24・64/64で正常応答し、開始時と同じ設定と演算ライブラリを照合した。専用CLIの既定は `off`。一つのZIPにソース、合成入力、raw時系列、数値行、短い・長い両方の代表native画像と再計算コードを収録した。長い画像は転送用に分割され、同じZIP内の断片だけで元のバイト列に戻せる。`python3 -B replay.py` でGPUやモデルなしに判定・数値・整合性を検証できる。[通常APIの復帰](SERVICE_RESULT.json)・[ZIPの取得先とSHA-256](download/)
"""
    (R/"RESULTS_JA.md").write_text(r)

if __name__=="__main__":main()
