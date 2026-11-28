"""Render a Japanese report from measured values and explicitly scoped controls."""
from collections import Counter
from pathlib import Path
import json
from statistics import median
R=Path(__file__).resolve().parent
def j(n):return json.loads((R/n).read_bytes())
def rows(n):return list(map(json.loads,(R/n).read_text().splitlines()))
def sec(v):return f"{v/1000:.2f}"
def main():
    s=j("SUMMARY.json"); ring=j("RING.json");svc=j("SERVICE.json");code=j("CODE.json"); replay=j("POLICY_REPLAY.json")
    ops=rows("workflow-evidence/workflows.jsonl");cfg=rows("workflow-evidence/plans.jsonl")
    order=("off","fixed","auto")
    layout=j("layout/evidence/create.json");compact=j("evidence/create.json")
    def size(q,p):return next(e["bytes"] for e in q["events"] if e["position"]==p)/2**20
    def table(k,attr):
        return " | ".join(sec(k[m][attr]) for m in order)
    def series(k,attr):
        return "／".join(sec(k[m][attr]) for m in order)
    speeds=[]
    for m in order:
        med=s["pooled_edits"][m];v=s["per_mode"][m]
        successes=sum(1 for z in ops if not z["session"].endswith("unprobed") and z["mode"]==m and z["op"]["name"].startswith("edit") and z["op"]["resume"]["total_ms"]<=3500)
        speeds.append(f"| {m} | {sec(med['resume_ms'])} | {sec(med['first_ms'])} | {sec(med['total_ms'])} | {med['old_replayed']:g} | {successes}/18 | {sec(v['median_session_wall_sum_ms'])} | {v['median_cumulative_bytes']/2**20:.1f} |")
    positions=[]
    for x in s["operations_by_index"]:
        if x["name"]=="A":continue
        p=x["name"][4:]
        origins=" / ".join(", ".join(str(c) for c in x[m]["origins"]) for m in order)
        positions.append(f"| {p} | {table(x,'resume_ms')} | {origins} |")
    pressure=[]
    for v in s["pressure"]:
        pressure.append(f"| {v['policy']} | {v['retained']} | {v['bytes']/2**20:.2f} | {v['rejected_budget']} |")
    intervals=Counter(r["op"]["receipt"]["checkpoint"]["interval_tokens"] for r in ops if r["mode"]=="auto")
    predicted=[(x["op"]["resume"]["total_ms"], x["op"]["resume"].get("estimated_ms")) for x in ops if x["mode"]!="off" and x["op"]["name"]!="A"]
    paired=[(a,b) for a,b in predicted if b is not None]
    drift=median(abs(a-b) for a,b in paired) if paired else None
    no_probe=j("ring-evidence/practical-off.json")
    practical=[]
    for name in ("edit127","edit256"):
        a=[r["op"] for r in no_probe if r["op"]["name"]==name]
        assert len(a)==3 and all(x["resume"] is None for x in a)
        practical.append(f"{name}の初回本文は中央値{sec(median(x['first_ms'] for x in a))}秒（{sec(min(x['first_ms'] for x in a))}〜{sec(max(x['first_ms'] for x in a))}秒）")
    extra_budget=j("ring-evidence/pressure56.json")
    extra_text="、".join(v["policy"]+"で"+str(v["preview"]["currently_available"]) for v in extra_budget)
    long=j("long-evidence/paired.json");longfix=j("long-evidence/fixture.json")
    assert len(long)==6
    long_table=[]
    for m in ("off","fixed"):
        a=[z for z in long if z["mode"]==m]
        long_table.append(f"| {m} | "+ " / ".join(sec(z["first_ms"]) for z in a) +
            f" | {sec(median(z['first_ms'] for z in a))} | {sec(median(z['total_ms'] for z in a))} | {a[0]['receipt']['timings']['prompt_n']} |")
    body=f"""# DeepSeekの入力・生成途中に保存し、編集位置に戻る

独立した会話CLIに、処理済みIDの絶対位置での同期保存と、復帰時間・容量を指定するプランナーを実装した。普段のAPIは24/24・64/64の構成で稼働している。実機試験はE5-2697 v2を2基、RAM 512 GiB、PG500-216、GPU 150 W、ECC有効、ファン上限70%、専用スロット1個、入力・生成24物理コア、しきい値128/128、batch/ubatch 1024で行った。

## 保存位置と数値

正規化後750トークンの入力から29個をサンプルした。同じ要求の入力中に256・512、生成中に768を保存した。生成末尾のサンプルは未評価のため保存長には含めない。圧縮境界前後も含む7画像を別プロセスで直接復元し、同じIDとCPU/GPU経路の9つの全語彙行を、保存せず同じ位置で分割した対照と比較した。対照の再実行は35/35行でビット一致した。

| 完全状態の保存方式 | 対照と全129,280値が一致した行 | 第一候補一致 | 最大絶対差 | 最大総変動距離 | 768位置の画像 |
|---|---:|---:|---:|---:|---:|
| 既存の圧縮された配置 | 0/9 | 9/9 | 2.89340 | 0.014983 | {size(compact,768):.2f} MiB |
| 専用の物理配置保存 XKV1 | 9/9 | 9/9 | 0 | 0 | {size(layout,768):.2f} MiB |

既存のKVライターは、スライディングウィンドウで隠れたセルを画像から除外する。元のスロット位置と次の割当位置も記録しない。専用コーデックでは使用中のセル・物理インデックス・次の書込み位置をひとまとめに復元した。この変更で上表の差が消えた。3項目の個別の寄与は測定していない。DSV4の圧縮KV、圧縮途中の状態、windowなどは従来の全状態シリアライザーを引き続き使用する。

実容量1,280セルのSWA領域をまたぐ追加試験では、1,279～1,537位置の6つの新規画像を別プロセスに戻した。空の追加入力の試験を含む7行も、対照と全値が一致した。空の入力末尾1,536の新要求は、直前の画像1,535から既知IDを1個だけ再評価した。スロットの受領数、元の入力ID、使用画像、実行先、全語彙行を保存している。さらに、通常CLIと同じく保存位置で追加logitsを要求しない対照を設けた。385からの33入力と続く2生成位置の3行もビット一致した。

固定間隔127でも1要求の入力中に127・254、生成中に381で保存した。処理済み{ring['fixed127_processed']}の非格子位置に手動markを置き、次のターンでは絶対位置508に保存した。途中保存後の生成を実際に中断したときは、会話・pending・世代を元の確定値に保ち、新しいエンジン世代から復帰できた。

分割自体には数値コストもある。元の未分割入力と細かく分割した対照の共通29出力は第一候補が29/29一致し、全値一致は0/29、最大総変動距離は0.05443だった。これは復元精度とは別に観察した、合成入力1件の結果である。

## 編集するときの待ち時間

校正には別の文章を使い、同じ温まった実行条件の観測を各組にコピーした。評価中のオンライン更新も原本に収録した。378トークンの入力Aと、同じ長さで1 IDだけを変えた編集を交互に使用した。各設定12要求、順番を回転して3組。履歴は空から作り、容量は各48 MiB、基準間隔128、autoの復帰目標3.5秒。時計はサーバー上の管理CLIで測った。

| 設定 | 共通部分への復帰 秒 | 編集時の初回本文 秒 | 編集要求全体 秒 | 既知ID再処理数 | 実測で3.5秒以内 | 12要求の合計 秒 | 書込み MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(speeds)}

編集時の各列は6編集×3組の中央値。末尾2列は各組の合計の中央値。戻らない最初のA単独では、全体時間の中央値はoff／fixed／autoで{series(s['operations_by_index'][0],'total_ms')}秒だった。同じ順の保存待ちの合計は{series(s['operations_by_index'][0],'save_ms')}秒、配信中の最大間隔の中央値は{series(s['operations_by_index'][0],'max_gap_ms')}秒。[全要求の集計](SUMMARY.json)・[各要求の生記録](workflow-evidence/workflows.jsonl)

| 共通prefixの長さ | off 秒 | fixed 秒 | auto 秒 | 実際に使った起点3組ずつ（off / fixed / auto） |
|---|---:|---:|---:|---|
{chr(10).join(positions)}

共通部分が使える時点をネイティブで同期して記録した。offの編集にも観測用の境界を1個置き、eraseを呼ぶ瞬間から測ったため、offの祖先選択の僅かな時間は含まれない。分割を加えない追加off対照のedit256は、初回本文{sec(s['unprobed_off'][-1]['first_ms'])}秒、全体{sec(s['unprobed_off'][-1]['total_ms'])}秒だった。全条件9回で回答本文が一致した操作位置は{s['answers_equal_positions']}/12だった。2位置では同じ英単語の列挙が、空白・カンマ区切りと改行に分かれ、12トークンの上限に収まる単語数も変わった。

計時分割を外した追加対照を別の保守区間で各3回繰り返した。{'。'.join(practical)}。保存済みモデル・設定は同じで、追加対照では診断用のobserverを待機状態に置いた。これらの短い入力について、手を加えないoffを上回る体感速度は確認できなかった。

autoが各要求で選んだ前進間隔は{'、'.join(f'{n:,}トークンが{k}回' for n,k in sorted(intervals.items()))}だった。時間を推定できた固定／自動の編集は{len(paired)}/{len(predicted)}回で、その復帰時間の絶対予測誤差の中央値は{(f'{drift:.0f} ms' if drift is not None else '未計測')}。少数の編集の結果であり、推定上の目標と実測時刻は別々に扱う。

## 3,500トークンの台帳の末尾近くを編集した場合

短い編集では初回本文が早くならなかったため、別の日本語台帳の空箱コードを1か所変更した。全3,500入力の共通部分は{longfix['p']:,}。1,024・2,048・3,072に履歴の構築中に作った3画像の合計は{sum(e["bytes"] for e in j("long-evidence/anchor.json")["positions"])/2**20:.1f} MiB。比較の出発点を揃えるための名前付き画像は別計上。6回すべて同じ確定済み台帳の完全画像から開始し、順番を替えて各3回測った。offには計時専用の分割を入れていない。容量512 MiB。固定版の起点は3回とも3,072で、既知IDの再入力は{longfix['p']-3072}、変更位置以降の入力は{3500-longfix['p']}だった。

| 設定 | 初回本文の3回 秒 | 初回本文中央値 秒 | 要求全体中央値 秒 | 実際に評価した入力 |
|---|---|---:|---:|---:|
{chr(10).join(long_table)}

台帳を初めて構築した1回の要求全体は、offが{sec(j('long-evidence/off-seed.json')['total_ms'])}秒、固定1,024が{sec(j('long-evidence/fixed-seed.json')['total_ms'])}秒だった。これらは3回の編集時間の外に計上した。長文の比較はこの1件の編集位置に限られる。[6回の復元・入力数・応答時刻](long-evidence/paired.json)

## 容量と未達

同じpinを含む別試験で実画像予算を18 MiBにした。128間隔で同じ入力を処理した後の実ファイル合計を示す。1バイトへの上限変更は両設定で拒否し、18 MiBとpinを保った。

| 保持方針 | 実際に戻れる一致位置 | 実画像合計 MiB | 1バイトへの変更 |
|---|---|---:|---|
{chr(10).join(pressure)}

18 MiBでは末尾384の画像がpinと同居できず、両方の方針で保持点が一致した。別の645入力、56 MiB、同じpinと直近130の試験では、実際の保持点は{extra_text}だった。

同一SHAの複数参照を1件として数え、書込み前に正確なnative画像長を予約する。SQLite・WAL、転送中のRAM、OSキャッシュ、VRAMはディスク画像予算と別会計。各画像を作る際の実バイトを記録した。初期の権限設定が異なった校正は一般ユーザーのnativeプロセスが画像を開けず、比較開始前に終了した。表はCLIと同じ一般ユーザーで取得したデータだけを使っている。

## 再現と運用

短文の順序比較の実装は `{code['timed_engine_source'][:12]}`、長文の実機試験と配布コードは `{code['final_source_commit'][:12]}`。最終版では共有位置の重複・保護領域での見積もり、低密度サンプルの表示、資源別の会計を補い、同じ位置の復元では名前付きheadを優先するようにした。通常比較での実測の容量判定{replay['capacity_admissions']}件と、各セッション終了時の評価位置計{replay['session_status_positions']}箇所について最終プランナーの再計算を照合した。実機の起動・基本CLI、固定127・空suffix・中断後の再起動にも別途受入試験がある。

通常APIの設定SHA-256は `{svc['normal_profile']}`。{len(svc['windows'])}回の保守区間がすべて閉じ、通常の4ライブラリ・HTTP応答・待機中スロットを照合した。専用版のモデルライブラリは `{code['physical_model_library_sha256']}`、サーバーアダプターは `{code['server_adapter_sha256']}`。

公開資料には、実行条件、正規ID、各操作の生時系列、画像のSHA、元の全語彙FP32行、容量と数値を再検算するスクリプトを収録した。実機用の操作はCLIのガイドにある。[CLIとビルド手順](../../../current/experiments/state-branches/README.md)
"""
    (R/"RESULTS_JA.md").write_text(body)
if __name__=="__main__":main()

