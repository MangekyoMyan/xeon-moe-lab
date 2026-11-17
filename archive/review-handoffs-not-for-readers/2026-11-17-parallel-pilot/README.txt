2026-11-17 GPU resident35 expert / CPU-GPU parallel pilot

RESULTS_JA.txt: 実測結果と採用判断。
CONSULT_PRO_JA.txt: 次にレビューしてほしい内容。
ROOT_TRIAGE_JA.txt: 人間の方針とPro提案の区別、実装上の変更・失敗も含む判断。
STATE.json: 各段階と実行runの所在。完了したcontrollerを再開しない。
RESTORED_RUNTIME.json: 通常APIの復帰とlibrary/config pin。

capture/: 実activation8件の最終sourceと二つの失敗run、成功run。
replay/: 元35planを保持し、O/C/S/Pの同一入力比較。生時間列と各出力。
integrated/: narrow context/graph port、build provenance、F-before/CSP/F-after。
incoming/: 外部Proの返却資料をそのまま保存。未実測提案と当方の実測を混同しない。
*_TERRA.txt: source・rawの独立レビュー。

モデル本体、秘密情報、ELF/.so/.oは含めていない。元engine/sourceと過去のrawは同repositoryの既存snapshotを参照。実機起動scriptは固定x9ram向けであり、一般Linuxでそのままモデルが起動できるという意味ではない。

FILES_SHA256.jsonは公開payloadのhash/size、RAW_MAPPING.jsonは元fileからの対応。
health.jsonは公開時にguard-health.jsonへ改名し内容はbyte同一。
replay_shared.pyは保存launch commandを実行せず、model/networkを使わずにrawを再計算する。前回2026-11-17-gpu-workspace archiveを隣に置く。

通常APIの設定/engineはこの研究では変更しない。今回の統合版はcallback=null、route collector=closed専用の実験。hybrid logical route origin分離は未実装。品質診断のreference-token NLLは一般回答の点数ではない。
