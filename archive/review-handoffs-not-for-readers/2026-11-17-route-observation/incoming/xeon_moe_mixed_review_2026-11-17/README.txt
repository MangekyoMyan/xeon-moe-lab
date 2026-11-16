xeon-moe-lab mixed-operator review — 2026-11-17
Pinned source: ff6168094858141aeeebf45abc5c309882fe5216

最初に読むもの
  ASTRA_NEXT_ACTION_JA.txt     実機担当へ渡す次の作業案と、観測の条件
  REVIEW_AND_NEXT_STEP_JA.txt  今回の結果の評価・理由・具体的source境界
  DECISION_AND_STAGE_GATES.json 採用保留と未実装の観測stageを機械可読化

詳細
  evidence/          全raw/manifest/正解byte/guardの独立監査
  cpu_diagnostic/    保留中のC0追加観測仕様と保存contextの監査
  gpu_observation/   37層route、private stream、ready計測、容量のsource契約
  charts/            全512 C0 chunkの図（32graph平均/point、除外なし）
  tools/             単発capacity snapshot、mock検査、親のraw再計算/plot script
  checks/            親の再計算とcapacity scriptのoffline検査結果
  VERIFICATION.json  このパッケージで検証した範囲、未実行範囲、原本copy照合
  FILES_SHA256.json  本manifest以外の同梱fileのsize/SHA

状態
  CPU abs: mixed利益あり、事前screen C0総時間条件は未達、採用保留。
  新route/ready collector: 契約のみ、未実装。
  capacity script: 単発のread-only補助、実GPU queryは未実施。mock7検査通過。
  新性能benchmark/model/API/GPU実行: なし。

実機Astraへの受け渡しにはZIP全体を使う。先にASTRA_NEXT_ACTION_JA.txtを読む。
このパッケージは通常APIを変更するinstallerや自動experiment runnerではない。

offline再現（元repoの上記commit checkoutが必要）
  python3 -B tools/parent_recheck_and_plot.py --repo /path/to/checkout --out /new/output --plot
    --plotだけmatplotlibを使用。外せばstdlibのみ。
  python3 -B evidence/reanalyze_mixed_evidence.py --repository /path/to/checkout \
    --prior-return /path/to/original/xeon_moe_abs_next_2026-11-16 --output-dir /new/evidence
    prior-returnは前回返却物の実bytesを使う。incomingを自分自身と比べて前回一致の証明にしない。
  python3 -B cpu_diagnostic/audit_saved_context.py --repo /path/to/checkout --output /new/context.json
  python3 -B gpu_observation/build_observation_evidence.py --repository /path/to/checkout \
    --output-dir /new/gpu-source-review
    元repo外の未作成ディレクトリを使う。source hashと過去容量の算術のみでGPU呼出なし。

外部一次文書URLと固定source SHAは各台帳にある。図・集計は保存rawから作り、新性能測定とは区別する。
復元した24expected binary、model weights、実library ELF、前回bundleの重複copyは同梱していない。
