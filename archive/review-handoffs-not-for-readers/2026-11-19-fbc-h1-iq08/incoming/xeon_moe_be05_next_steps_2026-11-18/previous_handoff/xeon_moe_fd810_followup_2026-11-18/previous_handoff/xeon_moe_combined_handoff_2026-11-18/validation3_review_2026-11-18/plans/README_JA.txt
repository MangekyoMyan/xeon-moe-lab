既存2cellのH1比較に使う固定診断plan
2026-11-18 JST / offline準備のみ。実機未実行、production候補ではない。

現実装に任意active-maskはない。同じbankに両候補を置くと両方がGPUへ行く。
したがって、membershipだけ異なる2planを別fresh processで使う。
COMPACT→NONCOMPACT→NONCOMPACT→COMPACTの順で、既存8cellを同じ手順で測る。
主要比較は事前にcell1/layer7とcell4/layer35に固定する。

H1_COMPACT.plan
  layer7=[19,154,331]、layer35=[0,1,78]
  対象cellでGPUへ行くのは331と78。
H1_NONCOMPACT.plan
  layer7=[19,85,154]、layer35=[0,1,100]
  対象cellでGPUへ行くのは85と100。

その他の層はoriginal35から変更なし。両planとも35expert、475453440 native bytes。
層ごとのexpert数、型、初期化するgraph形状が同じ。同一bank membershipではない。
layer35の0/1は対象top6にない形状合わせ用であり、配置推奨ではない。
生成scriptはparent/prepare_h1_plans.py。原planのSHAを固定し、出力を新規ディレクトリへ書く。
H1_PLAN_VERIFICATION.jsonに正確な入力hash、出力hash、8cell全部のHを記録した。

実機側の統合事項
・engine DSOsは変更せずに比較可能。既存operator replayのplan/hash期待値は明示的に更新する。
・guardを単に外さない。実際のruntime compact lookup、provider、payload、allocation、Hを照合する。
・既存callerの固定表示Original35を診断名へ直す。必要ならcallerだけ表示/pinを直してbuildする。
・同じcellのO=捕捉baseline、O=C、同じpartitionのS=P、cold slot一致を確認してから時間を読む。
・F/B/C費用分解のBと、この2planの名称を混同しない。こちらは固定H1機序試験。
・既存8cellのwarmup/first-use/定常sampleの定義を保持し、rawを削除しない。
・各process内のC−PとS−P、およびprocess間の変動を別に報告する。
・85と100はtrainingで2回/1回だけ。1回の節約が大きくてもそのままbankへ採用しない。
・頻出147/270の追加capture用planは別物であり、この2planに含まれると扱わない。
