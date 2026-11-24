# E5-2697 v2 ×2への移行結果

2026年11月24日にE5-2697 v2を2基、24物理コア・48論理CPUとして確認しました。両ソケットのメモリは合計16×32 GiB、設定速度1600 MT/sです。通常テキストAPIは左右12コアずつで入力・生成を行います。

| 固定入力の生成速度 | 日本語 | Python |
|---|---:|---:|
| 交換前の左右6＋6コア | 7.64 tok/s | 7.61 tok/s |
| 交換後の左右6＋6コア | 8.59 tok/s | 8.50 tok/s |
| 交換後の左右12＋12コア | 10.06 tok/s | 9.86 tok/s |

この表は入力を12コアにそろえた固定トークン計時です。通常APIの候補は生成24・入力24で別に試し、今回の採用時には通常待受で短いテキスト1要求を確認しました。[固定入力の比較と一時API試験](https://github.com/MangekyoMyan/xeon-moe-lab/blob/901c0d14f8b9b73065a8b59ce2516185934f84d7/archive/review-handoffs-not-for-readers/2026-11-24-postload-measurement/SUMMARY_JA.txt) · [通常APIでの受入と温度](https://github.com/MangekyoMyan/xeon-moe-lab/releases/tag/normal24-adoption-20261124)

この起動で物理コアの代表はCPU 0～23、マスクは両方とも0xffffffです。左右6コアずつの比較条件では0～5および12～17を使いました。次にCPU構成を変えるときは、<code>python3 current/check.py topology</code>で番号を取得できます。
