#!/usr/bin/env python3
import hashlib,json,statistics
from pathlib import Path
from full_replica_experiment import LAB,ROOT,OUT,BUILD,ACTIVE,MODEL
report=json.loads((OUT/'REPORT.json').read_text())
before=json.loads((OUT/'experiment-before.json').read_text())
changed=[]
for n,h in before['source_files'].items():
 p=LAB/'llama-private-prefetch'/n
 if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:changed.append(n)
artifacts={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in before['active_artifacts'].items()}
settings={n:Path(n).read_text()==v for n,v in before['settings'].items()}
assert not changed and all(artifacts.values()) and all(settings.values())
after={'active_source_unchanged':not changed,'active_artifacts_unchanged':artifacts,'os_settings_unchanged':settings,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
(OUT/'experiment-after.json').write_text(json.dumps(after,indent=2))
report['preserved_runtime']=after
report['page_queries']={}
for kind in ['large','fine']:
 d=OUT/('pages-'+kind)
 rows=json.loads((d/'page-node-query.json').read_text())
 report['page_queries'][kind]=[{'tensor':r['tensor'],'address':r['address'],'sampled_pages':len(r['nodes']),'node_counts':r['node_counts'],'max_same_node_run_bytes':max(v['pages'] for v in r['runs'])*4096} for r in rows]
 report['page_queries'][kind+'_smaps_rollup']=(d/'proc-smaps_rollup.txt').read_text()
report['commands']={'full_replica':str(LAB/'run_melody_full_replica_experiment.sh'),'single_fine_interleave':str(LAB/'run_melody_fine_interleave_experiment.sh')}
report['source_patch']='full-replica.patch'
report['limits']=['One model, Q6_K/Q8_0 formats, 24 threads, short decode contexts; not a DeepSeek result.','Core remote metric is retired DRAM-load uops, not total QPI bytes.','Page-size intervention also changes placement granularity and TLB behavior; finer-grain per-expert memory-controller contention remains an inference.','PMU windows are diagnostic runs separate from throughput comparisons.','No GPU, download, global kernel setting change, normal launcher replacement, or public contribution.']
report['excluded']={'placement-full':'Initial physical snapshot was taken before load completion; placement assertion failed. Replaced by placement-verified. Not used for performance conclusions.'}
assert len(report['main']['runs'])==8 and len(report['without_thp']['runs'])==4
assert report['main']['all_process_disk_reads_zero'] and report['without_thp']['all_process_disk_reads_zero']
assert report['correctness']['complete'] and report['correctness']['logits_bit_identical']
assert report['long_response']['identical'] and len(report['locality'])==3 and all(x['valid'] for x in report['locality'])
report['complete']=True
(OUT/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
m,n=report['main'],report['without_thp'];a,b=m['single_interleaved'],m['full_local'];c,e=n['single_interleaved'],n['full_local']
long=report['long_response'];la,lb,lc=[long['variants'][v]['median_timings_ms'] for v in 'ABC']
metrics={x['name'][-1]:x['metrics'] for x in report['locality']}
text=f'''26B-A4B 全重みNUMA複製 実測報告 / 2026-10-20

結論
全重みを左右のメモリに複製すると、同じページ設定で生成は {a['median_ts']:.3f} -> {b['median_ts']:.3f} token/s、{m['median_gain_percent']:.2f}% 改善した。
ただし、1コピーを4KiBごとに左右へ細かく分散するだけでも {c['median_ts']:.3f} token/s に改善した。
両条件を4KiBページにそろえた比較は {c['median_ts']:.3f} -> {e['median_ts']:.3f} token/s、完全複製の上乗せは {n['median_gain_percent']:.2f}% だった。
この26Bでは、まず1コピーの配置を整える効果が大きい。追加約21.6GiBを使う複製の価値は、その後に残る約2%と照らして判断する。
DeepSeekは別のモデル・量子化・演算構成であり、この割合をそのまま適用できない。

対象と環境
モデル: Melody1437-26B-A4B.i1-Q6_K.gguf。GGUF architecture=gemma4、expert_count=128、expert_used_count=8。
ファイル21.084GiB、GGUF重み21.069GiB。GGUF上の658テンソルをすべて複製した。
出力層用の埋め込み再配置を含む実行時659テンソルで、1コピー21.633GiB、2コピー43.266GiB。
実機: Sugon X9DR3-F、Xeon E5-2630L v2 x2、12物理コア/24論理CPU、128GB DDR3-1600（実用量約125GiB）、各CPU 2メモリチャネル。
GPU不使用。生成24スレッド、CPU 0-23へ固定、入力も24スレッド。CPUビルドのコンパイルフラグ18単位は従来版と一致。
同じプロセス内の1つの推論を両CPUで処理した。独立した2モデルプロセスの合計速度ではない。

生成速度（プロセスごとの平均速度を集計した中央値）
配置                         ページ条件       token/s    生成中RSS
1コピー・左右分散             通常THP設定      {a['median_ts']:.3f}     {a['median_generation_rss_gib']:.2f} GiB
左右に完全複製               通常THP設定      {b['median_ts']:.3f}    {b['median_generation_rss_gib']:.2f} GiB
1コピー・左右分散             4KiB固定         {c['median_ts']:.3f}    {c['median_generation_rss_gib']:.2f} GiB
左右に完全複製               4KiB固定         {e['median_ts']:.3f}    {e['median_generation_rss_gib']:.2f} GiB

通常THP設定はABBA+BAABの8プロセス、各128トークンを3回。
1コピー範囲 {a['min_ts']:.3f}-{a['max_ts']:.3f}、完全複製範囲 {b['min_ts']:.3f}-{b['max_ts']:.3f} token/s。
4KiB固定はABBAの4プロセス、各128トークンを2回。
1コピー範囲 {c['min_ts']:.3f}-{c['max_ts']:.3f}、完全複製範囲 {e['min_ts']:.3f}-{e['max_ts']:.3f} token/s。
各反復を独立したプロセス数として水増ししていない。同一ブート・同一ライブラリ・全プロセスの追加ディスク読み出し0。
ランダムトークン生成の合成試験で、深さ0から始めた。長い会話履歴での速度は今回未測定。

通常のmmap共有バッファも別に確認した。
12/24スレッド: 通常共有8.912/9.129、完全複製9.940/9.965、試験用1コピー8.162/9.089 token/s。
24スレッドでは試験用1コピーが通常共有に近く、試験用バッファを使うこと自体の大きな速度変化は見られなかった。
以降の主比較は同じ試験用バッファ・同じ参照切替処理の1コピー対2コピー。

実際の日本語応答
同じ質問、temp=0、seed=42、128トークン上限（evalされたのは127トークン）、6プロセスABCCBA。全出力が完全一致。
A=通常mmap共有、B=完全複製/通常THP、C=1コピー/4KiB分散。
                 起動load     入力処理       生成
A 通常共有       {la['load']/1000:.3f}秒      {la['prompt eval']/1000:.3f}秒       {la['eval']/1000:.3f}秒
B 完全複製       {lb['load']/1000:.3f}秒     {lb['prompt eval']/1000:.3f}秒       {lb['eval']/1000:.3f}秒
C 1コピー4KiB    {lc['load']/1000:.3f}秒     {lc['prompt eval']/1000:.3f}秒       {lc['eval']/1000:.3f}秒
生成速度はBがA比+{long['generation_gain_percent']:.2f}%、CがA比+{long['fine_interleave_generation_gain_percent']:.2f}%。BはCより+{long['full_vs_fine_interleave_generation_gain_percent']:.2f}%。
モデル常駐後の入力+生成の待ち時間はBがA比 {long['warm_request_latency_reduction_percent']:.2f}% 短縮。
入力処理の大きな改善はなかった。起動はコピーが必要な試験バッファで遅く、モデル常駐向けの改善。
これらはwarm cacheでのload値であり、冷えたHDDからの読み込み時間ではない。
短文（10トークン）の別試験は完全複製の生成改善+{report['short_response']['generation_gain_percent']:.2f}%だったがばらつきが大きく、長文試験を追加した。

CPU間読み出しとDRAM（独立した18秒の観察窓）
A=1コピー/通常THP、B=完全複製/通常THP、C=1コピー/4KiB。
条件   相手側DRAMから読む割合の指標   DRAM読み出し合計
A      {metrics['A']['remote_share_of_local_plus_remote_retired_dram_load_uops']*100:.2f}%                          {metrics['A']['dram_read_GB_s']:.2f} GB/s
B      {metrics['B']['remote_share_of_local_plus_remote_retired_dram_load_uops']*100:.2f}%                           {metrics['B']['dram_read_GB_s']:.2f} GB/s
C      {metrics['C']['remote_share_of_local_plus_remote_retired_dram_load_uops']*100:.2f}%                          {metrics['C']['dram_read_GB_s']:.2f} GB/s
完全複製は相手側DRAMからの読み出しを減らした。一方Cはその割合を減らさず、帯域利用と速度を改善している。
相手側割合は retired local/remote DRAM load uops の比率。QPI全通信量の比率やGB/sではなく、プリフェッチ・キャッシュ間転送・同期通信をすべて数えているわけではない。
カウンターは100%稼働、観察窓内ディスク読み出し0、major/minor fault0、CPU移動0。最大CPU温度46-48度、熱スロットリング増分0。
IMCはシステム全体の値。観察は速度比較と別実行で、計測の負荷を速度差に混ぜていない。

物理配置の確認
NUMA bindした両コピーは全ページ常駐し、それぞれ指定ノードに存在した。全658種類のGGUF重みが含まれ、両CPUの参照切替を確認、判定失敗によるfallbackは0回。
move_pagesは問い合わせだけ（flags=0、移動要求なし）で使い、各4MiB範囲の4KiBページごとの所属ノードを調べた。
通常THPの専門家重みでは同じノードが512ページ=2MiB連続する区間があり、4KiB固定では1ページごとにnode0/node1が交互だった。
全体の左右配分が50:50でも、一度に使う専門家重みの中の配分・細かさは異なる。
この違いが短時間のメモリコントローラ利用に影響する、というのが有力な解釈。ただしTHP無効化はTLB挙動も変えるため、原因を配置粒度だけに限定した証明ではない。

正しさと実装範囲
従来版・複製無効・1コピー対照・完全複製で34位置 x 262144語彙 = 8,912,896個のlogitがビット単位で一致、非有限値なし。
部分書き込み・memset・読み戻しとnode0/node1/両CPU/SMTの54グラフ比較を各コピー方式で実行、一致。
実装変更は隔離したsourceの3ファイル。既存のNUMA複製バッファを拡張し、演算の入力ポインターだけを各ワーカーのローカルコピーへ切り替える。
共有グラフと演算値を変更せず、RMS_NORM+MUL融合も対応。このモデルでは元々IQ2専用MoEペア融合の対象外。
試験用GGML_CPU_NUMA_FULL=1は2コピー、=2は1コピー。未指定では無効。
THPはwithout_thp.pyが試験プロセスと子プロセスだけで無効化する。OS全体のTHP/NUMA設定は変更していない。

大きいモデルの部分複製を考えるとき
『相手側の重みを自分側にも置くと通信が減る』という仮説は確認できた。
ただしこの26Bでは、1コピーの細かい分散を先に行うと、全複製による残りの改善は約2%だった。
数十GBを複製する前に、DeepSeekでもページサイズと配置粒度を比較する価値がある。
その後に残る待ちが大きければ、頻繁に使われる重みや通信負担の大きい部分を選んで複製する。
ファイルの先頭や単なるサイズ順を『上位』として選ぶ根拠はない。MoEの利用頻度、常時使う重み、通信量と追加メモリ量を基準にする。
別モデルでの改善率・最適な複製量は未測定で、今回の約2%も普遍的な上限ではない。

成果物と現在の状態
通常のMelody/DeepSeekランチャー、active source、active CPUライブラリはハッシュで変更なしを確認。
実験用ランチャー:
  {LAB/'run_melody_full_replica_experiment.sh'}
  {LAB/'run_melody_fine_interleave_experiment.sh'}
いずれも既存の通常ランチャーとは別。完全複製を4KiB条件で試す場合は、without_thp.pyで完全複製ランチャーを起動できる。
起動中の最大プロセスRSSは1コピー42.85GiB、完全複製64.48GiB。元ファイルのmmapを読みながらコピーを作るため、定常RSSより大きい。
CPU専用・今回のモデル専用の試験状態。DeepSeekへ全複製をそのまま有効にしない（容量制限・別の融合経路がある）。

再現用のコマンドと環境は各summary.json、ソース変更はfull-replica.patch、詳細値はREPORT.jsonを参照。
初回placement-fullはロード前の取得タイミング誤りで配置確認が失敗したため除外し、placement-verifiedで再検証済み。速度結論にも使っていない。
'''
(OUT/'REPORT_JA.txt').write_text(text)
print(json.dumps({'report':str(OUT/'REPORT_JA.txt'),'complete':True,'runtime_unchanged':True},ensure_ascii=False))
