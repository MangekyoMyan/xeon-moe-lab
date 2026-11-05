#!/usr/bin/env python3
import json
from pathlib import Path
import statistics as st
import xml.etree.ElementTree as ET

LAB=Path(__file__).resolve().parent.parent
OUT=LAB/'results/ram-upgrade-20261031'
d=json.loads((OUT/'comparison.json').read_text())
assert d['all_phases_complete']
before=json.loads((OUT/'before.json').read_text())
after=json.loads((OUT/'after.json').read_text())
matrix=d['matvec']['matvec-k4096-stream-t24']
local=d['local_matvec']['matvec-local-t24']
max_cpu=max([d['sampled_CPU_max_C']]+[v['after']['maximum_cpu_temperature_C'] for v in d['counters'].values()])
q8=matrix['q8_0'];q4=matrix['q4_K'];q3=matrix['q3_K']
lines=[f'''512GB増設後：llama.cppの短いCPUテストとメモリ帯域比較
測定日：2026-10-31（日本時間）

結論
増設は有効だった。Q8の大容量・24スレッド内積は、同じ形状・同じメモリ配置で
{q8['before']['seconds_per_matvec']*1000:.3f}→{q8['after']['seconds_per_matvec']*1000:.3f} ms、速度{q8['speedup']:.2f}倍。
Q4_Kは{q4['speedup']:.2f}倍。一方Q3_Kは{q3['speedup']:.2f}倍で、改善していない。
「512GB化によりすべてCPUボトルネックになった」わけではない。
Q8にはメモリ読み出し・NUMA配置の制約が残る。Q3_KはCPUの展開・内積処理の制約が強い。
同じ512GB内でQ8の重みをワーカー側RAMに置くと、さらに{q8['after']['seconds_per_matvec']*1000:.3f}→{local['q8_0']['after']['seconds_per_matvec']*1000:.3f} ms、
速度{local['q8_0']['speedup_vs_new_interleaved']:.2f}倍になった。これは容量増加とは別の配置変更による効果。
これは合成カーネルの実測。GPU併用の実モデルtokens/sやprefillの改善率ではない。

1. 増設後の認識
32GiB×16枚＝512GiB。各CPUに8枚＝256GiBずつ。
全DIMMのDMI Configured Memory Speedは1600 MT/s。各CPU4チャネル、各チャネル2枚。
IMCカウンターでも、両CPUの全8チャネルで読み出しを観測した。
増設前は各CPU2チャネルだったので、容量は4倍、使用チャネル数は2倍。
64bitデータ幅×1600MT/s×全8チャネルの理論データ帯域は102.4GB/s。実効値とは区別する。
DMIの内訳は既存Micron 72KSZS4G72L1Gが4枚、Samsung M386B4G70DM0-系が12枚。
Samsungの末尾はDMIの表示が一部切れており、正確な完全品番はここから断定しない。
CPUはE5-2630L v2×2、12物理/24論理CPU、AVX/F16Cあり・AVX2/FMAなし。
カーネルは6.12.113+deb13-amd64。GPUは待機状態、APIサービスは測定前後ともinactive。

2. メモリを読むだけのテスト（GB/s、各3回の中央値）
条件                                      128GB時       512GB時       比率
''']
labels={'read_n0_local':'CPU0側でローカルRAMを読む（6物理コア）',
        'read_n1_local':'CPU1側でローカルRAMを読む（6物理コア）',
        'read_both_physical':'両CPU、12物理コア、ページ交互配置',
        'read_both_SMT':'両CPU、24論理CPU、ページ交互配置',
        'read_n0_remote':'CPU0側からCPU1側RAMを読む',
        'read_n1_remote':'CPU1側からCPU0側RAMを読む'}
for name,label in labels.items():
    row=d['memory'][name]
    lines.append(f"{label}  {row['before']['logical_GBs']['median']:.2f} → {row['after']['logical_GBs']['median']:.2f}  ({row['after_over_before']:.2f}倍)")
lines.append('\n別の2GiB連続読出し試験（これも同一バイナリで過去と比較）：')
for key,label in [('interleave','両CPUでページを交互配置'),('local','各ワーカーが自分側RAMを読むfirst-touch配置')]:
    row=d['local_read'][key]
    lines.append(f"{label}: {row['before']:.2f} → {row['after']['median']:.2f} GB/s ({row['speedup']:.2f}倍)")
ratio=d['local_read']['local']['after']['median']/d['local_read']['interleave']['after']['median']
lines.append(f'''同じ512GB内でも、後者の配置が前者の{ratio:.2f}倍。CPU間をまたぐアクセスの影響が残っている。
2GiB試験と上の1GiB試験には実装・ページ設定の違いがあり、それぞれの過去値と比較する。
CPU間接続の生の最大帯域を測ったわけではない。遠隔RAM読出しにはQPIとメモリ遅延が両方関わる。
''')
for key in ('latency_n0_local','latency_n0_remote'):
    row=d['memory'][key]
    lines.append(f"{key}: 依存読出し {row['before']['dependent_load_ns']['median']:.2f} → {row['after']['dependent_load_ns']['median']:.2f} ns")
lines.append('容量と連続帯域は増えたが、ランダムな1回の読出し待ち時間が短縮したわけではない。')
lines.append('''
3. llama.cpp量子化内積（大容量、24スレッド、1行列あたりms）
K4096×M65536。Q8の重み272MiB、短い右入力を再利用。6回の中央値。
主比較は前回と同じNUMA interleave。最後の列だけ新たにlocalallocへ変更。
形式       128GB交互配置   512GB交互配置   速度比   512GBローカル配置
''')
for kind in ('q8_0','q6_K','q5_K','q4_K','q3_K','q2_K','q4_0','iq4_xs'):
    r=matrix[kind];l=local[kind]
    lines.append(f"{kind:<9} {r['before']['seconds_per_matvec']*1000:>9.3f} {r['after']['seconds_per_matvec']*1000:>15.3f} {r['speedup']:>8.2f}x {l['after']['seconds_per_matvec']*1000:>13.3f}")
lines.append('''
量子化の初期変換・メモリ確保は内積計時から除外。全形式で内積を参照計算と照合し通過。
Q4_K_Mなどのモデル名は混合形式なので、ここでの単一Q4_Kと同一ではない。
量子化形式ごとの回答品質はテストしていない。

4. CPU側の制約をどう判定したか
test-quantize-perfのCPU0・4096要素のキャッシュ内内積では：''')
for kind in ('q8_0','q6_K','q4_K','q3_K'):
    ratios=[r['speedup'] for name,rows in d['stock'].items() if name.startswith('quant-perf-cache-long')
            for r in rows if r['after']['type']==kind and r['after']['operation']=='vec_dot_q']
    lines.append(f"  {kind}: 増設前比{st.median(ratios):.3f}倍（2セットの中央値）")
lines.append('''CPU内キャッシュで処理できる場合はほぼ据え置き。これを対照とし、大容量での改善を評価した。

大容量内積の15秒観測窓（24スレッド、交互配置）：
形式    実DRAM読出しGB/s 旧→新   IPC 旧→新    L2待ちを伴う停止/cycles 旧→新''')
for kind in ('q8_0','q6_K','q4_K','q3_K'):
    r=d['counters'][kind];o=r['before'];n=r['after']
    lines.append(f"{kind:<7} {o['dram_read_GB_s']:6.2f} → {n['dram_read_GB_s']:6.2f}   {o['ipc']:.3f} → {n['ipc']:.3f}     {o['stall_l2_pending_fraction']*100:.1f}% → {n['stall_l2_pending_fraction']*100:.1f}%")
lines.append('''この停止割合は重なりうるイベントの比率で、実行時間の独立した内訳ではない。
Q8は帯域を多く使い、メモリ待ちも残る。ローカル配置でさらに速くなるかを直接照合した。
Q3_Kは少ない帯域でも速度が伸びず、L2待ちを伴う停止も小さい。
このためQ3_Kは「RAMが遅いから」よりも、CPUカーネルの処理能力が主な制約と判断する。
Q6/Q4は両者の中間で、SMTによって処理能力を補う効果もある。

同じ512GB・交互配置で12→24スレッドの速度比：''')
for kind in ('q8_0','q6_K','q4_K','q3_K'):
    twelve=d['matvec']['matvec-k4096-stream-t12'][kind]['after']['seconds_per_matvec']
    twentyfour=matrix[kind]['after']['seconds_per_matvec']
    lines.append(f"  {kind}: {twelve/twentyfour:.3f}倍")
lines.append(f'''
5. 安全・比較条件・限界
測定中のCPU最高温度は{max_cpu:.0f}℃。EDACの訂正可能/訂正不能ECCは前後とも0、熱制限カウンター増加0。
カウンター観測窓での対象処理のHDD読出し・major faultは0。
ファン設定、GPU電力制限、ネットワーク、API、OS設定は変更していない。モデルもロードしていない。
これは数GB以下の合成負荷であり、512GB全域のメモリ健全性や長時間安定性を保証する試験ではない。
増設前の10月19/21/23日の測定後にはカーネル更新やGPU搭載もあるため、RAMだけを同時点で差し替えた完全なA/Bではない。
ただし測定バイナリとlibggml-cpuのSHA256一致を確認し、形状・スレッド・環境変数も揃えた。
これは比較用の保存済みllama.cppカーネルであり、現在の独自エンジン全体の性能測定ではない。
通常エンジンでのexpert共有・ルーティング・同期・GPU転送は含まれず、ローカル配置の改善率をそのまま適用できない。

保存先
このレポート：{OUT/'REPORT_JA.txt'}
集計・全比較：{OUT/'comparison.json'}
生ログ：同ディレクトリのmemory、quant、counters、local。
実行前後のハードウェア：before.json、after.json。
一致確認：binary-verification.json、historical-hash-matches.json。
測定方法と再現手順：{LAB/'ram-upgrade-20261031/README_REPRODUCE.txt'}
測定ドライバー：{LAB/'ram-upgrade-20261031/run.py'}
''')
(OUT/'REPORT_JA.txt').write_text('\n'.join(lines)+'\n')
print(OUT/'REPORT_JA.txt')
