#!/usr/bin/env python3
"""Write a conditional simulation report; all GPU inputs remain assumptions."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

SRC = Path(__file__).resolve().parent
LAB = SRC.parent
OUT = LAB/'results/hybrid-sim-20261023'


def read(mode):
    rows = list(csv.DictReader((OUT/f'{mode}-results.tsv').open(),delimiter='\t'))
    for row in rows:
        for key in row:
            if key not in ('id','card'):
                row[key] = float(row[key])
    return rows


def main():
    screen_all, sweep_all = read('screen'), read('sweep')
    screen = {r['id']:r for r in screen_all if r['turn']==-1}
    sweep = {r['id']:r for r in sweep_all if r['turn']==-1}
    for name in ['screen','sweep']:
        assert json.loads((OUT/f'{name}-status.json').read_text())['exit_code']==0
    inputs = json.loads((OUT/'inputs.json').read_text())
    cpu = inputs['CPU_calibration']['24']
    assert all(r['tokens']==2256 for r in [*screen.values(),*sweep.values()])
    assert abs(screen['control_ideal_lru']['hit_fraction']-.9234208776595745)<1e-11
    expected_cpu = cpu['ms'][6]*40 + .04*40
    for card,common in [('P40',35),('V100',15)]:
        assert abs(screen[card+'_none']['ms_per_token']-(expected_cpu+.8*common))<1e-6
    crossovers=[]
    for reserve in [8,10,12]:
        for scale in [1,1.5,2]:
            for pcie in [6,12]:
                def key(g,card):return f'grid_r{reserve}_b{scale}_p{pcie}_g{g}_{card}'
                v=sweep[key(0,'V100')]['ms_per_token']
                points=[(g,sweep[key(g,'P40')]['ms_per_token']-v) for g in [0,10,20,30,40]]
                crossing=None
                for (x0,y0),(x1,y1) in zip(points,points[1:]):
                    if y0<=0<=y1:crossing=x0+(x1-x0)*(-y0)/(y1-y0)
                crossovers.append({'reserve_GiB':reserve,'memory_scale':scale,'pcie_GBs':pcie,
                                   'extra_common_ms_P40_break_even_interpolated':crossing})
    examples=[]
    for scale in [1,1.5]:
        for gap in [10,20,30]:
            examples.append({'memory_scale':scale,'P40_common_ms':15+gap,'V100_common_ms':15,
                **{card:sweep[f'grid_r10_b{scale}_p12_g{gap}_{card}'] for card in ['P40','V100']}})
    p,v=screen['P40_adaptive8'],screen['V100_adaptive8']
    report={'complete':True,'inputs':inputs,'scenario_count':len(screen)+len(sweep),
            'screen':screen_all,'sweep':sweep_all,'crossovers':crossovers,'examples':examples,
            'checks':['C++ analytic transfer/completion/capacity/overlap/full-cache checks',
                      '4500-slot instantaneous LRU matches all 102080 public per-layer hit counts',
                      'No-cache end-to-end time equals calibrated CPU time plus specified serial GPU time',
                      'All 233 scenarios completed and preserve 2256 scored decode steps'],
            'caveats':['GPU timings/bandwidths are assumed, not measurements of P40/V100 at 200W.',
                       'The compute floor is a cache-kernel proxy, not a proof of the CPU architectural peak.',
                       'Memory-side scaling 1.5 or 2 is hypothetical; 512GB does not imply these multipliers.',
                       'Only one 8-turn short-context trace from the September release was used, not 1017.',
                       'GPU non-expert time is distributed uniformly over 40 layers; real layer timings differ.',
                       'CPU microbenchmark omits nonlinear activation and per-expert activation quantization; 40us handoff/dispatch is an assumed allowance.',
                       'Copy bandwidth contention uses a simplified fluid model, not a PCIe/DRAM cycle simulator.',
                       'No batched prefill, quality, long-context attention, speculative decode, or multi-user performance prediction.',
                       'Calibration streamed measurements preceded the user-triggered reboot; compute-proxy measurements followed it. Same CPU/library, separate runs.']}
    (OUT/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['P40 24GB / V100 16GB: CPUフォールバック付きexpertキャッシュのシミュレーション',
           '2026-10-23 日本時間','',
           '結果の位置づけ','これは公開実ルーティング履歴とXeon実測に基づく、条件付きの離散イベント計算。GPU実機の速度予測を検証したものではない。',
           'GPUは両方とも未入手。200W制限そのものを電力モデルで計算せず、200W運用時の実効性能を未知の入力として扱った。',
           '一つの値に決めず、GPU共通処理時間・expert演算速度・PCIe帯域・VRAM予約量・メモリ帯域を振った。','',
           '分かったこと',
           '1. CPUでミスを計算しつつ、最近よく選ばれるexpertだけを非同期に入れる方式が有利。転送完了待ちを毎層挟むLRUは遅い。',
           '2. 共通予約10GiB、PCIe12GB/s、最大8個/tokenの更新では、P40のGPU共通処理がV100より約13〜15ms/token余計にかかる所が逆転点。',
           '3. 共通予約8〜12GiBまで広げると逆転点は約9〜20ms/tokenへ動く。GPUの作業領域も結論に効く。',
           '4. MXFP4では現在のCPU処理がキャッシュ内演算能力に比較的近い。帯域だけ2倍にする仮定でもCPU時間を半分にはしていない。','',
           'CPU実測（両Xeon、24スレッド、5回の中央値）',
           'H5120 / 中間2304、gate/up/downの3行列をMXFP4で処理。64 expert分1.2GBの物理メモリを回してLLC内の繰返しを避けた。',
           'CPU担当expert数    1層分の時間ms']
    lines += [f'{i:>6}               {cpu["ms"][i]:.4f}' for i in range(1,7)]
    lines += [f'キャッシュ内カーネルからの計算時間目安: 1 expert {cpu["compute_floor_ms"][1]:.4f}ms。',
              '入力量子化・SwiGLU・実エンジンのグラフ処理はこのマイクロ測定に含まれない。CPU/GPU受渡し等に別途40us/層を仮定。','',
              '代表ケースの仮定',
              'V100: expert用6GiB/342個、expert実効550GB/s、共通処理15ms/token。',
              'P40 : expert用14GiB/799個、expert実効220GB/s、共通処理35ms/token。',
              '共通処理にはattention・shared expert・dense投影等を含む。20%はCPU expertと並列可能と仮定。',
              'PCIeは12GB/s。転送とCPUのメモリ帯域競合は0.5係数、GPU側帯域競合も含めた。',
              '上記はGPU実測ではない。特にP40のFP16回避経路のコストは共通処理時間に現れる。','',
              '代表ケース: 方針別のシミュレーション（tok/s）','方針                      P40        V100']
    for label,key in [('expert GPU配置なし','none'),('入力時だけ更新','prefill'),('非同期LRU/queue32','async_lru'),('毎層転送完了待ちLRU','sync_lru'),('頻度更新 最大4個/token','adaptive4'),('頻度更新 最大8個/token','adaptive8'),('頻度更新 最大16個/token','adaptive16')]:
        lines.append(f'{label:<22} {screen["P40_"+key]["tok_s"]:>7.2f}     {screen["V100_"+key]["tok_s"]:>7.2f}')
    lines += ['', '最大8個更新の中身（コピー完了前はヒットに数えない）',
              'GPU      ヒット率   CPU担当個数/token   完了転送MB/token   新規コピー個数/token']
    for card,row in [('P40',p),('V100',v)]:
        lines.append(f'{card:<8} {row["hit_fraction"]*100:>6.2f}%       {(1-row["hit_fraction"])*240:>7.2f}            {row["DMA_MB_per_token"]:>7.2f}                {row["copies_per_token"]:.2f}')
    lines += ['頻度は実際に観測した選択だけで更新。毎token0.97倍減衰、候補1.5回相当以上、既存expertより15%以上熱いものを優先。',
              '1token最大8個、同時待ち行列32個。空きも含めキャッシュ容量を厳守し、転送中の枠も使用量に数える。',
              'CPUとGPUのexpert処理は並行するが、各層の最後で両方が終わるまで次の層へ進まない。expertの省略・次層への先送りはしていない。','',
              '優劣が逆転する例（予約10GiB、PCIe12GB/s、最大8個更新）','メモリ側倍率  P40共通ms  V100共通ms   P40 tok/s   V100 tok/s']
    for row in examples:
        lines.append(f'{row["memory_scale"]:>7}          {row["P40_common_ms"]:>3}          15        {row["P40"]["tok_s"]:>6.2f}       {row["V100"]["tok_s"]:>6.2f}')
    lines += ['', '予約量別の逆転点（P40がV100より余計に使ってよい共通処理時間）','予約GiB   メモリ側倍率   追加ms/tokenの境目（近似）']
    for row in crossovers:
        if row['pcie_GBs']==12 and row['memory_scale'] in [1,1.5]:
            lines.append(f'{row["reserve_GiB"]:>5}           {row["memory_scale"]:>3}                {row["extra_common_ms_P40_break_even_interpolated"]:.1f}')
    lines += ['この境目は仮定したexpert演算速度・同期時間・共通処理の並行割合でも動く。普遍的なGPU性能基準ではない。','',
              '512GB化の扱い','全expertとEngram等がRAMに載り、HDD読出しを必要としない状態を仮定。',
              'メモリ側の時間を1/1.5または1/2にしても、キャッシュ内演算時間の目安より短くしないモデル。1.5倍と2倍で結果がほぼ同じになるのは、この計算上限に当たるため。',
              'これは512GB増設による改善量を測った結果ではない。DIMM構成・動作クロック・演算最適化により変わる。','',
              '検証と限界',*report['checks'],*report['caveats'],'',
              'データの由来','https://github.com/acupof-ai/cachedMoE/blob/84ba1ab723b1397e04e5532bdd50e8e0668594b2/bench/results/hitrate/base_auto/route.bin',
              '同じrepoのturns.jsonで入力/生成を区別。2552全ステップのうち生成2256ステップを報告。',
              '元の4500スロットのヒット率を全102080層で再現。CPU測定は既存AVX版libggml-cpu.soを使用、実装・OS設定を変更していない。','',
              '成果物','REPORT.json / screen-results.tsv / sweep-results.tsv: 全条件と集計値。',
              'details/: 代表ケースのtoken単位の時間・ヒット・転送量。',
              'cpu-calibration/ と compute-*/: 実CPUの校正ログ。',
              'hybrid-sim.cpp / run_simulation.py: 再現用コード。仮定はinputs.jsonと*-configs.jsonへ保存。']
    (OUT/'REPORT_JA.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete':True,'scenarios':len(screen)+len(sweep),'central_P40':p,
        'central_V100':v,'crossovers':crossovers},indent=2))


if __name__=='__main__':main()
