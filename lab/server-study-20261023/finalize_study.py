#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import statistics

SRC=Path(__file__).resolve().parent
OUT=SRC.parent/'results/server-study-20261023'
def main():
    before=json.loads((OUT/'inventory-before/inventory.json').read_text())
    after=json.loads((OUT/'inventory-after/inventory.json').read_text())
    checks={'same_boot':before['boot_id']==after['boot_id'],
            'runtime_hashes_unchanged':before['runtime_sha256']==after['runtime_sha256'],
            'global_settings_unchanged':{k:v for k,v in before['sys'].items() if k!='/proc/meminfo'}=={k:v for k,v in after['sys'].items() if k!='/proc/meminfo'},
            'EDAC_error_counts_unchanged':all(before['edac'][mc][k]==after['edac'][mc][k] for mc in before['edac'] for k in ['ce_count','ue_count']),
            'thermal_throttle_counts_unchanged':before['telemetry']['throttle_counts']==after['telemetry']['throttle_counts']}
    frequency_keys=['scaling_driver','scaling_governor','scaling_min_freq','scaling_max_freq','cpuinfo_min_freq','cpuinfo_max_freq']
    checks['frequency_policy_unchanged']=all(before['telemetry']['frequency'][p][k]==after['telemetry']['frequency'][p][k] for p in before['telemetry']['frequency'] for k in frequency_keys)
    checks['power_limits_unchanged']=all(before['telemetry']['powercap'][p][k]==after['telemetry']['powercap'][p][k] for p in before['telemetry']['powercap'] for k in before['telemetry']['powercap'][p] if k not in ['energy_uj'])
    def modules(d):return set(line.split()[0] for line in d['commands']['modules'].get('stdout','').splitlines()[1:] if line.strip())
    validation={'checks':checks,'modules_added':sorted(modules(after)-modules(before)),
                'modules_removed':sorted(modules(before)-modules(after)),
                'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.iterdir() if p.suffix in ['.py','.cpp'] and p.name!='expert_probe.cpp'},
                'note':'No global CPU/NUMA/THP/power policy changes; process-local affinity and new anonymous mappings were used for measurement.'}
    assert all(checks.values()),checks
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    s=json.loads((OUT/'summary.json').read_text());disk=json.loads((OUT/'storage-read.json').read_text())
    idle=json.loads((OUT/'idle-power.json').read_text())['power']
    def v(name,key='logical_GBs'):return s['cases'][name][key]['median']
    def interval(name,key='logical_GBs'):
        x=s['cases'][name][key];return f"{x['min']:.2f}〜{x['max']:.2f}"
    report=f'''増設前の実機観察：CPU・メモリ・NUMA・電力・PCIe
2026-10-23 日本時間

GPU選定のシミュレーションは、ユーザーの希望により終了した。
ここにある数値は現在のサーバーで実際に測定したもの。GPU性能予測、GPU処理、推論エンジン変更は行っていない。

主な結果

現在の128GB構成で、片側CPUから自分側のRAMを連続読み出しすると約24.5GB/s。
両CPUの12物理コアでインターリーブされたRAMを読むと{v('read_both_physical'):.2f}GB/s。
24論理CPUへ増やしても、この読み出し処理では{v('read_both_SMT'):.2f}GB/sで伸びなかった。
相手側のRAMだけを読む場合は約19〜21GB/s。依存するランダム読み出しでは待ち時間が約100nsから約198nsへ増えた。
1コアによるコピーは、自分側RAM内なら約3.40GB/s、コピー先を相手側RAMに変えると約1.64GB/s。
NUMAをまたぐ影響は、連続読出し、ランダム読出し、書込みで異なる。

1. 実機の状態

基板の報告値：Sugon X9DR3-F、Rev1.11A、BIOS3.4（2020-06-30）。
CPU：Xeon E5-2630L v2 ×2、合計12物理コア/24論理CPU。AVX/F16Cあり、AVX2/FMAなし。
CPU1＝NUMA node0＝Linux CPU0〜5/12〜17。
CPU2＝NUMA node1＝Linux CPU6〜11/18〜23。
RAM：32GB×4枚＝128GB。Micron 72KSZS4G72L1G、4rank、設定1600MT/s。
CPU1にはchannel0/1、CPU2にはchannel2/3が実装され、現在は各CPU2channel、合計4channelを使用している。
最終16枚の512GB構成では各CPU8枚となる。チャネル数と容量は別なので、容量4倍を帯域4倍とは解釈しない。増設後の動作クロックと帯域は実測する。
現在PCIeに列挙されるGPUはオンボードMatroxだけ。PG500-216を使った試験は行っていない。
OS：Debian13、kernel6.12.107+deb13-amd64。現在NVIDIAドライバー/CUDA開発環境は未導入。
測定時boot ID：{before['boot_id']}

2. メモリの連続読出し（各3回、中央値）

実行するCPUとRAM                    GB/s
CPU1 → CPU1側RAM、6物理コア          {v('read_n0_local'):.2f}
CPU1 → CPU2側RAM、6物理コア          {v('read_n0_remote'):.2f}（範囲{interval('read_n0_remote')}）
CPU2 → CPU2側RAM、6物理コア          {v('read_n1_local'):.2f}
CPU2 → CPU1側RAM、6物理コア          {v('read_n1_remote'):.2f}（範囲{interval('read_n1_remote')}）
両CPU、12物理コア、RAMインターリーブ {v('read_both_physical'):.2f}
両CPU、24論理CPU、RAMインターリーブ  {v('read_both_SMT'):.2f}

1GiBの実配列を繰り返し読むAVX処理。L3だけに収まる試験ではない。
両CPU12物理コア時のメモリコントローラー実測も約{v('read_both_physical','IMC_read_GBs'):.2f}GB/sで、プログラム側の読出量と整合した。
この処理では、読み出しが完了する前にCPUが待つ割合が高い。これを全LLM処理のボトルネック割合へ流用しない。
SMTがこの読出し試験に効かなかったことは、量子化展開やLLMでもSMTが不要という意味ではない。

3. 依存するランダム読出し（1コア、128MiB、4KiBページ）

実行CPU → RAM                     1回の実効待ち時間
CPU1 → CPU1側RAM                  {v('latency_n0_local','dependent_load_ns'):.2f} ns
CPU1 → CPU2側RAM                  {v('latency_n0_remote','dependent_load_ns'):.2f} ns
CPU2 → CPU2側RAM                  {v('latency_n1_local','dependent_load_ns'):.2f} ns
CPU2 → CPU1側RAM                  {v('latency_n1_remote','dependent_load_ns'):.2f} ns

次のアドレスが直前の読出し結果で決まるpointer chase。多数の読出しを先行させる連続アクセスとは条件が違う。
TLBやcacheの影響を含む実効時間であり、DRAMチップ単体の遅延ではない。QPIの転送帯域そのものを測った数値でもない。

4. メモリコピー（CPU1の1コア、512MiB）

コピー元 → コピー先               コピーしたデータ量/秒
CPU1側RAM → CPU1側RAM              {v('copy_n0_local1'):.3f} GB/s
CPU2側RAM → CPU1側RAM              {v('copy_n0_from_remote1'):.3f} GB/s
CPU1側RAM → CPU2側RAM              {v('copy_n0_to_remote1'):.3f} GB/s

最後の条件は自分側コピーの約{100*s['copy_remote_destination_vs_local_ratio']:.1f}%。
同じ容量・同じCPUでも、書き込み先がどちらにあるかで大きな差が出た。
コピーしたpayloadのGB/sであり、DRAMの読み出し＋書き込みの合計とは違う。
遠いコピー先の場合、DRAM読み出しは約{v('copy_n0_to_remote1','IMC_read_GBs'):.2f}GB/s、書き込みは約{v('copy_n0_to_remote1','IMC_write_GBs'):.2f}GB/sだった。
この追加読み出しの原因までは切り分けていない。CPUによるコピーの測定であり、GPUのDMA転送速度を示すものではない。

5. クロック・温度・電力

全コア読出し中のturbostat表示は約2.6GHz。1コアの試験では約2.8GHzが観測された。
現在の設定はintel_cpufreq＋schedutil、範囲1.2〜2.8GHz、Turbo有効。設定は変更していない。
CPUパッケージの長期電力制限は各60W、短期制限各72Wと報告されている。
40秒連続の24論理CPU読出し：
  読出速度 {v('read_both_SMT_sustained'):.2f}GB/s
  CPUパッケージ2個のRAPL合計 約{v('read_both_SMT_sustained','RAPL_package_W'):.1f}W
  DRAMのRAPL合計 約{v('read_both_SMT_sustained','RAPL_DRAM_W'):.1f}W
  合計 約{v('read_both_SMT_sustained','RAPL_package_plus_DRAM_W'):.1f}W
  今回の全測定で観測したCPU最高温度 {s['max_observed_CPU_C']:.1f}℃

アイドル寄りの10秒観測はCPUパッケージ＋DRAMの合計約{idle['package_plus_dram_W']:.1f}Wだった。
RAPLはCPU/DRAMのエネルギーカウンターから求めた値。電源入力やコンセントの実測ではなく、マザーボード・HDD・ファン・電源損失を含まない。
今回の128GBと読出し負荷に限る。512GB化後、GPU搭載後、演算中心の負荷にそのまま適用してはいけない。
これだけを根拠にGPUの電力上限を上げない。運用の出発点は合意したECC有効・150Wを維持する。

6. ECC・熱制限

sb_edacが両CPUのメモリコントローラーを監視し、x4 DIMM/S4ECD4EDモードを報告している。
測定前後の訂正エラー/訂正不能エラーはすべて0。熱制限カウンターの増加も0。
これは今回の起動後と試験中の観測。生涯エラー0や、128GB全域を検査済みという意味ではない。

7. PCIeの実機対応

基板上の名前      Linuxルートポート     NUMA   最大リンク
CPU1_SLOT2        0000:00:02.0          0      PCIe3.0 x16
CPU2_SLOT4        0000:80:02.0          1      PCIe3.0 x16
CPU2_SLOT6        0000:80:03.0          1      PCIe3.0 x16

SMBIOSと実際のバス接続を照合した。現在は空スロットなのでリンク幅x0。故障を示すものではない。
CPU2側のSMBIOS Bus AddressはLinuxのdomain表記と一致しない部分があるので、搭載後の実BDFを読む。
64-bit bridgeの対応能力は確認できるが、空のwindowだけからAbove4G Decodingの設定やV100のBAR割当成功を断定しない。
DIMMやPCHヒートシンクとの物理的な干渉、ライザー/固定具、ファンの吸気スペースは現物で確認する。

8. HDDと空き容量

ディスクはWDC WD10SPZX-75Z10T0、約1TBのSATA HDD。
既存のGLMモデルの大きいシャードから、先頭付近/中間/末尾付近を各1GiB、O_DIRECTで読み出した。
観測速度：{disk['results'][0]['MB_s']:.1f} / {disk['results'][1]['MB_s']:.1f} / {disk['results'][2]['MB_s']:.1f} MB/s。
Linuxのpage cacheを全体破棄せず、既存ファイルは書き換えていない。これはダウンロード速度やランダムI/O性能ではない。
現時点の空きは約{disk['filesystem']['available_unprivileged_bytes']/1e9:.1f}GB。
将来510GB級のモデル一式を保存すると、変換や別コピーのための余裕は小さくなる。今回はモデル削除やダウンロードを行っていない。

測定方法と再現

18条件を順序反転を挟んで3回ずつ、各約4秒測定し、別に40秒の連続読出しを1回行った。主測定55件。
全配列を実際に確保・初期化し、mbindとmove_pagesによるサンプリングでNUMA配置を確認。新規配列だけMADV_NOHUGEPAGEを指定。
OpenMP workerを指定CPUに固定し、開始前と終了後にaffinityを検証。読出し/コピーのchecksumも検証した。
perfのenable/disable確認を使って初期化を計測外にし、CPU/IMCカウンター、RAPL、turbostat、温度、ECCを記録。
全core/IMCイベントのrunning率99.9%以上。主測定中のmodel disk readとmajor faultは0。観測の前後処理を含む区間には少数のminor faultがある。
プログラムの速度はC++timed loop、カウンターとRAPLは短い開始/終了処理も含む観測区間。両者の時間窓が完全一致するという主張はしない。
初回smokeは出力ファイルのflush確認で記録検査に失敗し、修正後の3種smokeは通過。初回は主集計から除外。
HDD試験の初回は小さな先頭シャードを検出して読み始める前に中断し、大きいシャードを選んで測定した。

通常のllamaランタイムとglobal設定の前後一致をvalidation.jsonで確認済み。
GPU選定シミュレーションやGPU負荷試験は実行していない。GPU向けの処理設計は到着後の実測を待つ。

主なファイル：
inventory-before/after、pcie-map.json、measurements.tsv、counter-details.json、summary.json、storage-read.json、validation.json。
再測定手順はREADME_REPRODUCE.txt。元の55件は保存して、増設後は別の出力先で比較する。

参照：
Linux powercap/RAPLのエネルギーカウンター仕様
https://docs.kernel.org/power/powercap/powercap.html
Supermicro基板配置（実機データと照合）
https://www.supermicro.com/QuickRefs/motherboard/C606_602/QRG-X9DR3_i-F.pdf
'''
    (OUT/'REPORT_JA.txt').write_text(report)
    print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
