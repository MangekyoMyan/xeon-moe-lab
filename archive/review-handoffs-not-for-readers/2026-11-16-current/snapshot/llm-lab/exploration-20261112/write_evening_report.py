"""Close the bounded exploration with measured tradeoffs and a stable default."""
from pathlib import Path
import hashlib,json,re,statistics,time
R=Path(__file__).resolve().parent;O=R/'hot-ffn'
def one(pattern):
    paths=list(R.glob(pattern));assert len(paths)==1,(pattern,len(paths));return paths[0]
def load(path):return json.loads(path.read_text())
def main():
    baseline=load(R/'FINAL_ABBA.json');miss=load(O/'MISS_ABBA.json');quality=load(O/'QUALITY_COMPARISON.json')
    candidate_path=one('heldout/*-hot14-r1/RESULT.json');candidate=load(candidate_path);assert candidate['complete'] and candidate['error'] is None
    control=load(Path(miss['sources']['control'][0]));rate=sum(c['timings']['predicted_n']-1 for c in candidate['cases'])*1000/sum(c['timings']['predicted_ms'] for c in candidate['cases'])
    rate0=miss['aggregate']['control']['decode_tok_s'];gain=(rate/rate0-1)*100
    cpp=load(one('runs/*-hot-final-control-prefill-count-plain-c4096-n16/RESULT.json'));gpp=load(one('runs/*-hot14-prefill-count-plain-c4096-n16/RESULT.json'))
    pp0=cpp['final']['timings']['prompt_ms'];pp1=gpp['final']['timings']['prompt_ms']
    long_path=one('long-cache/*-hot14-16k/RESULT.json');long=load(long_path);vision_path=one('vision/*-hot14-newspaper/RESULT.json');vision=load(vision_path)
    assert long['passed'] and long['error'] is None and vision['headline_terms_present']
    text=(R.parent/'deepseek41-bringup-20261101/server.stderr.log').read_text(errors='replace')
    rows=[]
    for m in re.finditer(r'xeon-hot-stats layer=(\d+) calls=(\d+) H=\[([0-9,]+)\]',text):
        if int(m[2])>1000:rows.append({'layer':int(m[1]),'calls':int(m[2]),'H_histogram':list(map(int,m[3].split(',')))})
    assert len(rows)==40 and {x['layer'] for x in rows}==set(range(40)) and len({x['calls'] for x in rows})==1
    assert all(sum(x['H_histogram'])==x['calls'] for x in rows)
    hits=sum(sum(h*n for h,n in enumerate(x['H_histogram'])) for x in rows);routes=sum(x['calls']*6 for x in rows)
    observations=[long['initial_watch'],long['final_watch']]+[x['watch'] for x in long['samples']]
    gpu_keys=set().union(*(x.get('gpu',{}).keys() for x in observations))
    memory_key=next((x for x in ['used_MiB','memory_used_MiB','vram_used_MiB'] if x in gpu_keys),None)
    peak=max(x['gpu'][memory_key] for x in observations if memory_key in x.get('gpu',{})) if memory_key else None
    per_case=[]
    controls=[load(Path(p)) for p in miss['sources']['control']]
    for index,row in enumerate(candidate['cases']):
        cr=[x['cases'][index] for x in controls];base_rate=sum(x['timings']['predicted_n']-1 for x in cr)*1000/sum(x['timings']['predicted_ms'] for x in cr)
        per_case.append({'name':row['name'],'control_tok_s':base_rate,'candidate_tok_s':row['timings']['predicted_per_second'],'gain_percent':(row['timings']['predicted_per_second']/base_rate-1)*100,'same128tokens':row['token_sha256']==cr[0]['token_sha256']})
    result={'adopted_as_default':False,'default_profile':'final','candidate_profile':'hot14-trace','scope':'Control two runs before candidate;candidate one run,8fixedinputs,128tokens. Not a balanced positive-residency ABBA. Different generated outputs;fixedprefix comparison is separate.',
        'control_tok_s':rate0,'candidate_tok_s':rate,'free_generation_gain_percent':gain,'full128token_matches':sum(x['same128tokens'] for x in per_case),'cases':per_case,
        'prefill4096_control_ms':pp0,'prefill4096_candidate_ms':pp1,'prefill_time_increase_percent':(pp1/pp0-1)*100,
        'actual_hot_routes':hits,'actual_total_routes':routes,'actual_hot_fraction':hits/routes,'dispatch_layers':rows,
        'routing_scope':'Entire positivepilot: freegeneration,fixedprefix,short/longcount,vision. Not training coverage or per-case hit distribution.',
        'quality':{k:v for k,v in quality.items() if k!='cases'},'vision_result':str(vision_path),'long_cache_result':str(long_path),'observed_long_GPU_peak_MiB':peak,
        'decision':'Preserve as experiment: generation gains depend on input,4096prefill slower,outputs change;general quality not established. Keep directly ABBA-verified output-identical checkpoint as ordinary API.'}
    (O/'MODEL_PILOT_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    # Keep every previously measured profile immutable; add a documented optional profile.
    c=load(R/'profiles/hot14.json');c['tested_on_real_model']=True
    c['optimization_study'].update(profile='hot14-reviewed',status='experimental-functionally-verified-not-default',validation_record=str(O/'MODEL_PILOT_RESULTS.json'),production_recommended=False,
        validation_difference='Same algorithm/libraries/plan as hot14-trace;ordinary logging level3 instead of4')
    target=R/'profiles/hot14-reviewed.json';raw=(json.dumps(c,indent=2)+'\n').encode()
    if target.exists():assert target.read_bytes()==raw
    else:target.write_bytes(raw);target.chmod(0o600)
    detail='\n'.join(f"  {x['name']}: {x['control_tok_s']:.3f} → {x['candidate_tok_s']:.3f} tok/s ({x['gain_percent']:+.2f}%) /128 IDs一致={x['same128tokens']}" for x in per_case)
    report=f'''2026-11-12 追加探索・終了記録

通常APIの採用構成
  profiles/final.json。今回開始時との直接ABBAで生成6.719→7.590 tok/s、+12.96%。
  新しい8入力×各設定2回、各128生成tokenの全文・全token IDsが一致。
  CPU RAMのexpert単位compact選択、MXFP4入力補正/scale再利用、CUDA expert ID分類改善を採用。
  以前の96→120GiB cache拡大は開始時点に含まれるため、今回の改善率へ重複計上しない。
  詳細はcheckpoint-1600/RESULTS_JA.txt。終了時の実稼働確認はEVENING_FINAL_STATUS.json。

追加したGPU expert常駐の実験
  約14GiBに1,035個の(layer,expert)をgate/up/down一組で配置。
  原重みをCPU RAMへ保持し、T=1のGPU hitとCPU missを並行計算。GPU側の結果は別pinned領域へ回収後、元の6slot順で合成。
  schedulerへは入力とIDsだけを渡し、元のrouter重み・加算順を維持。T>1は既存の処理。
  既存GPU常駐3層をCPUへ戻し、GPUと重なるCPU compact2,402slotsを除去して約14.844GiB回収。
  CPU planは80→87行へ更新し、CPU compact payloadは19,200→16,798slots。
  現12物理core/Volta/DeepSeek41/1context用の明示的な実験。別CPU・モデルには再検証が必要。

実モデルの結果
  生成: {rate0:.3f} → {rate:.3f} tok/s ({gain:+.2f}%)。
  対照2回、候補1回の観測。両対照は候補より前で、positive案の完全ABBAではない。
  入力と生成数は同じだが、自由生成の128 IDsが完全一致したのは8件中{sum(x['same128tokens'] for x in per_case)}件。
{detail}
  4096入力: {pp0/1000:.3f} → {pp1/1000:.3f}秒、{(pp1/pp0-1)*100:.2f}%長くなった（各1回）。
  実pilot全体のGPU hitは{hits:,}/{routes:,}={hits/routes*100:.2f}%。
  40層全てで実行カウンタを確認。層37–39には訓練censusがなく、GPU cache選択も0。
  訓練集計44.77%はCPU層0–36のみの値で、この実測hit率と分母・入力が異なる。

固定prefixでの数値比較
  baselineの64token列を固定し、候補の生成tokenを次の入力に混ぜず、毎回1tokenだけ追加入力。
  対照は256/256地点で自己再現。候補のgreedy一致は251/256 (98.05%)。
  255地点でraw top100 logprobを比較。UTF8表示都合の1地点欠落を補完せず扱った。
  最大共通token確率差は約19.86 percentage points、平均共通部分L1差は約0.01976。
  教師token logprob差の平均は-0.005726 nats。
  同じprefixのT1処理時間平均133.302→119.033ms、10.70%短縮。
  n_probs100付きの測定であり、連続自由生成tok/sとは別。完全KL/TVD、PPL、一般回答品質の評価ではない。

機能とメモリ
  新聞画像: {vision['answer'].strip()}。画像の観測VRAM最大{vision['observed_gpu_peak_MiB']:.0f}MiB。
  16k入力: {long['requests'][0]['response']['timings']['prompt_ms']/1000:.3f}秒。
  継続はcache_n=16399/prompt_n=1、入力処理{long['requests'][1]['response']['timings']['prompt_ms']/1000:.6f}秒。
  長文側の観測GPU最大MiB: {peak if peak is not None else 'raw long-cache記録を参照'}。
  context131072設定は維持。128k全域の性能・品質を実測した意味ではない。

採否
  GPU常駐は有効な候補として保存。今回の静的配置では入力処理と一部入力の生成が遅くなり、出力も変わる。
  一般品質をこの比較だけで判断せず、通常APIは出力一致を確認したfinal構成へ戻す。
  必要な次の検討は全40層の実routingと、入力による配置の偏り、CPU/GPU計算差の評価。今回はここで区切る。

他の候補
  native pthread poll50はpoll0比+3.09%（ABBA、出力一致）。通常OpenMPを上回る強い根拠は得られず。
  入力native24/生成OpenMP12の混合は生成-0.11%、入力も少し遅く、採用しない。
  TLBの2入力観測でPMH busy/user cyclesは約0.7–0.8%。stall率ではなく、今回はTHP変更へ進まなかった。
  既存のTHP無効・interleave・12生成/24入力・batch1024・150W/ECC/fan<=70を維持。

検証上の注意と記録
  専用backendの実scheduler検証は2層×90cases。normal/ASAN+UBSANで再実行し、リーク検査も通過。
  CUDA初期化とASANのshadow領域が衝突し、初回san試験は推論前に失敗した。
  最小再現でprotect_shadow_gap=0だけを追加すると解消。意図的host heap越境が検出されることも確認し、detect_leaks=1は維持。
  NVIDIAの参考: https://github.com/NVIDIA/cutlass/blob/main/CMakeLists.txt
  変更provider/testはinstrument済み。再利用したggml/CUDA libraryの全内部をinstrumentした検査ではない。
  compute-sanitizerは未導入。新しいGPU kernelは追加せず、既存allocator/MMVQを使用。
  サーバーlv3ではlibrary INFOが抑制され、all-miss ABBA後のcounter照合だけ失敗した。
  同じ凍結binaryへlv4を付けた別試験で37層×15callsを確認。元のABBA生記録とfailureを残した。

利用・復帰
  通常起動: sudo python3 /home/codex/projects/llm-lab/deepseek41-bringup-20261101/start.py
  実験を明示して試す: sudo python3 {R}/switch_and_wait.py hot14-reviewed
  安定版へ戻す: sudo python3 {R}/switch_and_wait.py final
  切替ではAPIを再起動する。GPU150W/ECC、128k f16 KV、fan<=70の方針は共通。

主要証拠
  FINAL_ABBA.json / checkpoint-1600/ / EVENING_FINAL_STATUS.json
  hot-ffn/BACKEND_VALIDATION.json / ASAN_CUDA_COMPAT.json / MISS_ABBA.json / MISS_DISPATCH.json
  hot-ffn/MODEL_PILOT_RESULTS.json / QUALITY_COMPARISON.json / FINAL_PILOT_CONTROLLER.json
  hot-ffn/ENGINE_BUILD.json / release-a/ / CAPACITY.json / FIXTURE_RESULTS.json
  hybrid-pool/VALIDATION.json / LIVE_ABBA.json / DECISION.json / cpu-wait/TLB_OBSERVATION.json
  元GGUF・以前の凍結構成・公開repositoryは保全。公開push、ダウンロード、ハードウェア設定変更は行っていない。
'''
    (R/'EVENING_RESULTS_JA.txt').write_text(report)
    print(json.dumps({'report':str(R/'EVENING_RESULTS_JA.txt'),'GPU_hot_fraction':hits/routes,'GPU_peak_MiB':peak,'default':'final','optional_profile':'hot14-reviewed'}),flush=True)
if __name__=='__main__':main()
