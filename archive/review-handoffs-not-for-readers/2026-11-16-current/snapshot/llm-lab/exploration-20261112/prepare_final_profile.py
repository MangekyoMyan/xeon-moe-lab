"""Prepare the final configuration only after a direct, accepted whole-stack comparison."""
from pathlib import Path
import copy,hashlib,json
from heldout_probe import private_write
R=Path(__file__).resolve().parent
def main():
    selection=json.loads((R/'FINAL_SELECTION.json').read_text());result=json.loads((R/'FINAL_ABBA.json').read_text())
    assert result['comparison_accepted'] and result['all_outputs_identical'] and result['aggregate_gain_percent']>0
    baseline_sha='e808d438d4ad4d852febb405dbf3d083cdf57f10e56e023ee6186eae43d340bd'
    assert selection['baseline_config_sha256']==baseline_sha
    input_sha=hashlib.sha256((R/'FRESH_INPUTS.json').read_bytes()).hexdigest();assert result['input_sha256']==input_sha
    assert all(len(result['sources'][arm])==2 for arm in ['control','candidate'])
    paths=[p for arm in ['control','candidate'] for p in result['sources'][arm]];assert len(set(paths))==4
    chronology=[]
    for arm,expected in [('control',baseline_sha),('candidate',selection['candidate_config_sha256'])]:
        for path in result['sources'][arm]:
            proof=json.loads(Path(path).read_text())
            assert proof['complete'] and proof['error'] is None and proof['config_sha256']==expected and proof['input_sha256']==input_sha
            assert proof['token_collection']=='Complete generated token IDs from native nonstream final response'
            assert len(proof['cases'])==8 and all(c['output_token_count']==c['timings']['predicted_n']==128 for c in proof['cases'])
            chronology.append((proof['started_unix'],arm))
    assert [arm for _,arm in sorted(chronology)]==['control','candidate','candidate','control']
    source=R/'profiles'/(selection['candidate_profile']+'.json');raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==selection['candidate_config_sha256']
    original=json.loads(raw);profile=copy.deepcopy(original)
    profile['tested_on_real_model']=True
    profile['gpu_stream_profile'].update(name='gpu3-stream64-verified',full_model_validation=str(R/'FINAL_ABBA.json'))
    profile['cpu_iq3_compact_profile'].update(name='frequency19200-120GiB-verified',full_model_validation=str(R/'FINAL_ABBA.json'))
    profile['vision_profile'].pop('full_image_inference_verified',None)
    profile['vision_profile']['verification_record']=str(R/'FINAL_STATUS.json')
    profile['optimization_study'].update(profile='final',status='direct-final-ABBA-verified',selected_profile=selection['candidate_profile'],final_comparison=str(R/'FINAL_ABBA.json'),final_process_checks=str(R/'FINAL_STATUS.json'))
    assert profile['command']==original['command'] and profile['environment']==original['environment']
    assert profile['study_libraries']==original['study_libraries'] and profile['study_files']==original['study_files']
    target=R/'profiles/final.json'
    if target.exists():assert json.loads(target.read_text())==profile
    else:private_write(target,profile)
    manifest={'profile':'final','profile_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'selected_profile':selection['candidate_profile'],'selected_config_sha256':selection['candidate_config_sha256'],
              'runtime_command_and_environment_unchanged':True,'benchmark_gain_percent':result['aggregate_gain_percent'],'post_switch_checks_record':str(R/'FINAL_STATUS.json'),'checks_state_at_preparation':'not yet run','rollback_profile':'baseline'}
    output=R/'FINAL_PROFILE_MANIFEST.json'
    if output.exists():assert json.loads(output.read_text())==manifest
    else:private_write(output,manifest)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
