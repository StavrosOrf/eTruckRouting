"""Compare frozen-policy sampling temperatures without selecting lucky rollouts.

Only opened development scenarios are used. Physical scenario RNG and policy
sampling RNG are separate. Repeated policy seeds are not independent trainings.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
from scripts.evaluation import run_canonical_campaign as campaign
from EVRoutingEnv.state.action_mask import policy_action_mask

ORIGINAL_BUILD_POLICY=campaign.build_policy


def read(path):return json.loads(path.read_text())


def dump(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class TemperatureRunner:
    def __init__(self,settings):
        self.base=campaign.CanonicalPolicyRunner(Path(settings['checkpoint']),prefix=settings.get('prefix','best'))
        self.temperature=float(settings['sampling_temperature'])
        self.sampling_seed=int(settings['sampling_seed'])
        self.scenario_seed=None
        self.generator=torch.Generator(device='cpu')

    @torch.no_grad()
    def __call__(self,env,observation,info):
        if self.scenario_seed!=env.scenario_seed:
            self.scenario_seed=env.scenario_seed
            key=f'GraphPPO independent policy RNG|{self.sampling_seed}|{self.scenario_seed}'
            seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63)
            self.generator.manual_seed(seed)
        batch=torch.as_tensor(np.expand_dims(observation,0),dtype=torch.float32)
        mask=torch.as_tensor(np.expand_dims(policy_action_mask(env),0),dtype=torch.bool)
        output=self.base.policy(batch,mask)
        if self.temperature==0:
            action=output.logits.argmax(dim=-1)
        else:
            probabilities=torch.softmax(output.logits/self.temperature,dim=-1)
            action=torch.multinomial(probabilities,1,generator=self.generator).flatten()
        if not bool(output.feasible_mask[0,action[0]]):
            raise ValueError('sampled action violates the trained mask')
        return int(action[0])


def build_policy(name,settings):
    if 'sampling_temperature' in settings:
        return TemperatureRunner(settings)
    return ORIGINAL_BUILD_POLICY(name,settings)


campaign.build_policy=build_policy


def verify(root):
    for filename,expected in read(root/'input_hashes.json').items():
        if digest(Path(filename))!=expected:raise ValueError('frozen temperature-study input changed: '+filename)


def summarize(root):
    module=load(root/'evaluation_pipeline.py','temperature_pipeline')
    reporter=module.reporter(root)
    methods=read(root/'methods.json')
    cached=read(root/'cached_methods.json')
    inputs=[(entry['method'],Path(entry['directory'])) for entry in read(root/'reference_inputs.json')]
    inputs.extend((name,root/'scores/validation'/name) for name in methods)
    baseline=Path(read(root/'protocol.json')['baseline_evaluation'])
    for name in cached:
        inputs.extend((name,baseline/'shards'/str(index)/'validation'/name) for index in range(4))
    rows={};manifests={}
    for name,directory in inputs:
        values=module.load_rows(directory/'episode_rows.jsonl');module.validate_task_rows(values,name)
        target=rows.setdefault(name,{})
        for row in values:
            seed=row['scenario_seed']
            if seed in target:raise ValueError('duplicate physical scenario')
            target[seed]=row
        manifests[str(directory)]=read(directory/'manifest.json')
    expected=set(range(1000002000,1000002128))
    if any(set(values)!=expected for values in rows.values()):raise ValueError('incomplete scenario coverage')
    module.check_manifests(manifests)
    anchor=next(iter(rows.values()))
    for values in rows.values():
        for seed,row in values.items():
            if reporter.physical_scenario(row)!=reporter.physical_scenario(anchor[seed]):raise ValueError('physical scenario differs')
    replay_fields=('success','terminated','truncated','termination_reason','customers_served','customers_total',
                   'total_operating_time','total_travel_time','total_charging_time','total_queue_time',
                   'total_service_time','total_time_window_waiting','vehicles_used','policy_calls')
    replayed=0
    for name,settings in methods.items():
        if settings['sampling_temperature']!=0:continue
        original=settings['base_method']
        for seed,row in rows[name].items():
            reference=rows[original][seed]
            if any(row[key]!=reference[key] for key in replay_fields):
                raise ValueError('zero-temperature replay differs from cached greedy: '+name+' '+str(seed))
            replayed+=1
    if replayed!=384:raise ValueError('greedy control did not replay all three actors and 128 cases')
    dump(root/'greedy_replay_audit.json',{'status':'passed','episodes':replayed,'fields':list(replay_fields),
                                       'same_physical_cases':True,'same_outcomes_and_costs':True})
    references=read(root/'protocol.json')['reference_methods']
    report=reporter.summarize(rows,references)
    report['evaluation_role']='opened development decoding study; one executed trajectory per predefined policy RNG and case; no best-of-rollouts oracle'
    dump(root/'independent_reference_report.json',report)
    groups={}
    for name,settings in {**methods,**cached}.items():groups.setdefault(settings['arm'],[]).append((name,settings))
    keys=('success_rate','mean_operating_hours_successful','mean_paired_excess_percent')
    metrics={}
    for arm,entries in groups.items():
        per_training_seed={}
        for seed in range(3):
            selected=[report['methods'][name] for name,settings in entries if settings['training_seed']==seed]
            per_training_seed[str(seed)]={key:sum(row[key] for row in selected)/len(selected) for key in keys}
            per_training_seed[str(seed)]['sampling_replicates']=len(selected)
        metrics[arm]={key:sum(row[key] for row in per_training_seed.values())/3 for key in keys}
        metrics[arm].update(training_seeds=3,per_training_seed=per_training_seed,
                            sampling_replicates_are_independent_training_seeds=False)
    dump(root/'summary.json',{'status':'completed','metrics':metrics,'role':report['evaluation_role'],
         'all_failures_retained':True,'costs_conditioned_on_success':True,'optimizer_updates':0,
         'final_validation_opened':False,'test_opened':False,'goal_achieved':False})
    dump(root/'state.json',{'status':'completed','goal_achieved':False})
    print(json.dumps(metrics,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();root=args.directory.resolve()
    torch.set_num_threads(1);verify(root)
    if (root/'summary.json').exists():raise RuntimeError('study already completed; inspect its report')
    methods=read(root/'methods.json');config=yaml.safe_load((root/'config.yaml').read_text())
    # Require exact completed files before treating a method as cached.
    for name in methods:
        folder=root/'scores/validation'/name
        if folder.exists() and not (folder/'summary.json').exists():
            raise RuntimeError('partial method execution; inspect before any retry: '+name)
    jobs=[(name,settings,config,list(range(1000002000,1000002128)),'validation',str(root/'scores/validation'),10000,tuple(sys.argv)) for name,settings in methods.items()]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('fork')) as pool:
        for name,_ in pool.map(campaign.score_method,jobs):print('completed temperature method',name,flush=True)
    verify(root);summarize(root)


if __name__=='__main__':main()
