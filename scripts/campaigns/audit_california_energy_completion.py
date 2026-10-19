"""Query optimistic energy-completion bounds along recorded development actions."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing
from pathlib import Path
import numpy as np
import yaml
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.feasibility import joint_action_feasibility
from EVRoutingEnv.state.action_mask import policy_action_mask

_ENV=None


def replay(task):
    global _ENV
    trace_path,config_path,helper_path,output=map(Path,task[:4])
    factor,compare_batched,effective_policy_mask=task[4:]
    record=json.loads(trace_path.read_text())
    if _ENV is None:
        _ENV=EventDrivenTruckEnv(yaml.safe_load(config_path.read_text()),verbose=False,enable_plotting=False)
    env=_ENV
    spec=importlib.util.spec_from_file_location('optimistic_completion_audit',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    env.reset(seed=record['scenario_seed'])
    bounds=helper.OptimisticCompletionBounds(env)
    if factor is not None:bounds.factor=factor
    compared_decisions=0
    compared_candidates=0
    rejected=[]
    strict_rejected=[]
    fallback_steps=[]
    for row in record['trace']:
        if int(env.active_truck_id)!=row['truck']:
            raise ValueError('recorded active truck changed')
        decisions=joint_action_feasibility(env)
        decision=decisions[row['action']]
        if compare_batched:
            legal=np.asarray([item.feasible for item in decisions],dtype=bool)
            scalar=np.asarray([bounds.candidate_possible(item) if legal[index] else False for index,item in enumerate(decisions)],dtype=bool)
            batched=bounds.candidate_mask(decisions,legal)
            if not np.array_equal(scalar,batched):
                raise ValueError(f'vectorized energy bounds disagree: {trace_path.name}, step {row["step"]}')
            compared_decisions+=1
            compared_candidates+=int(legal.sum())
        if not decision.feasible:
            raise ValueError('recorded reference action became execution-illegal')
        strict_allowed=bounds.candidate_possible(decision)
        details={'step':row['step'],'truck':row['truck'],'kind':decision.action_kind.value,'target':decision.target_node,'customers_served':row['customers_served']}
        if not strict_allowed:strict_rejected.append(details)
        allowed=strict_allowed
        if effective_policy_mask:
            original=dict(env.config['environment'])
            try:
                env.config['environment'].update(completion_shield=True,completion_capacity_lookahead=True,completion_capacity_exact_lookahead=True,completion_work_escape_lookahead=False)
                baseline=policy_action_mask(env)
                restricted=bounds.candidate_mask(decisions,baseline)
                if baseline[row['action']] and bool(restricted[row['action']])!=strict_allowed:
                    raise ValueError('chosen scalar and batched bounds disagree')
                if baseline.any() and not restricted.any():
                    restricted=baseline
                    fallback_steps.append(row['step'])
                allowed=bool(restricted[row['action']])
            finally:
                env.config['environment'].clear();env.config['environment'].update(original)
        if not allowed:rejected.append(details)
        _,_,terminated,truncated,_=env.step(row['action'])
    metrics=extract_operational_metrics(env).as_dict()
    if bool(metrics['success'])!=record['success'] or not np.isclose(metrics['total_operating_time'],record['operating_hours']):
        raise ValueError('query-only replay changed recorded outcome')
    result={'method':record['method'],'scenario_seed':record['scenario_seed'],'success':metrics['success'],
            'termination_reason':metrics['termination_reason'],'operating_hours':metrics['total_operating_time'],
            'decisions':len(record['trace']),'compared_mask_decisions':compared_decisions,'compared_legal_candidates':compared_candidates,'rejected_steps':rejected,'strict_rejected_steps':strict_rejected,'energy_fallback_steps':fallback_steps,'frozen_outcome_matched':True}
    output.mkdir(parents=True,exist_ok=True)
    (output/trace_path.name).write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--helper',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--energy-factor',type=float)
    p.add_argument('--compare-batched',action='store_true')
    p.add_argument('--effective-policy-mask',action='store_true')
    args=p.parse_args()
    paths=sorted(args.traces.glob('*.json'))
    paths=[path for path in paths if path.name!='summary.json']
    tasks=[(*map(str,(path,args.config,args.helper,args.output/'cases')),args.energy_factor,args.compare_batched,args.effective_policy_mask) for path in paths]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for row in pool.map(replay,tasks):
            results.append(row)
            if len(results)%16==0:print(f'queried {len(results)}/{len(tasks)} frozen episodes',flush=True)
    success=[row for row in results if row['success']]
    report={'status':'completed','role':'recorded development-action replay; optimistic mask queries only, not counterfactual policy execution',
            'effective_policy_mask_including_fallback':args.effective_policy_mask,
            'strict_successful_routes_rejected':sum(bool(row['strict_rejected_steps']) for row in success),
            'strict_successful_decisions_rejected':sum(len(row['strict_rejected_steps']) for row in success),
            'energy_fallback_queries':sum(len(row['energy_fallback_steps']) for row in results),
            'energy_factor_override':args.energy_factor,
            'restriction_semantics':'optimistic necessary condition' if args.energy_factor is None else 'experimental energy-risk restriction; not a necessary physical-feasibility condition',
            'compared_mask_decisions':sum(row['compared_mask_decisions'] for row in results),
            'compared_legal_candidates':sum(row['compared_legal_candidates'] for row in results),
            'episodes':len(results),'successful_episodes':len(success),
            'successful_decisions':sum(row['decisions'] for row in success),
            'successful_routes_rejected':sum(bool(row['rejected_steps']) for row in success),
            'successful_decisions_rejected':sum(len(row['rejected_steps']) for row in success),
            'all_frozen_outcomes_matched':all(row['frozen_outcome_matched'] for row in results),'optimizer_updates':0,
            'final_validation_opened':False,'test_opened':False,'results':results}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='results'},indent=2),flush=True)

if __name__=='__main__':main()
