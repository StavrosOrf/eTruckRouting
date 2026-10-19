"""Replay opened frozen trajectories and query mask fallbacks without changing actions."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
import yaml
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics
from EVRoutingEnv.state import action_mask as masks
from EVRoutingEnv.state.feasibility import joint_action_feasibility


def strict_completion_mask():
    # Expose the pre-fallback restriction for diagnostics only. Neither the
    # pinned source nor the mask used to choose the recorded actions is edited.
    source=inspect.getsource(masks.completion_action_mask)
    fallback='    if not result.any() and legal_mask.any():\n        return legal_mask\n'
    if source.count(fallback)!=1:
        raise ValueError('inspect completion-mask fallback before diagnosing it')
    source=source.replace(fallback,'').replace('def completion_action_mask(', 'def audit_strict_completion_mask(',1)
    namespace=dict(vars(masks))
    exec(compile(source,'<diagnostic completion mask>','exec'),namespace)
    return namespace['audit_strict_completion_mask']


def diagnose(evaluation,output):
    strict=strict_completion_mask()
    methods=json.loads((evaluation/'methods.json').read_text())
    cases=[]
    for path in sorted((evaluation/'failure_traces').glob('large_warm*.json')):
        recorded=json.loads(path.read_text())
        config=yaml.safe_load((evaluation/'config.yaml').read_text())
        settings=methods[recorded['method']]
        for section,values in settings['environment_overrides'].items():
            config[section].update(values)
        env=EventDrivenTruckEnv(config,verbose=False,enable_plotting=False)
        try:
            env.reset(seed=recorded['scenario_seed'])
            rows=[]
            for expected in recorded['trace']:
                active=env.trucks[env.active_truck_id]
                if active.truck_id!=expected['truck'] or active.current_node!=expected['node']:
                    raise ValueError('prefix replay state changed')
                legal=masks.get_action_mask(env)
                baseline=strict(env,legal)
                old=config['environment'].get('completion_work_escape_lookahead',False)
                try:
                    config['environment']['completion_work_escape_lookahead']=True
                    escape=strict(env,legal)
                finally:
                    config['environment']['completion_work_escape_lookahead']=old
                action=expected['action']
                decisions=joint_action_feasibility(env)
                if not legal[action]:
                    raise ValueError('recorded action is no longer execution-legal')
                row={key:expected[key] for key in ('step','truck','node','battery','payload','kind','target','customers_served')}
                row.update(legal_candidates=int(legal.sum()),strict_candidates=int(baseline.sum()),
                           fallback_to_legal=bool(legal.any() and not baseline.any()),
                           chosen_survives_strict=bool(baseline[action]),
                           work_escape_strict_candidates=int(escape.sum()),
                           chosen_survives_work_escape=bool(escape[action]),
                           newly_rejected_by_work_escape=bool(baseline[action] and not escape[action]),
                           legal_kinds={kind:sum(bool(legal[i]) and decision.action_kind.value==kind for i,decision in enumerate(decisions)) for kind in ('customer','charger','charge','depot')},
                           unfinished_customers=[{'node':task.node_id,'available':bool(task.is_available),'claimed_by':task.claimed_by,'demand':task.demand} for task in env.task_registry.tasks() if not task.is_served])
                rows.append(row)
                _,_,terminated,truncated,_=env.step(action)
            metrics=extract_operational_metrics(env).as_dict()
            if metrics['success']!=recorded['success'] or not np.isclose(metrics['total_operating_time'],recorded['operating_hours']):
                raise ValueError('mask-query replay changed the physical outcome')
            case={'method':recorded['method'],'scenario_seed':recorded['scenario_seed'],'termination_reason':metrics['termination_reason'],
                  'frozen_replay_matched':True,'decisions':len(rows),
                  'fallback_steps':[row for row in rows if row['fallback_to_legal']],
                  'work_escape_new_rejections':[row for row in rows if row['newly_rejected_by_work_escape']]}
            cases.append(case)
            output.mkdir(parents=True,exist_ok=True)
            (output/f"{recorded['method']}_{recorded['scenario_seed']}.json").write_text(json.dumps({'summary':case,'all_steps':rows},indent=2)+'\n')
            print(recorded['method'],metrics['termination_reason'],'fallbacks',len(case['fallback_steps']),'work-escape rejections',len(case['work_escape_new_rejections']),flush=True)
        finally:
            env.close()
    (output/'summary.json').write_text(json.dumps({'status':'completed','role':'recorded trajectory replay and counterfactual mask queries, no counterfactual policy execution or optimizer updates','cases':cases},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    diagnose(args.evaluation,args.output)
