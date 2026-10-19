"""Observable routing relations between candidate action destinations.

Node IDs are used only to match existing typed observation rows. No network,
case-study map, witness, solver route or future disturbance is queried here.
"""
from __future__ import annotations
import torch
from algo.canonical_state import NODE_FEATURE_NAMES
from EVRoutingEnv.state.features import ACTION_FEATURES, EDGE_FEATURES, TRUCK_FEATURES
from EVRoutingEnv.state.feasibility import ActionKind

ACTION_PAIR_FEATURE_DIM=8


@torch.no_grad()
def action_pair_features(tensors,feasible):
    counts=feasible.sum(dim=1)
    width=int(counts.max().item())
    batch_size=feasible.shape[0]
    positions=feasible.nonzero(as_tuple=False)
    pointers=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
    local=torch.arange(positions.shape[0],device=feasible.device)-pointers[positions[:,0]]
    slots=torch.zeros((batch_size,width),dtype=torch.long,device=feasible.device)
    slots[positions[:,0],local]=positions[:,1]
    valid=torch.arange(width,device=feasible.device).unsqueeze(0)<counts.unsqueeze(1)
    packed=tensors.action.gather(1,slots.unsqueeze(-1).expand(-1,-1,tensors.action.shape[-1]))
    target=packed[...,ACTION_FEATURES.index('target_node')]
    endpoint_type=torch.full_like(slots,-1)
    endpoint_index=torch.zeros_like(slots)
    kinds=('customer','charger','truck')
    for code,kind in enumerate(kinds):
        names=NODE_FEATURE_NAMES[kind]
        identity='current_node' if kind=='truck' else 'node_id'
        nodes=tensors.nodes[kind][...,names.index(identity)]
        matches=(target.unsqueeze(-1)==nodes.unsqueeze(1)) & tensors.node_masks[kind].unsqueeze(1)
        found=matches.any(dim=-1) & valid & (endpoint_type<0)
        endpoint_type[found]=code
        endpoint_index[found]=matches.to(torch.long).argmax(dim=-1)[found]
    routing=packed.new_zeros((batch_size,width,width,len(EDGE_FEATURES)))
    known=torch.zeros((batch_size,width,width),dtype=torch.bool,device=feasible.device)
    batch=torch.arange(batch_size,device=feasible.device).view(-1,1,1)
    for i,source in enumerate(kinds):
        for j,destination in enumerate(kinds):
            table=tensors.pairwise[(source,destination)]
            source_index=endpoint_index.clamp(0,table.shape[1]-1)
            target_index=endpoint_index.clamp(0,table.shape[2]-1)
            values=table[batch,source_index.unsqueeze(2),target_index.unsqueeze(1)]
            mask=tensors.pairwise_mask[(source,destination)][batch,source_index.unsqueeze(2),target_index.unsqueeze(1)]
            use=(endpoint_type==i).unsqueeze(2) & (endpoint_type==j).unsqueeze(1) & mask
            routing=torch.where(use.unsqueeze(-1),values,routing)
            known |= use
    # Incoming depot costs are explicit node features even when no truck is
    # currently at the depot. Directed costs from the depot are never inferred
    # by reversing these edges; unknown edges retain the baseline GCN message.
    depot_code=list(ActionKind).index(ActionKind.DEPOT)
    all_depots=(tensors.action[...,ACTION_FEATURES.index('kind_code')]==depot_code) & tensors.action_padding_mask
    depot_slot=all_depots.to(torch.long).argmax(dim=1)
    depot=tensors.action[torch.arange(batch_size,device=feasible.device),depot_slot,ACTION_FEATURES.index('target_node')]
    is_depot=(target==depot.unsqueeze(1)) & all_depots.any(dim=1).unsqueeze(1)
    for i,kind in enumerate(kinds):
        names=NODE_FEATURE_NAMES[kind]
        nodes=tensors.nodes[kind]
        row_index=endpoint_index.clamp(0,nodes.shape[1]-1)
        values=nodes.gather(1,row_index.unsqueeze(-1).expand(-1,-1,nodes.shape[-1]))
        home=torch.stack([values[...,names.index(name)] for name in ('depot_energy_kwh','depot_travel_hours','depot_reachable')],dim=-1)
        use=(endpoint_type==i).unsqueeze(2) & is_depot.unsqueeze(1)
        routing=torch.where(use.unsqueeze(-1),home.unsqueeze(2),routing)
        known |= use
    same=(target.unsqueeze(2)==target.unsqueeze(1)) & valid.unsqueeze(2) & valid.unsqueeze(1)
    routing=torch.where(same.unsqueeze(-1),torch.zeros_like(routing),routing)
    routing[...,EDGE_FEATURES.index('reachable')]=torch.where(same,torch.ones_like(routing[...,2]),routing[...,2])
    known |= same
    known &= valid.unsqueeze(2) & valid.unsqueeze(1)
    active=tensors.nodes['truck'][...,TRUCK_FEATURES.index('is_active')]
    capacities=tensors.nodes['truck'][...,TRUCK_FEATURES.index('battery_capacity_kwh')]
    energy_scale=(active*capacities).sum(dim=1).clamp_min(1.0)
    travel=packed[...,ACTION_FEATURES.index('leg_travel_hours')]
    positive=(travel>0) & valid
    time_scale=(travel.masked_fill(~positive,0).sum(dim=1)/positive.sum(dim=1).clamp_min(1)).clamp_min(1e-3)
    customer_code=list(ActionKind).index(ActionKind.CUSTOMER)
    customer=(packed[...,ACTION_FEATURES.index('kind_code')]==customer_code).to(packed.dtype)
    charge=packed[...,ACTION_FEATURES.index('charge_value')]
    energy=(routing[...,EDGE_FEATURES.index('nominal_energy_kwh')]/energy_scale[:,None,None]).clamp(0,20)
    hours=(routing[...,EDGE_FEATURES.index('nominal_travel_hours')]/time_scale[:,None,None]).clamp(0,20)
    result=torch.stack((energy,hours,routing[...,EDGE_FEATURES.index('reachable')],same.to(packed.dtype),
          customer.unsqueeze(2).expand(-1,-1,width),customer.unsqueeze(1).expand(-1,width,-1),
          charge.unsqueeze(2).expand(-1,-1,width),charge.unsqueeze(1).expand(-1,width,-1)),dim=-1)
    return result,known
