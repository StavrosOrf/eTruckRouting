"""Observable, optimistic energy bounds for candidate fleet continuations.

These necessary conditions ignore time, queues and charger outages, allow full
recharging at every station, and use the minimum possible energy multiplier.
Passing is not a completion certificate. No event-queue disturbance, witness,
solver plan or future random draw is read. The helper is not enabled in training.
"""
from __future__ import annotations
import math
import numpy as np
from EVRoutingEnv.state.feasibility import ActionKind


class OptimisticCompletionBounds:
    def __init__(self, env):
        self.env=env
        self.depot=int(env.joint_instance.depot_node)
        self.stations=tuple(sorted(env.charging_nodes))
        self.station_set=set(self.stations)
        traffic=env.traffic_config
        self.factor=(float(traffic['min_energy_multiplier'])
                     if traffic['enable_traffic'] and traffic['enable_energy_uncertainty'] else 1.0)
        if not math.isfinite(self.factor) or self.factor<=0:
            raise ValueError('energy lower multiplier must be positive and finite')
        self.tasks=tuple(env.task_registry.tasks())
        self.nodes=tuple(int(task.node_id) for task in self.tasks)
        self.indices={node:index for index,node in enumerate(self.nodes)}
        self._energy_cache={}
        self._capacity_cache={}
        self._reach_cache={}
        self.station_energy=np.asarray([[self.energy(i,j) for j in self.stations] for i in self.stations],dtype=float).reshape(len(self.stations),len(self.stations))
        self.station_customer=np.asarray([[self.energy(i,j) for j in self.nodes] for i in self.stations],dtype=float).reshape(len(self.stations),len(self.nodes))
        self.customer_station=np.asarray([[self.energy(i,j) for j in self.stations] for i in self.nodes],dtype=float).reshape(len(self.nodes),len(self.stations))
        self.customer_depot=np.asarray([self.energy(node,self.depot) for node in self.nodes])
        self.station_depot=np.asarray([self.energy(node,self.depot) for node in self.stations])

    def energy(self,origin,target):
        key=(int(origin),int(target))
        if key not in self._energy_cache:
            if key[0]==key[1]:
                value=0.0
            else:
                try:value=float(self.env.transport_graph.get_path_energy(*key))
                except (KeyError,ValueError):value=math.inf
            if math.isnan(value) or value<0:
                raise ValueError('path energy must be nonnegative or infinity')
            self._energy_cache[key]=value
        return self._energy_cache[key]

    def capacity_graph(self,capacity):
        capacity=float(capacity)
        if capacity not in self._capacity_cache:
            closure=self.factor*self.station_energy <= capacity+1e-8
            for middle in range(len(self.stations)):
                closure |= closure[:,middle,None] & closure[None,middle,:]
            direct_home=self.factor*self.station_depot <= capacity+1e-8
            home=(closure & direct_home[None,:]).any(axis=1)
            escape=self.customer_depot.copy()
            if home.any():
                escape=np.minimum(escape,self.customer_station[:,home].min(axis=1))
            after_station=self.factor*(self.station_customer+escape[None,:]) <= capacity+1e-8
            self._capacity_cache[capacity]=(closure,home,escape,after_station)
        return self._capacity_cache[capacity]

    def snapshot(self,truck,all_tasks):
        """Upper bound energy after the publicly announced in-flight leg.

        Battery is debited at arrival in this simulator. Subtract only the
        public nominal leg's minimum possible energy, not its sampled discharge.
        At stations permit a future full recharge as an optimistic relaxation.
        """
        origin=int(truck.current_node)
        battery=float(truck.current_battery)
        destination=getattr(truck,'route_destination',None)
        if destination is not None:
            destination=int(destination)
            battery-=self.factor*self.energy(origin,destination)
            origin=destination
        capacity=float(truck.battery_capacity)
        if origin in self.station_set:
            battery=capacity
        committed=sum(task.demand for task in all_tasks if not task.is_available and not task.is_served and task.claimed_by==truck.truck_id)
        payload=math.inf if truck.remaining_payload is None else max(0.0,float(truck.remaining_payload)-committed)
        return (truck.truck_id,origin,battery,capacity,payload)

    def reachability(self,state):
        if state in self._reach_cache:
            return self._reach_cache[state]
        _,origin,battery,capacity,_=state
        closure,station_home,escape,after_station=self.capacity_graph(capacity)
        if origin in self.station_set:
            battery=capacity
        station_legs=np.asarray([self.energy(origin,node) for node in self.stations])
        seeds=self.factor*station_legs <= battery+1e-8
        reachable=(seeds[:,None] & closure).any(axis=0)
        home=(self.factor*self.energy(origin,self.depot) <= battery+1e-8
              or bool((reachable & station_home).any()))
        customer_legs=np.asarray([self.energy(origin,node) for node in self.nodes])
        customers=self.factor*(customer_legs+escape) <= battery+1e-8
        if reachable.any():
            customers |= after_station[reachable].any(axis=0)
        self._reach_cache[state]=(bool(home),customers)
        return self._reach_cache[state]

    def candidate_possible(self,decision):
        """Necessary home reachability and energy/payload assignment cuts."""
        env=self.env
        active=env.trucks[env.active_truck_id]
        all_tasks=list(env.task_registry.tasks())
        pending=[task for task in all_tasks if task.is_available]
        states=[]
        for truck in env.trucks:
            if (truck.truck_id==active.truck_id or truck.failed or truck.is_complete
                or getattr(truck,'return_to_depot_pending',False)
                or getattr(truck,'route_destination',None)==self.depot):
                continue
            states.append(self.snapshot(truck,all_tasks))
        kind=decision.action_kind
        target=decision.target_node
        retires=(kind is ActionKind.DEPOT or (target is not None and int(target)==self.depot and kind in (ActionKind.CUSTOMER,ActionKind.CHARGER)))
        if not retires:
            payload=math.inf if active.remaining_payload is None else float(active.remaining_payload)
            origin=int(active.current_node)
            battery=float(active.current_battery)
            capacity=float(active.battery_capacity)
            if kind in (ActionKind.CUSTOMER,ActionKind.CHARGER):
                target=int(target)
                battery-=self.factor*self.energy(origin,target)
                origin=target
            if kind is ActionKind.CUSTOMER:
                selected=next(task for task in pending if task.node_id==target)
                payload-=selected.demand
                pending=[task for task in pending if task.node_id!=target]
            if origin in self.station_set or kind is ActionKind.CHARGE:
                battery=capacity
            states.append((active.truck_id,origin,battery,capacity,payload))
        reach=[]
        for state in states:
            home,eligible=self.reachability(state)
            if not home:
                return False
            reach.append(eligible)
        if not pending:
            return True
        if not states:
            return False
        masks=[]
        for task in pending:
            bits=0
            for index,(state,eligible) in enumerate(zip(states,reach)):
                owner=task.preassigned_to
                if (task.demand<=state[4]+1e-8 and (owner is None or owner==state[0]) and eligible[self.indices[task.node_id]]):
                    bits |= 1<<index
            if bits==0:
                return False
            masks.append(bits)
        # Necessary capacity cuts for each observed eligible-truck subset and
        # their union. This relaxation permits split demand and is not an
        # exact integral packing or a route-feasibility certificate.
        union=0
        for bits in masks:union |= bits
        for subset in set(masks)|{union}:
            demand=sum(task.demand for task,bits in zip(pending,masks) if bits & ~subset==0)
            capacity=sum(state[4] for index,state in enumerate(states) if subset & (1<<index))
            if demand>capacity+1e-8:
                return False
        return True

    def candidate_mask(self,decisions,legal_mask):
        """Vectorized equivalent of candidate_possible for execution-legal slots."""
        allowed=np.flatnonzero(legal_mask)
        result=np.zeros_like(legal_mask,dtype=bool)
        if not len(allowed):return result
        env=self.env
        active=env.trucks[env.active_truck_id]
        all_tasks=list(env.task_registry.tasks())
        other_states=[];other_reach=[]
        for truck in env.trucks:
            if (truck.truck_id==active.truck_id or truck.failed or truck.is_complete
                or getattr(truck,'return_to_depot_pending',False)
                or getattr(truck,'route_destination',None)==self.depot):continue
            state=self.snapshot(truck,all_tasks)
            home,eligible=self.reachability(state)
            if not home:return result
            other_states.append(state);other_reach.append(eligible)
        if len(other_states)>=63:
            for index in allowed:result[index]=self.candidate_possible(decisions[index])
            return result
        count=len(allowed);customers=len(self.tasks)
        pending=np.asarray([task.is_available for task in self.tasks],dtype=bool)
        demands=np.asarray([task.demand for task in self.tasks],dtype=float)
        owners=np.asarray([-1 if task.preassigned_to is None else task.preassigned_to for task in self.tasks])
        pending_after=np.broadcast_to(pending,(count,customers)).copy()
        origins=np.full(count,int(active.current_node),dtype=int)
        batteries=np.full(count,float(active.current_battery))
        payloads=np.full(count,math.inf if active.remaining_payload is None else float(active.remaining_payload))
        retires=np.zeros(count,dtype=bool)
        capacity=float(active.battery_capacity)
        for row,index in enumerate(allowed):
            decision=decisions[index];kind=decision.action_kind;target=decision.target_node
            retires[row]=(kind is ActionKind.DEPOT or (target is not None and int(target)==self.depot and kind in (ActionKind.CUSTOMER,ActionKind.CHARGER)))
            if retires[row]:continue
            if kind in (ActionKind.CUSTOMER,ActionKind.CHARGER):
                origins[row]=int(target)
                batteries[row]-=self.factor*self.energy(active.current_node,target)
            if kind is ActionKind.CUSTOMER:
                column=self.indices[int(target)]
                payloads[row]-=demands[column]
                pending_after[row,column]=False
            if origins[row] in self.station_set or kind is ActionKind.CHARGE:batteries[row]=capacity
        closure,station_home,escape,after_station=self.capacity_graph(capacity)
        # Cache public energy rows across decisions within this scenario.
        if not hasattr(self,'_origin_vectors'):self._origin_vectors={}
        unique,inverse=np.unique(origins,return_inverse=True)
        for origin in unique:
            if int(origin) not in self._origin_vectors:
                self._origin_vectors[int(origin)]=(np.asarray([self.energy(origin,node) for node in self.stations]),
                    np.asarray([self.energy(origin,node) for node in self.nodes]),self.energy(origin,self.depot))
        station_legs=np.stack([self._origin_vectors[int(node)][0] for node in unique])[inverse]
        customer_legs=np.stack([self._origin_vectors[int(node)][1] for node in unique])[inverse]
        depot_legs=np.asarray([self._origin_vectors[int(node)][2] for node in unique])[inverse]
        seeds=self.factor*station_legs<=batteries[:,None]+1e-8
        reachable=(seeds.astype(np.int32) @ closure.astype(np.int32))>0
        own_home=(self.factor*depot_legs<=batteries+1e-8) | (reachable & station_home[None,:]).any(axis=1)
        eligible=(self.factor*(customer_legs+escape[None,:])<=batteries[:,None]+1e-8) | ((reachable.astype(np.int32) @ after_station.astype(np.int32))>0)
        eligible &= demands[None,:]<=payloads[:,None]+1e-8
        eligible &= ((owners<0)|(owners==active.truck_id))[None,:]
        eligible[retires]=False
        base_bits=np.zeros(customers,dtype=np.int64)
        for index,(state,reach) in enumerate(zip(other_states,other_reach)):
            can=reach & (demands<=state[4]+1e-8) & ((owners<0)|(owners==state[0]))
            base_bits[can] |= 1<<index
        active_bit=1<<len(other_states)
        bits=base_bits[None,:] | np.where(eligible,active_bit,0)
        possible=(retires | own_home) & ((bits!=0)|~pending_after).all(axis=1)
        unions=np.bitwise_or.reduce(np.where(pending_after,bits,0),axis=1,initial=0)
        subsets=set(map(int,bits[pending_after])) | set(map(int,unions))
        for subset in subsets:
            applicable=((bits==subset)&pending_after).any(axis=1) | (unions==subset)
            forced=pending_after & ((bits & ~subset)==0)
            required=(forced*demands[None,:]).sum(axis=1)
            other_capacity=sum(state[4] for index,state in enumerate(other_states) if subset & (1<<index))
            available=other_capacity+np.where(bool(subset & active_bit),payloads,0.0)
            possible &= ~(applicable & (required>available+1e-8))
        result[allowed]=possible
        return result
