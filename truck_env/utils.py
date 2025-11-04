from gymnasium.spaces import Discrete, Box, Dict, MultiDiscrete
import os
import pickle
import numpy as np
import networkx as nx


BATTERY_CAPACITY = 300.0
LEVEL2_CHARGING_RATE = 20.0
LEVEL_CHARGING_RATE = 50.0
DCFAST_CHARGING_RATE = 200.0
MAX_CHARGERS = 100


GRAPH_MAPPINGS = {
    "node_to_index": None,
    "index_to_node": None
}

path = os.getcwd()


def read_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


edge_distance_file = f"{path}/data/shortest_path_energy_dict.pkl"
edge_time_file = f"{path}/data/shortest_path_time_dict.pkl"
chargers_file = f"{path}/data/station_info_dict.pkl"

isNew = True

checkpoint_dir = f"{path}/checkpoints"


def get_graph():
    if isNew:
        return get_graph_new()
    else:
        return get_graph_old()


def get_truck_configs():
    if isNew:
        return get_truck_configs_new()
    else:
        return get_truck_configs_old()


def fast_charger():
    return DCFAST_CHARGING_RATE


def slow_charger():
    return LEVEL_CHARGING_RATE


charger_function_map = {
    "fast": fast_charger,
    "slow": slow_charger
}


def get_node_to_index():
    if GRAPH_MAPPINGS["node_to_index"] is None:  # Proper None check
        raise RuntimeError("node to index mapping is None")  # Proper exception
    return GRAPH_MAPPINGS["node_to_index"]


def get_index_to_node():
    if GRAPH_MAPPINGS["index_to_node"] is None:  # Proper None check
        raise RuntimeError("index to node mapping is None")  # Proper exception
    return GRAPH_MAPPINGS["index_to_node"]


def get_route_sequence():
    return [
        [5026447875, 65657291, 5433392625],
        [54382864, 90796641],
        [9512913929, 49291774, 90810515],
        [9512913929, 90539746, 90403004],
        [9512913929, 353478871, 9509748626]]


def charge_standard(current_charge, time=1):
    return 0.8 * time


def discharge_standard(current_charge, distance=1):
    return 0.2 * distance


def get_truck_types():
    """Define truck type specifications."""
    return {
        "standard": {
            "battery_capacity": BATTERY_CAPACITY,
            "base_speed": 50.0,  # km/h
            "base_discharge_function": discharge_standard,  # battery per km
            "base_charge_function": charge_standard
        },
        "heavy": {
            "battery_capacity": BATTERY_CAPACITY,
            "base_speed": 40.0,
            "base_discharge_function": discharge_standard,  # battery per km
            "base_charge_function": charge_standard
        }
    }

# def get_charger_configs():
#    """Define charger specifications at each node."""
#    return {
#        1: {"type": "slow", "capacity": 1, "base_charge_rate": 5.0},   # 5 battery/hour
#        3: {"type": "fast", "capacity": 2, "base_charge_rate": 15.0},  # 15 battery/hour
#        5: {"type": "fast", "capacity": 1, "base_charge_rate": 15.0},
#    }


def get_charger_configs(graph):
    """Extract charger capacities from graph nodes."""
    return {
        node: data["charger_type"]
        for node, data in graph.nodes(data=True)
        if data["has_charger"]
    }


def get_charger_occupancy_template(graph):
    """Initialize empty occupancy tracking structure."""
    return {
        node: {ctype: 0 for ctype in data["charger_type"]}
        for node, data in graph.nodes(data=True)
        if data["has_charger"]
    }
# =============================================================================
# CUSTOM DISCHARGE AND CHARGE FUNCTIONS
# =============================================================================


def discharge_function(truck_config: dict, edge_data: dict, travel_time: float, current_time: float) -> float:
    """Custom non-linear discharge function."""
    truck_type = get_truck_types()[truck_config["truck_type"]]

    # Base discharge based on distance and truck type
    # edge_data["distance"] * truck_type["base_discharge_rate"]
    base_discharge = truck_type["base_discharge_function"](
        truck_config["current_battery"], edge_data["distance"])

    # Terrain factor affects discharge
    terrain_modifier = edge_data.get("terrain_factor", 1.0)

    # Time-based modifier (traffic/weather simulation)
    # + 0.1 * np.sin(current_time / 10.0)  # Varies with time
    time_modifier = 1.0

    # Non-linear battery efficiency (discharge increases as battery gets low)
    # if truck_config["current_battery"] > 15 else 1.3
    battery_efficiency = 1.0

    total_discharge = base_discharge * terrain_modifier * \
        time_modifier * battery_efficiency

    return max(0, total_discharge)


def charge_function(graph, truck_config: dict, charger_node: int, charge_time: float, current_time: float, charger_type) -> float:
    """Custom non-linear charge function."""
    charger_configs = get_charger_configs(graph)
    truck_types = get_truck_types()

    if charger_node not in charger_configs:
        return 0.0

    charger = charger_configs[charger_node]
    truck_type = truck_types[truck_config["truck_type"]]

    current_battery = truck_config["current_battery"]

    # Base charge rate
    print(f"truck_type is {truck_type}")
    base_charge = truck_type["base_charge_function"](
        current_battery, charge_time)  # * charger_function_map[charger_type]

    # Truck charge efficiency
    efficiency = 1  # truck_type["charge_efficiency"]

    # Non-linear charging (slower as battery gets fuller)
    battery_capacity = truck_type["battery_capacity"]
    battery_ratio = current_battery / battery_capacity

    if battery_ratio < 0.5:
        charge_efficiency = 1.0
    elif battery_ratio < 0.8:
        charge_efficiency = 0.7
    else:
        charge_efficiency = 0.4
    charge_efficiency = 1
    # Time-based modifier (grid load simulation)
    time_modifier = 1.0  # - 0.2 * np.sin(current_time / 8.0)

    total_charge = base_charge * efficiency * charge_efficiency * time_modifier

    # Ensure we don't exceed battery capacity
    max_possible_charge = battery_capacity - current_battery

    return min(total_charge, max_possible_charge)

# =============================================================================
# ACTION AND OBSERVATION SPACES
# =============================================================================


def get_high_level_action_space(graph):
    """High-level agent chooses next node to route to."""
    return Discrete(graph.number_of_nodes())


def get_low_level_action_space():
    """Low-level agent manages charging decisions."""
    # 0: do nothing, 1: start charging, 2: stop charging, 3: wait for charger
    return Discrete(4)


def map_charger_type(station_type):
    """Map charger types to internal names."""
    if station_type == 'Level2':
        return 'slow'
    elif station_type == 'DCFC' or station_type == 'DCFast':
        return 'fast'
    return station_type  # Fallback for unknown types


def get_graph_new():
    edge_distance = read_file(edge_distance_file)
    edge_time = read_file(edge_time_file)
    chargers = read_file(chargers_file)

    """Build road network graph using index-based nodes (0,1,2,...)"""
    G = nx.DiGraph()
    all_nodes = set()

    # Collect all nodes from edges and chargers
    for (u, v) in edge_distance.keys():
        all_nodes.add(u)
        all_nodes.add(v)
    all_nodes.update(chargers.keys())

    # Create sorted list of nodes and mappings
    node_list = sorted(all_nodes)  # Sort for consistent ordering
    node_to_index = {node: idx for idx, node in enumerate(node_list)}
    index_to_node = {idx: node for node, idx in node_to_index.items()}

    # Process chargers - handle multiple types per node
    charger_aggregated = {}
    for node, info in chargers.items():
        # Handle single charger type per node
        if 'station_type' in info:
            mapped_type = map_charger_type(info['station_type'])
            count = int(info['total_capacity'])
            idx = node_to_index[node]
            charger_aggregated.setdefault(idx, {})[mapped_type] = count

        # Handle multiple charger types per node
        elif 'chargers' in info:
            for charger in info['chargers']:
                mapped_type = map_charger_type(charger['station_type'])
                count = int(charger['total_capacity'])
                idx = node_to_index[node]
                charger_aggregated.setdefault(idx, {})[mapped_type] = charger_aggregated.get(
                    idx, {}).get(mapped_type, 0) + count

    # Add nodes with properties using INDEXES
    for idx in range(len(node_list)):
        if idx in charger_aggregated:
            props = {
                "has_charger": True,
                "charger_type": charger_aggregated[idx],
                # Store original ID for reference
                "original_id": index_to_node[idx]
            }
        else:
            props = {
                "has_charger": False,
                "charger_type": None,
                # Store original ID for reference
                "original_id": index_to_node[idx]
            }
        G.add_node(idx, **props)

    # Add edges with attributes using INDEXES
    for (u_orig, v_orig) in edge_distance.keys():
        u_idx = node_to_index[u_orig]
        v_idx = node_to_index[v_orig]
        distance = edge_distance[(u_orig, v_orig)]
        time_val = edge_time.get((u_orig, v_orig), 0)
        G.add_edge(u_idx, v_idx, distance=distance,
                   time=time_val, terrain_factor=1.0)
    GRAPH_MAPPINGS["node_to_index"] = node_to_index
    GRAPH_MAPPINGS["index_to_node"] = index_to_node
    return G


def get_graph_old():
    """Define the road network graph with charging stations."""
    G = nx.DiGraph()

    # Add nodes with properties
    G.add_nodes_from([
        (0, {"has_charger": False, "charger_type": None}),      # Start node
        (1, {"has_charger": True, "charger_type": {
         "slow": 2, "fast": 3}}),     # Charging station
        (2, {"has_charger": False, "charger_type": None}),      # Regular node
        # Charging station
        (3, {"has_charger": True, "charger_type": {"fast": 5}}),
        (4, {"has_charger": False, "charger_type": None}),      # End node
        # Charging station
        (5, {"has_charger": True, "charger_type": {"slow": 1}}),
    ])

    # Add edges with distance and terrain difficulty
    edges = [
        (0, 1, {"distance": 10, "terrain_factor": 1.0}),
        (1, 2, {"distance": 15, "terrain_factor": 1.2}),
        (2, 3, {"distance": 8, "terrain_factor": 0.8}),
        (3, 4, {"distance": 12, "terrain_factor": 1.1}),
        (1, 3, {"distance": 20, "terrain_factor": 1.3}),
        (0, 2, {"distance": 18, "terrain_factor": 1.0}),
        (2, 5, {"distance": 6, "terrain_factor": 0.9}),
        (5, 4, {"distance": 9, "terrain_factor": 1.0}),
    ]
    G.add_edges_from([(u, v, attr) for u, v, attr in edges])

    return G


def get_truck_configs_new():
    """Define individual truck configurations based on route_sequence."""
    route_sequence = get_route_sequence()
    node_to_index = get_node_to_index()
    return [
        {
            "id": idx,
            "start_node": node_to_index[route[0]],   # Convert to index
            "end_node": node_to_index[route[-1]],     # Convert to index
            "initial_battery": BATTERY_CAPACITY,
            "truck_type": "standard"
        }
        for idx, route in enumerate(route_sequence)
    ]


def get_truck_configs_old():
    """Define individual truck configurations with start/end points."""
    return [
        {
            "id": 0,
            "start_node": 0,
            "end_node": 4,
            "initial_battery": 25.0,
            "truck_type": "standard"
        },
        {
            "id": 1,
            "start_node": 0,
            "end_node": 5,
            "initial_battery": 20.0,
            "truck_type": "heavy"
        }
    ]


def get_charging_nodes(graph):
    """Get only nodes that have charging stations."""
    return [node for node, data in graph.nodes(data=True) if data["has_charger"]]


def get_transit_nodes(graph):
    """Get all nodes (for high-level routing)."""
    return list(graph.nodes())


def get_observation_space(graph):  # , num_trucks):
    """Observation space for both agent levels."""
    num_nodes = graph.number_of_nodes()

    return Dict({
        # "truck_id": Discrete(num_trucks), random trucks
        "id": Box(0.0, 1.0, shape=(), dtype=np.float32),  # for debug
        "current_node": Discrete(num_nodes),
        "destination_node": Discrete(num_nodes),
        "battery_level": Box(0.0, BATTERY_CAPACITY, shape=(), dtype=np.float32),
        "battery_capacity": Box(0.0, BATTERY_CAPACITY, shape=(), dtype=np.float32),
        "is_charging": Discrete(2),
        "charger_available": Discrete(2),
        # "charger_occupancy": Box(0, 5, shape=(), dtype=np.float32),
        # "charger_capacity": Box(0, 5, shape=(), dtype=np.float32),
        "charger_occupancy_fast": Box(0.0, MAX_CHARGERS, (), np.float32),
        "charger_occupancy_slow": Box(0.0, MAX_CHARGERS, (), np.float32),
        # "charger_capacity_fast": Box(0.0, MAX_CHARGERS, (), np.float32),
        # "charger_capacity_slow": Box(0.0, MAX_CHARGERS, (), np.float32),
        "time_elapsed": Box(0.0, 1000.0, shape=(), dtype=np.float32),
        "waiting_time": Box(0.0, 800.0, shape=(), dtype=np.float32),
        "can_reach_destination": Discrete(2),
        "nearest_charger_distance": Box(0.0, 900.0, shape=(), dtype=np.float32),
    })
