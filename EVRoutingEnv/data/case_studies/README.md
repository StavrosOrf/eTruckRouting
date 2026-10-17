# Real-network case studies (asymmetric energy and time)

Each folder is a drop-in replacement for the California data in `EVRoutingEnv/data/`: same file
names, same key format, same units (kWh, hours). Point `network.data_path` at the folder and the
environment, baselines and plotting work unchanged.

| Case | Folder | Config | Terrain | Nodes (chargers / depots / delivery stops) | DC-fast ports | Battery |
|---|---|---|---|---|---|---|
| Puget Sound (Seattle, WA) | `seattle/` | `config_files/config_joint_seattle.yaml` | hilly, water-constrained, 0-590 m | 396 (163 / 3 / 230) | 952 | 400 kWh |
| Chicagoland (IL) | `chicago/` | `config_files/config_joint_chicago.yaml` | flat (165-306 m), dense grid | 517 (283 / 4 / 230) | 1644 | 400 kWh |
| California (existing) | `../` | `config_files/config_joint.yaml` | state-wide | 258 (25 stations) | 209 | 400 kWh |

The generated configs are `config_joint.yaml` with only `network.data_path` changed, so fleet size,
payload, charging curves (150/350/750 kW classes), traffic noise and rewards match the California runs.

Run one: `python scripts/casestudies/smoke_episode.py EVRoutingEnv/config_files/config_joint_seattle.yaml --plot`

## What is real and what is modelled

| Layer | Source | Used for |
|---|---|---|
| Delivery stops and depots | Amazon Last Mile Routing Research Challenge 2021 (real driver routes, CC BY-NC 4.0) | 230 randomly drawn real stops per case, plus the real depot ("station") locations |
| Road network | OpenStreetMap via Geofabrik (ODbL). Drivable ways, one-ways and speed limits respected; service roads excluded | directed graph, largest strongly connected component |
| Slope | AWS terrain tiles (USGS 3DEP/NED at about 13 m per pixel over the US) | node elevation, smoothed along the road graph (about 60 m scale); bridge and tunnel nodes are interpolated between their land ends |
| Chargers | NREL/NLR Alternative Fuels Data Center, public open DC-fast stations only | one node per merged site, `total_capacity` = number of DC-fast ports |
| Energy and time | modelled (see below) | all-pairs minimum-energy routes between the nodes |

Delivery demand, service times and time windows are **not** taken from Amazon; they come from the
environment config exactly as in the California runs.

### Charger nodes
Only DC-fast ports are counted, and **every** public open DC-fast site in the study area is kept
(`max_chargers: 0`). Stations closer than 100 m (single linkage) are one node and their ports add up
(`vis_data/stations.json` lists the member stations, networks, connectors and cluster diameter; chains
can slightly exceed 100 m). Sites that snap to the same road node are also combined. The AFDC lists
Tesla-only connectors too; set `exclude_tesla_only: true` in the preset to drop them.

Merge radius versus node count (ports are always preserved, only the node count changes):

| radius | Seattle (213 stations) | Chicago (315 stations) |
|---|---|---|
| none | 213 | 315 |
| 100 m (default) | 167 | 283 |
| 500 m | 134 | 245 |
| 1 km | 100 | 210 |

Larger radii lump more ports into one node (Chicago: up to 114 ports at 500 m-1 km) and merge sites up to
about 2 km apart, so the node is no longer a single place. Change `charger_merge_m` in the preset to try
them. Set `max_chargers: N` to keep only N sites chosen for coverage.

### Energy and time model
Per directed road segment, from the slope of the smoothed elevation profile:

- time = length / (posted speed x class factor) + signal (20 s) / stop-sign (8 s) delay at the head node;
- battery energy = traction (gravity + rolling + drag) divided by drivetrain efficiency, or multiplied by
  the regeneration efficiency on net-downhill segments, + auxiliary load x time + kinetic losses at
  signals and stop signs;
- vehicle profile `heavy_truck` (default): 36 t class-8 battery-electric truck, Crr 0.007, CdA 6.5 m2,
  drivetrain 0.90, regeneration 0.65, 6 kW auxiliary. These were **calibrated to the California data**
  (median 1.57 kWh/km, 10th-90th percentile 1.13-1.97, taken from its lane table) rather than read from a
  publication: the case studies give 1.47 (Seattle) and 1.25 (Chicago) kWh/km median over routes. A
  `van` profile (4.2 t, about 0.3 kWh/km) is also provided. Parameters are recorded in `metadata.json`
  and overridable in the preset (`vehicle_profile:`, `vehicle:`).

Uphill and downhill, and one-way versus two-way streets, make both costs depend on direction.
Routes minimise battery energy (regeneration allowed on the road graph; exact, via an elevation
potential so Dijkstra applies); the time stored for a pair is the time **on that same route**, as in
the California data. `metadata.json` records the check that energy re-accumulated along each route
equals the Dijkstra value (error below 5e-6 kWh) and that no reduced edge cost was negative.

The environment cannot use negative leg energy, so a pair whose net energy is below 0.02 kWh is
floored to 0.02 kWh (411 of 156k legs in Seattle, 4 of 267k in Chicago). The unfloored values are in
`vis_data/od_matrices.npz` (`energy_net_kwh`).

### Measured asymmetry (node pairs where both directions exceed 1 kWh / 3 min)

| | energy: median diff | energy: pairs > 5 % | energy: pairs > 20 % | time: pairs > 5 % |
|---|---|---|---|---|
| California | 6.4 % | 57 % | - | 21 % |
| Seattle | 15.3 % | 82 % | 39 % | 55 % |
| Chicago | 9.8 % | 76 % | 10 % | 21 % |

Seattle's energy asymmetry follows net climb between the two ends (`figures/asymmetry.png`). Its time
asymmetry is high because a 36 t truck's minimum-energy route differs by direction (it avoids climbs), and
time is read off that route; a lighter vehicle gives about 22 %. Chicago is flat (99th-percentile segment
grade 3.9 %), so it is close to California. `comparison.png` overlays all three.

### Energy versus battery
With the 400 kWh battery the median leg is 56 kWh (Seattle) and 52 kWh (Chicago), 95th percentile 132 and
103 kWh, maximum 204 and 168 kWh. In test episodes with the heuristic policy (10 stops, 2 trucks) no
charging was needed, because tours average about 6 legs; California's median leg is 326 kWh. If you want charging to matter as
much as in California, reduce `battery_kwh` in the preset (about 150-200 kWh), add stops, or scale
leg energy with the existing `energy_scale` of `TransportationGraph.scale_network`.

## Folder contents

```
<case>/
  shortest_path_energy_dict.json   "(u, v)" -> kWh, all ordered pairs of the nodes   (get_graph reads this)
  shortest_path_time_dict.json     "(u, v)" -> hours on the same route               (get_graph reads this)
  station_info_dict.json           node -> {"station_type": "DCFast", "total_capacity": ports}
  waiting_time_lookup.json         charger queue model (copy of the California file; used automatically)
  metadata.json                    provenance, parameters, counts, distributions, asymmetry statistics
  vis_data/nodes.json              node -> role (customer/depot/charger), lat, lon, elevation, snap distance
  vis_data/stations.json           per charger node: ports, member stations, networks, connectors
  vis_data/all_dcfast_sites.json   every merged DC-fast site and whether it was selected
  vis_data/roads.npz, terrain.npz  road segments (with grade) and a 250 m elevation grid, for plotting
  vis_data/od_matrices.npz         unfloored net energy, floored energy, time, distance for the nodes
  figures/overview.png             map: stops, depots, chargers (marker size = ports) on hillshade
  figures/terrain.png              road-grade map, grade distribution, node elevations
  figures/asymmetry.png            forward vs reverse energy and time, asymmetry vs net climb
```

Node ids are OpenStreetMap node ids (like California). Depots are ordinary nodes here too: the
environment still draws its depot at random from the non-charger nodes; the real depot node ids are
the `role == "depot"` rows of `vis_data/nodes.json`.

## Rebuild or add a case study

```bash
# 1. a preset (copy seattle.yaml): state, Geofabrik region, Amazon depot codes of that metro, sizes,
#    charger_merge_m, battery_kwh / vehicle_profile if you want something other than 400 kWh / heavy_truck
#    Amazon metros: Seattle DSE*, Chicago DCH*, Los Angeles DLA*, Boston DBO*, Austin DAU1
# 2. build (needs `pip install osmium pillow pyyaml scipy` on top of the project deps; downloads the open data into .casestudy_raw/ on first use: about 0.5 GB per metro, a few minutes)
python scripts/casestudies/build_case_study.py scripts/casestudies/presets/<name>.yaml
# 3. figures, comparison, episode check
python scripts/casestudies/visualize_case_study.py EVRoutingEnv/data/case_studies/<name>
python scripts/casestudies/compare_case_studies.py
python scripts/casestudies/smoke_episode.py EVRoutingEnv/config_files/config_joint_<name>.yaml --plot
pytest tests/unit/test_case_studies.py
```

Set `NREL_API_KEY` for the AFDC download (the shared `DEMO_KEY` is rate limited) and
`EVRP_CASESTUDY_RAW` to keep the raw cache elsewhere. The build is deterministic for a given
`seed` and raw cache; the OSM and AFDC downloads are "latest" snapshots (these case studies:
October 2026), so a fresh download can differ slightly.

## Limits to keep in mind

- Loading a case into the environment builds all-pairs routes once (about 1-3 minutes for these
  sizes, cached in the system temp folder afterwards); the energy and time JSON files are 5-10 MB each.
- Time ignores turn penalties, time-of-day congestion and real speed data (the environment adds
  stochastic traffic on top); the speed factors are plausible defaults, not measurements.
- The DEM is bare earth at about 13 m; very short steep ramps and overpasses without a `bridge` tag can be
  mis-graded, and smoothing flattens real grades shorter than about 60 m.
- Amazon stops come from 2018 routes; coordinates are as published (median 19-20 m from the nearest
  road node, 95th percentile 77-79 m).
- The Amazon data is CC BY-NC 4.0, so these derived folders are for non-commercial research;
  OSM data requires attribution (ODbL).
