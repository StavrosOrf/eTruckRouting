# Depot Removal Fix - Completed Trucks

## Issue
Depots were remaining in the GNN state even after no trucks were pointing at them (i.e., after trucks had departed).

## Root Cause
The original logic only checked if a truck was active (not complete, not failed) but didn't verify if the truck was actually at the depot. This meant that depots would remain as long as there was an active truck anywhere in the system, even if that truck had already departed.

## Solution
Changed the depot creation logic to only create a depot if:
1. The truck is **not complete** (still has work to do)
2. The truck is **not failed** (still operational)
3. The truck is **currently at the depot** (`truck.current_node == truck.delivery_sequence[0]`)

All three conditions must be true for the depot to appear in the graph.

## Code Changes

**File:** `/home/sorfanouda/EVPR/truck_env/models/gnn_state_space.py` (Lines 107-137)

**Key Change:**
```python
# OLD: Created depot if truck was active anywhere
if truck_is_active and truck_at_depot and starting_node not in unique_starting_nodes:

# NEW: Only create depot if truck is physically at it
if truck.current_node == starting_node:
```

## Test Results

| State | T0 Position | T1 Position | Depots | Result |
|-------|------------|------------|--------|--------|
| Initial | at 194 (depot) | at 217 (depot) | [194, 217] | ✓ Both present |
| T1 departs | at 194 (depot) | at 195 (en route) | [194] | ✓ D217 removed |
| Both complete | — | — | [] | ✓ All removed |
| T1 failed | at 194 (depot) | failed | [194] | ✓ D194 remains, D217 gone |

## Impact

✅ **Graph now properly excludes unused depots**
- Depots removed when truck departs
- Graph shrinks as trucks leave depots
- Cleaner, more accurate state representation

✅ **Maintains all other functionality**
- Depot-to-delivery edges still exist (added separately)
- Failed trucks still excluded
- Truck connectivity preserved

✅ **Visualization**
- Depots automatically excluded from visualization
- No phantom nodes or edges
