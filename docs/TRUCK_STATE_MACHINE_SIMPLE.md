# Truck State Machine - Simplified

## State Diagram

```mermaid
graph TD
    Start([Start]) --> Ready[Ready]
    
    Ready -->|Assign delivery| Routing[Routing]
    
    Routing -->|Low battery| WaitingToCharge[Waiting to Charge]
    Routing -->|All deliveries done| Complete[Complete]
    Routing -->|Routing impossible| Failed[Failed<br/>-1000 Penalty]
    
    WaitingToCharge -->|Charger available| Charging[Charging]
    WaitingToCharge -->|Queue full| WaitingToCharge
    
    Charging -->|Battery full| Ready
    
    Complete --> End([End])
    Failed --> End
    
    style Failed fill:#f88,stroke:#d33,stroke-width:3px
    style Complete fill:#8f8,stroke:#3d3,stroke-width:3px
    style Ready fill:#aaf,stroke:#33d,stroke-width:2px
```

## States

| State | Description |
|-------|-------------|
| **Ready** | Truck is idle, ready for assignment |
| **Routing** | Traveling to delivery or charger |
| **Waiting to Charge** | In queue for charger |
| **Charging** | Actively charging battery |
| **Complete** | All deliveries finished |
| **Failed** | Routing impossible (-1000 penalty) |

## Failure Conditions (-1000 Reward)

The truck enters the **Failed** state when routing is impossible:

1. **No Valid Path** - Target is unreachable
2. **Insufficient Battery** - Not enough charge for the trip
3. **Would Strand Truck** - Delivery would leave truck unable to reach charger or complete mission

## Key Transitions

- **Ready → Routing**: Dispatcher assigns next delivery
- **Routing → Failed**: Path validation fails
- **Routing → WaitingToCharge**: Battery below threshold, needs charging
- **Routing → Complete**: Last delivery finished
- **WaitingToCharge → Charging**: Spot opens in charging queue
- **Charging → Ready**: Battery recharged to capacity

---

## Action Feasibility Graphs

### 1. When Truck is Ready (at Charger)

```mermaid
graph TD
    Start([Truck Ready at Charger]) --> Decision{Action Type?}
    
    Decision -->|Navigate to Delivery| NavCheck[Navigation Feasibility Check]
    Decision -->|Charge Battery| ChargeCheck[Charging Action]
    
    NavCheck --> Check1{Path exists?}
    Check1 -->|No| Fail1[❌ FAIL: -1000<br/>No valid path]
    Check1 -->|Yes| Check2{Sufficient battery<br/>for trip?}
    
    Check2 -->|No| Fail2[❌ FAIL: -1000<br/>Insufficient battery]
    Check2 -->|Yes| Check3{Will have feasible<br/>action after arrival?}
    
    Check3 -->|No| Fail3[❌ FAIL: -1000<br/>Would strand truck]
    Check3 -->|Yes| Success1[✅ SUCCESS<br/>Navigate to delivery]
    
    ChargeCheck --> ChargeValid{At charger<br/>location?}
    ChargeValid -->|No| Fail4[❌ Navigate to delivery instead]
    ChargeValid -->|Yes| Success2[✅ SUCCESS<br/>Charge for duration]
    
    style Fail1 fill:#f88,stroke:#d33,stroke-width:2px
    style Fail2 fill:#f88,stroke:#d33,stroke-width:2px
    style Fail3 fill:#f88,stroke:#d33,stroke-width:2px
    style Fail4 fill:#fa8,stroke:#d63,stroke-width:2px
    style Success1 fill:#8f8,stroke:#3d3,stroke-width:2px
    style Success2 fill:#8f8,stroke:#3d3,stroke-width:2px
```

**Feasibility Checks for Navigation (3-step algorithm):**

1. **Can reach any charger** from target delivery?
   - If YES → Action is feasible
   
2. **Can complete all remaining deliveries** from target without charging?
   - If YES → Action is feasible
   
3. **Can reach next delivery then a charger** from target?
   - If YES → Action is feasible
   - If NO to all 3 → **FAIL: -1000 penalty**

### 2. When Truck is Ready (not at Charger)

```mermaid
graph TD
    Start([Truck Ready<br/>Not at Charger]) --> Decision{Action Type?}
    
    Decision -->|Navigate to Delivery| NavCheck[Navigation Feasibility Check]
    Decision -->|Navigate to Charger| ChargerCheck[Charger Navigation]
    
    NavCheck --> Check1{Path exists?}
    Check1 -->|No| Fail1[❌ FAIL: -1000<br/>No valid path]
    Check1 -->|Yes| Check2{Sufficient battery<br/>for trip?}
    
    Check2 -->|No| Fail2[❌ FAIL: -1000<br/>Insufficient battery]
    Check2 -->|Yes| Check3{Will have feasible<br/>action after arrival?}
    
    Check3 -->|No| Fail3[❌ FAIL: -1000<br/>Would strand truck]
    Check3 -->|Yes| Success1[✅ SUCCESS<br/>Navigate to delivery]
    
    ChargerCheck --> ChargerPath{Path exists?}
    ChargerPath -->|No| Fail4[❌ FAIL: -1000<br/>No valid path]
    ChargerPath -->|Yes| ChargerBattery{Sufficient battery?}
    
    ChargerBattery -->|No| Fail5[❌ FAIL: -1000<br/>Insufficient battery]
    ChargerBattery -->|Yes| Success2[✅ SUCCESS<br/>Navigate to charger]
    
    style Fail1 fill:#f88,stroke:#d33,stroke-width:2px
    style Fail2 fill:#f88,stroke:#d33,stroke-width:2px
    style Fail3 fill:#f88,stroke:#d33,stroke-width:2px
    style Fail4 fill:#f88,stroke:#d33,stroke-width:2px
    style Fail5 fill:#f88,stroke:#d33,stroke-width:2px
    style Success1 fill:#8f8,stroke:#3d3,stroke-width:2px
    style Success2 fill:#8f8,stroke:#3d3,stroke-width:2px
```

**Key Differences:**
- When not at a charger, the truck can navigate to either:
  - Next delivery (with full feasibility check)
  - A charging station (basic path + battery check)
- Charging action is not available when not at a charger location

---

## Detailed Failure Conditions

| Check | Condition | Penalty |
|-------|-----------|---------|
| **Path Validity** | `energy == infinity` | -1000 |
| **Battery Sufficiency** | `discharge > current_battery` | -1000 |
| **Post-Delivery Feasibility** | Cannot reach charger, complete remaining deliveries, OR reach next delivery + charger | -1000 |
| **Battery Depletion** | `battery <= 0` during travel (safety catch) | -1000 |
