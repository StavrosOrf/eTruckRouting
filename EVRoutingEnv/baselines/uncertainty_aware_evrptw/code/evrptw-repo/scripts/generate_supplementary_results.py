"""
Generate supplementary results: all remaining data points from the paper
plus 4 additional figures not in the initial batch.

Data sourced exclusively from paper text (§5, §6).
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from pathlib import Path

FIG_DIR  = Path("/home/claude/evrptw-repo/results/figures")
TAB_DIR  = Path("/home/claude/evrptw-repo/results/tables")
DATA_DIR = Path("/home/claude/evrptw-repo/results/data")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

C = {"hybrid":"#e63946","rks":"#6c757d","deploy":"#2d6a4f","warn":"#e9c46a","positive":"#52b788"}
print("Generating supplementary data and figures...")

# ═══ A. SUPPLEMENTARY DATA TABLES ══════════════════════════════════════════════

df_resilience = pd.DataFrame({
    "Scenario":["Highway closure","Charging station outage","Demand surge (+28%)"],
    "Replanning_time_DRL_s":[8.4, None, 252.0],
    "Speedup_vs_baseline":["84× faster", "N/A (route rerouting)", "N/A"],
    "Strandings_DRL":[None, 0, None],
    "Strandings_historical":[None, 3, None],
    "On_time_delivery_pct":[None, None, 96.3],
    "Notes":["8.4 s replanning; 84× faster than baseline","Zero strandings vs 3 historical incidents","4.2 min reoptimization, maintained 96.3% on-time"],
})
df_resilience.to_csv(TAB_DIR/"table_resilience_scenarios.csv", index=False)

df_driver = pd.DataFrame({
    "Dimension":["Route quality","Range prediction reliability","Work-life balance"],
    "Before_mean":[3.2, 2.8, 2.4],
    "Before_sd":[0.8, 0.9, 0.9],
    "After_mean":[4.6, 4.7, 4.1],
    "After_sd":[0.3, 0.4, 0.6],
    "Change":["+1.4","+1.9","+1.7"],
    "Note":["Likert 1–5; n=8; descriptive only"]*3,
})
df_driver.to_csv(TAB_DIR/"table_driver_satisfaction.csv", index=False)

df_co2 = pd.DataFrame({
    "Scenario":["Diesel baseline","Electric + RKS","Electric + DRL-Hybrid","Electric + DRL-Hybrid (2030 grid)"],
    "Annual_CO2_tons":[128.5, 158.3, 133.9, 87.3],
    "Grid_intensity_kg_per_kWh":[None, 0.418, 0.418, 0.273],
    "vs_Diesel_pct":[0.0, +23.2, +4.2, -32.0],
    "Notes":["Diesel ICE; local operational emissions","Italy grid 0.418 kg CO2/kWh","15.4% reduction vs EV+RKS","Grid decarbonises to 0.273 kg/kWh by 2030"],
})
df_co2.to_csv(TAB_DIR/"table_co2_comparison.csv", index=False)

df_env = pd.DataFrame({
    "Metric":[
        "Annual CO2 avoided (routing optimisation)","PM2.5 reduction (delivery zones)","NOx reduction (delivery zones)",
        "Air quality benefit (avoided healthcare costs)","Noise level — EV fleet","Noise level — diesel fleet",
        "Noise reduction","Extended delivery window","Energy per package — baseline","Energy per package — Week 9–13",
        "Energy per package improvement","Off-peak charging — baseline","Off-peak charging — final",
        "Grid CO2 intensity — current (Italy)","Grid CO2 intensity — 2030 projected","CO2 at 2030 grid",
    ],
    "Value":[
        "~19 tons","0.73 tons/year","2.14 tons/year","€47,300","58 dB","73 dB","15 dB",
        "6:00–22:00 (extended)","1.89 kWh/package","1.52 kWh/package","−19.6%",
        "51% of sessions","67% of sessions","0.418 kg CO2/kWh","0.273 kg CO2/kWh","87.3 tons/year",
    ],
    "Source":["§6.3"]*16,
})
df_env.to_csv(TAB_DIR/"table_environmental_detail.csv", index=False)

df_econ = pd.DataFrame({
    "Metric":[
        "Operator annual savings","Total societal value","Air quality externality",
        "Grid optimisation value","Avoided carbon cost (€35/tonne)","Operator share of societal value",
        "5-year TCO advantage vs diesel","Payback period","Min fleet for 18-month payback",
        "Optimal fleet size (min per-vehicle cost)","Infrastructure sharing cost reduction","Dispatcher hours freed",
    ],
    "Value":[
        "€143,000","€232,000","€47,300","€9,900","€6,200","62%",
        "23.4%","4.2 months","12–15 vehicles","40–80 vehicles","31–47%","6.5 h/dispatcher/week",
    ],
    "Source":["§6.3"]*12,
})
df_econ.to_csv(TAB_DIR/"table_economic_detail.csv", index=False)

df_phases = pd.DataFrame({
    "Phase":["Infrastructure setup","System training","Pilot expansion","Full deployment"],
    "Duration_weeks":[6, 4, 4, 4],
    "Vehicles":[None, None, 8, 25],
    "Cumulative_weeks":[6, 10, 14, 18],
    "Description":[
        "Integration with existing TMS; data pipeline setup",
        "Transfer learning using 423 historical routes (4 hours)",
        "8-vehicle pilot; online policy adaptation",
        "Full 25-vehicle fleet; steady-state performance",
    ],
})
df_phases.to_csv(TAB_DIR/"table_deployment_phases.csv", index=False)

df_mech = pd.DataFrame({
    "Component":[
        "GAT Encoder","GAT Encoder","GAT Encoder",
        "Uncertainty-Aware Energy","Uncertainty-Aware Energy","Uncertainty-Aware Energy",
        "Uncertainty-Aware Energy","Uncertainty-Aware Energy","Uncertainty-Aware Energy",
        "MILP Refinement","MILP Refinement","MILP Refinement",
    ],
    "Metric":[
        "Attention entropy — high battery (>70%)",
        "Attention entropy — low battery (<30%)",
        "Route length reduction vs MLP",
        "Highway segment variance (σ²_E)",
        "Highway segment CV",
        "Urban segment variance (σ²_E)",
        "Urban segment CV",
        "Charging time reduction",
        "Stranding risk before → after",
        "Instances with customer reassignment",
        "Instances with station switch",
        "Instances with time-window shift",
    ],
    "Value":[
        "H = 2.87 nats","H = 1.43 nats","−6.2%",
        "12.4 kWh²","38%","3.8 kWh²","22%","−17.3%",
        "8.2% → 0.7%",
        "67% (avg 2.4 reassign.)","18%","42%",
    ],
    "Source":["§5.3.1"]*12,
})
df_mech.to_csv(TAB_DIR/"table_mechanistic_analysis.csv", index=False)

df_tl = pd.DataFrame({
    "Metric":[
        "Routes used (actual deployment)","Actual adaptation time","Full retraining time",
        "Time reduction","Performance vs full training","Regions validated",
        "Stage 1 duration","Stage 2 duration","Stage 2 episodes",
        "Stage 2 BC/RL weight (λ_BC / λ_RL)","Stage 3 validation routes","Stage 3 coverage target",
    ],
    "Value":[423,"4 hours","72 hours","~94%","94%",4,"2–3 hours","1–2 hours",2000,"0.3 / 0.7",100,"95%"],
    "Source":["§4.4/§6"]*12,
})
df_tl.to_csv(TAB_DIR/"table_transfer_detail.csv", index=False)

print("  ✓ 8 supplementary CSV tables")

# Update JSON
with open(DATA_DIR/"all_results.json") as f:
    existing = json.load(f)
existing.update({
    "resilience_scenarios":df_resilience.to_dict(orient="records"),
    "driver_satisfaction":df_driver.to_dict(orient="records"),
    "co2_comparison":df_co2.to_dict(orient="records"),
    "environmental_detail":df_env.to_dict(orient="records"),
    "economic_detail":df_econ.to_dict(orient="records"),
    "deployment_phases":df_phases.to_dict(orient="records"),
    "mechanistic_analysis":df_mech.to_dict(orient="records"),
    "transfer_learning_detail":df_tl.to_dict(orient="records"),
})
with open(DATA_DIR/"all_results.json","w") as f:
    json.dump(existing, f, indent=2, default=str)
print("  ✓ JSON updated (16 sections)")

# Rebuild Excel
all_dfs = {
    "Table2_Compute":pd.read_csv(TAB_DIR/"table2_computational_resources.csv"),
    "Table3_Static":pd.read_csv(TAB_DIR/"table3_static_benchmark.csv"),
    "Table4_Dynamic":pd.read_csv(TAB_DIR/"table4_dynamic_conditions.csv"),
    "Table5_Ablation":pd.read_csv(TAB_DIR/"table5_ablation.csv"),
    "Table6_MILP":pd.read_csv(TAB_DIR/"table6_milp_refinement.csv"),
    "Table7_Deployment":pd.read_csv(TAB_DIR/"table7_deployment_evolution.csv"),
    "Key_Metrics":pd.read_csv(TAB_DIR/"key_metrics.csv"),
    "Sustainability":pd.read_csv(TAB_DIR/"sustainability_impact.csv"),
    "S1_Resilience":df_resilience,
    "S2_Driver_Satisfaction":df_driver,
    "S3_CO2_Comparison":df_co2,
    "S4_Environmental":df_env,
    "S5_Economic":df_econ,
    "S6_Deployment_Phases":df_phases,
    "S7_Mechanistic":df_mech,
    "S8_Transfer_Detail":df_tl,
}
with pd.ExcelWriter(TAB_DIR/"all_results_tables.xlsx", engine="openpyxl") as xw:
    for sn, df in all_dfs.items():
        df.to_excel(xw, sheet_name=sn, index=False)
print("  ✓ Excel workbook: 16 sheets")

# ═══ B. FIG 13 — DRIVER SATISFACTION ══════════════════════════════════════════
dims = ["Route Quality","Range Prediction\nReliability","Work-Life\nBalance"]
before=[3.2,2.8,2.4]; after=[4.6,4.7,4.1]; sd_b=[0.8,0.9,0.9]; sd_a=[0.3,0.4,0.6]
x=np.arange(3); w=0.35

fig, axes = plt.subplots(1,2,figsize=(13,5.5))
fig.suptitle("Driver Satisfaction Survey — Likert Scale 1–5 (n=8, descriptive)", fontweight="bold")
ax=axes[0]
b1=ax.bar(x-w/2,before,w,label="Before",color="#adb5bd",yerr=sd_b,capsize=6,zorder=3,error_kw=dict(elinewidth=1.5))
b2=ax.bar(x+w/2,after, w,label="After", color=C["deploy"],yerr=sd_a,capsize=6,zorder=3,error_kw=dict(elinewidth=1.5))
for bar,v in zip(b1,before): ax.text(bar.get_x()+bar.get_width()/2,v-0.38,f"{v}",ha="center",fontsize=11,fontweight="bold",color="white")
for bar,v in zip(b2,after):  ax.text(bar.get_x()+bar.get_width()/2,v-0.38,f"{v}",ha="center",fontsize=11,fontweight="bold",color="white")
ax.set_xticks(x); ax.set_xticklabels(dims)
ax.set_ylabel("Likert Score (1–5)"); ax.set_ylim(0,5.8); ax.legend()
ax.set_title("Mean Scores Before vs After\n(error bars = ±1 SD)")

ax2=axes[1]
deltas=[a-b for a,b in zip(after,before)]
ax2.bar(dims,deltas,color=C["deploy"],width=0.5,zorder=3)
for i,(d,v) in enumerate(zip(dims,deltas)):
    ax2.text(i,v+0.04,f"+{v:.1f}",ha="center",fontsize=13,fontweight="bold",color=C["deploy"])
ax2.set_ylabel("Score Improvement (Likert points)"); ax2.set_ylim(0,2.6)
ax2.set_title("Improvement per Dimension")
ax2.text(0.5,0.04,"⚠ Small sample (n=8); descriptive evidence only",transform=ax2.transAxes,
         ha="center",fontsize=9,color="grey",style="italic")
fig.tight_layout()
fig.savefig(FIG_DIR/"fig_driver_satisfaction.png",dpi=150,bbox_inches="tight")
plt.close(); print("  ✓ Figure 13 — driver satisfaction")

# ═══ C. FIG 14 — CO₂ TRAJECTORY ══════════════════════════════════════════════
fig,axes=plt.subplots(1,2,figsize=(13,5.5))
fig.suptitle("Carbon Emissions: Current Grid vs 2030 Projection",fontweight="bold")
ax=axes[0]
scenarios=["Diesel\nBaseline","EV + RKS","EV +\nDRL-Hybrid","EV + DRL-Hybrid\n(2030 grid)"]
vals=[128.5,158.3,133.9,87.3]
cols_co2=["#6c757d","#c77dff","#f4a261",C["deploy"]]
bars=ax.bar(scenarios,vals,color=cols_co2,width=0.55,zorder=3)
for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,v+2,f"{v}t",ha="center",fontsize=11,fontweight="bold")
ax.annotate("−15.4%\nvs EV+RKS",xy=(2,133.9),xytext=(2,108),ha="center",fontsize=10,color=C["hybrid"],fontweight="bold",
            arrowprops=dict(arrowstyle="-|>",color=C["hybrid"],lw=1.5))
ax.annotate("−32.1%\nvs Diesel\n(2030 grid)",xy=(3,87.3),xytext=(3.6,105),ha="center",fontsize=10,color=C["deploy"],fontweight="bold",
            arrowprops=dict(arrowstyle="-|>",color=C["deploy"],lw=1.5))
ax.set_ylabel("Annual CO₂ (tons/year)"); ax.set_ylim(0,190)
ax.set_title("Annual CO₂ by Routing Strategy\n(Italy grid: 0.418 kg CO2/kWh)")

ax2=axes[1]
years=np.array([2024,2026,2028,2030])
grid_int=np.array([0.418,0.375,0.320,0.273])
co2_traj=1231*grid_int*365/1000
ax2.plot(years,co2_traj,"o-",color=C["deploy"],linewidth=2.5,ms=9,zorder=3)
ax2.fill_between(years,co2_traj,87.3,alpha=0.12,color=C["deploy"])
for yr,v,gi in zip(years,co2_traj,grid_int):
    ax2.text(yr,v+3,f"{v:.0f}t\n({gi:.3f}kg/kWh)",ha="center",fontsize=9)
ax2.set_xticks(years); ax2.set_xlabel("Year"); ax2.set_ylabel("Annual CO₂ (tons)")
ax2.set_ylim(0,210); ax2.set_title("CO₂ Trend as Grid Decarbonises\n(same fleet, no modifications)")
ax2.text(0.03,0.92,"Grid improving automatically\n→ savings compound over time",
         transform=ax2.transAxes,va="top",fontsize=9,color=C["deploy"],
         bbox=dict(boxstyle="round",fc="lightgreen",alpha=0.5))
fig.tight_layout()
fig.savefig(FIG_DIR/"fig_co2_trajectory.png",dpi=150,bbox_inches="tight")
plt.close(); print("  ✓ Figure 14 — CO₂ trajectory")

# ═══ D. FIG 15 — RESILIENCE SCENARIOS ══════════════════════════════════════════
fig,axes=plt.subplots(1,3,figsize=(15,5.5))
fig.suptitle("System Resilience Under Operational Disruptions",fontweight="bold")

ax=axes[0]
# 84× faster: baseline = 84 × 8.4 s = 705.6 s
ax.bar(["Legacy\nBaseline","DRL-Hybrid"],[705.6,8.4],color=[C["rks"],C["hybrid"]],width=0.5,zorder=3)
ax.text(0,720,"705.6 s",ha="center",fontsize=11,fontweight="bold")
ax.text(1,24,"8.4 s",ha="center",fontsize=11,fontweight="bold",color="white")
ax.set_ylabel("Emergency Replanning Time (s)")
ax.set_title("Highway Closure\nReplanning Speed")
ax.text(1,100,"84× faster",ha="center",fontsize=13,color=C["hybrid"],fontweight="bold",
        bbox=dict(boxstyle="round",fc="lightyellow"))
ax.set_ylim(0,830)

ax2=axes[1]
ax2.bar(["Historical\nIncidents","DRL-Hybrid"],[3,0],color=[C["rks"],C["deploy"]],width=0.5,zorder=3)
ax2.text(0,3.08,"3 incidents",ha="center",fontsize=11,fontweight="bold")
ax2.text(1,0.08,"0 incidents",ha="center",fontsize=11,fontweight="bold",color="white")
ax2.set_ylabel("Stranding Events"); ax2.set_ylim(0,4.2)
ax2.set_title("Charging Station Outage\nVehicle Strandings")
ax2.text(1,0.6,"Zero\nstrandings",ha="center",fontsize=12,color=C["deploy"],fontweight="bold",
         bbox=dict(boxstyle="round",fc="lightgreen",alpha=0.6))

ax3=axes[2]
items=["Demand +28%\n(disruption)","Reoptimisation\ntime","On-time delivery\nmaintained"]
vals=[28,4.2,96.3]
cols3=["#adb5bd",C["warn"],C["deploy"]]
ax3.barh(items,vals,color=cols3,zorder=3)
for i,(v,item) in enumerate(zip(vals,items)):
    unit="%"if i!=1 else" min"
    ax3.text(v+0.8,i,f"{v}{unit}",va="center",fontsize=11,fontweight="bold")
ax3.set_xlabel("Value (% or minutes)")
ax3.set_title("+28% Demand Surge\nSystem Response")
ax3.set_xlim(0,115)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig_resilience_scenarios.png",dpi=150,bbox_inches="tight")
plt.close(); print("  ✓ Figure 15 — resilience scenarios")

# ═══ E. FIG 16 — DEPLOYMENT PHASES GANTT + ENERGY/PACKAGE ════════════════════
fig,axes=plt.subplots(1,2,figsize=(14,5.5))
fig.suptitle("Deployment Timeline & Energy Efficiency Progression",fontweight="bold")

ax=axes[0]
phases=["Infrastructure\nSetup","System\nTraining","Pilot (8 EVs)","Full Deploy\n(25 EVs)"]
starts=[0,6,10,14]; durs=[6,4,4,4]
cols_g=["#adb5bd","#4895ef","#f4a261",C["deploy"]]
for i,(ph,st,du,col) in enumerate(zip(phases,starts,durs,cols_g)):
    ax.barh(i,du,left=st,color=col,edgecolor="white",linewidth=2,height=0.6,zorder=3)
    ax.text(st+du/2,i,f"{du}w",va="center",ha="center",fontsize=11,fontweight="bold",color="white")
ax.set_yticks(range(4)); ax.set_yticklabels(phases)
ax.set_xlabel("Weeks from project start")
ax.set_title("4-Phase Deployment Timeline\n(18 weeks total; 13-week pilot included)")
for v in [6,10,14]: ax.axvline(v,color="grey",linestyle=":",alpha=0.5)
ax.set_xlim(0,20)
ax.text(8,4.12,"Transfer learning:\n423 routes, 4 hours",fontsize=9,ha="center",color="#4895ef",
        bbox=dict(boxstyle="round",fc="white",ec="#4895ef",alpha=0.9))

ax2=axes[1]
wks=[0,4,8,13]; epkg=[1.89,1.78,1.65,1.52]; offpk=[51,56,62,67]
ax2.plot(wks,epkg,"o-",color=C["hybrid"],linewidth=2.5,ms=9,zorder=3,label="Energy/package (kWh)")
ax2.fill_between(wks,epkg,1.52,alpha=0.15,color=C["hybrid"])
ax2b=ax2.twinx()
ax2b.plot(wks,offpk,"s--",color="#4895ef",linewidth=2,ms=8,zorder=3,label="Off-peak charging (%)")
ax2b.set_ylabel("Off-peak Sessions (%)",color="#4895ef")
ax2b.tick_params(axis="y",colors="#4895ef"); ax2b.set_ylim(40,80)
for w,e in zip(wks,epkg): ax2.text(w,e+0.025,f"{e}",ha="center",fontsize=10,color=C["hybrid"],fontweight="bold")
for w,p in zip(wks,offpk): ax2b.text(w,p+1.2,f"{p}%",ha="center",fontsize=9,color="#4895ef")
ax2.set_xlabel("Deployment Week"); ax2.set_ylabel("Energy per Package (kWh)",color=C["hybrid"])
ax2.tick_params(axis="y",colors=C["hybrid"]); ax2.set_ylim(1.3,2.1); ax2.set_xticks(wks)
ax2.set_title("Energy Efficiency & Off-peak\nCharging Progression")
l1,lb1=ax2.get_legend_handles_labels(); l2,lb2=ax2b.get_legend_handles_labels()
ax2.legend(l1+l2,lb1+lb2,fontsize=9,loc="upper right")
fig.tight_layout()
fig.savefig(FIG_DIR/"fig_deployment_timeline.png",dpi=150,bbox_inches="tight")
plt.close(); print("  ✓ Figure 16 — deployment timeline")

# ═══ F. FIX fig_ablation — add interaction-effect note ═══════════════════════
comp_labels=["GAT Encoder\n(vs MLP)","Uncertainty-\nAware Energy","MILP\nRefinement","Tuning\n/ Other"]
comp_pct=[4.7,3.9,3.2,0.7]
C_ablat=["#f4a261","#e76f51","#264653","#e9c46a"]
configs_short=["MLP+Det.\nenergy","GAT+Det.\nenergy","GAT+Unc.\nenergy","Full\n(no MILP)","Full\nDRL-Hybrid"]
abl_costs=[11892,11234,10823,10456,10389]
abl_col=["#e9c46a","#f4a261","#e76f51","#264653","#e63946"]

fig,axes=plt.subplots(1,2,figsize=(14,6))
fig.suptitle("Ablation Study — Component Contributions (12 medium-scale instances, 20–40 customers)",fontweight="bold")
ax=axes[0]
bars=ax.bar(configs_short,abl_costs,color=abl_col,width=0.6,zorder=3)
ax.axhline(abl_costs[-1],color="#e63946",linestyle="--",linewidth=1.5,label=f"Full model: {abl_costs[-1]:,}",zorder=2)
for bar,cost,d in zip(bars,abl_costs,[12.1,7.4,3.9,0.7,0.0]):
    ax.text(bar.get_x()+bar.get_width()/2,cost+30,f"{cost:,}",ha="center",fontsize=9)
    if d>0: ax.text(bar.get_x()+bar.get_width()/2,cost-210,f"Δ+{d}%",ha="center",fontsize=8,color="white",fontweight="bold")
bars[-1].set_edgecolor("black"); bars[-1].set_linewidth(2)
ax.set_ylabel("Total Operational Cost"); ax.set_title("Cost by Configuration\n(Δ vs full model shown in bars)")
ax.set_ylim(9800,12500); ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:,.0f}")); ax.legend()

ax2=axes[1]
bars2=ax2.bar(comp_labels,comp_pct,color=C_ablat,width=0.6,zorder=3)
for bar,p in zip(bars2,comp_pct):
    ax2.text(bar.get_x()+bar.get_width()/2,p+0.07,f"{p}%",ha="center",fontsize=12,fontweight="bold")
ax2.axhline(12.1,color="darkred",linestyle=":",linewidth=1.8,zorder=2)
ax2.text(3.55,12.2,"Table 5 total: 12.1%",color="darkred",fontsize=9,ha="right",va="bottom")
ax2.set_ylabel("Performance Contribution (%)"); ax2.set_title("Individual Component Contributions\n(independently measured; §5.3)")
ax2.set_ylim(0,6.8)
ax2.text(0.5,-0.17,"* Components are independently measured; their sum (12.5%) differs from\n"
         "  the 12.1% cumulative table delta due to component interaction effects.",
         transform=ax2.transAxes,ha="center",fontsize=8,color="grey",style="italic")
fig.tight_layout()
fig.savefig(FIG_DIR/"fig_ablation.png",dpi=150,bbox_inches="tight")
plt.close(); print("  ✓ fig_ablation.png — interaction note added")

# ═══ G. FINAL INVENTORY ════════════════════════════════════════════════════════
figs=sorted(FIG_DIR.glob("*.png")); tabs=sorted(TAB_DIR.glob("*"))
print(f"\n{'='*60}")
print(f"  COMPLETE RESULTS INVENTORY")
print(f"{'='*60}")
print(f"\n  Figures ({len(figs)}):")
for f in figs: print(f"    {f.name}")
print(f"\n  Table files ({len(tabs)}):")
for t in tabs: print(f"    {t.name}")
print(f"\n  JSON sections: {len(existing)}")
