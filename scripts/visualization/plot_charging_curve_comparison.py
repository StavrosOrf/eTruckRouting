"""
Generate a paper-ready comparison of linear vs CCCV charging curves.

Outputs:
- docs/figures/charging_curve_comparison.pdf

Notes:
- Uses importlib to load the ChargingCurveModel without importing gym-dependent envs.
- Defaults match config.yaml DC-fast settings (50 kW, 0.85 eff, taper at 80%, min 25 kW).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_charging_curve_model():
    """Load ChargingCurveModel directly from file to avoid pulling gym dependencies."""
    module_path = Path("EVRoutingEnv/models/simulation/charging_curve.py")
    spec = importlib.util.spec_from_file_location("charging_curve", module_path)
    charging_curve = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(charging_curve)
    return charging_curve.ChargingCurveModel


def generate_figure():
    ChargingCurveModel = load_charging_curve_model()
    model = ChargingCurveModel(verbose=False)

    # Base settings (from config.yaml)
    battery_capacity = 400.0  # kWh
    initial_soc = 0.10  # 10%
    charge_hours = 10.0  # generous window to reach 100%
    dcfast_cfg = {
        "charge_rate": 50.0,
        "efficiency": 0.85,
        "use_realistic_curve": True,
        "taper_start_soc": 0.8,
        "taper_power_min": 25.0,
    }
    linear_cfg = dict(dcfast_cfg, use_realistic_curve=False)

    def run(cfg):
        _, details = model.calculate_charge(
            initial_soc=initial_soc,
            charge_hours=charge_hours,
            battery_capacity=battery_capacity,
            charger_config=cfg,
            charger_type="DCFast",
        )
        curve = np.array(details["power_curve"])  # cols: time, power, soc
        return curve[:, 0], curve[:, 1], curve[:, 2], details

    lin_t, lin_p, lin_soc, lin_details = run(linear_cfg)
    cccv_t, cccv_p, cccv_soc, cccv_details = run(dcfast_cfg)
    max_time = max(lin_details["actual_charge_hours"], cccv_details["actual_charge_hours"])

    # Matplotlib styling for paper
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    colors = {"linear": "#1f77b4", "cccv": "#d62728"}

    # Power vs SOC
    axes[0].plot(lin_soc * 100, lin_p, label="Linear (constant power)", color=colors["linear"])
    axes[0].plot(cccv_soc * 100, cccv_p, label="CCCV (tapered)", color=colors["cccv"])
    axes[0].axvline(dcfast_cfg["taper_start_soc"] * 100, color="#666666", linestyle="--", linewidth=1.3, label="Taper start")
    axes[0].set_xlabel("State of charge (%)")
    axes[0].set_ylabel("Power (kW)")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, max(lin_p.max(), cccv_p.max()) * 1.1)
    axes[0].set_title("Power profile")
    axes[0].legend(loc="lower left")

    # SOC vs time
    axes[1].plot(
        lin_t,
        lin_soc * 100,
        label=f"Linear ({lin_details['actual_charge_hours']:.1f} h to 100%)",
        color=colors["linear"],
    )
    axes[1].plot(
        cccv_t,
        cccv_soc * 100,
        label=f"CCCV ({cccv_details['actual_charge_hours']:.1f} h to 100%)",
        color=colors["cccv"],
    )
    axes[1].axhline(dcfast_cfg["taper_start_soc"] * 100, color="#666666", linestyle="--", linewidth=1.3)
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("State of charge (%)")
    axes[1].set_xlim(0, max_time)
    axes[1].set_ylim(0, 105)
    axes[1].set_title("SOC over time")
    axes[1].legend(loc="lower right")

    # Save
    outdir = Path("docs/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "charging_curve_comparison.pdf"
    fig.savefig(outfile, bbox_inches="tight")
    print(f"Saved figure to {outfile} (linear={lin_details['actual_charge_hours']:.2f}h, cccv={cccv_details['actual_charge_hours']:.2f}h)")


if __name__ == "__main__":
    generate_figure()
