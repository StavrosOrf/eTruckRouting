"""Generate charging curve comparison figures from the CCCV model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from EVRoutingEnv.models.simulation.charging_curve import ChargingCurveModel


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "charging_curves"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def simulate_cccv_charge(
    *,
    initial_soc: float,
    battery_capacity: float,
    peak_power: float,
    efficiency: float,
    taper_start_soc: float,
    taper_power_min: float,
    dt_hours: float,
) -> tuple[list[float], list[float], list[float]]:
    """Simulate a dense CCCV charging trace from initial SOC to 100%."""
    model = ChargingCurveModel(verbose=False)

    time_hours = [0.0]
    soc_values = [initial_soc]
    power_values = [
        model.cccv_power_at_soc(
            initial_soc,
            peak_power,
            taper_start_soc,
            taper_power_min,
        )
    ]

    current_time = 0.0
    current_soc = initial_soc

    while current_soc < 1.0:
        current_power = model.cccv_power_at_soc(
            current_soc,
            peak_power,
            taper_start_soc,
            taper_power_min,
        )
        energy_step = current_power * efficiency * dt_hours
        remaining_capacity = (1.0 - current_soc) * battery_capacity

        if energy_step >= remaining_capacity:
            partial_dt = remaining_capacity / (current_power * efficiency)
            current_time += partial_dt
            current_soc = 1.0
        else:
            current_time += dt_hours
            current_soc += energy_step / battery_capacity

        time_hours.append(current_time)
        soc_values.append(current_soc)
        power_values.append(
            model.cccv_power_at_soc(
                current_soc,
                peak_power,
                taper_start_soc,
                taper_power_min,
            )
        )

    return time_hours, soc_values, power_values


def simulate_linear_charge(
    *,
    initial_soc: float,
    battery_capacity: float,
    peak_power: float,
    efficiency: float,
) -> tuple[list[float], list[float], list[float]]:
    """Simulate constant-power charging from initial SOC to 100%."""
    charge_hours = ((1.0 - initial_soc) * battery_capacity) / (
        peak_power * efficiency
    )
    time_hours = np.linspace(0.0, charge_hours, 200)
    soc_values = initial_soc + (
        (peak_power * efficiency * time_hours) / battery_capacity
    )
    soc_values = np.clip(soc_values, initial_soc, 1.0)
    power_values = np.full_like(time_hours, peak_power)
    return time_hours.tolist(), soc_values.tolist(), power_values.tolist()


def plot_charging_curve_comparison(
    *,
    output_path: Path,
    initial_soc: float,
    battery_capacity: float,
    peak_power: float,
    efficiency: float,
    taper_start_soc: float,
    taper_power_min: float,
    dt_hours: float,
) -> None:
    """Create a two-panel linear vs CCCV comparison figure."""
    model = ChargingCurveModel(verbose=False)

    linear_time, linear_soc, _ = simulate_linear_charge(
        initial_soc=initial_soc,
        battery_capacity=battery_capacity,
        peak_power=peak_power,
        efficiency=efficiency,
    )
    cccv_time, cccv_soc, _ = simulate_cccv_charge(
        initial_soc=initial_soc,
        battery_capacity=battery_capacity,
        peak_power=peak_power,
        efficiency=efficiency,
        taper_start_soc=taper_start_soc,
        taper_power_min=taper_power_min,
        dt_hours=dt_hours,
    )

    soc_grid = np.linspace(initial_soc, 1.0, 400)
    linear_power = np.full_like(soc_grid, peak_power)
    cccv_power = [
        model.cccv_power_at_soc(soc, peak_power, taper_start_soc, taper_power_min)
        for soc in soc_grid
    ]

    fig, (ax_power, ax_soc) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax_power.plot(
        soc_grid * 100.0,
        linear_power,
        color="#1f77b4",
        linewidth=1.8,
        label="Linear (constant power)",
    )
    ax_power.plot(
        soc_grid * 100.0,
        cccv_power,
        color="#d62728",
        linewidth=1.8,
        label="CCCV (tapered)",
    )
    ax_power.axvline(
        taper_start_soc * 100.0,
        color="#6e6e6e",
        linestyle="--",
        linewidth=1.2,
        label="Taper start",
    )
    ax_power.set_xlabel("State of charge (%)")
    ax_power.set_ylabel("Power (kW)")
    ax_power.set_xlim(0, 100)
    ax_power.set_ylim(0, peak_power * 1.15)
    ax_power.set_xticks(np.arange(0, 101, 20))
    ax_power.grid(True, color="#b0b0b0", alpha=0.35, linewidth=0.6)
    ax_power.legend(loc="lower left", frameon=True, framealpha=0.95)

    ax_soc.plot(
        linear_time,
        [soc * 100.0 for soc in linear_soc],
        color="#1f77b4",
        linewidth=1.8,
        label=f"Linear ({linear_time[-1]:.1f} h to 100%)",
    )
    ax_soc.plot(
        cccv_time,
        [soc * 100.0 for soc in cccv_soc],
        color="#d62728",
        linewidth=1.8,
        label=f"CCCV ({cccv_time[-1]:.1f} h to 100%)",
    )
    ax_soc.axhline(
        taper_start_soc * 100.0,
        color="#6e6e6e",
        linestyle="--",
        linewidth=1.2,
    )
    ax_soc.set_xlabel("Time (hours)")
    ax_soc.set_ylabel("State of charge (%)")
    ax_soc.set_xlim(0, max(linear_time[-1], cccv_time[-1]) * 1.05)
    ax_soc.set_ylim(0, 105)
    ax_soc.set_yticks(np.arange(0, 101, 20))
    ax_soc.grid(True, color="#b0b0b0", alpha=0.35, linewidth=0.6)
    ax_soc.legend(loc="lower right", frameon=True, framealpha=0.95)

    ax_power.text(
        0.5,
        -0.32,
        "(a) Power profile",
        transform=ax_power.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    ax_soc.text(
        0.5,
        -0.32,
        "(b) State of charge over time",
        transform=ax_soc.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    fig.subplots_adjust(bottom=0.28, left=0.085, right=0.985, top=0.98, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a CCCV full-charge SOC and power profile."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "charging_curve_comparison.pdf",
        help="Path for the generated figure.",
    )
    parser.add_argument("--initial-soc", type=float, default=0.0)
    parser.add_argument("--battery-capacity", type=float, default=400.0)
    parser.add_argument("--peak-power", type=float, default=50.0)
    parser.add_argument("--efficiency", type=float, default=0.85)
    parser.add_argument("--taper-start-soc", type=float, default=0.8)
    parser.add_argument("--taper-power-min", type=float, default=25.0)
    parser.add_argument(
        "--dt-hours",
        type=float,
        default=0.01,
        help="Simulation time step in hours.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    plot_charging_curve_comparison(
        output_path=output_path,
        initial_soc=args.initial_soc,
        battery_capacity=args.battery_capacity,
        peak_power=args.peak_power,
        efficiency=args.efficiency,
        taper_start_soc=args.taper_start_soc,
        taper_power_min=args.taper_power_min,
        dt_hours=args.dt_hours,
    )
    print(f"Saved charging curve plot to {output_path}")


if __name__ == "__main__":
    main()
