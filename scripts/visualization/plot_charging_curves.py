"""
Visualization script for charging curves.

Creates plots showing:
1. Power vs SOC for linear and realistic (CCCV) charging
2. SOC progression over time
3. Comparison of charging duration for different initial SOC values
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from EVRoutingEnv.models.charging_curve import ChargingCurveModel


def plot_power_vs_soc():
    """Plot power delivery as a function of SOC."""
    model = ChargingCurveModel(verbose=False)
    
    # Configuration
    battery_capacity = 400.0  # kWh (e.g., Tesla Semi, Silverado EV)
    peak_power = 150.0  # kW
    efficiency = 0.85
    taper_start_soc = 0.8
    taper_power_min = 30.0
    
    # Generate SOC points
    soc_points = np.linspace(0, 1, 200)
    
    # Calculate power for each SOC point
    linear_power = []
    realistic_power = []
    
    for soc in soc_points:
        # For power vs SOC, we charge for a very small duration from each SOC
        # and track the instantaneous power
        
        # Linear: always peak power until full
        if soc < 1.0:
            linear_power.append(peak_power)
        else:
            linear_power.append(0)
        
        # Realistic: get power from a tiny charge session
        if soc < 1.0:
            _, details = model.calculate_charge(
                initial_soc=soc,
                charge_hours=0.01,  # Very small duration
                battery_capacity=battery_capacity,
                charger_config={
                    'charge_rate': peak_power,
                    'efficiency': efficiency,
                    'use_realistic_curve': True,
                    'taper_start_soc': taper_start_soc,
                    'taper_power_min': taper_power_min
                },
                charger_type='DCFast'
            )
            # Get the first power sample (initial power at this SOC)
            if details['power_curve']:
                realistic_power.append(details['power_curve'][0][1])
            else:
                realistic_power.append(0)
        else:
            realistic_power.append(0)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Power vs SOC
    ax1.plot(soc_points * 100, linear_power, 'b-', linewidth=2.5, label='Linear (Constant Rate)', alpha=0.7)
    ax1.plot(soc_points * 100, realistic_power, 'r-', linewidth=2.5, label='Realistic (CCCV - Silverado EV Style)', alpha=0.8)
    ax1.axvline(x=taper_start_soc * 100, color='gray', linestyle='--', alpha=0.5, label=f'Taper Start ({taper_start_soc*100:.0f}% SOC)')
    ax1.axhline(y=peak_power, color='green', linestyle=':', alpha=0.4, label=f'Peak Power ({peak_power} kW)')
    
    ax1.set_xlabel('State of Charge (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Charging Power (kW)', fontsize=12, fontweight='bold')
    ax1.set_title('DC Fast Charging Power Profile', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, peak_power * 1.1])
    
    # Add annotations
    ax1.annotate('Ramp-up Phase\n(0-50% SOC)', xy=(25, peak_power * 0.8), fontsize=9, 
                ha='center', color='darkred', style='italic')
    ax1.annotate('Plateau Phase\n(50-80% SOC)', xy=(65, peak_power * 1.05), fontsize=9,
                ha='center', color='darkred', style='italic')
    ax1.annotate('Taper Phase\n(80-100% SOC)', xy=(90, peak_power * 0.6), fontsize=9,
                ha='center', color='darkred', style='italic')
    
    # Plot 2: Energy added vs SOC (integral of power)
    linear_energy = [peak_power * efficiency * i / 200 for i in range(len(soc_points))]
    realistic_energy = []
    cumulative = 0
    for i in range(len(realistic_power)):
        if i > 0:
            cumulative += realistic_power[i] * efficiency * 0.01  # Approximate integral
        realistic_energy.append(cumulative)
    
    ax2.plot(soc_points * 100, linear_energy, 'b-', linewidth=2.5, label='Linear', alpha=0.7)
    ax2.plot(soc_points * 100, realistic_energy, 'r-', linewidth=2.5, label='Realistic (CCCV)', alpha=0.8)
    
    ax2.set_xlabel('State of Charge (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Energy Rate (kWh/step)', fontsize=12, fontweight='bold')
    ax2.set_title('Energy Delivery Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_xlim([0, 100])
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             'results', 'charging_curves')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'power_vs_soc.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Saved: {output_path}')
    plt.close()


def plot_soc_progression():
    """Plot SOC progression over time for different scenarios."""
    model = ChargingCurveModel(verbose=False)
    
    battery_capacity = 400.0
    peak_power = 150.0
    efficiency = 0.85
    
    # Test scenarios: charge from different initial SOCs
    scenarios = [
        ('20% → 100%', 0.2, 5.0),
        ('50% → 100%', 0.5, 3.0),
        ('80% → 100%', 0.8, 2.0),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (label, initial_soc, max_hours) in enumerate(scenarios):
        ax = axes[idx]
        
        # Linear charging
        charge_linear, details_linear = model.calculate_charge(
            initial_soc=initial_soc,
            charge_hours=max_hours,
            battery_capacity=battery_capacity,
            charger_config={
                'charge_rate': peak_power,
                'efficiency': efficiency,
                'use_realistic_curve': False
            },
            charger_type='DCFast'
        )
        
        # Realistic charging
        charge_realistic, details_realistic = model.calculate_charge(
            initial_soc=initial_soc,
            charge_hours=max_hours,
            battery_capacity=battery_capacity,
            charger_config={
                'charge_rate': peak_power,
                'efficiency': efficiency,
                'use_realistic_curve': True,
                'taper_start_soc': 0.8,
                'taper_power_min': 30.0
            },
            charger_type='DCFast'
        )
        
        # Extract time series
        times_linear = [t for t, p, s in details_linear['power_curve']]
        socs_linear = [s * 100 for t, p, s in details_linear['power_curve']]
        powers_linear = [p for t, p, s in details_linear['power_curve']]
        
        times_realistic = [t for t, p, s in details_realistic['power_curve']]
        socs_realistic = [s * 100 for t, p, s in details_realistic['power_curve']]
        powers_realistic = [p for t, p, s in details_realistic['power_curve']]
        
        # Convert times to minutes
        times_linear_min = [t * 60 for t in times_linear]
        times_realistic_min = [t * 60 for t in times_realistic]
        
        # Create dual-axis plot
        ax_soc = ax
        ax_power = ax.twinx()
        
        # Plot SOC progression
        line1 = ax_soc.plot(times_linear_min, socs_linear, 'b-', linewidth=2.5, 
                           label='Linear SOC', alpha=0.7)
        line2 = ax_soc.plot(times_realistic_min, socs_realistic, 'r-', linewidth=2.5,
                           label='Realistic SOC', alpha=0.8)
        
        # Plot power curves (lighter, thinner)
        line3 = ax_power.plot(times_linear_min, powers_linear, 'b--', linewidth=1.5,
                             label='Linear Power', alpha=0.4)
        line4 = ax_power.plot(times_realistic_min, powers_realistic, 'r--', linewidth=1.5,
                             label='Realistic Power', alpha=0.4)
        
        ax_soc.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
        ax_soc.set_ylabel('State of Charge (%)', fontsize=11, fontweight='bold', color='black')
        ax_power.set_ylabel('Power (kW)', fontsize=11, fontweight='bold', color='gray')
        ax_soc.set_title(f'Charging Session: {label}', fontsize=12, fontweight='bold')
        
        ax_soc.grid(True, alpha=0.3)
        ax_soc.set_ylim([initial_soc * 100 - 5, 105])
        ax_power.set_ylim([0, peak_power * 1.2])
        
        # Color y-axis labels
        ax_soc.tick_params(axis='y', labelcolor='black')
        ax_power.tick_params(axis='y', labelcolor='gray')
        
        # Combined legend
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax_soc.legend(lines, labels, loc='lower right', fontsize=8)
        
        # Add charging time annotations
        time_linear_total = times_linear[-1] * 60
        time_realistic_total = times_realistic[-1] * 60
        time_diff = time_realistic_total - time_linear_total
        
        textstr = f'Linear: {time_linear_total:.0f} min\n'
        textstr += f'Realistic: {time_realistic_total:.0f} min\n'
        textstr += f'Difference: +{time_diff:.0f} min ({time_diff/time_linear_total*100:.1f}%)'
        
        ax_soc.text(0.05, 0.95, textstr, transform=ax_soc.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'results', 'charging_curves')
    output_path = os.path.join(output_dir, 'soc_progression.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Saved: {output_path}')
    plt.close()


def plot_charging_comparison_summary():
    """Create comprehensive comparison summary plot."""
    model = ChargingCurveModel(verbose=False)
    
    battery_capacity = 400.0
    peak_power = 150.0
    efficiency = 0.85
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Charging time vs initial SOC
    ax1 = fig.add_subplot(gs[0, 0])
    initial_socs = np.linspace(0.1, 0.9, 20)
    times_linear = []
    times_realistic = []
    
    for soc in initial_socs:
        # Calculate time to charge to 100%
        _, details_l = model.calculate_charge(
            initial_soc=soc, charge_hours=10.0, battery_capacity=battery_capacity,
            charger_config={'charge_rate': peak_power, 'efficiency': efficiency, 'use_realistic_curve': False},
            charger_type='DCFast'
        )
        _, details_r = model.calculate_charge(
            initial_soc=soc, charge_hours=10.0, battery_capacity=battery_capacity,
            charger_config={'charge_rate': peak_power, 'efficiency': efficiency, 'use_realistic_curve': True,
                          'taper_start_soc': 0.8, 'taper_power_min': 30.0},
            charger_type='DCFast'
        )
        times_linear.append(details_l['actual_charge_hours'])
        times_realistic.append(details_r['actual_charge_hours'])
    
    ax1.plot(initial_socs * 100, times_linear, 'b-o', linewidth=2, markersize=6, label='Linear', alpha=0.7)
    ax1.plot(initial_socs * 100, times_realistic, 'r-s', linewidth=2, markersize=6, label='Realistic (CCCV)', alpha=0.8)
    ax1.fill_between(initial_socs * 100, times_linear, times_realistic, alpha=0.2, color='orange')
    ax1.set_xlabel('Initial SOC (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Time to 100% (hours)', fontsize=11, fontweight='bold')
    ax1.set_title('Charging Duration Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Taper factor vs initial SOC
    ax2 = fig.add_subplot(gs[0, 1])
    taper_factors = []
    avg_powers = []
    
    for soc in initial_socs:
        _, details = model.calculate_charge(
            initial_soc=soc, charge_hours=10.0, battery_capacity=battery_capacity,
            charger_config={'charge_rate': peak_power, 'efficiency': efficiency, 'use_realistic_curve': True,
                          'taper_start_soc': 0.8, 'taper_power_min': 30.0},
            charger_type='DCFast'
        )
        taper_factors.append(details['taper_factor'])
        avg_powers.append(details['average_power'])
    
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(initial_socs * 100, taper_factors, 'g-o', linewidth=2, markersize=6, 
                     label='Taper Factor', alpha=0.7)
    line2 = ax2_twin.plot(initial_socs * 100, avg_powers, 'purple', linestyle='--', marker='s',
                          linewidth=2, markersize=6, label='Avg Power', alpha=0.7)
    
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=0.8, color='orange', linestyle=':', alpha=0.5, label='80% efficiency')
    ax2.axvline(x=80, color='red', linestyle='--', alpha=0.4, label='Taper starts')
    
    ax2.set_xlabel('Initial SOC (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Taper Factor (Avg/Peak Power)', fontsize=11, fontweight='bold', color='green')
    ax2_twin.set_ylabel('Average Power (kW)', fontsize=11, fontweight='bold', color='purple')
    ax2.set_title('Charging Efficiency Metrics', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='green')
    ax2_twin.tick_params(axis='y', labelcolor='purple')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='lower left', fontsize=9)
    
    # Plot 3: Detailed power curve for one scenario
    ax3 = fig.add_subplot(gs[1, :])
    
    # Charge from 20% to 100%
    _, details_realistic = model.calculate_charge(
        initial_soc=0.2, charge_hours=5.0, battery_capacity=battery_capacity,
        charger_config={'charge_rate': peak_power, 'efficiency': efficiency, 'use_realistic_curve': True,
                      'taper_start_soc': 0.8, 'taper_power_min': 30.0},
        charger_type='DCFast'
    )
    
    times = [t * 60 for t, p, s in details_realistic['power_curve']]  # Convert to minutes
    powers = [p for t, p, s in details_realistic['power_curve']]
    socs = [s * 100 for t, p, s in details_realistic['power_curve']]
    
    # Create filled area plot
    ax3.fill_between(times, 0, powers, alpha=0.3, color='steelblue', label='Power Output')
    ax3.plot(times, powers, 'b-', linewidth=2.5, label='Charging Power')
    
    # Add SOC line on secondary axis
    ax3_twin = ax3.twinx()
    ax3_twin.plot(times, socs, 'r-', linewidth=2.5, alpha=0.7, label='State of Charge')
    
    # Add phase markers
    phase_markers = [
        (0, 'Ramp-up\nPhase', 'green'),
        (30, 'Plateau\nPhase', 'blue'),
        (60, 'Taper\nPhase', 'orange'),
    ]
    
    for time_mark, phase_name, color in phase_markers:
        if time_mark < max(times):
            ax3.axvline(x=time_mark, color=color, linestyle='--', alpha=0.4, linewidth=1.5)
            ax3.text(time_mark + 2, peak_power * 0.85, phase_name, fontsize=10, 
                    color=color, fontweight='bold', alpha=0.7)
    
    ax3.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Charging Power (kW)', fontsize=12, fontweight='bold', color='blue')
    ax3_twin.set_ylabel('State of Charge (%)', fontsize=12, fontweight='bold', color='red')
    ax3.set_title('Detailed Charging Profile: 20% → 100% SOC (Realistic CCCV Model)', 
                 fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='both')
    ax3.set_ylim([0, peak_power * 1.1])
    ax3_twin.set_ylim([0, 110])
    
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    
    # Add statistics box
    total_time = times[-1]
    total_energy = details_realistic['actual_charge_hours'] * details_realistic['average_power'] * efficiency
    avg_power = details_realistic['average_power']
    taper_factor = details_realistic['taper_factor']
    
    stats_text = f'Session Statistics:\n'
    stats_text += f'Duration: {total_time:.1f} min ({total_time/60:.2f} h)\n'
    stats_text += f'Energy Added: {total_energy:.1f} kWh\n'
    stats_text += f'Avg Power: {avg_power:.1f} kW\n'
    stats_text += f'Taper Factor: {taper_factor:.2f} ({taper_factor*100:.1f}%)'
    
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')
    
    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.suptitle('DC Fast Charging Analysis: Linear vs Realistic (CCCV) Models',
                fontsize=15, fontweight='bold', y=0.995)
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'results', 'charging_curves')
    output_path = os.path.join(output_dir, 'charging_comparison_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Saved: {output_path}')
    plt.close()


if __name__ == '__main__':
    print('='*60)
    print('Generating Charging Curve Visualizations')
    print('='*60)
    print()
    
    print('1. Plotting power vs SOC...')
    plot_power_vs_soc()
    
    print('2. Plotting SOC progression...')
    plot_soc_progression()
    
    print('3. Plotting comprehensive comparison...')
    plot_charging_comparison_summary()
    
    print()
    print('='*60)
    print('✅ All visualizations generated successfully!')
    print('='*60)
    print()
    print('Output directory: results/charging_curves/')
