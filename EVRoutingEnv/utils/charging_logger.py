"""
Charging Logger for tracking and analyzing charging sessions.

Logs detailed charging metrics including:
- SOC progression over time
- Power delivery curves
- Charging efficiency metrics
- Taper behavior for realistic curves
"""

import json
import os
from typing import Dict, List, Optional
import numpy as np


class ChargingLogger:
    """
    Tracks charging sessions and generates detailed logs for analysis.
    """
    
    def __init__(self, output_dir: str, verbose: bool = False):
        """
        Initialize charging logger.
        
        Args:
            output_dir: Directory to save log files
            verbose: Enable verbose printing
        """
        self.output_dir = output_dir
        self.verbose = verbose
        self.charging_sessions = []
        
        # Create charging logs subdirectory
        self.charging_log_dir = os.path.join(output_dir, "charging_logs")
        os.makedirs(self.charging_log_dir, exist_ok=True)
        
    def log_charging_session(
        self,
        truck_id: int,
        charger_node: int,
        charger_type: str,
        start_time: float,
        end_time: float,
        initial_soc: float,
        final_soc: float,
        charge_amount: float,
        battery_capacity: float,
        charging_details: Dict
    ):
        """
        Log a completed charging session.
        
        Args:
            truck_id: Truck identifier
            charger_node: Charger node ID
            charger_type: "DCFast" or "Level2"
            start_time: Global clock when charging started (hours)
            end_time: Global clock when charging ended (hours)
            initial_soc: SOC at start (0.0-1.0)
            final_soc: SOC at end (0.0-1.0)
            charge_amount: Energy delivered (kWh)
            battery_capacity: Battery capacity (kWh)
            charging_details: Dict from ChargingCurveModel containing:
                - model_used: "linear" or "cccv"
                - average_power: Average power (kW)
                - taper_factor: Ratio of avg to peak power
                - power_curve: List of (time, power, soc) tuples
                - (for CCCV) taper_start_soc, peak_power, taper_power_min
        """
        session = {
            "truck_id": truck_id,
            "charger_node": int(charger_node),
            "charger_type": charger_type,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "initial_soc": initial_soc,
            "final_soc": final_soc,
            "soc_gain": final_soc - initial_soc,
            "charge_amount": charge_amount,
            "battery_capacity": battery_capacity,
            "model_used": charging_details.get("model_used", "unknown"),
            "average_power": charging_details.get("average_power", 0.0),
            "taper_factor": charging_details.get("taper_factor", 1.0),
            "power_curve": charging_details.get("power_curve", [])
        }
        
        # Add CCCV-specific fields if applicable
        if charging_details.get("model_used") == "cccv":
            session["taper_start_soc"] = charging_details.get("taper_start_soc")
            session["peak_power"] = charging_details.get("peak_power")
            session["taper_power_min"] = charging_details.get("taper_power_min")
        
        self.charging_sessions.append(session)
        
        if self.verbose:
            print(f"\n[ChargingLogger] Session logged for Truck {truck_id}")
            print(f"  Charger: {charger_type} @ node {charger_node}")
            print(f"  Time: {start_time:.2f}h → {end_time:.2f}h ({session['duration']:.2f}h)")
            print(f"  SOC: {initial_soc*100:.1f}% → {final_soc*100:.1f}% (+{session['soc_gain']*100:.1f}%)")
            print(f"  Energy: {charge_amount:.2f} kWh")
            print(f"  Model: {session['model_used']}")
            print(f"  Avg Power: {session['average_power']:.1f} kW (taper: {session['taper_factor']*100:.1f}%)")
    
    def save_session_logs(self, episode_id: Optional[str] = None):
        """
        Save all logged sessions to JSON file.
        
        Args:
            episode_id: Optional episode identifier for filename
        """
        if not self.charging_sessions:
            if self.verbose:
                print("[ChargingLogger] No sessions to save")
            return
        
        # Generate filename
        if episode_id:
            filename = f"charging_sessions_{episode_id}.json"
        else:
            filename = "charging_sessions.json"
        
        filepath = os.path.join(self.charging_log_dir, filename)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(self.charging_sessions, f, indent=2)
        
        if self.verbose:
            print(f"\n[ChargingLogger] Saved {len(self.charging_sessions)} sessions to {filepath}")
    
    def save_summary_statistics(self, episode_id: Optional[str] = None):
        """
        Calculate and save summary statistics for all charging sessions.
        
        Args:
            episode_id: Optional episode identifier for filename
        """
        if not self.charging_sessions:
            return
        
        # Separate by model type and charger type
        stats = {
            "total_sessions": len(self.charging_sessions),
            "by_model": {},
            "by_charger_type": {},
            "overall": {}
        }
        
        # Group sessions
        by_model = {}
        by_charger = {}
        
        for session in self.charging_sessions:
            model = session["model_used"]
            charger = session["charger_type"]
            
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(session)
            
            if charger not in by_charger:
                by_charger[charger] = []
            by_charger[charger].append(session)
        
        # Calculate statistics for each group
        for model, sessions in by_model.items():
            stats["by_model"][model] = self._calculate_group_stats(sessions)
        
        for charger, sessions in by_charger.items():
            stats["by_charger_type"][charger] = self._calculate_group_stats(sessions)
        
        stats["overall"] = self._calculate_group_stats(self.charging_sessions)
        
        # Save to JSON
        if episode_id:
            filename = f"charging_summary_{episode_id}.json"
        else:
            filename = "charging_summary.json"
        
        filepath = os.path.join(self.charging_log_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        if self.verbose:
            print(f"\n[ChargingLogger] Charging Summary Statistics")
            print(f"{'='*60}")
            print(f"Total Sessions: {stats['total_sessions']}")
            print(f"\nBy Model:")
            for model, model_stats in stats["by_model"].items():
                print(f"  {model}:")
                print(f"    Count: {model_stats['count']}")
                print(f"    Avg Duration: {model_stats['avg_duration']:.2f}h")
                print(f"    Avg Power: {model_stats['avg_power']:.1f} kW")
                print(f"    Avg Taper Factor: {model_stats['avg_taper_factor']*100:.1f}%")
            print(f"\nSaved to {filepath}")
    
    def _calculate_group_stats(self, sessions: List[Dict]) -> Dict:
        """Calculate statistics for a group of sessions."""
        if not sessions:
            return {}
        
        durations = [s["duration"] for s in sessions]
        soc_gains = [s["soc_gain"] for s in sessions]
        charge_amounts = [s["charge_amount"] for s in sessions]
        avg_powers = [s["average_power"] for s in sessions]
        taper_factors = [s["taper_factor"] for s in sessions]
        
        return {
            "count": len(sessions),
            "avg_duration": np.mean(durations),
            "std_duration": np.std(durations),
            "avg_soc_gain": np.mean(soc_gains),
            "avg_charge_amount": np.mean(charge_amounts),
            "avg_power": np.mean(avg_powers),
            "avg_taper_factor": np.mean(taper_factors),
            "min_taper_factor": np.min(taper_factors),
            "max_taper_factor": np.max(taper_factors)
        }
    
    def reset(self):
        """Clear all logged sessions."""
        self.charging_sessions = []
