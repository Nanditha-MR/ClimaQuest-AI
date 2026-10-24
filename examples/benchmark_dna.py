"""
ClimaQuest AI — Benchmark & Research Demonstration
Demonstrates how Storm DNA enables:
1. Forecast Reality Gap calculation (Observed vs Forecasted DNA divergence).
2. Historical Analog Retrieval (finding twin storms in past seasons).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.storm_dna import StormDNA


def main():
    print("=" * 75)
    print("CLIMAQUEST AI — RESEARCH BENCHMARK: THE POWER OF STORM DNA")
    print("How Member 2's CV output powers Member 3's Forecast Reality Gap Engine")
    print("=" * 75)

    # 1. Simulate Observed Storm Cell at t = 14:00Z
    # A severe supercell developing unexpected explosive northward expansion
    obs_dna = StormDNA(
        vector=np.array([0.72, 0.88, 0.82, 0.45, 0.65, 0.42, 0.85, 0.68], dtype=np.float32),
        archetype="Severe Supercell",
        raw_metrics={
            "area_km2": 4500.0,
            "max_intensity": 62.5,
            "speed_kmh": 42.0,
            "growth_rate_pct_hr": 48.0,
        },
    )

    # 2. Simulate 3 Competing Numerical Forecast Model Scenarios (e.g. GFS vs ECMWF vs NCMRWF)
    forecast_scenarios = {
        "Scenario A (ECMWF Control)": StormDNA(
            vector=np.array([0.68, 0.82, 0.80, 0.40, 0.60, 0.40, 0.55, 0.62], dtype=np.float32),
            archetype="Severe Supercell",
            raw_metrics={"area_km2": 4100.0, "max_intensity": 58.0, "speed_kmh": 40.0, "growth_rate_pct_hr": 10.0},
        ),
        "Scenario B (GFS Ensemble Member 4)": StormDNA(
            vector=np.array([0.35, 0.50, 0.45, 0.75, 0.25, 0.60, 0.35, 0.40], dtype=np.float32),
            archetype="Squall Line / Linear Multicell",
            raw_metrics={"area_km2": 1800.0, "max_intensity": 45.0, "speed_kmh": 60.0, "growth_rate_pct_hr": -15.0},
        ),
        "Scenario C (Decaying Model Baseline)": StormDNA(
            vector=np.array([0.20, 0.30, 0.40, 0.30, 0.10, 0.20, 0.20, 0.25], dtype=np.float32),
            archetype="Decaying Stratiform Shield",
            raw_metrics={"area_km2": 950.0, "max_intensity": 36.0, "speed_kmh": 20.0, "growth_rate_pct_hr": -40.0},
        ),
    }

    print("\n[Case 1] Forecast Reality Gap Analysis:")
    print("Observed Event Archetype:", obs_dna.archetype)
    print("Observed DNA Vector:    ", [round(float(v), 2) for v in obs_dna.vector])
    print("-" * 75)

    print(f"{'Forecast Scenario':<36} | {'DNA Dist':<10} | {'Cosine Sim':<12} | {'Reality Gap Status'}")
    print("-" * 75)

    for name, f_dna in forecast_scenarios.items():
        dist = obs_dna.distance_to(f_dna)
        sim = obs_dna.similarity_to(f_dna)

        if dist < 0.35:
            gap_status = "MINIMAL GAP (Aligned)"
        elif dist < 0.70:
            gap_status = "MODERATE GAP (Diverging)"
        else:
            gap_status = "CRITICAL REALITY GAP"

        print(f"{name:<36} | {dist:<10.3f} | {sim:<12.3f} | {gap_status}")

    print("-" * 75)
    print("Key SIH Takeaway:")
    print("Instead of simply displaying a raw satellite picture, ClimaQuest AI calculates")
    print("the Forecast Reality Gap directly from Member 2's Storm DNA vectors.")
    print("Member 3 uses this quantitative distance to calibrate scenario weights in Engine 3 (Future Tree)!\n")


if __name__ == "__main__":
    main()
