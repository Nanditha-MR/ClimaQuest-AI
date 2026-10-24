"""
ClimaQuest AI — Member 2 Executable Demonstration
Runs complete Visual Weather Intelligence Pipeline (Engine 1):
Simulates 12 convective radar frames, performs storm cell segmentation,
tracks events across time, synthesizes Storm DNA, and renders mission control outputs.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic_radar import MeteorologicalSequenceGenerator
from src.pipeline.engine import VisualWeatherIntelligencePipeline


def main():
    print("=" * 75)
    print("CLIMAQUEST AI — ENGINE 1: VISUAL WEATHER INTELLIGENCE")
    print("Member 2: Computer Vision, Storm Tracking, and Storm DNA")
    print("=" * 75)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate simulated meteorological radar sequence
    print("\n[Phase 1] Generating high-resolution meteorological sequence (12 frames)...")
    generator = MeteorologicalSequenceGenerator(
        height=350,
        width=350,
        lon_min=73.5,
        lon_max=82.5,
        lat_min=13.0,
        lat_max=22.0,
        seed=101,
    )
    frames = generator.generate_scenario(num_frames=12)
    print(f"Generated {len(frames)} frames with resolution 2.5 km/pixel.")

    # Step 2: Initialize Visual Weather Intelligence Pipeline
    print("\n[Phase 2] Initializing Computer Vision & Tracking Engine...")
    pipeline = VisualWeatherIntelligencePipeline(
        envelope_threshold=25.0,  # 25 dBZ rain shield
        core_threshold=42.0,      # 42 dBZ severe convective core
        min_cell_area_px=25,
        resolution_km=2.5,
        dt_hours=0.25,            # 15 minutes per frame
        max_search_radius_px=55.0,
        enable_watershed=True,
    )

    # Step 3: Execute streaming pipeline
    print("\n[Phase 3] Processing sequence and rendering mission-control telemetry...")
    results = pipeline.process_sequence(frames, output_dir=output_dir)

    # Step 4: Display telemetry for the final frame
    latest_result = results[-1]
    summary = latest_result["summary"]
    active_tracks = latest_result["active_tracks"]
    dna_map = latest_result["dna_map"]

    print("\n" + "=" * 75)
    print(f"SURVEILLANCE REPORT — TIMESTAMP: {summary['timestamp']}")
    print(f"Active Tracked Storms: {summary['active_event_count']}")
    print("=" * 75)

    for track in active_tracks:
        dna = dna_map[track.track_id]
        feat = track.history_features[-1]
        raw = dna.raw_metrics

        print(f"\n STORM CELL [ID #{track.track_id:02d}] — {dna.archetype.upper()}")
        print(f"   Status:           {track.status} (Track Age: {track.age_frames} frames / {track.age_frames * 15} mins)")
        print(f"   Centroid:         Lon {feat.lon:.2f}°, Lat {feat.lat:.2f}° (Pixel: x={feat.centroid_px[0]:.1f}, y={feat.centroid_px[1]:.1f})")
        print(f"   Physical Extent:  {feat.area_km2:.1f} km² (Perimeter: {feat.perimeter_km:.1f} km)")
        print(f"   Kinematics:       Speed = {track.speed_kmh:.1f} km/h | Bearing = {track.bearing_deg:.1f}°")
        print(f"   Convective Core:  Peak = {feat.max_intensity:.1f} dBZ | Mean = {feat.mean_intensity:.1f} dBZ | Core Ratio = {feat.core_ratio * 100:.1f}%")
        print(f"   Evolution Trend:  Growth Rate = {track.growth_rate_pct_hr:+.1f}%/hr | Intensity Trend = {track.intensity_trend_dbz_hr:+.1f} dBZ/hr")
        print(f"   Storm DNA Genes:  {dna.vector.tolist()}")

    print("\n" + "=" * 75)
    print("[Phase 4] Artifact Generation Summary:")
    print(f" - Mission Control Visuals: {os.path.join(output_dir, 'frames')}")
    print(f" - Event Ledger JSON:       {os.path.join(output_dir, 'event_tracking_ledger.json')}")
    print(f" - MapLibre GeoJSON:        {os.path.join(output_dir, 'latest_events.geojson')}")
    print("=" * 75)
    print("Member 2 CV Pipeline executed with 100% success!\n")


if __name__ == "__main__":
    main()
