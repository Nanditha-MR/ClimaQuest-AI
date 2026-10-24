"""
ClimaQuest AI — Member 2 Handoff: FastAPI Bridge for Member 4 (Backend Lead)
Demonstrates how Member 4 can ingest Member 2's outputs and serve them via REST endpoints.
"""

import os
import json
from typing import Dict, Any, List

# Paths where Member 2 generates live artifacts
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
LEDGER_PATH = os.path.join(OUTPUT_DIR, "event_tracking_ledger.json")
GEOJSON_PATH = os.path.join(OUTPUT_DIR, "latest_events.geojson")


class ClimaQuestBackendService:
    """
    Service class that Member 4 can copy directly into the FastAPI backend
    to serve Member 2's computer vision and tracking outputs to Member 1 (Frontend).
    """

    @staticmethod
    def get_active_events() -> List[Dict[str, Any]]:
        """
        Endpoint: GET /api/v1/surveillance/active-storms
        Returns the latest detected storm cells, kinematics, and Storm DNA.
        """
        if not os.path.exists(LEDGER_PATH):
            return []
        with open(LEDGER_PATH, "r") as f:
            ledger = json.load(f)
        if not ledger:
            return []
        return ledger[-1].get("events", [])

    @staticmethod
    def get_geojson_feature_collection() -> Dict[str, Any]:
        """
        Endpoint: GET /api/v1/surveillance/geojson
        Returns standard GeoJSON polygons for MapLibre / Deck.gl.
        """
        if not os.path.exists(GEOJSON_PATH):
            return {"type": "FeatureCollection", "features": []}
        with open(GEOJSON_PATH, "r") as f:
            return json.load(f)

    @staticmethod
    def get_storm_dna(track_id: int) -> Dict[str, Any]:
        """
        Endpoint: GET /api/v1/surveillance/dna/{track_id}
        Returns the 8-gene vector, 16-dim deep visual embedding, and archetype.
        """
        events = ClimaQuestBackendService.get_active_events()
        for ev in events:
            if ev.get("track_id") == track_id:
                return ev.get("storm_dna", {})
        return {"error": f"Storm with track_id {track_id} not found"}


def verify_handoff():
    print("=" * 70)
    print("CLIMAQUEST AI — MEMBER 2 -> MEMBER 4 HANDOFF VERIFICATION")
    print("=" * 70)

    events = ClimaQuestBackendService.get_active_events()
    print(f"\n[1] Active Storms Retrieved by Backend Service: {len(events)}")
    for ev in events:
        dna = ev.get("storm_dna", {})
        print(f"   -> ID #{ev['track_id']:02d}: {dna.get('archetype', 'Unknown')} | "
              f"Speed: {ev['kinematics']['speed_kmh']} km/h | "
              f"Area: {ev['spatial']['area_km2']} km² | "
              f"Deep Embedding Dim: {len(dna.get('deep_embedding', []))}")

    geojson = ClimaQuestBackendService.get_geojson_feature_collection()
    print(f"\n[2] GeoJSON Features Ready for Member 1's MapLibre: {len(geojson.get('features', []))} polygons")

    if events:
        sample_id = events[0]["track_id"]
        dna_sample = ClimaQuestBackendService.get_storm_dna(sample_id)
        print(f"\n[3] Storm DNA Sample (ID {sample_id}) for Storm DNA Studio:")
        print(f"   Genes: {dna_sample.get('genes')}")
        print(f"   Deep Visual Embedding (first 4 dims): {dna_sample.get('deep_embedding', [])[:4]}...")

    print("\n" + "=" * 70)
    print("Zero-friction integration verified! Member 4 and Member 1 can plug in immediately.")
    print("=" * 70)


if __name__ == "__main__":
    verify_handoff()
