"""
ClimaQuest AI — Export & Schema Module
Defines standardized JSON and GeoJSON data contracts for Member 4 (FastAPI Backend),
Member 1 (Deck.gl/React Frontend), and Member 3 (Forecast Reality Gap Engine).
"""

from typing import List, Dict, Any, Optional
import json
import numpy as np
from ..vision.tracker import StormTrack
from ..vision.storm_dna import StormDNA


class ClimaQuestDataExporter:
    """
    Serializes tracked storm events, kinematics, contours, and Storm DNA
    into industry-standard GeoJSON and REST-ready JSON payloads.
    """

    @staticmethod
    def to_event_summary_json(
        timestamp: str,
        frame_idx: int,
        tracks: List[StormTrack],
        dna_map: Dict[int, StormDNA],
    ) -> Dict[str, Any]:
        """
        Generates master frame summary for Backend API (Member 4)
        and Forecast Reality Gap Engine (Member 3).
        """
        events = []
        for t in tracks:
            dna = dna_map.get(t.track_id)
            latest_feat = t.history_features[-1]

            event_record = {
                "track_id": t.track_id,
                "status": t.status,
                "age_frames": t.age_frames,
                "spatial": {
                    "centroid_px": [round(c, 2) for c in latest_feat.centroid_px],
                    "centroid_geo": {
                        "lon": round(latest_feat.lon, 4) if latest_feat.lon else None,
                        "lat": round(latest_feat.lat, 4) if latest_feat.lat else None,
                    },
                    "bbox_px": list(latest_feat.bbox_px),
                    "area_km2": round(latest_feat.area_km2, 1),
                    "perimeter_km": round(latest_feat.perimeter_km, 1),
                    "solidity": round(latest_feat.solidity, 3),
                    "circularity": round(latest_feat.circularity, 3),
                    "eccentricity": round(latest_feat.eccentricity, 3),
                    "aspect_ratio": round(latest_feat.aspect_ratio, 2),
                },
                "kinematics": {
                    "velocity_kmh": [round(v, 2) for v in t.velocity_kmh],
                    "speed_kmh": round(t.speed_kmh, 2),
                    "bearing_deg": round(t.bearing_deg, 1),
                    "growth_rate_pct_hr": round(t.growth_rate_pct_hr, 1),
                    "intensity_trend_dbz_hr": round(t.intensity_trend_dbz_hr, 1),
                },
                "radiometric": {
                    "max_intensity": round(latest_feat.max_intensity, 1),
                    "mean_intensity": round(latest_feat.mean_intensity, 1),
                    "p90_intensity": round(latest_feat.p90_intensity, 1),
                    "core_area_km2": round(latest_feat.core_area_km2, 1),
                    "core_ratio": round(latest_feat.core_ratio, 3),
                },
                "storm_dna": dna.to_dict() if dna else None,
            }
            events.append(event_record)

        return {
            "timestamp": timestamp,
            "frame_idx": frame_idx,
            "active_event_count": len(events),
            "events": events,
        }

    @staticmethod
    def to_geojson(
        tracks: List[StormTrack],
        dna_map: Dict[int, StormDNA],
        bounds: Optional[Dict[str, float]] = None,
        height: int = 400,
        width: int = 400,
    ) -> Dict[str, Any]:
        """
        Converts active storm contours and motion vectors to standard GeoJSON FeatureCollection
        optimized for Deck.gl / MapLibre (Member 1).
        """
        features = []

        for t in tracks:
            dna = dna_map.get(t.track_id)
            latest_feat = t.history_features[-1]

            # Convert contour points to Geo coordinates
            contour_pts = t.latest_contour.squeeze()
            if contour_pts.ndim == 1:
                contour_pts = contour_pts.reshape(-1, 2)

            geo_coords = []
            for pt in contour_pts:
                px, py = float(pt[0]), float(pt[1])
                if bounds:
                    lon = bounds["lon_min"] + (px / (width - 1)) * (bounds["lon_max"] - bounds["lon_min"])
                    lat = bounds["lat_max"] - (py / (height - 1)) * (bounds["lat_max"] - bounds["lat_min"])
                else:
                    lon, lat = px, py
                geo_coords.append([round(lon, 5), round(lat, 5)])

            # Close polygon ring
            if len(geo_coords) > 2:
                geo_coords.append(geo_coords[0])

            feature = {
                "type": "Feature",
                "properties": {
                    "track_id": t.track_id,
                    "status": t.status,
                    "archetype": dna.archetype if dna else "Unknown",
                    "speed_kmh": round(t.speed_kmh, 1),
                    "bearing_deg": round(t.bearing_deg, 1),
                    "area_km2": round(latest_feat.area_km2, 1),
                    "max_intensity": round(latest_feat.max_intensity, 1),
                    "growth_rate_pct_hr": round(t.growth_rate_pct_hr, 1),
                    "dna_vector": dna.vector.tolist() if dna else [],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geo_coords],
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
        }
