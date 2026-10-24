"""
ClimaQuest AI — Engine 1: Visual Weather Intelligence Pipeline
Coordinates Detection, Feature Extraction, Tracking, Storm DNA Synthesis,
and Serialization into a unified streaming processing pipeline.
"""

from typing import List, Dict, Any, Optional
import os
import json
import numpy as np

from ..vision.detector import StormDetector
from ..vision.features import FeatureExtractor
from ..vision.tracker import StormTracker, StormTrack
from ..vision.storm_dna import StormDNAGenerator, StormDNA
from ..export.schema import ClimaQuestDataExporter
from ..viz.renderer import MissionControlRenderer


class VisualWeatherIntelligencePipeline:
    """
    Complete end-to-end Computer Vision Engine (Engine 1) for ClimaQuest AI.
    Processes consecutive meteorological frames and outputs structured events + Storm DNA.
    """

    def __init__(
        self,
        envelope_threshold: float = 25.0,
        core_threshold: float = 40.0,
        min_cell_area_px: int = 25,
        resolution_km: float = 2.5,
        dt_hours: float = 0.25,
        max_search_radius_px: float = 60.0,
        enable_watershed: bool = True,
    ):
        self.detector = StormDetector(
            envelope_threshold=envelope_threshold,
            core_threshold=core_threshold,
            min_cell_area_px=min_cell_area_px,
            enable_watershed_split=enable_watershed,
        )
        self.feature_extractor = FeatureExtractor(resolution_km=resolution_km)
        self.tracker = StormTracker(
            max_search_radius_px=max_search_radius_px,
            dt_hours=dt_hours,
            resolution_km=resolution_km,
        )
        self.resolution_km = resolution_km
        self.dt_hours = dt_hours

    def process_frame(
        self,
        grid: np.ndarray,
        timestamp: str,
        frame_idx: int,
        bounds: Optional[Dict[str, float]] = None,
        render_output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes a single meteorological observation grid frame.
        """
        # Step 1: Storm Detection & Segmentation
        candidates = self.detector.detect(grid)

        # Step 2: Feature Extraction for all candidates
        features_list = []
        for cand in candidates:
            feat = self.feature_extractor.extract(
                grid=grid,
                mask=cand.mask,
                contour=cand.contour,
                core_mask=cand.core_mask,
                bounds=bounds,
            )
            features_list.append(feat)

        # Step 3: Multi-Object Spatiotemporal Tracking
        active_tracks = self.tracker.update_frame(
            candidates=candidates,
            features_list=features_list,
            timestamp=timestamp,
        )

        # Step 4: Extract Storm DNA for each active track
        dna_map: Dict[int, StormDNA] = {}
        for track in active_tracks:
            latest_feat = track.history_features[-1]
            dna = StormDNAGenerator.extract_dna(grid, track, latest_feat)
            dna_map[track.track_id] = dna

        # Step 5: Export JSON Summary & GeoJSON
        summary_json = ClimaQuestDataExporter.to_event_summary_json(
            timestamp=timestamp,
            frame_idx=frame_idx,
            tracks=active_tracks,
            dna_map=dna_map,
        )

        geojson = ClimaQuestDataExporter.to_geojson(
            tracks=active_tracks,
            dna_map=dna_map,
            bounds=bounds,
            height=grid.shape[0],
            width=grid.shape[1],
        )

        # Step 6: Render Mission Control Frame (if directory specified)
        rendered_image_path = None
        if render_output_dir:
            os.makedirs(render_output_dir, exist_ok=True)
            rendered_image_path = os.path.join(
                render_output_dir, f"frame_{frame_idx:04d}.png"
            )
            MissionControlRenderer.render_frame(
                grid=grid,
                tracks=active_tracks,
                dna_map=dna_map,
                timestamp=timestamp,
                frame_idx=frame_idx,
                output_path=rendered_image_path,
            )

        return {
            "summary": summary_json,
            "geojson": geojson,
            "active_tracks": active_tracks,
            "dna_map": dna_map,
            "rendered_image_path": rendered_image_path,
        }

    def process_sequence(
        self,
        frames: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Processes a full time sequence of frames.
        """
        results = []
        render_dir = os.path.join(output_dir, "frames") if output_dir else None

        for item in frames:
            res = self.process_frame(
                grid=item["grid"],
                timestamp=item["timestamp"],
                frame_idx=item["frame_idx"],
                bounds=item.get("bounds"),
                render_output_dir=render_dir,
            )
            results.append(res)

        if output_dir:
            # Save complete event ledger JSON
            all_summaries = [r["summary"] for r in results]
            ledger_path = os.path.join(output_dir, "event_tracking_ledger.json")
            with open(ledger_path, "w") as f:
                json.dump(all_summaries, f, indent=2)

            # Save latest GeoJSON
            latest_geojson_path = os.path.join(output_dir, "latest_events.geojson")
            with open(latest_geojson_path, "w") as f:
                json.dump(results[-1]["geojson"], f, indent=2)

        return results
