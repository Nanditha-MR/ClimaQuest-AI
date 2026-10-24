"""
Unit tests for ClimaQuest AI Computer Vision & Storm Tracking Pipeline (Member 2).
"""

import unittest
import numpy as np
import os
import sys

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic_radar import MeteorologicalSequenceGenerator
from src.vision.detector import StormDetector
from src.vision.features import FeatureExtractor
from src.vision.tracker import StormTracker
from src.vision.storm_dna import StormDNAGenerator
from src.pipeline.engine import VisualWeatherIntelligencePipeline


class TestClimaQuestCVPipeline(unittest.TestCase):

    def setUp(self):
        self.generator = MeteorologicalSequenceGenerator(height=200, width=200, seed=42)
        self.frames = self.generator.generate_scenario(num_frames=5)

    def test_synthetic_data_generation(self):
        """Verify simulated frames have correct shape, values, and timestamps."""
        self.assertEqual(len(self.frames), 5)
        grid0 = self.frames[0]["grid"]
        self.assertEqual(grid0.shape, (200, 200))
        self.assertGreaterEqual(grid0.min(), 0.0)
        self.assertLessEqual(grid0.max(), 75.0)

    def test_storm_cell_detector(self):
        """Verify multi-threshold detection finds convective storms."""
        detector = StormDetector(envelope_threshold=25.0, core_threshold=40.0, min_cell_area_px=20)
        grid0 = self.frames[0]["grid"]
        cells = detector.detect(grid0)
        self.assertGreater(len(cells), 0, "Detector should locate convective storms in test grid.")

        for c in cells:
            self.assertGreaterEqual(c.mask.sum(), 20 * 255)
            self.assertTrue(c.contour is not None)

    def test_feature_extractor(self):
        """Verify physical and geometric feature calculations."""
        detector = StormDetector(envelope_threshold=25.0, core_threshold=40.0, min_cell_area_px=20)
        extractor = FeatureExtractor(resolution_km=2.5)

        grid0 = self.frames[0]["grid"]
        cells = detector.detect(grid0)
        feat = extractor.extract(grid0, cells[0].mask, cells[0].contour, cells[0].core_mask)

        self.assertGreater(feat.area_km2, 0.0)
        self.assertGreater(feat.max_intensity, 25.0)
        self.assertGreaterEqual(feat.solidity, 0.0)
        self.assertLessEqual(feat.solidity, 1.0)
        self.assertGreaterEqual(feat.circularity, 0.0)
        self.assertLessEqual(feat.circularity, 1.0)

    def test_tracking_and_kinematics(self):
        """Verify tracking maintains persistent identity across sequential frames."""
        tracker = StormTracker(max_search_radius_px=60.0, dt_hours=0.25, resolution_km=2.5)
        detector = StormDetector(envelope_threshold=25.0, core_threshold=40.0, min_cell_area_px=20)
        extractor = FeatureExtractor(resolution_km=2.5)

        for f in self.frames:
            grid = f["grid"]
            ts = f["timestamp"]
            cands = detector.detect(grid)
            feats = [extractor.extract(grid, c.mask, c.contour, c.core_mask) for c in cands]
            active = tracker.update_frame(cands, feats, ts)

        # After 5 frames, persistent tracks should exist
        self.assertGreater(len(active), 0)
        t0 = active[0]
        self.assertGreaterEqual(t0.age_frames, 2)
        self.assertGreater(t0.speed_kmh, 0.0)
        self.assertGreaterEqual(t0.bearing_deg, 0.0)
        self.assertLess(t0.bearing_deg, 360.0)

    def test_storm_dna_synthesis(self):
        """Verify Storm DNA 8-gene normalized vector properties."""
        pipeline = VisualWeatherIntelligencePipeline()
        res = pipeline.process_sequence(self.frames)
        self.assertEqual(len(res), 5)

        last_res = res[-1]
        self.assertGreater(len(last_res["active_tracks"]), 0)

        for track_id, dna in last_res["dna_map"].items():
            self.assertEqual(len(dna.vector), 8)
            self.assertTrue(np.all(dna.vector >= 0.0))
            self.assertTrue(np.all(dna.vector <= 1.0))
            self.assertIn(dna.archetype, [
                "Severe Supercell",
                "Squall Line / Linear Multicell",
                "Mesoscale Convective System (MCS)",
                "Rapidly Intensifying Pulse Cell",
                "Decaying Stratiform Shield",
                "Isolated Convective Cluster",
            ])
            self.assertIn("genes", dna.to_dict())

    def test_data_export_schema(self):
        """Verify JSON and GeoJSON outputs conform to Member 4 contract."""
        pipeline = VisualWeatherIntelligencePipeline()
        res = pipeline.process_sequence(self.frames)
        last_summary = res[-1]["summary"]
        last_geojson = res[-1]["geojson"]

        self.assertIn("active_event_count", last_summary)
        self.assertIn("events", last_summary)
        self.assertEqual(last_geojson["type"], "FeatureCollection")
        self.assertGreater(len(last_geojson["features"]), 0)


if __name__ == "__main__":
    unittest.main()
