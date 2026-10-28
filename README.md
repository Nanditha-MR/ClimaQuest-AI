# ClimaQuest AI — Engine 1: Visual Weather Intelligence
## Computer Vision, Spatiotemporal Storm Tracking, and Storm DNA
**Technical Architecture & System Reference Manual**

---

## 1. System Overview

**ClimaQuest AI** is a software-powered extreme-weather intelligence platform designed to investigate the relationship between real-time atmospheric observation and numerical forecast model predictions.

Within ClimaQuest AI, **Engine 1: Visual Weather Intelligence** provides the core perception layer. It ingests high-dimensional meteorological observation grids (Doppler Weather Radar, Geostationary Satellite Infrared, and NASA GPM precipitation fields), segments convective systems, tracks continuous storm lifecycles, and synthesizes a compact multi-dimensional fingerprint called **Storm DNA**.

```
                   +--------------------------------------------+
                   |  Raw Weather Imagery / Grids (IR / Radar)  |
                   +--------------------------------------------+
                                         |
                                         v
                   +--------------------------------------------+
                   |   StormDetector (Multi-Threshold & Core)   |
                   +--------------------------------------------+
                                         |
                                         v
                   +--------------------------------------------+
                   |  FeatureExtractor (Geometry & Radiometry)  |
                   +--------------------------------------------+
                                         |
                                         v
                   +--------------------------------------------+
                   |   StormTracker (Kalman + Hungarian Match)  |
                   +--------------------------------------------+
                                         |
                                         v
                   +--------------------------------------------+
                   |  StormDNAGenerator (8-Gene + Deep Latents) |
                   +--------------------------------------------+
                                         |
            +----------------------------+----------------------------+
            |                                                         |
            v                                                         v
+------------------------+                               +------------------------+
| Backend Data Service   |                               | AI/ML Forecast Module  |
| JSON & GeoJSON API     |                               | Reality Gap Engine     |
+------------------------+                               +------------------------+
            |                                                         |
            v                                                         v
+------------------------+                               +------------------------+
| Mission Control UI     |                               | Scenario Calibration & |
| MapLibre / Deck.gl     |                               | Watchpoint AI          |
+------------------------+                               +------------------------+
```

---

## 2. Repository Structure

```
climaquest-cv/
├── README.md                      # Technical documentation and architecture reference
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git exclusion rules
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic_radar.py     # 2D meteorological radar & precipitation grid generator
│   │   └── grid_loader.py         # Calibrator & loader for GeoTIFF / NetCDF / PNG rasters
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── detector.py            # Dual-threshold convective core & envelope segmentation
│   │   ├── features.py            # Geometric, physical, and radiometric feature extractor
│   │   ├── tracker.py             # Spatiotemporal Kalman filter + Hungarian association
│   │   ├── cnn_embedder.py        # 16-dimensional multi-scale convolutional structural embedder
│   │   └── storm_dna.py           # Hybrid Storm DNA synthesis & archetype classification
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── engine.py              # End-to-end streaming Visual Weather Intelligence engine
│   ├── export/
│   │   ├── __init__.py
│   │   └── schema.py              # Standardized JSON & GeoJSON contracts
│   └── viz/
│       ├── __init__.py
│       └── renderer.py            # Mission-control scientific visual renderer
├── examples/
│   ├── run_demo.py                # 12-frame end-to-end demonstration script
│   ├── benchmark_dna.py           # Forecast Reality Gap benchmark & scenario comparison
│   ├── api_bridge_for_member4.py  # Backend service bridge and endpoint simulation
│   └── make_animation.py          # Mission control animated GIF compiler
├── tests/
│   └── test_cv_pipeline.py        # Automated unit test suite (6 passing tests)
└── output/
    ├── frames/                    # Rendered mission-control PNG frames
    ├── event_tracking_ledger.json # Historical tracking telemetry ledger in JSON
    ├── latest_events.geojson      # Standard GeoJSON FeatureCollection for mapping
    └── mission_control_surveillance.gif # Looping surveillance animation
```

---

## 3. Core Scientific Innovations

### 3.1 Multi-Threshold Convective Core Isolation (TITAN / SCIT Methodology)
In meteorology, simple binary thresholding fails because convective storms consist of an extensive **stratiform rain envelope** (20–30 dBZ) surrounding compact, energetic **convective updraft cores** (> 40 dBZ).
- **Dual-Threshold Segmentation**: Separates the outer system envelope ($25\text{ dBZ}$ / $5\text{ mm/h}$) from intense convective updrafts ($42\text{ dBZ}$ / $20\text{ mm/h}$).
- **Morphological Filtering**: Applies elliptical structuring elements to filter out radar clutter and sensor speckle while preserving fluid cloud geometry.
- **Watershed Decomposition**: Employs distance transforms and seeded watershed segmentation to cleanly isolate distinct convective cells sharing a common rain shield.

### 3.2 Spatiotemporal Kalman Filter + Hungarian Bipartite Matching
Convective storms are deformable, non-rigid atmospheric phenomena that advect, grow, and decay.
- **State Estimation**: Tracks maintain a continuous state vector $\mathbf{x} = [x, y, v_x, v_y]^T$ via a discrete linear Kalman filter predicting movement at $t+1$.
- **Cost Formulation**: Combines spatial displacement, boundary IoU, intensity deviation, and area consistency:
  $$\text{Cost} = 0.45 \cdot D_{\text{spatial}} + 0.25 \cdot (1 - \text{IoU}) + 0.15 \cdot \Delta I + 0.15 \cdot \Delta A$$
- **Optimal Assignment**: Solves the global assignment matrix via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).
- **Kinematic Derivations**: Computes ground speed ($km/h$), compass bearing ($0^\circ = \text{North}, 90^\circ = \text{East}$), area growth rate ($\%/\text{hr}$), and intensity trend ($\text{dBZ}/\text{hr}$).

### 3.3 "Storm DNA" — Hybrid Meteorological Fingerprinting
Each tracked event is distilled into a standardized 8-dimensional normalized physical vector $\vec{D} \in [0.0, 1.0]^8$ plus a 16-dimensional deep visual structural embedding:

| Gene | Dimension | Meteorological Significance | Mathematical Formulation |
|---|---|---|---|
| $G_1$ | **Spatial Scale** | Total footprint / coverage | $\min(1.0, \text{Area} / 10000\text{ km}^2)$ |
| $G_2$ | **Convective Vigour** | Peak updraft intensity | $\min(1.0, \max(0.0, (I_{max} - 25) / 50))$ |
| $G_3$ | **Compactness** | Solidity and edge coherence | $\text{Solidity} = \text{Area} / \text{Convex Hull Area}$ |
| $G_4$ | **Shear Asymmetry** | Elongation along frontal boundary | $\text{Eccentricity} = \sqrt{1 - (b/a)^2}$ |
| $G_5$ | **Core Density** | Severity of convective updraft cores | $\min(1.0, \text{Core Ratio} \times 3.0)$ |
| $G_6$ | **Ground Speed** | Velocity of forward propagation | $\min(1.0, \text{Speed} / 100\text{ km/h})$ |
| $G_7$ | **Growth Momentum** | Explosive intensification rate | $\min(1.0, \max(0.0, 0.5 + \text{Growth Rate} / 100))$ |
| $G_8$ | **Textural Gradient** | Convective bubbling & turbulence | Laplacian variance of core intensity |

#### Deep Visual Structural Embedding
To capture complex cloud spiral bands and texture patterns, `DeepStormEmbedder` uses multi-scale, multi-orientation convolutional spatial filter banks (evaluating 4 orientations $\times$ 4 wavelengths) with global average pooling to yield a normalized 16-dimensional latent representation.

#### Meteorological Archetype Classifier
Based on the synthesized DNA, events are automatically classified into recognized meteorological archetypes:
- **Severe Supercell**: Vigour $> 0.65$, Core $> 0.40$, Compactness $> 0.70$
- **Squall Line / Linear Multicell**: Asymmetry $> 0.70$, Speed $> 0.35$
- **Mesoscale Convective System (MCS)**: Scale $> 0.60$, Area $> 3000\text{ km}^2$
- **Rapidly Intensifying Pulse Cell**: Growth $> 0.70$, Area $< 1500\text{ km}^2$
- **Decaying Stratiform Shield**: Growth $< 0.30$, Core $< 0.15$

---

## 4. System Integration & Data Contracts

### 4.1 Backend Service Interface (FastAPI Integration)
The tracking pipeline exports directly to `output/event_tracking_ledger.json` and `output/latest_events.geojson`. A backend service can expose these endpoints directly:
- `GET /api/v1/surveillance/active-storms` $\to$ Returns current active events, kinematics, and Storm DNA.
- `GET /api/v1/surveillance/geojson` $\to$ Returns standard GeoJSON FeatureCollections for map overlays.
- `GET /api/v1/surveillance/dna/{track_id}` $\to$ Returns normalized genes, deep embeddings, and archetype.

### 4.2 Frontend Visualization Interface (MapLibre / Deck.gl)
- GeoJSON polygon coordinates are pre-formatted with longitude/latitude coordinates for immediate rendering in mapping libraries.
- The `genes` dictionary inside Storm DNA maps directly to radar/spider chart visualization components.

### 4.3 Forecast Reality Gap Integration
When numerical weather prediction models (e.g., GFS, ECMWF) issue forecast grids for $t+1$, the forecast evaluation engine runs `StormDNAGenerator` on the forecasted grid and calculates the **Forecast Reality Gap Distance**:
$$\text{Reality Gap} = \|\vec{D}_{\text{Observed}} - \vec{D}_{\text{Forecasted}}\|_2$$
A Reality Gap distance exceeding $0.70$ signals significant forecast divergence, triggering automated scenario reassessment.

---

## 5. Quickstart & Verification

### Running the End-to-End Pipeline Demo
```powershell
python examples/run_demo.py
```
*Processes 12 consecutive meteorological observation frames, tracks active cells, extracts Storm DNA, and saves output frames to `output/frames/`.*

### Running the Forecast Reality Gap Benchmark
```powershell
python examples/benchmark_dna.py
```
*Validates Storm DNA distance calculations against competing forecast model scenarios.*

### Running the Backend Integration Bridge
```powershell
python examples/api_bridge_for_member4.py
```
*Verifies active storm retrieval, 16-dimensional deep embeddings, and GeoJSON polygon generation.*

### Compiling the Animated Surveillance GIF
```powershell
python examples/make_animation.py
```
*Generates `output/mission_control_surveillance.gif` from rendered frames.*

### Running the Automated Unit Test Suite
```powershell
python -m unittest tests/test_cv_pipeline.py
```
*Executes all unit tests verifying detection, tracking, feature calculation, and export contracts.*

---

## 6. Technical FAQ & Architectural Decisions

| Challenge / Question | Engineering & Scientific Rationale |
|---|---|
| **Why not rely strictly on object detection networks (e.g. YOLO)?** | Bounding-box object detectors are designed for rigid, discrete RGB objects. Severe weather systems are continuous, deformable physical fluid fields governed by thermodynamics and fluid dynamics. Our approach pairs physically calibrated dual-thresholding (TITAN/SCIT) with morphological decomposition and 8-dimensional Storm DNA, making results mathematically verifiable and directly comparable to numerical forecast grids. |
| **How are cell splitting and merging dynamics handled?** | The `StormTracker` constructs a continuous bipartite cost matrix combining Kalman-predicted locations, boundary IoU, area ratios, and intensity deltas. During cell splitting, the matrix associates candidate cores originating from the parent mask, enabling the logging of topological events while preserving lineage. |
| **How is sensor noise and atmospheric clutter suppressed?** | The pipeline applies spatial Gaussian smoothing and elliptical morphological opening to eliminate high-frequency clutter. Cross-referencing radiometric thresholds ensures stratiform cloud shields are differentiated from convective cores. |
| **What real-world data sources are supported?** | `MeteorologicalGridLoader` supports satellite infrared rasters (e.g. INSAT-3D/3DR TIR-1 brightness temperatures), NASA GPM IMERG precipitation grids, and Doppler Weather Radar (DWR) reflectivity products in GeoTIFF, NetCDF, or array formats. |
