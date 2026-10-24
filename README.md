# ClimaQuest AI — Engine 1: Visual Weather Intelligence
## Member 2: Computer Vision, Storm Tracking, and Storm DNA
**Smart India Hackathon (SIH) — Technical Architecture & Implementation Guide**

---

## 1. Executive Summary & Role Definition

In **ClimaQuest AI**, Member 2 builds **Engine 1: Visual Weather Intelligence**.
Your primary mission is to transform raw, noisy, high-dimensional meteorological observation grids (Satellite Infrared, Doppler Weather Radar, NASA GPM IMERG precipitation grids) into **discrete, tracked, physically quantified storm entities equipped with a unique "Storm DNA" fingerprint**.

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
                   |        StormDNAGenerator (8-Gene Vector)   |
                   +--------------------------------------------+
                                         |
            +----------------------------+----------------------------+
            |                                                         |
            v                                                         v
+------------------------+                               +------------------------+
| Member 4 (FastAPI)     |                               | Member 3 (ML & Engine) |
| JSON & GeoJSON API     |                               | Reality Gap Engine     |
+------------------------+                               +------------------------+
            |                                                         |
            v                                                         v
+------------------------+                               +------------------------+
| Member 1 (React)       |                               | Future Tree &          |
| MapLibre / Deck.gl     |                               | Watchpoint AI          |
+------------------------+                               +------------------------+
```

---

## 2. Directory Structure

```
climaquest-cv/
├── README.md                      # This comprehensive guide & SIH defense manual
├── requirements.txt               # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic_radar.py     # Realistic 2D meteorological radar simulator
│   │   └── grid_loader.py         # Loader & calibrator for real GeoTIFF / NetCDF / PNG
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── detector.py            # Convective core & rain-shield multi-threshold segmentation
│   │   ├── features.py            # Geometric, physical, and radiometric feature extractor
│   │   ├── tracker.py             # Spatiotemporal Kalman filter + Hungarian association
│   │   └── storm_dna.py           # 8-Gene Storm DNA synthesis & archetype classification
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── engine.py              # End-to-end streaming Visual Weather Intelligence engine
│   ├── export/
│   │   ├── __init__.py
│   │   └── schema.py              # JSON & GeoJSON contracts for Member 4 & Member 1
│   └── viz/
│       ├── __init__.py
│       └── renderer.py            # Dark scientific mission-control visual renderer
├── examples/
│   ├── run_demo.py                # 12-frame end-to-end demonstration script
│   └── benchmark_dna.py           # Research benchmark for Forecast Reality Gap calculation
├── tests/
│   └── test_cv_pipeline.py        # Automated test suite (6 passing unit tests)
└── output/
    ├── frames/                    # Rendered mission-control PNGs (frame_0000.png - frame_0011.png)
    ├── event_tracking_ledger.json # Complete historical tracking telemetry in JSON
    └── latest_events.geojson      # Ready-to-render GeoJSON for Deck.gl / MapLibre
```

---

## 3. The Core Innovations (Why This Wins SIH)

### Innovation 1: Multi-Threshold Convective Core Isolation (Beyond Naive Thresholding)
- Standard CV projects apply a flat binary threshold (`cv2.threshold(img, 127, 255, ...)`). In meteorology, this fails because a storm has a diffuse **stratiform rain envelope** (20–30 dBZ) and tight **convective updraft cores** (> 40 dBZ).
- Our `StormDetector` uses a 2-tier hierarchical strategy inspired by meteorological radar algorithms (TITAN and SCIT):
  1. Isolates the system boundary at the envelope threshold ($25\text{ dBZ}$ or $5\text{ mm/h}$).
  2. Isolates severe convective cores at the core threshold ($40\text{ dBZ}$ or $20\text{ mm/h}$).
  3. Uses watershed distance-transforms to separate coalescing or touching cells.

### Innovation 2: Spatiotemporal Kalman Filter + Hungarian Bipartite Matching
- Storms deform, expand, and drift. Our tracker maintains a continuous state vector $\mathbf{x} = [x, y, v_x, v_y]^T$ for every active track.
- Predicts expected position at $t+1$ and builds a cost matrix combining:
  $$\text{Cost} = 0.45 \cdot D_{\text{spatial}} + 0.25 \cdot (1 - \text{IoU}) + 0.15 \cdot \Delta I + 0.15 \cdot \Delta A$$
- Uses the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) for globally optimal multi-object association.
- Computes meteorological kinematics: ground speed ($km/h$), compass bearing ($0^\circ = \text{North}, 90^\circ = \text{East}$), growth rate ($\%/\text{hr}$), and intensity trend ($\text{dBZ}/\text{hr}$).

### Innovation 3: "Storm DNA" — The Signature Scientific Vector
A standardized 8-dimensional normalized fingerprint vector $\vec{D} \in [0.0, 1.0]^8$:

| Gene | Dimension Name | Physical Meaning | Normalization Formula |
|---|---|---|---|
| $G_1$ | **Spatial Scale** | Storm coverage / footprint | $\min(1.0, \text{Area} / 10000\text{ km}^2)$ |
| $G_2$ | **Convective Vigour** | Peak updraft intensity | $\min(1.0, \max(0.0, (I_{max} - 25) / 50))$ |
| $G_3$ | **Compactness** | Solidity / boundary coherence | $\text{Solidity} = \text{Area} / \text{Convex Hull Area}$ |
| $G_4$ | **Shear Asymmetry** | Elongation along frontal boundary | $\text{Eccentricity} = \sqrt{1 - (b/a)^2}$ |
| $G_5$ | **Core Density** | Severity of convective cores | $\min(1.0, \text{Core Ratio} \times 3.0)$ |
| $G_6$ | **Ground Speed** | Rapid propagation vs stall | $\min(1.0, \text{Speed} / 100\text{ km/h})$ |
| $G_7$ | **Growth Momentum** | Explosive intensification vs decay | $\min(1.0, \max(0.0, 0.5 + \text{Growth Rate} / 100))$ |
| $G_8$ | **Textural Gradient** | Convective bubbling & turbulence | Laplacian variance of core intensity |

#### Meteorological Archetype Classifier
Based on $\vec{D}$, the engine automatically classifies the event into:
- **Severe Supercell**: Vigour $> 0.65$, Core $> 0.40$, Compactness $> 0.70$
- **Squall Line / Linear Multicell**: Asymmetry $> 0.70$, Speed $> 0.35$
- **Mesoscale Convective System (MCS)**: Scale $> 0.60$, Area $> 3000\text{ km}^2$
- **Rapidly Intensifying Pulse Cell**: Growth $> 0.70$, Area $< 1500\text{ km}^2$
- **Decaying Stratiform Shield**: Growth $< 0.30$, Core $< 0.15$

---

## 4. Integration Contracts with Team Members

### How Member 2 Powers Member 4 (Backend API)
Your pipeline writes directly to `output/event_tracking_ledger.json` and `output/latest_events.geojson`. Member 4's FastAPI backend can mount these endpoints directly:
- `GET /api/v1/events/active` $\to$ Returns current tracked storms and Storm DNA summaries.
- `GET /api/v1/events/{track_id}/history` $\to$ Returns trajectory coordinates and growth rate history.
- `GET /api/v1/events/geojson` $\to$ Returns standard GeoJSON FeatureCollection for frontend maps.

### How Member 2 Powers Member 1 (Frontend / UI)
- The GeoJSON polygon coordinates allow Member 1 to render animated storm contours on **MapLibre** or **Deck.gl** with zero custom conversion.
- The `genes` dictionary inside `Storm DNA` provides the exact data points for Member 1's **Storm DNA Studio Radar/Spider Chart**.

### How Member 2 Powers Member 3 (Forecast Reality Gap Engine)
When the numerical weather forecast (e.g. GFS or ECMWF) makes a prediction for $t+1$, Member 3 runs `StormDNAGenerator` on the forecasted precipitation grid.
Member 3 then computes the **Forecast Reality Gap Distance**:
$$\text{Reality Gap} = \|\vec{D}_{\text{Observed}} - \vec{D}_{\text{Forecasted}}\|_2$$
If $\text{Reality Gap} > 0.70$, ClimaQuest AI flags a **Critical Prediction Divergence** and activates Engine 4 (Watchpoint AI) to identify which observational sensors can resolve the uncertainty!

---

## 5. Quickstart & Verification

### Running the Complete Demo
```powershell
# From the climaquest-cv directory
.\.venv\Scripts\python.exe examples/run_demo.py
```

### Running the Research Benchmark (Storm DNA vs Forecast Scenarios)
```powershell
.\.venv\Scripts\python.exe examples/benchmark_dna.py
```

### Running the Unit Test Suite
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_cv_pipeline.py
```

---

## 6. SIH Defense Playbook: Handling Judge Questions

| Tough Question | Your Winning Defense |
|---|---|
| *"Why didn't you just train a standard YOLOv8 or Mask R-CNN model?"* | *"YOLO and Mask R-CNN are trained on bounding boxes of rigid RGB objects (cars, people, chairs). Extreme weather events are continuous, deformable physical fluid fields governed by the Navier-Stokes equations and thermodynamic lapse rates. Training YOLO on arbitrary cloud pictures fails to capture physical rain rates, internal convective updrafts, or volumetric flux. Our approach combines physically calibrated thresholding (TITAN/SCIT) with morphological decomposition and 8-dimensional Storm DNA, making it mathematically verifiable and physically grounded."* |
| *"How do you handle cell splitting and merging?"* | *"Our `StormTracker` builds a continuous bipartite cost matrix using Kalman-predicted position, morphological IoU, area ratio, and intensity delta. When an intensifying supercell undergoes dynamic splitting, the cost matrix identifies multiple candidate cores originating from the parent envelope mask, allowing the system to log a topological 'CELL_SPLIT' event and allocate a daughter track ID while preserving the parent heritage."* |
| *"What happens if real satellite data is cloudy or noisy?"* | *"We use Gaussian spatial pre-filtering and elliptical morphological opening to eliminate sensor speckle noise. Furthermore, by evaluating infrared brightness temperature ($T_B < 210\text{ K}$) and radar reflectivity ($Z > 40\text{ dBZ}$) simultaneously, we distinguish severe towering cumulonimbus clouds with active updrafts from non-severe cirrus shields."* |
| *"Where does the project get real data?"* | *"The system is designed to ingest standard ISRO INSAT-3D/3DR TIR-1 brightness temperature rasters from MOSDAC, NASA GPM IMERG 0.1° half-hourly precipitation grids, and IMD Doppler Weather Radar (DWR) NetCDF/GeoTIFF files via `MeteorologicalGridLoader`."* |
