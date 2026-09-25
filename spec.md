Canopy: Technical Build Spec
Sep 25, 2026 · @Ray Flanagan
Overview
Canopy is a 24-hour hackathon build: a simulated drone swarm autonomously maps a procedurally generated house, finds the electric meter, and outputs a battery placement decision plus an auto-captured Site Survey Review (SSR) photo packet.
Problem. SSR needs standardized photos of each home (meter, proposed battery site, conduit route, blockers). Customers take them badly, causing rework.
Goals (in priority order)
1. A visually compelling, working multi-drone mapping demo: the world starts grayscale and turns to true color as drones observe it.
2. Swarm coordination that is robust: frontier exploration with live reassignment when a drone is killed.
3. Meter detection, then a ranked battery site, conduit route, and bush removal list.
4. An SSR packet (JSON + HTML with photos) generated at mission end.
Non-goals. Real drone hardware, photogrammetry, flight-dynamics fidelity, production-grade detection accuracy, real Tesla SSR rules (a YAML placeholder stands in).
Key architecture decisions (do not revisit without cause)
• Classical planning is the critical path: frontier exploration, Hungarian assignment, grid path planning, and a safety shield. RL is optional and shielded.
• Sensing is ray casting (Open3D RaycastingScene) against the scene meshes, not the physics engine's camera renderer. Each ray returns distance, object ID and triangle ID, so semantics are free.
• Default motion model is kinematic. PyBullet physics is an optional flag.
• Battery placement is a constrained optimization over a 2D cost map, not a classifier. Only the meter uses a detector, and only as a stretch goal.
• One scene manifest is the single source of truth for physics, sensing, and visualization.
• The product output is the SSR packet; the 3D map is the means.
Tech stack and environment
Python 3.10 in a conda-forge environment; the same environment.yml works on Windows, macOS (Apple Silicon) and Linux. Primary dev machine is Windows.
Technology
Used for
gym-pybullet-drones
Forked swarm sim: quadcopter dynamics, PID control, multi-drone, RL scaffold (physics mode only)
PyBullet (conda-forge)
Physics engine under gym-pybullet-drones; prebuilt, so no MSVC compile on Windows
Open3D RaycastingScene
360 depth sensor and photo camera; returns distance, object ID, triangle ID per ray
trimesh + shapely + mapbox-earcut
Procedural house, bush, tree and meter meshes; footprint polygons
numpy
Voxel occupancy grid, ray sampling, coverage
scipy.ndimage
Frontier detection and clustering
scipy.optimize linear_sum_assignment
Hungarian drone-to-frontier assignment
scikit-image MCP_Geometric
3D flight path planning and 2D conduit routing
Rerun (rerun-sdk)
Live 3D visualization and .rrd recordings
Jinja2 + matplotlib
SSR packet HTML and top-down site plan
PyYAML
rules.yaml placement constraints and weights
Ultralytics YOLO (stretch)
Meter detector
Stable-Baselines3 (stretch)
PPO for the optional local-avoidance policy
environment.yml
name: canopy
channels: [conda-forge]
dependencies:
  - python=3.10
  - pybullet
  - numpy
  - scipy
  - scikit-image
  - shapely
  - pyyaml
  - jinja2
  - matplotlib
  - pytest
  - pip
  - pip:
      - open3d
      - trimesh
      - mapbox-earcut
      - rerun-sdk
Setup
conda env create -f environment.yml
conda activate canopy
git clone https://github.com/utiasDSL/gym-pybullet-drones third_party/gym-pybullet-drones
pip install -e third_party/gym-pybullet-drones
• If the last step tries to compile pybullet, rerun with --no-deps and install missing imports by hand.
• NumPy ABI error on import: pip install "numpy<2".
• Windows spawns processes instead of forking: every entry point uses if __name__ == "__main__":, and each worker builds its own Open3D scene (not picklable).
• Skip the repo's Betaflight / Crazyflie firmware SITL steps.
• Add .gitattributes with * text=auto.
• Launch the viewer with rerun --memory-limit 4GB on a 16 GB laptop.
• Fallback if native Windows setup fails: WSL2 Ubuntu 22.04, with the Rerun viewer run natively on Windows and networkingMode=mirrored in .wslconfig.
Repo structure and entry points
One Python package, canopy, with one subpackage per module; every module is testable alone against a generated scene.
canopy/
  environment.yml
  config/
    default.yaml          # all tunables (see Config defaults)
    rules.yaml            # site placement constraints and weights
  canopy/
    contracts.py          # shared dataclasses (Data contracts)
    worldgen/
      house.py            # footprint, walls, roof, garage, doors, windows
      yard.py             # meter, bushes, trees, fence, driveway, distractors
      generate.py         # seed -> SceneManifest + OBJ files
    sim/
      scene.py            # loads manifest into Open3D (and PyBullet if enabled)
      sensors.py          # 360 ray sensor, pinhole photo camera
      dynamics.py         # KinematicDynamics, PyBulletDynamics (same interface)
      world.py            # SimWorld: step(), drones, kill_drone()
    mapping/
      occupancy.py        # voxel grid updates
      coverage.py         # per-triangle observed mask
      semantics.py        # discovered objects
    planning/
      frontier.py         # frontier extraction + clustering + viewpoints
      assign.py           # Hungarian assignment
      pathing.py          # MCP_Geometric 3D paths, path smoothing
      mission.py          # state machine + per-drone behavior
      safety.py           # shield checks
    perception/
      detector.py         # GTDetector, YoloDetector (same interface)
    site/
      costmap.py          # 2D grid layers
      solver.py           # candidates, scoring, conduit, bush removal
    report/
      photos.py           # SSR shot list and viewpoint computation
      packet.py           # JSON + HTML (Jinja2), site plan PNG
      templates/packet.html.j2
    viz/
      rerun_log.py        # all Rerun logging lives here
    rl/                   # optional track
  scripts/
    gen_scene.py
    run_mission.py
    batch_eval.py
  tests/
  out/                    # generated scenes, recordings, packets (gitignored)
CLI entry points
• python scripts/gen_scene.py --seed 42 --out out/scenes/42 writes manifest.json and meshes/*.obj.
• python scripts/run_mission.py --seed 42 --drones 4 --dynamics kinematic --detector gt --viz live runs a full mission and writes out/runs/<seed>_<n>/ (packet, .rrd, metrics.json).
• python scripts/batch_eval.py --seeds 0-29 --drones 1,2,4,6 --workers 4 runs headless and writes out/eval/summary.csv.
• Common flags: --config config/default.yaml, --kill-drone <id>@<t_seconds>, --no-viz, --record <path.rrd>.
Data contracts
All cross-module data goes through these dataclasses in canopy/contracts.py; write them first and do not change field names without updating every consumer. Units are meters, seconds, radians; world frame is Z-up, origin at lot center, ground at z=0.
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

class Cls(IntEnum):
    GROUND = 0; DRIVEWAY = 1; WALL = 2; ROOF = 3; DOOR = 4; WINDOW = 5
    GARAGE_DOOR = 6; METER = 7; BUSH = 8; TREE = 9; FENCE = 10
    AC_UNIT = 11; GAS_METER = 12

@dataclass
class SceneObject:
    obj_id: int                 # unique, == Open3D geometry id order
    cls: Cls
    mesh_path: str              # OBJ in world coordinates (pose baked in)
    color: tuple[int, int, int] # true RGB for reveal + photos
    wall_normal: np.ndarray | None = None  # for meter/door/window: outward normal

@dataclass
class SceneManifest:
    seed: int
    lot_bounds: np.ndarray      # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    footprint: list[tuple[float, float]]   # house outline polygon (CCW)
    objects: list[SceneObject]
    home: np.ndarray            # (3,) launch pad position
    gt_meter_id: int            # for evaluation only, never read by planner

@dataclass
class DroneState:
    drone_id: int
    pos: np.ndarray             # (3,)
    vel: np.ndarray             # (3,)
    yaw: float
    alive: bool = True
    battery: float = 1.0        # 0..1, simple linear drain

@dataclass
class Scan:
    drone_id: int
    t: float
    origin: np.ndarray          # (3,)
    dirs: np.ndarray            # (N,3) unit
    dist: np.ndarray            # (N,) inf on miss
    obj_ids: np.ndarray         # (N,) int, -1 on miss
    tri_ids: np.ndarray         # (N,) int, -1 on miss (global triangle index)

@dataclass
class DiscoveredObject:
    obj_id: int
    cls: Cls
    pos: np.ndarray             # mean hit point
    n_hits: int
    confidence: float           # 1.0 for GT detector
    first_seen_t: float
    first_seen_by: int

@dataclass
class MapState:
    occ: np.ndarray             # (X,Y,Z) uint8: 0 UNKNOWN, 1 FREE, 2 OCC
    origin: np.ndarray          # world pos of voxel (0,0,0)
    voxel: float
    tri_seen: np.ndarray        # (T,) bool
    discovered: dict[int, DiscoveredObject] = field(default_factory=dict)

@dataclass
class DroneTask:
    drone_id: int
    kind: str                   # 'orbit' | 'frontier' | 'inspect' | 'rth' | 'hold'
    goal: np.ndarray            # (3,)
    look_at: np.ndarray | None
    path: np.ndarray            # (K,3) waypoints

@dataclass
class SiteCandidate:
    pos: np.ndarray             # (3,) ground point at wall
    wall_normal: np.ndarray
    cost: float
    breakdown: dict[str, float] # conduit_m, bushes, penalties...
    conduit: np.ndarray         # (K,3) polyline meter -> site
    bushes_to_remove: list[int]

@dataclass
class Photo:
    label: str                  # e.g. 'meter_close'
    path: str
    cam_pos: np.ndarray
    look_at: np.ndarray

@dataclass
class SiteResult:
    meter: DiscoveredObject | None
    candidates: list[SiteCandidate]   # sorted, best first, max 3
    photos: list[Photo]
    coverage_ground_band: float
    mission_time_s: float
• Global triangle index: triangles of all objects concatenated in manifest order; keep an obj_tri_offset array to map tri_id back to an object.
• The site solver may read SceneManifest only in --oracle debug mode; in normal runs it reads MapState.discovered only.
Module 1: World gen
generate(seed, cfg) -> SceneManifest builds a deterministic random property from one numpy.random.default_rng(seed) and writes one OBJ per object with its pose baked in.
Lot and ground
• Lot 30 x 40 m, centered at origin. Ground = one plane mesh (class GROUND), subdivided to 0.5 m triangles so the reveal reads.
• Driveway: 3.5 x 10 m flat box 2 cm tall from lot edge to garage door.
• Fence: 1.8 m tall, 5 cm thick walls on 3 lot edges (front open).
House
• Footprint: union of 1 to 3 axis-aligned rectangles (shapely unary_union), each side 8 to 16 m; reject if concave angles < 90 degrees or area > 250 m². Extrude with trimesh.creation.extrude_polygon.
• Wall height 3.0 m (1 story) or 5.8 m (2 story), 50/50.
• Roof: gable per rectangle (triangular prism, ridge along long axis, pitch 30 degrees). Flat roof allowed at 20%.
• Garage: attached 6 x 6 m box on a random side facing the driveway, with a GARAGE_DOOR panel.
• Doors: 1 front + 0 to 1 back, 1.0 x 2.1 m panels offset 2 cm from wall. Windows: 0.9 x 1.2 m panels, sill at 0.9 m, spaced every 3 to 4 m on each wall.
• Subdivide all surfaces to max edge 0.2 m (trimesh.remesh.subdivide_to_size) for a smooth gray-to-color reveal. Budget: total triangles under 250k.
Meter and distractors
• METER: 0.3 w x 0.15 d x 0.45 h m box (+ 0.2 m cylinder face) centered at 1.5 m height on a random exterior wall segment, at least 1 m from corners, doors and windows. Store wall_normal.
• GAS_METER (60% chance): 0.3 m box at 0.5 m height, different wall, same shape family as meter so detection is non-trivial.
• AC_UNIT (80% chance): 0.8 m cube on ground, 0.5 m from a wall.
Vegetation
• BUSH: icosphere (subdivisions 2) scaled to ellipsoid 0.6 to 1.6 m wide, 0.5 to 1.4 m tall, vertices jittered by 5 to 10% noise. Place along the foundation band 0.3 to 1.2 m from walls, density bush_density bushes per 10 m of wall (default 2.0).
• p_meter_occluded (default 0.5): force a bush directly in front of the meter, 0.6 to 1.0 m from the wall.
• TREE: trunk cylinder r=0.15 m, h=2.5 to 4 m + crown icosphere r=1.5 to 3 m; 2 to 5 trees, at least 2 m from house; at least one crown within 1.5 m of the roof edge (real flight obstacle).
Colors (true colors for reveal)
Class
RGB
GROUND
96, 140, 70
DRIVEWAY
150, 150, 150
WALL
225, 205, 170
ROOF
120, 60, 50
DOOR / GARAGE_DOOR
110, 75, 45
WINDOW
120, 170, 210
METER
230, 200, 30
GAS_METER
200, 120, 30
AC_UNIT
180, 185, 190
BUSH
40, 110, 40
TREE
70, 130, 50 (crown), 100, 70, 40 (trunk)
FENCE
160, 130, 100
Outputs. out/scenes/<seed>/manifest.json (numpy arrays as lists) + meshes/<obj_id>_<cls>.obj. Add a load_manifest(path) helper. Same seed must produce byte-identical files.
Module 2: Sim + sensors
SimWorld owns time, drone states, dynamics and sensors; it runs at a fixed 20 Hz control tick with sensing at 5 Hz, faster than real time when --no-viz.
Scene loading. Load every OBJ into an Open3D o3d.t.geometry.RaycastingScene via add_triangles, in manifest order, so the returned geometry id equals obj_id. Build obj_tri_offset for global triangle ids. In physics mode, also load each OBJ into PyBullet as a static concave collision body.
Dynamics interface
class Dynamics(Protocol):
    def reset(self, states: list[DroneState]) -> None: ...
    def step(self, targets: dict[int, np.ndarray], dt: float) -> list[DroneState]: ...
• KinematicDynamics (default): move toward the current waypoint target at v_max = 3.0 m/s with a_max = 4.0 m/s²; yaw turns toward velocity at 90 deg/s. No physics.
• PyBulletDynamics (flag --dynamics pybullet): gym-pybullet-drones CtrlAviary + DSLPIDControl, DIRECT mode (no GUI), targets = waypoint positions. Must match the same interface.
360 sensor
• Equirectangular ray grid: 180 azimuth x 60 elevation (elevation -75 to +30 degrees), 10,800 rays per scan.
• Max range 12 m; misses set dist = inf, ids = -1.
• cast_rays returns t_hit, geometry_ids, primitive_ids, primitive_normals; map to Scan.
• Add 1 cm Gaussian range noise (config flag, default on) so the map is not suspiciously perfect.
Photo camera
• render_photo(cam_pos, look_at, fov=60, w=640, h=480) -> np.ndarray (H,W,3): pinhole rays from create_rays_pinhole, shade = class color x (0.35 + 0.65 * max(0, n·l)) with a fixed sun direction, sky color for misses.
• Also return per-pixel obj_ids so the GT detector and YOLO auto-labeling can use it.
SimWorld API
class SimWorld:
    def __init__(self, manifest, n_drones, cfg): ...
    def step(self, tasks: dict[int, DroneTask]) -> list[Scan]  # scans only on sensor ticks
    def kill_drone(self, drone_id): ...   # alive=False, drone lands in place
    def photo(self, cam_pos, look_at) -> tuple[np.ndarray, np.ndarray]
    @property
    def t(self) -> float
    @property
    def drones(self) -> list[DroneState]
• Drones spawn on a 1 m spaced line at manifest.home, z = 0, and take off to 2 m.
• Collision check (ground truth): if a drone position is inside or within 0.25 m of any mesh (Open3D compute_distance), log a COLLISION event. Used by tests and eval, never by the planner.
• Battery drains 1% per 10 s of flight (config); below 15% triggers RTH.
Module 3: Mapper
Mapper.integrate(scan) -> None fuses every scan into one shared MapState: occupancy for planning, triangle coverage for the reveal and metrics, discovered objects for the site solver.
Occupancy grid
• Voxel 0.25 m over lot bounds x height 0 to 12 m: 120 x 160 x 48 = 921,600 voxels, uint8.
• Per scan, fully vectorized numpy (no Python loops over rays):
    1. Subsample to every 2nd ray for free-space carving (5,400 rays).
    2. Sample points along each ray at 0.2 m steps up to min(dist, 12) - 0.3 and mark voxels FREE, but never downgrade OCC to FREE.
    3. Mark voxel of each hit point OCC.
• Keep an occ_changed flag so frontier extraction only reruns when the map changed.
Coverage
• tri_seen[scan.tri_ids[scan.tri_ids >= 0]] = True, but only for hits with range under 8 m and incidence angle under 70 degrees (photo-quality rule).
• Precompute per-triangle area and centroid z. Metrics:
    ◦ coverage_total = seen area / total area of non-ground triangles.
    ◦ coverage_ground_band = seen area / total area of WALL, DOOR, METER, BUSH triangles with centroid z in 0 to 2.5 m. This is the mission completion metric.
Semantics
• Accumulate per obj_id hit count and running mean hit position.
• An object becomes a DiscoveredObject when hits reach min_hits (default 30).
• The Detector (Module 6) decides the class and confidence; with GTDetector, class = manifest class.
• Emit an event the first time a METER is discovered (drives the INSPECT state and the viz callout).
Bushes as footprints. For each discovered BUSH, compute a 2D footprint = convex hull of its observed hit points projected to z=0, buffered by 0.1 m. The site solver uses these, not the ground-truth meshes.
Modules 4-5: Swarm planner and safety shield
A centralized MissionController replans at 1 Hz from the shared MapState and hands each live drone a DroneTask; the shield vets every path and waypoint before it reaches dynamics.
Mission state machine
stateDiagram-v2
    [*] --> TAKEOFF
    TAKEOFF --> ORBIT: all drones at 2 m
    ORBIT --> FRONTIER: orbit laps done
    FRONTIER --> INSPECT: meter discovered
    INSPECT --> FRONTIER: close-ups captured
    FRONTIER --> PHOTOS: no reachable frontiers and ground-band coverage >= 0.90
    PHOTOS --> RTH: SSR shot list captured
    RTH --> [*]
The state is global; INSPECT pulls only the nearest drone off frontier duty, the rest keep exploring. Hard timeout: 300 s sim time, then jump to PHOTOS with what exists.
ORBIT
• House center c = footprint centroid; radius r = max footprint half-extent + 4 m.
• Rings at altitudes {2.0, 4.5, 8.0} m; drone i gets ring i % 3 and a start angle offset of 2π i / n.
• One lap per drone, 24 waypoints per ring, camera look-at = c. Waypoints that are not FREE or UNKNOWN-safe (see shield) are skipped.
FRONTIER
• Frontier voxel = FREE with at least one 6-neighbor UNKNOWN, inside the geofence, z in 1.0 to 10 m. Compute with array shifts, no loops.
• Cluster with scipy.ndimage.label (26-connectivity); drop clusters under 8 voxels.
• Viewpoint per cluster: the FREE voxel within 1.5 to 3 m of the cluster centroid that maximizes unknown voxels in a 3 m sphere; look-at = cluster centroid. Weight clusters with centroid z < 2.5 m by 2x (ground band matters most).
• Assignment: cost[d, f] = euclidean(d, f) - λ * gain[f] + 5.0 * [f within 3 m of another drone's current goal]; λ = 0.05 per voxel. Solve with linear_sum_assignment; extra drones get the next-best cluster or HOLD.
• Keep current goal unless new cost is 20% better (hysteresis, prevents thrashing).
• Blacklist a goal for 30 s after it fails (unreachable or stuck).
INSPECT
• On first METER discovery, the nearest live drone gets viewpoints at meter + wall_normal * {1.5, 3.0} m, z = 1.5 m, and ±30 degrees around the normal. Each viewpoint must have line of sight to the meter (single ray cast against the mapped occupancy, not ground truth); if blocked, rotate by 15 degrees up to ±60 degrees.
Pathing
• Plan on a 0.5 m downsampled grid: cost 1 for FREE, impassable for OCC or UNKNOWN (after inflation). skimage.graph.MCP_Geometric from start to goal; convert back to world coordinates; shortcut-smooth by line-of-sight checks.
• Goal viewpoints are FREE by construction; start voxel is forced passable.
Safety shield (the fail-safe story)
• Known-free only: inflate OCC by drone radius 0.25 m + margin 0.35 m, treat UNKNOWN as OCC for pathing. A drone never enters unobserved space.
• Geofence: lot bounds shrunk by 0.5 m; ceiling 10 m.
• Separation: minimum 1.5 m between drones. Priority = drone id; before each tick, if the next waypoints of two drones come within 1.5 m, the lower-priority one HOLDs for that tick. Replan for it if held more than 3 s.
• Stuck detection: under 0.5 m progress in 5 s means blacklist goal and replan.
• Drone kill: kill_drone marks it dead; the next 1 Hz assignment excludes it, so its frontiers are absorbed automatically. Log a DRONE_LOST event.
• Low battery (under 15%): RTH regardless of state.
Termination. No reachable frontier clusters with z < 10 m and coverage_ground_band >= 0.90, or timeout.
Module 6: Perception
The MVP uses ground-truth object IDs from ray casts; a YOLO meter detector is a stretch goal behind the same interface, selected by --detector gt|yolo.
class Detector(Protocol):
    def classify(self, obj_id: int, hits: np.ndarray) -> tuple[Cls, float]: ...
    def on_photo(self, img: np.ndarray, obj_ids: np.ndarray, cam) -> list[tuple[int, Cls, float]]: ...
• GTDetector: returns the manifest class with confidence 1.0. Say this plainly in the pitch; it is standard practice in sim.
• YoloDetector (stretch A): every 2 s each drone renders a 320 x 240 photo toward its look-at; run Ultralytics YOLO nano; a METER box whose center pixel's obj_id matches an object confirms it. Other classes still come from GT.
Stretch A: synthetic training data
• scripts/make_yolo_data.py --seeds 100-199: for each seed, render 20 photos from random free viewpoints 1 to 6 m from walls; derive METER boxes from the per-pixel obj_ids mask (skip boxes under 12 px). Include GAS_METER as a hard negative class.
• Train yolo detect train model=yolo11n.pt data=out/yolo/data.yaml imgsz=320 epochs=30 (use the latest nano model available).
Stretch B: real photos. Fine-tune on a public electric meter set (for example Roboflow Universe detector-electric-meter, CC BY 4.0) and show one real house photo detection in the pitch. Not wired into the sim.
Module 7: Site solver
solve(map_state, rules) -> list[SiteCandidate] picks the top 3 battery sites by minimizing conduit length plus bush removal plus penalties on a 2D cost map built only from discovered objects.
Inputs. Meter position and wall normal, house wall segments (from OCC voxels at z 0.5 to 2.5 m, fit with the footprint polygon approximation: use the convex outline of occupied wall voxels, or the manifest footprint in --oracle mode only), bush footprints, and positions of doors, windows, AC unit, gas meter, driveway.
Cost map layers (0.1 m grid over lot)
• house: inside footprint.
• wall_band: cells 0 to 0.4 m outside a wall (where a wall-mounted battery sits).
• keepout: doors and windows projected to the ground, buffered by door_clearance_m / window_clearance_m.
• obstacle: AC unit, gas meter, fence, trees (not removable).
• bush: bush footprints (removable at cost).
Candidates. Walk the house perimeter in 0.25 m steps. At each point, place the battery rectangle (battery_w_m along the wall, battery_d_m deep) plus a front clearance rectangle (front_clearance_m). Reject if it overlaps house, keepout or obstacle, or if conduit length exceeds max_conduit_m.
Conduit. 2D MCP_Geometric over the wall band from meter base to candidate: cost 1 along the band, 3 elsewhere outside the house, impassable through house, keepouts and obstacles. Conduit length = path length + vertical run (meter height).
Scoring
J = w_c \cdot L_{conduit} + w_b \cdot N_{bushes} + w_s \cdot \text{south\_facing} + w_d \cdot \text{driveway\_adjacent}
• bushes_to_remove = bushes intersecting the battery rectangle, its front clearance, or a 0.3 m corridor around the conduit.
• Keep the 3 lowest-J candidates at least 2 m apart; store each term in breakdown.
rules.yaml (placeholder values; replace with the real SSR checklist)
battery_w_m: 1.1
battery_d_m: 0.3
front_clearance_m: 0.9
door_clearance_m: 0.9
window_clearance_m: 0.3
max_conduit_m: 12.0
weights:
  conduit_per_m: 1.0
  bush_removal: 3.0
  south_facing: 2.0
  driveway_adjacent: -0.5
If no meter is discovered, return an empty candidate list and flag meter_not_found in the packet.
Module 8: Visualization and SSR packet
All Rerun calls live in canopy/viz/rerun_log.py; the rest of the code emits plain data and events so viz can be disabled with --no-viz for batch runs.
Rerun entity layout
Entity path
Type
Update rate
world/scene
Mesh3D, per-vertex colors (gray until seen, then class color)
2 Hz
world/drones/<id>
Boxes3D or small mesh + label
20 Hz
world/drones/<id>/trail
LineStrips3D, last 30 s
5 Hz
world/drones/<id>/rays
LineStrips3D, 60 random rays of the latest scan
5 Hz
world/frontiers
Points3D, orange
1 Hz
world/goals
Arrows3D drone to goal
1 Hz
world/meter
Points3D + label "METER"
on discovery
world/site/candidates
Boxes3D (best green, others yellow)
at end
world/site/conduit
LineStrips3D
at end
world/site/remove
Boxes3D red around bushes to remove
at end
topdown/costmap
Image heatmap
at end
metrics/coverage/*
Scalars (total, ground band)
1 Hz
events
TextLog (METER_FOUND, DRONE_LOST, COLLISION, PHASE)
on event
• Reveal: gray = luminance of the class color x 0.6. For per-triangle color, unmerge vertices once at load (each triangle owns its 3 vertices), then color by tri_seen. Log colors only when tri_seen changed.
• Send a blueprint: large 3D view left, top-down + coverage plot + event log stacked right.
• Always save a .rrd of each run (--record) as the backup demo.
SSR packet
In the PHOTOS state, assign shots to the nearest drones and capture with SimWorld.photo:
1. meter_close: 1.5 m from meter along normal, z 1.5 m.
2. meter_wide: 4 m along normal, z 2 m.
3. site_<k> for each top candidate: 3 m out, z 1.8 m, looking at the battery rectangle center.
4. conduit_route: midpoint of conduit, 4 m out.
5. bush_<id> for each bush to remove: 2.5 m out.
6. elevation_{n,e,s,w}: 10 m from footprint center at z 4 m.
Each viewpoint is line-of-sight checked (rotate up to ±60 degrees). Write out/runs/<run>/packet/:
• packet.json: SiteResult serialized plus meter position, coverage, mission time, events.
• site_plan.png (matplotlib, top-down): footprint, bushes (red if removed), meter, candidates with rank, conduit, cost heatmap underlay.
• index.html from templates/packet.html.j2: headline decision, cost breakdown table, photo grid with labels, site plan.
• metrics.json: coverage over time, time to meter, collisions, drones lost.
Optional RL track
Only start after milestone M4 passes; RL replaces the local waypoint follower, never the global planner or the shield, and ships only if it beats the classical follower on held-out seeds.
• Env: canopy/rl/avoid_env.py, Gymnasium API, kinematic dynamics, random obstacle fields (boxes, cylinders, ellipsoids) then generated houses as curriculum.
• Observation: 48 ray distances (16 azimuth x 3 elevation, normalized by 6 m), goal vector in body frame, velocity.
• Action: velocity setpoint in [-3, 3] m/s per axis.
• Reward: +1.0 x progress toward waypoint (m), -10 on collision (terminate), -0.2 x (1 - d_min / 1.0) when within 1 m of an obstacle, -0.01 per step.
• Train: Stable-Baselines3 PPO, 8 SubprocVecEnv workers (with the __main__ guard), 2M steps. On an NVIDIA GPU, Genesis is the faster alternative.
• Integration: --local-policy rl; the shield checks each predicted next position against the inflated map and falls back to the classical follower on violation. Log fallback rate.
• Kill criterion: on seeds 30-39, if success rate or mission time is worse than classical by milestone M6, drop it from the demo and keep training curves for a slide.
Build order and milestones
Build a thin end-to-end skeleton first, then deepen each module; never leave run_mission.py broken for more than one milestone. Each milestone ends with its tests green and a commit.
Milestone
Target hour
Deliverable
Exit check
M0 Setup
H1
Env, repo skeleton, contracts.py, default.yaml
pytest runs; import open3d, rerun, trimesh OK
M1 Skeleton
H4
Hand-built box house, 1 drone kinematic orbit, 360 scan, gray-to-color reveal in Rerun
Visual check; coverage_total rises each lap
M2 World gen
H7
generate(seed) with all classes, colors, subdivision
Determinism + validity tests pass on seeds 0-9
M3 Mapping
H9
Occupancy, coverage, semantics, discovered meter
Meter discovered on seed 42 with 1 drone orbit
M4 Swarm
H13
Frontiers, assignment, MCP pathing, shield, multi-drone, --kill-drone
4 drones, seeds 0-9: 0 collisions, ground-band coverage >= 0.90
M5 Site + packet
H16
Cost map, solver, INSPECT and PHOTOS states, packet HTML
Packet generated on seeds 0-9; meter found in all
M6 Robustness
H18
batch_eval.py, fixes for stuck and crash cases, scaling chart
Acceptance tests below pass
M7 Polish
H21
Blueprint layout, golden seeds, recorded .rrd + video
Demo script runs end to end twice in a row
Stretch
after M6
YOLO (A, B), physics mode, RL
Must not break M6 checks
Agent rules
• Work milestones in order; do not start stretch work until M6 passes.
• Write the module's tests before or with the module.
• Keep all tunables in config/default.yaml; no magic numbers in code.
• Profile any per-tick function that exceeds 50 ms at 4 drones; vectorize before optimizing elsewhere.
• Stub freely to keep the pipeline running (e.g. a site solver returning the meter position as the only candidate) and replace the stub later.
Acceptance tests and definition of done
The project is done when batch_eval.py --seeds 0-29 --drones 4 meets every threshold below and the demo script runs end to end.
Unit tests (pytest)
• test_worldgen: same seed gives identical manifest and OBJ bytes; every manifest has exactly 1 METER at 1.3 to 1.7 m height on a wall; no two objects intersect except bushes with ground; triangle count under 250k.
• test_sensors: a ray straight at a known wall returns the expected distance within 2 cm and the wall's obj_id; misses return inf and -1.
• test_occupancy: after one scan in an empty box room, voxels between drone and wall are FREE, the wall voxel is OCC, voxels behind the wall stay UNKNOWN.
• test_frontier: a synthetic grid with a known boundary yields frontier voxels exactly on it.
• test_assign: 3 drones, 3 separated frontiers yields a one-to-one assignment with no shared goals.
• test_pathing: path never crosses an OCC or UNKNOWN voxel of the inflated grid.
• test_shield: two drones on a collision course never get within 1.5 m.
• test_solver: on a hand-built scene with a bush in front of the only valid wall, the top candidate lists that bush for removal; a closer valid site with no bush beats a farther one.
System thresholds (seeds 0-29, 4 drones, kinematic, GT detector)
Metric
Threshold
Collisions (ground-truth check)
0 across all runs
Meter found
30 of 30
Ground-band coverage at end
>= 0.90 mean, >= 0.80 min
Mission time (sim)
<= 240 s mean
Crashes / exceptions
0
Site result with >= 1 candidate
>= 28 of 30
Drone kill at t=60 s (seeds 0-9)
Mission still completes, coverage >= 0.85
Scaling
Mean time to 0.90 coverage decreases from 1 to 2 to 4 drones
summary.csv columns: seed, drones, dynamics, detector, collisions, meter_found, t_meter_s, coverage_ground_band, coverage_total, mission_time_s, n_candidates, best_conduit_m, bushes_removed, drones_lost.
Config defaults and demo script
Every number in this spec lives in config/default.yaml; start from these values and tune only against batch_eval.py.
sim:
  control_hz: 20
  sensor_hz: 5
  replan_hz: 1
  timeout_s: 300
  dynamics: kinematic
  v_max: 3.0
  a_max: 4.0
  battery_drain_per_10s: 0.01
  rth_battery: 0.15
sensor:
  az_rays: 180
  el_rays: 60
  el_min_deg: -75
  el_max_deg: 30
  max_range_m: 12.0
  range_noise_m: 0.01
worldgen:
  lot_m: [30, 40]
  bush_density_per_10m: 2.0
  p_meter_occluded: 0.5
  trees: [2, 5]
  max_edge_m: 0.2
map:
  voxel_m: 0.25
  height_m: 12.0
  plan_voxel_m: 0.5
  min_hits: 30
  coverage_max_range_m: 8.0
  coverage_max_incidence_deg: 70
planner:
  orbit_altitudes: [2.0, 4.5, 8.0]
  orbit_margin_m: 4.0
  frontier_min_voxels: 8
  gain_lambda: 0.05
  goal_conflict_radius_m: 3.0
  goal_conflict_penalty: 5.0
  hysteresis: 0.2
  blacklist_s: 30
  done_ground_coverage: 0.90
safety:
  drone_radius_m: 0.25
  margin_m: 0.35
  min_separation_m: 1.5
  ceiling_m: 10.0
  geofence_inset_m: 0.5
  stuck_window_s: 5
  stuck_min_progress_m: 0.5
Demo script (about 3 minutes)
1. run_mission.py --seed <golden> --drones 4: gray house appears; drones take off and fan into orbit rings; the house paints in color.
2. Orbit ends; drones break formation for frontiers behind bushes, garage and trees.
3. METER_FOUND callout; one drone peels off for close-ups while the others keep exploring.
4. Kill a drone live (--kill-drone 2@60 or a hotkey); its frontiers are absorbed.
5. Mission ends: site overlay shows the chosen battery site, conduit line and bushes to remove in red.
6. Open the SSR packet HTML.
7. Show the coverage vs time chart for 1, 2, 4, 6 drones, then rerun with a new seed live.
Pick 3 golden seeds: one with the meter behind a bush, one L-shaped two-story house, one with a tree over the roof edge. Keep their .rrd files as the backup if live runs fail.