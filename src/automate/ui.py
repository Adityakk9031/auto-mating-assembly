from __future__ import annotations

import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
import numpy as np

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GCPnts import GCPnts_UniformDeflection
from OCP.TopAbs import TopAbs_EDGE, TopAbs_REVERSED
from OCP.TopExp import TopExp
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher

from .features import extract_assembly_features, extract_part_features
from .matcher import find_mating_instances
from .models import MatingConfig, PartData
from .step_io import is_same_part_instance, load_step_assembly, load_step_part
from .visualizer import render_mating_results

_CLIENT_DISCONNECT_ERRORS = (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)
_ENGINE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute() and path.exists():
        return path
    cand = PROJECT_ROOT / path
    if cand.exists():
        return cand
    return path


def _face_to_mesh_dict(face) -> Optional[dict[str, Any]]:
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None or tri.NbNodes() == 0 or tri.NbTriangles() == 0:
        return None

    trsf = loc.Transformation()
    is_identity = loc.IsIdentity()
    rev = face.Orientation() == TopAbs_REVERSED

    n_nodes = tri.NbNodes()
    verts: list[float] = []
    for i in range(1, n_nodes + 1):
        p = tri.Node(i)
        if not is_identity:
            p = p.Transformed(trsf)
        verts.extend([round(float(p.X()), 4), round(float(p.Y()), 4), round(float(p.Z()), 4)])

    indices: list[int] = []
    n_tris = tri.NbTriangles()
    for i in range(1, n_tris + 1):
        a, b, c = tri.Triangle(i).Get()
        if rev:
            b, c = c, b
        indices.extend([a - 1, b - 1, c - 1])

    return {"vertices": verts, "indices": indices}


def _shape_edges_to_segments(shape, deflection: float = 0.3) -> list[float]:
    emap = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, emap)
    segs: list[float] = []
    for i in range(1, emap.Extent() + 1):
        edge = TopoDS.Edge(emap.FindKey(i))
        try:
            curve = BRepAdaptor_Curve(edge)
            disc = GCPnts_UniformDeflection(curve, deflection)
            if not disc.IsDone() or disc.NbPoints() < 2:
                continue
            for j in range(1, disc.NbPoints()):
                p1 = disc.Value(j)
                p2 = disc.Value(j + 1)
                segs.extend([
                    round(float(p1.X()), 4), round(float(p1.Y()), 4), round(float(p1.Z()), 4),
                    round(float(p2.X()), 4), round(float(p2.Y()), 4), round(float(p2.Z()), 4),
                ])
        except Exception:
            continue
    return segs


def _serialize_part_mesh(part: PartData) -> dict[str, Any]:
    diag = float(np.linalg.norm(part.bbox_max - part.bbox_min))
    defl = max(0.04, diag * 0.005)
    BRepMesh_IncrementalMesh(part.shape, defl, False, 0.35, True)

    faces_out = []
    for f in part.faces:
        mesh = _face_to_mesh_dict(f.occt_face)
        if mesh is None:
            continue
        faces_out.append({
            "global_id": f.global_id,
            "local_face_id": f.local_face_id,
            "part_id": f.part_id,
            "part_name": f.part_name,
            "surface_type": f.surface_type.value,
            "sense": f.sense,
            "area": round(float(f.area), 2),
            "radius": round(float(f.radius), 3) if f.radius is not None else None,
            "half_angle_deg": round(float(np.degrees(f.half_angle)), 2) if f.half_angle is not None else None,
            "centroid": [round(float(x), 3) for x in f.centroid],
            "normal": [round(float(x), 3) for x in f.normal] if f.normal is not None else None,
            "vertices": mesh["vertices"],
            "indices": mesh["indices"],
        })

    return {
        "part_id": part.part_id,
        "name": part.name,
        "volume": round(float(part.volume), 2),
        "bbox_min": [round(float(x), 3) for x in part.bbox_min],
        "bbox_max": [round(float(x), 3) for x in part.bbox_max],
        "faces": faces_out,
        "edges": _shape_edges_to_segments(part.shape, defl),
    }


def _run_scenario_payload(
    part_path: str,
    assembly_path: str,
    selected_faces: list[int],
    exact_tol: float = 0.08,
    max_clearance: float = 2.5,
    scenario_name: str = "custom",
    force_render: bool = False,
) -> dict[str, Any]:
    with _ENGINE_LOCK:
        resolved_part = str(_resolve_path(part_path))
        resolved_asm = str(_resolve_path(assembly_path))

        cache_key = f"part:{resolved_part}"
        if cache_key not in _CACHE:
            p = load_step_part(resolved_part)
            extract_part_features(p)
            _CACHE[cache_key] = (p, _serialize_part_mesh(p))
        query_part, query_mesh = _CACHE[cache_key]

        asm_key = f"asm:{resolved_asm}"
        if asm_key not in _CACHE:
            a = load_step_assembly(resolved_asm)
            extract_assembly_features(a)
            asm_meshes = [_serialize_part_mesh(pt) for pt in a.parts]
            _CACHE[asm_key] = (a, asm_meshes)
        assembly, asm_meshes = _CACHE[asm_key]

        cfg = MatingConfig(exact_fit_tol_mm=exact_tol, max_clearance_mm=max_clearance)
        matches = find_mating_instances(
            query_part=query_part,
            selected_face_ids=selected_faces,
            assembly=assembly,
            config=cfg,
        )

        out_dir = PROJECT_ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        faces_tag = "_".join(str(fid) for fid in selected_faces[:8])
        png_name = (
            f"{scenario_name}_render.png"
            if not force_render
            else f"{scenario_name}_f{faces_tag}_render.png"
        )
        png_path = out_dir / png_name
        if not png_path.exists() or force_render or scenario_name == "custom":
            render_mating_results(
                query_part, selected_faces, assembly, matches, png_path, title=scenario_name
            )

        self_part_ids = [
            pt.part_id for pt in assembly.parts if is_same_part_instance(query_part, pt)
        ]

        return {
            "scenario": scenario_name,
            "part_path": part_path,
            "assembly_path": assembly_path,
            "selected_face_ids": selected_faces,
            "self_part_ids": self_part_ids,
            "query_part": query_mesh,
            "assembly_parts": asm_meshes,
            "matches": [m.to_dict() for m in matches],
            "render_url": f"/outputs/{png_name}",
        }


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Auto-Mating CAD Studio — SE(3) B-Rep Assembly Matcher</title>
<style>
  :root {
    --bg: #121418;
    --panel: #1a1d24;
    --border: #2c313c;
    --text: #e6edf3;
    --muted: #8b949e;
    --green: #2ecc71;
    --cyan: #22d3ee;
    --amber: #f59e0b;
    --accent: #3b82f6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    height: 100vh;
    width: 100vw;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    background: #0d0f12;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 14px;
    gap: 10px;
    flex-wrap: wrap;
    flex-shrink: 0;
  }
  .brand {
    font-weight: 700;
    font-size: 14.5px;
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
  }
  .badge {
    font-size: 11px;
    padding: 2px 7px;
    border-radius: 999px;
    font-weight: 600;
    background: rgba(46, 204, 113, 0.16);
    color: var(--green);
    border: 1px solid rgba(46, 204, 113, 0.35);
    white-space: nowrap;
  }
  .controls {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    min-width: 0;
  }
  .ctrl-group {
    display: flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
  }
  #scenarioSelect {
    max-width: 240px;
    text-overflow: ellipsis;
  }
  select, input, button {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 12px;
  }
  button {
    cursor: pointer;
    font-weight: 600;
    transition: all 0.15s;
    white-space: nowrap;
  }
  button.primary {
    background: #238636;
    border-color: #2ea043;
    color: #fff;
  }
  button.primary:hover { background: #2ea043; }
  button.tab-btn {
    padding: 3px 8px;
    font-size: 11px;
  }
  button.tab-btn.active {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .workspace {
    flex: 1;
    display: grid;
    grid-template-columns: 270px minmax(0, 1fr);
    min-height: 0;
    min-width: 0;
    overflow: hidden;
  }
  .sidebar {
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }
  .sidebar-section {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .sidebar-section h3 {
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .face-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-height: 0;
  }
  .face-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 8px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: #14171d;
    cursor: pointer;
    font-size: 11.5px;
    gap: 6px;
  }
  .face-item.selected {
    border-color: var(--green);
    background: rgba(46, 204, 113, 0.14);
  }
  .main-area {
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }
  .viewports {
    flex: 1;
    display: grid;
    grid-template-columns: 42% 58%;
    position: relative;
    min-height: 0;
    overflow: hidden;
  }
  .vp-col {
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
    min-height: 0;
    overflow: hidden;
  }
  .vp-header {
    background: #14171d;
    border-bottom: 1px solid var(--border);
    padding: 6px 10px;
    min-height: 36px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
    flex-shrink: 0;
  }
  .vp-title-text {
    font-size: 12px;
    font-weight: 600;
    color: var(--text);
    white-space: nowrap;
  }
  .vp-toolbar {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }
  .vp-canvas-wrap {
    flex: 1;
    position: relative;
    min-height: 0;
    overflow: hidden;
  }
  #renderModal {
    display: none;
    position: absolute;
    inset: 0;
    background: rgba(12, 14, 18, 0.96);
    z-index: 30;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 18px;
  }
  #renderModal img {
    max-width: 96%;
    max-height: 85%;
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  .results-bar {
    height: 195px;
    background: var(--panel);
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    overflow: hidden;
  }
  .results-header {
    padding: 7px 14px;
    background: #14171c;
    border-bottom: 1px solid var(--border);
    font-size: 12.5px;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .table-wrap {
    flex: 1;
    overflow-y: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th, td {
    text-align: left;
    padding: 7px 12px;
    border-bottom: 1px solid var(--border);
  }
  th {
    color: var(--muted);
    font-weight: 600;
    background: #161920;
    position: sticky;
    top: 0;
    z-index: 10;
    box-shadow: 0 2px 0 var(--border);
  }
  tr.match-row { cursor: pointer; }
  tr.match-row:hover { background: rgba(59, 130, 246, 0.12); }
  tr.match-row.active-row { background: rgba(46, 204, 113, 0.16); }
  .fit-pill {
    padding: 2px 7px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 11px;
    white-space: nowrap;
  }
  .fit-EXACT_FIT { background: rgba(46, 204, 113, 0.2); color: #4ade80; }
  .fit-SLIDING_CLEARANCE_FIT { background: rgba(34, 211, 238, 0.2); color: #22d3ee; }
  .fit-INTERFERENCE_FIT { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
<header>
  <div class="brand">
    <span>⚙️ Auto-Mating CAD Studio</span>
    <span class="badge">SE(3) B-Rep Engine</span>
  </div>
  <div class="controls">
    <div class="ctrl-group">
      <label style="font-size:12px;color:var(--muted)">Scenario:</label>
      <select id="scenarioSelect"></select>
    </div>
    <div class="ctrl-group">
      <label style="font-size:12px;color:var(--muted)">Exact Tol:</label>
      <input type="number" id="exactTolInput" value="0.08" step="0.01" style="width:64px" />
    </div>
    <div class="ctrl-group">
      <label style="font-size:12px;color:var(--muted)">Max Clear:</label>
      <input type="number" id="maxClearInput" value="2.5" step="0.1" style="width:62px" />
    </div>
    <button class="primary" id="runBtn" onclick="runCurrentMatch()">▶ Run Auto-Mate</button>
    <button onclick="toggleRenderModal()">🖼️ 3-Panel VTK Render</button>
  </div>
</header>

<div class="workspace">
  <div class="sidebar">
    <div class="sidebar-section">
      <h3>Part & Assembly Paths</h3>
      <div style="display:flex;flex-direction:column;gap:5px">
        <input id="partPathInput" placeholder="Part STEP path" style="font-size:11px" />
        <input id="asmPathInput" placeholder="Assembly STEP path" style="font-size:11px" />
      </div>
    </div>
    <div class="sidebar-section" style="display:flex;justify-content:space-between;align-items:center;gap:4px">
      <h3 style="margin:0">Query Faces (Click 3D/List)</h3>
      <div style="display:flex;gap:4px">
        <button onclick="resetDefaultFaces()" style="padding:2px 6px;font-size:10.5px" title="Restore scenario default mating faces">Reset</button>
        <button onclick="clearSelectedFaces()" style="padding:2px 6px;font-size:10.5px">Clear</button>
      </div>
    </div>
    <div class="face-list" id="faceList"></div>
  </div>

  <div class="main-area">
    <div class="viewports">
      <div class="vp-col">
        <div class="vp-header">
          <span class="vp-title-text" id="queryTitle">1. Query Part (Click 3D faces to select)</span>
        </div>
        <div class="vp-canvas-wrap" id="vpQuery"></div>
      </div>

      <div class="vp-col" style="border-right:none">
        <div class="vp-header">
          <span class="vp-title-text" id="asmTitle">2. Assembly & Mated Instances</span>
          <div class="vp-toolbar">
            <button class="tab-btn active" id="btnModeBoth" onclick="setAsmMode('both')">Faces + Poses</button>
            <button class="tab-btn" id="btnModeFaces" onclick="setAsmMode('faces')">Faces Only</button>
            <button class="tab-btn" id="btnModePoses" onclick="setAsmMode('poses')">Poses Only</button>
          </div>
        </div>
        <div class="vp-canvas-wrap" id="vpAsm"></div>
      </div>

      <div id="renderModal">
        <div style="margin-bottom:10px;display:flex;gap:12px;align-items:center">
          <span style="font-weight:600">3-Panel Verification Render</span>
          <button onclick="downloadRenderImg()">⬇ Download PNG</button>
          <button onclick="toggleRenderModal()">✕ Close</button>
        </div>
        <img id="renderImg" src="" alt="3-Panel Render" />
      </div>
    </div>

    <div class="results-bar">
      <div class="results-header">
        <span id="resultsCount">Matched Instances</span>
        <span style="font-size:11.5px;color:var(--muted)">Click any match row to isolate that instance in 3D</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Match #</th>
              <th>Target Part(s)</th>
              <th>Target Local Face IDs</th>
              <th>Fit Classification</th>
              <th>Kinematic Pair (DOFs)</th>
              <th>Mean Gap</th>
              <th>Opposing Clearance</th>
              <th>Normal Err</th>
              <th>Overlap</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody id="resultsBody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
let scenarios = [];
let currentData = null;
let selectedFaceIds = new Set();
let activeMatchId = null;
let asmDisplayMode = 'both';
let currentAbortCtrl = null;

// Three.js scenes
let qScene, qCam, qRen, qCtrl, qFaceMeshes = [];
let aScene, aCam, aRen, aCtrl;

function syncViewportSizes() {
  const qBox = document.getElementById('vpQuery');
  const aBox = document.getElementById('vpAsm');
  if (!qRen || !aRen) return;
  const qw = Math.max(10, qBox.clientWidth), qh = Math.max(10, qBox.clientHeight);
  const aw = Math.max(10, aBox.clientWidth), ah = Math.max(10, aBox.clientHeight);
  qCam.aspect = qw / qh;
  qCam.updateProjectionMatrix();
  qRen.setSize(qw, qh);
  aCam.aspect = aw / ah;
  aCam.updateProjectionMatrix();
  aRen.setSize(aw, ah);
}

function initThree() {
  const qBox = document.getElementById('vpQuery');
  qScene = new THREE.Scene();
  qScene.background = new THREE.Color(0x16181d);
  qCam = new THREE.PerspectiveCamera(40, Math.max(1, qBox.clientWidth) / Math.max(1, qBox.clientHeight), 0.1, 5000);
  qRen = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  qBox.appendChild(qRen.domElement);
  qCtrl = new THREE.OrbitControls(qCam, qRen.domElement);
  addLights(qScene);

  // Raycaster for clicking faces on Query Part
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let downPos = { x: 0, y: 0 };
  qRen.domElement.addEventListener('pointerdown', e => { downPos = { x: e.clientX, y: e.clientY }; });
  qRen.domElement.addEventListener('pointerup', e => {
    if (Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y) > 4) return;
    const rect = qRen.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, qCam);
    const hits = raycaster.intersectObjects(qFaceMeshes);
    if (hits.length > 0) {
      const fid = hits[0].object.userData.local_face_id;
      toggleFace(fid);
    }
  });

  const aBox = document.getElementById('vpAsm');
  aScene = new THREE.Scene();
  aScene.background = new THREE.Color(0x131519);
  aCam = new THREE.PerspectiveCamera(40, Math.max(1, aBox.clientWidth) / Math.max(1, aBox.clientHeight), 0.1, 8000);
  aRen = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  aBox.appendChild(aRen.domElement);
  aCtrl = new THREE.OrbitControls(aCam, aRen.domElement);
  addLights(aScene);

  syncViewportSizes();
  window.addEventListener('resize', syncViewportSizes);
  if (window.ResizeObserver) {
    new ResizeObserver(syncViewportSizes).observe(qBox);
    new ResizeObserver(syncViewportSizes).observe(aBox);
  }

  function animate() {
    requestAnimationFrame(animate);
    qCtrl.update();
    aCtrl.update();
    qRen.render(qScene, qCam);
    aRen.render(aScene, aCam);
  }
  animate();
}

function addLights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const d1 = new THREE.DirectionalLight(0xffffff, 0.75);
  d1.position.set(200, 300, 250);
  scene.add(d1);
  const d2 = new THREE.DirectionalLight(0x90b0ff, 0.45);
  d2.position.set(-250, -200, -150);
  scene.add(d2);
}

function clearScene(scene) {
  const toRemove = [];
  scene.traverse(obj => { if (obj.isMesh || obj.isLineSegments) toRemove.push(obj); });
  toRemove.forEach(obj => {
    scene.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  });
}

function makeBufferGeom(f) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(f.vertices, 3));
  geom.setIndex(f.indices);
  geom.computeVertexNormals();
  return geom;
}

function makeEdgeLines(edgesArr, colorHex, opacity = 0.5) {
  if (!edgesArr || edgesArr.length === 0) return null;
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(edgesArr, 3));
  const mat = new THREE.LineBasicMaterial({ color: colorHex, transparent: opacity < 1.0, opacity });
  return new THREE.LineSegments(geom, mat);
}

function fitCameraToBox(cam, ctrl, bmin, bmax) {
  const center = new THREE.Vector3(
    0.5 * (bmin[0] + bmax[0]),
    0.5 * (bmin[1] + bmax[1]),
    0.5 * (bmin[2] + bmax[2])
  );
  const diag = Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]) || 50;
  ctrl.target.copy(center);
  cam.position.set(center.x + diag * 1.1, center.y - diag * 1.1, center.z + diag * 0.85);
  cam.lookAt(center);
  ctrl.update();
}

function buildQueryScene(resetCam = true) {
  clearScene(qScene);
  qFaceMeshes = [];
  if (!currentData) return;
  const qp = currentData.query_part;
  qp.faces.forEach(f => {
    const isSel = selectedFaceIds.has(f.local_face_id);
    const geom = makeBufferGeom(f);
    const mat = new THREE.MeshPhongMaterial({
      color: isSel ? 0x2ecc71 : 0x7c8490,
      transparent: !isSel,
      opacity: isSel ? 1.0 : 0.45,
      side: THREE.DoubleSide,
      shininess: 40,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.userData = { local_face_id: f.local_face_id };
    qScene.add(mesh);
    qFaceMeshes.push(mesh);
  });
  const lines = makeEdgeLines(qp.edges, 0x111318, 0.75);
  if (lines) qScene.add(lines);
  if (resetCam) fitCameraToBox(qCam, qCtrl, qp.bbox_min, qp.bbox_max);
}

function applyTransform4x4(obj, m4) {
  const mat = new THREE.Matrix4();
  mat.set(
    m4[0][0], m4[0][1], m4[0][2], m4[0][3],
    m4[1][0], m4[1][1], m4[1][2], m4[1][3],
    m4[2][0], m4[2][1], m4[2][2], m4[2][3],
    m4[3][0], m4[3][1], m4[3][2], m4[3][3]
  );
  obj.applyMatrix4(mat);
}

function buildAssemblyScene(resetCam = true) {
  clearScene(aScene);
  if (!currentData) return;

  const matches = currentData.matches || [];
  const filteredMatches = activeMatchId !== null
    ? matches.filter(m => m.match_id === activeMatchId)
    : matches;

  const matchedGlobalFaces = new Map();
  const matchedPartIds = new Set();
  filteredMatches.forEach(m => {
    m.part_ids.forEach(pid => matchedPartIds.add(pid));
    m.target_global_face_ids.forEach(gid => matchedGlobalFaces.set(gid, m.fit_type));
  });

  const selfSet = new Set(currentData.self_part_ids || []);
  let gmin = [1e9, 1e9, 1e9], gmax = [-1e9, -1e9, -1e9];

  currentData.assembly_parts.forEach(pt => {
    if (asmDisplayMode === 'poses' && selfSet.has(pt.part_id)) return;
    for (let k = 0; k < 3; k++) {
      gmin[k] = Math.min(gmin[k], pt.bbox_min[k]);
      gmax[k] = Math.max(gmax[k], pt.bbox_max[k]);
    }
    const isMatchedPart = matchedPartIds.has(pt.part_id);
    pt.faces.forEach(f => {
      const geom = makeBufferGeom(f);
      const fit = matchedGlobalFaces.get(f.global_id);
      let color = 0x757c88, opacity = 0.18;
      if (fit && asmDisplayMode !== 'poses') {
        color = fit === 'SLIDING_CLEARANCE_FIT' ? 0x22d3ee : (fit === 'INTERFERENCE_FIT' ? 0xf59e0b : 0x2ecc71);
        opacity = 0.96;
      } else if (isMatchedPart) {
        color = 0x457b5b;
        opacity = 0.28;
      }
      const mat = new THREE.MeshPhongMaterial({
        color,
        transparent: opacity < 1.0,
        opacity,
        side: THREE.DoubleSide,
        depthWrite: opacity > 0.85,
      });
      aScene.add(new THREE.Mesh(geom, mat));
    });
    const lines = makeEdgeLines(pt.edges, 0x22262e, isMatchedPart ? 0.55 : 0.25);
    if (lines) aScene.add(lines);
  });

  // Render placed query part at each match's SE(3) transform_4x4
  if (asmDisplayMode === 'both' || asmDisplayMode === 'poses') {
    const qp = currentData.query_part;
    filteredMatches.forEach(m => {
      qp.faces.forEach(f => {
        const isSel = selectedFaceIds.has(f.local_face_id);
        const geom = makeBufferGeom(f);
        const mat = new THREE.MeshPhongMaterial({
          color: isSel ? 0x2ecc71 : 0xf5b932,
          transparent: true,
          opacity: isSel ? 0.95 : 0.78,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geom, mat);
        applyTransform4x4(mesh, m.transform_4x4);
        aScene.add(mesh);
      });
      const qLines = makeEdgeLines(qp.edges, 0x332508, 0.75);
      if (qLines) {
        applyTransform4x4(qLines, m.transform_4x4);
        aScene.add(qLines);
      }
    });
  }

  if (resetCam && gmin[0] < gmax[0]) fitCameraToBox(aCam, aCtrl, gmin, gmax);
}

function renderSidebarFaces() {
  const list = document.getElementById('faceList');
  list.innerHTML = '';
  if (!currentData) return;
  currentData.query_part.faces.forEach(f => {
    const div = document.createElement('div');
    const isSel = selectedFaceIds.has(f.local_face_id);
    div.className = 'face-item' + (isSel ? ' selected' : '');
    let meta = `${f.surface_type} | A=${f.area}`;
    if (f.radius !== null) meta += ` | R=${f.radius}`;
    div.innerHTML = `<span><b>Face #${f.local_face_id}</b></span><span style="color:var(--muted)">${meta}</span>`;
    div.onclick = () => toggleFace(f.local_face_id);
    list.appendChild(div);
  });
}

function renderMatchesTable() {
  const tbody = document.getElementById('resultsBody');
  tbody.innerHTML = '';
  const matches = (currentData && currentData.matches) || [];
  const selArr = Array.from(selectedFaceIds).sort((a, b) => a - b);
  document.getElementById('resultsCount').textContent =
    `Matched Instances (${matches.length} found for faces [${selArr.join(', ')}]) — Scenario: ${currentData ? currentData.scenario : ''}`;

  if (matches.length === 0) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="10" style="text-align:center;color:#f87171;padding:18px;">
      No compatible mating instances found in assembly for selected face(s) <code>[${selArr.join(', ')}]</code>.
      Try selecting a mating face group or click <b>Reset</b> in the left sidebar.
    </td>`;
    tbody.appendChild(tr);
    return;
  }

  matches.forEach(m => {
    const tr = document.createElement('tr');
    tr.className = 'match-row' + (activeMatchId === m.match_id ? ' active-row' : '');
    tr.onclick = () => {
      activeMatchId = (activeMatchId === m.match_id) ? null : m.match_id;
      renderMatchesTable();
      buildAssemblyScene(false);
    };
    const opp = m.opposing_clearance_mm !== null ? `${m.opposing_clearance_mm.toFixed(3)} mm` : '—';
    tr.innerHTML = `
      <td><b>#${m.match_id}</b></td>
      <td>${m.part_names.join(', ')}</td>
      <td><code>[${m.target_local_face_ids.join(', ')}]</code></td>
      <td><span class="fit-pill fit-${m.fit_type}">${m.fit_type}</span></td>
      <td>${m.kinematic_pair} (T:${m.dof_translation}, R:${m.dof_rotation})</td>
      <td>${m.mean_gap_mm.toFixed(3)} mm</td>
      <td>${opp}</td>
      <td>${m.normal_error_deg.toFixed(2)}°</td>
      <td>${(m.overlap_score * 100).toFixed(0)}%</td>
      <td><b>${m.score.toFixed(3)}</b></td>
    `;
    tbody.appendChild(tr);
  });
}

function toggleFace(fid) {
  if (selectedFaceIds.has(fid)) selectedFaceIds.delete(fid);
  else selectedFaceIds.add(fid);
  renderSidebarFaces();
  buildQueryScene(false);
}

function clearSelectedFaces() {
  selectedFaceIds.clear();
  renderSidebarFaces();
  buildQueryScene(false);
}

function resetDefaultFaces() {
  const sc = scenarios[document.getElementById('scenarioSelect').selectedIndex];
  if (sc) {
    selectedFaceIds = new Set(sc.faces);
    renderSidebarFaces();
    buildQueryScene(false);
    runCurrentMatch();
  }
}

function setAsmMode(mode) {
  asmDisplayMode = mode;
  document.getElementById('btnModeBoth').classList.toggle('active', mode === 'both');
  document.getElementById('btnModeFaces').classList.toggle('active', mode === 'faces');
  document.getElementById('btnModePoses').classList.toggle('active', mode === 'poses');
  buildAssemblyScene(false);
}

function generateLive3PanelDataURL() {
  if (!currentData || !qRen || !aRen) return '';
  const w1 = 440, w2 = 560, w3 = 560, panelH = 520;
  const headerH = 44;
  const matches = currentData.matches || [];
  const maxRows = Math.min(matches.length, 8);
  const rowH = 22;
  const footerH = 50 + Math.max(1, maxRows) * rowH;
  const totalW = w1 + w2 + w3;
  const totalH = headerH + panelH + footerH;

  const canvas = document.createElement('canvas');
  canvas.width = totalW;
  canvas.height = totalH;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#181a1e';
  ctx.fillRect(0, 0, totalW, totalH);

  // Render Panel 1 (Query Part)
  qRen.render(qScene, qCam);
  ctx.drawImage(qRen.domElement, 0, headerH, w1, panelH);

  // Render Panel 2 (Target Faces Only)
  const savedMode = asmDisplayMode;
  asmDisplayMode = 'faces';
  buildAssemblyScene(false);
  aRen.render(aScene, aCam);
  ctx.drawImage(aRen.domElement, w1, headerH, w2, panelH);

  // Render Panel 3 (Mated Poses Only)
  asmDisplayMode = 'poses';
  buildAssemblyScene(false);
  aRen.render(aScene, aCam);
  ctx.drawImage(aRen.domElement, w1 + w2, headerH, w3, panelH);

  // Restore user display mode
  asmDisplayMode = savedMode;
  buildAssemblyScene(false);
  aRen.render(aScene, aCam);

  // Header bar
  ctx.fillStyle = '#101216';
  ctx.fillRect(0, 0, totalW, headerH);
  ctx.font = 'bold 13px Consolas, monospace';
  const selArr = Array.from(selectedFaceIds).sort((a, b) => a - b);
  ctx.fillStyle = '#ebeff5';
  ctx.fillText(`1. Query: ${currentData.query_part.name} (faces [${selArr.join(', ')}])`, 14, 27);
  ctx.fillStyle = '#91ebaf';
  ctx.fillText(`2. Target Face Matches: ${matches.length} [Green=Exact, Cyan=Sliding]`, w1 + 14, 27);
  ctx.fillStyle = '#f5d273';
  ctx.fillText('3. SE(3) Mated Pose Verification [Gold=Placed Query Part]', w1 + w2 + 14, 27);

  ctx.strokeStyle = '#3c424d';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(w1, 0); ctx.lineTo(w1, headerH + panelH);
  ctx.moveTo(w1 + w2, 0); ctx.lineTo(w1 + w2, headerH + panelH);
  ctx.stroke();

  // Footer summary
  const y0 = headerH + panelH + 22;
  ctx.fillStyle = '#dce1e6';
  ctx.font = 'bold 13px Consolas, monospace';
  ctx.fillText(`Scenario '${currentData.scenario}' — Matched Instances (${matches.length}):`, 16, y0);
  ctx.font = '12px Consolas, monospace';
  if (matches.length === 0) {
    ctx.fillStyle = '#f87171';
    ctx.fillText(`No compatible mating instances found for selected faces [${selArr.join(', ')}].`, 22, y0 + 24);
  } else {
    matches.slice(0, maxRows).forEach((m, i) => {
      const y = y0 + 22 + i * rowH;
      const opp = m.opposing_clearance_mm !== null ? `, opp_clear=${m.opposing_clearance_mm.toFixed(3)}mm` : '';
      const line = `[#${m.match_id}] Part(s): ${m.part_names.join(', ')} | TargetFaces: [${m.target_local_face_ids.join(', ')}] | ${m.fit_type} (${m.kinematic_pair} T:${m.dof_translation} R:${m.dof_rotation}) | gap=${m.mean_gap_mm.toFixed(3)}mm${opp} | overlap=${(m.overlap_score * 100).toFixed(0)}% | score=${m.score.toFixed(3)}`;
      ctx.fillStyle = m.fit_type === 'SLIDING_CLEARANCE_FIT' ? '#6ee7f5' : '#73f091';
      ctx.fillText(line, 20, y);
    });
  }
  return canvas.toDataURL('image/png');
}

function updateRenderModalImage() {
  const img = document.getElementById('renderImg');
  if (!currentData) return;
  const sc = scenarios[document.getElementById('scenarioSelect').selectedIndex];
  const selArr = Array.from(selectedFaceIds).sort((a, b) => a - b);
  const defArr = sc ? [...sc.faces].sort((a, b) => a - b) : [];
  const isDefault = JSON.stringify(selArr) === JSON.stringify(defArr);

  img.onerror = () => {
    img.onerror = null;
    img.src = generateLive3PanelDataURL();
  };

  if (!isDefault || !currentData.render_url || currentData._clientRendered) {
    img.src = generateLive3PanelDataURL();
  } else {
    img.src = currentData.render_url + '?t=' + Date.now();
  }
}

function downloadRenderImg() {
  const url = generateLive3PanelDataURL();
  if (!url) return;
  const a = document.createElement('a');
  a.href = url;
  a.download = `${currentData ? currentData.scenario : 'automate'}_3panel_render.png`;
  a.click();
}

function toggleRenderModal() {
  const modal = document.getElementById('renderModal');
  if (modal.style.display === 'flex') {
    modal.style.display = 'none';
  } else {
    updateRenderModalImage();
    modal.style.display = 'flex';
  }
}

async function loadScenarioByIndex(idx) {
  const sc = scenarios[idx];
  document.getElementById('partPathInput').value = sc.part;
  document.getElementById('asmPathInput').value = sc.assembly;
  selectedFaceIds = new Set(sc.faces);
  activeMatchId = null;
  await executeMatchRequest(sc.part, sc.assembly, sc.faces, sc.name, true, false);
}

async function runCurrentMatch() {
  const partPath = document.getElementById('partPathInput').value.trim();
  const asmPath = document.getElementById('asmPathInput').value.trim();
  const faces = Array.from(selectedFaceIds).sort((a, b) => a - b);
  if (faces.length === 0) {
    alert('Please select at least 1 face on the query part.');
    return;
  }
  activeMatchId = null;
  const sc = scenarios[document.getElementById('scenarioSelect').selectedIndex];
  const isDefaultFaces = sc && JSON.stringify(faces) === JSON.stringify([...sc.faces].sort((a, b) => a - b));
  const name = (sc && sc.part === partPath && sc.assembly === asmPath) ? sc.name : 'custom';
  await executeMatchRequest(partPath, asmPath, faces, name, false, !isDefaultFaces);
}

function _vDot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function _vSub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function _vAdd(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function _vScale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function _vCross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function _vNorm(a) {
  const len = Math.hypot(a[0], a[1], a[2]);
  return len < 1e-12 ? [0, 0, 1] : [a[0] / len, a[1] / len, a[2] / len];
}

function _singleFaceCompatJS(q, t, maxClear) {
  if (q.surface_type !== t.surface_type) return false;
  if (q.surface_type === 'PLANE') {
    const ratio = Math.min(q.area, t.area) / Math.max(q.area, t.area, 1e-3);
    return ratio >= 0.01;
  }
  if (q.surface_type === 'CYLINDER' || q.surface_type === 'SPHERE') {
    if (q.sense * t.sense !== -1) return false;
    const rConv = q.sense > 0 ? q.radius : t.radius;
    const rConc = t.sense < 0 ? t.radius : q.radius;
    if (rConv == null || rConc == null) return false;
    const c = rConc - rConv;
    const effMax = Math.min(maxClear, Math.max(0.35, 0.08 * 2.0 * rConv));
    return c >= -0.15 && c <= effMax;
  }
  if (q.surface_type === 'CONE') {
    if (q.sense * t.sense !== -1) return false;
    return Math.abs((q.half_angle_deg || 0) - (t.half_angle_deg || 0)) <= 3.0;
  }
  return q.sense * t.sense !== 1 || q.sense === 0;
}

function _pairwiseCompatJS(q1, q2, t1, t2, maxClear) {
  if (t1.global_id === t2.global_id) return false;
  // BBox proximity check
  let qMaxD2 = 0, tMinD2 = 0;
  for (let k = 0; k < 3; k++) {
    const farQ = Math.max(Math.abs(q1.bbox_max[k] - q2.bbox_min[k]), Math.abs(q2.bbox_max[k] - q1.bbox_min[k]));
    qMaxD2 += farQ * farQ;
    if (t1.bbox_max[k] < t2.bbox_min[k]) tMinD2 += (t2.bbox_min[k] - t1.bbox_max[k]) ** 2;
    else if (t2.bbox_max[k] < t1.bbox_min[k]) tMinD2 += (t1.bbox_min[k] - t2.bbox_max[k]) ** 2;
  }
  if (Math.sqrt(tMinD2) > Math.sqrt(qMaxD2) + maxClear + 2.0) return false;

  const vq1 = q1.normal || q1.axis, vq2 = q2.normal || q2.axis;
  const vt1 = t1.normal || t1.axis, vt2 = t2.normal || t2.axis;
  if (vq1 && vq2 && vt1 && vt2) {
    const dq = Math.max(-1, Math.min(1, _vDot(vq1, vq2)));
    const dt = Math.max(-1, Math.min(1, _vDot(vt1, vt2)));
    const sq1 = (q1.surface_type === 'PLANE' || q1.surface_type === 'FREEFORM') ? -1 : (q1.surface_type === 'CONE' ? 1 : 0);
    const sq2 = (q2.surface_type === 'PLANE' || q2.surface_type === 'FREEFORM') ? -1 : (q2.surface_type === 'CONE' ? 1 : 0);
    if (sq1 !== 0 && sq2 !== 0) {
      const expDt = Math.max(-1, Math.min(1, (sq1 * sq2) * dq));
      if (Math.abs(Math.acos(dt) - Math.acos(expDt)) > 0.08) return false;
    } else {
      if (Math.abs(Math.acos(Math.abs(dq)) - Math.acos(Math.abs(dt))) > 0.08) return false;
    }
  }

  if (q1.surface_type === 'PLANE' && q2.surface_type === 'PLANE' && q1.normal && q2.normal) {
    const dotN = _vDot(q1.normal, q2.normal);
    if (Math.abs(dotN) >= 0.998) {
      const dq = _vDot(_vSub(q2.origin || q2.centroid, q1.origin || q1.centroid), q1.normal);
      const dt = _vDot(_vSub(t2.origin || t2.centroid, t1.origin || t1.centroid), t1.normal);
      const g = dt + dq;
      if (dotN < 0) {
        const effMax = Math.min(maxClear, Math.max(0.35, 0.08 * Math.max(Math.abs(dq), Math.abs(dt), 1.0)));
        if (g < -0.15 || g > effMax) return false;
      } else if (Math.abs(g) > maxClear) {
        return false;
      }
    } else if (q1.sample_points && q2.sample_points && t1.sample_points && t2.sample_points) {
      let minSq12 = 1e9, maxSq12 = -1e9, minSt12 = 1e9, maxSt12 = -1e9;
      let minSq21 = 1e9, maxSq21 = -1e9, minSt21 = 1e9, maxSt21 = -1e9;
      const oq1 = q1.origin || q1.centroid, oq2 = q2.origin || q2.centroid;
      const ot1 = t1.origin || t1.centroid, ot2 = t2.origin || t2.centroid;
      for (const p of q2.sample_points) { const v = _vDot(_vSub(p, oq1), q1.normal); minSq12 = Math.min(minSq12, v); maxSq12 = Math.max(maxSq12, v); }
      for (const p of t2.sample_points) { const v = _vDot(_vSub(p, ot1), t1.normal); minSt12 = Math.min(minSt12, v); maxSt12 = Math.max(maxSt12, v); }
      for (const p of q1.sample_points) { const v = _vDot(_vSub(p, oq2), q2.normal); minSq21 = Math.min(minSq21, v); maxSq21 = Math.max(maxSq21, v); }
      for (const p of t1.sample_points) { const v = _vDot(_vSub(p, ot2), t2.normal); minSt21 = Math.min(minSt21, v); maxSt21 = Math.max(maxSt21, v); }
      const ov1 = Math.min(maxSq12, -minSt12) - Math.max(minSq12, -maxSt12);
      const ov2 = Math.min(maxSq21, -minSt21) - Math.max(minSq21, -maxSt21);
      if (ov1 < -0.5 || ov2 < -0.5) return false;
    }
  }
  return true;
}

function _matMul3(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    }
  }
  return C;
}

function _matVec3(R, v) {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

function _rodriguesJS(u, v) {
  const un = _vNorm(u), vn = _vNorm(v);
  const dot = Math.max(-1, Math.min(1, _vDot(un, vn)));
  if (dot > 1 - 1e-7) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  if (dot < -1 + 1e-7) {
    const ortho = Math.abs(un[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const ax = _vNorm(_vCross(un, ortho));
    return [
      [-1 + 2 * ax[0] * ax[0], 2 * ax[0] * ax[1], 2 * ax[0] * ax[2]],
      [2 * ax[1] * ax[0], -1 + 2 * ax[1] * ax[1], 2 * ax[1] * ax[2]],
      [2 * ax[2] * ax[0], 2 * ax[2] * ax[1], -1 + 2 * ax[2] * ax[2]],
    ];
  }
  const c = _vCross(un, vn);
  const s = Math.hypot(c[0], c[1], c[2]);
  const k = (1 - dot) / (s * s);
  const K = [
    [0, -c[2], c[1]],
    [c[2], 0, -c[0]],
    [-c[1], c[0], 0],
  ];
  const K2 = _matMul3(K, K);
  return [
    [1 + K[0][0] + k * K2[0][0], K[0][1] + k * K2[0][1], K[0][2] + k * K2[0][2]],
    [K[1][0] + k * K2[1][0], 1 + K[1][1] + k * K2[1][1], K[1][2] + k * K2[1][2]],
    [K[2][0] + k * K2[2][0], K[2][1] + k * K2[2][1], 1 + K[2][2] + k * K2[2][2]],
  ];
}

function _solvePoseJS(qFaces, tFaces) {
  const k = qFaces.length;
  const uList = [], vList = [];
  for (let i = 0; i < k; i++) {
    const q = qFaces[i], t = tFaces[i];
    if (q.normal && t.normal) {
      uList.push(_vNorm(q.normal));
      vList.push(_vScale(_vNorm(t.normal), -1));
    } else if (q.axis && t.axis) {
      const sgn = q.surface_type === 'CONE' ? 1 : 1;
      uList.push(_vNorm(q.axis));
      vList.push(_vScale(_vNorm(t.axis), sgn));
    }
  }

  let R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  if (uList.length === 1) {
    R = _rodriguesJS(uList[0], vList[0]);
  } else if (uList.length >= 2) {
    // Find two non-collinear direction vectors
    let idx2 = -1;
    const e1q = uList[0], e1t = vList[0];
    for (let i = 1; i < uList.length; i++) {
      if (Math.abs(_vDot(e1q, uList[i])) < 0.98) {
        idx2 = i;
        break;
      }
    }
    if (idx2 === -1) {
      R = _rodriguesJS(e1q, e1t);
    } else {
      const u2 = uList[idx2], v2 = vList[idx2];
      const e2q = _vNorm(_vSub(u2, _vScale(e1q, _vDot(u2, e1q))));
      const e3q = _vCross(e1q, e2q);
      const e2t = _vNorm(_vSub(v2, _vScale(e1t, _vDot(v2, e1t))));
      const e3t = _vCross(e1t, e2t);
      const Et = [
        [e1t[0], e2t[0], e3t[0]],
        [e1t[1], e2t[1], e3t[1]],
        [e1t[2], e2t[2], e3t[2]],
      ];
      const EqT = [
        [e1q[0], e1q[1], e1q[2]],
        [e2q[0], e2q[1], e2q[2]],
        [e3q[0], e3q[1], e3q[2]],
      ];
      R = _matMul3(Et, EqT);
    }
  }

  // Verify all directional constraints are satisfied under R
  for (let i = 0; i < uList.length; i++) {
    const ru = _matVec3(R, uList[i]);
    if (_vDot(ru, vList[i]) < 0.97) return null;
  }

  // Compute translation aligning centroids and iteratively projecting onto plane/axis constraints
  const qCent = [0, 0, 0], tCent = [0, 0, 0];
  for (let i = 0; i < k; i++) {
    for (let d = 0; d < 3; d++) {
      qCent[d] += qFaces[i].centroid[d] / k;
      tCent[d] += tFaces[i].centroid[d] / k;
    }
  }
  const rqCent = _matVec3(R, qCent);
  let tVec = _vSub(tCent, rqCent);

  // Refine tVec by Gauss-Seidel projections onto matched plane/cylinder constraints
  for (let iter = 0; iter < 4; iter++) {
    for (let i = 0; i < k; i++) {
      const q = qFaces[i], t = tFaces[i];
      const pq = _vAdd(_matVec3(R, q.origin || q.centroid), tVec);
      const pt = t.origin || t.centroid;
      if (t.surface_type === 'PLANE' && t.normal) {
        const err = _vDot(_vSub(pt, pq), t.normal);
        tVec = _vAdd(tVec, _vScale(t.normal, err * 0.5));
      } else if (t.surface_type === 'CYLINDER' && t.axis) {
        const diff = _vSub(pt, pq);
        const perp = _vSub(diff, _vScale(t.axis, _vDot(diff, t.axis)));
        tVec = _vAdd(tVec, perp);
      } else if (t.surface_type === 'SPHERE' || t.surface_type === 'CONE') {
        tVec = _vSub(pt, _matVec3(R, q.origin || q.centroid));
      }
    }
  }

  return [
    [R[0][0], R[0][1], R[0][2], tVec[0]],
    [R[1][0], R[1][1], R[1][2], tVec[1]],
    [R[2][0], R[2][1], R[2][2], tVec[2]],
    [0, 0, 0, 1],
  ];
}

function clientSideMatch(data, selFaces, exactTol, maxClear) {
  // Fast client-side SE(3) B-Rep matcher for static Vercel deployment
  const defFaces = (data.default_face_ids || []).slice().sort((a, b) => a - b);
  const curFaces = selFaces.slice().sort((a, b) => a - b);
  if (JSON.stringify(defFaces) === JSON.stringify(curFaces) && data.default_matches) {
    return data.default_matches
      .filter(m => m.mean_gap_mm <= maxClear && (m.opposing_clearance_mm === null || m.opposing_clearance_mm <= maxClear))
      .map((m, idx) => {
        const copy = JSON.parse(JSON.stringify(m));
        copy.match_id = idx;
        const gapInd = copy.opposing_clearance_mm !== null ? copy.opposing_clearance_mm : copy.mean_gap_mm;
        if (copy.min_gap_mm < -0.01) copy.fit_type = 'INTERFERENCE_FIT';
        else if (gapInd <= exactTol && copy.max_gap_mm <= exactTol + 0.05) copy.fit_type = 'EXACT_FIT';
        else copy.fit_type = 'SLIDING_CLEARANCE_FIT';
        return copy;
      });
  }

  const qMap = new Map(data.query_part.faces.map(f => [f.local_face_id, f]));
  const qFaces = curFaces.map(id => qMap.get(id)).filter(Boolean);
  if (qFaces.length === 0) return [];

  // General k-face SE(3) subgraph search across all non-self assembly parts
  const selfSet = new Set(data.self_part_ids || []);
  const results = [];
  const verifiedSets = new Set();
  const k = qFaces.length;

  data.assembly_parts.forEach(pt => {
    if (selfSet.has(pt.part_id) || pt.faces.length < k) return;
    const candsPerQ = qFaces.map(q => pt.faces.filter(t => _singleFaceCompatJS(q, t, maxClear)));
    if (candsPerQ.some(c => c.length === 0)) return;

    const assignedT = [];
    const usedIds = new Set();

    function backtrack(depth) {
      if (results.length >= 40) return;
      if (depth === k) {
        const setKey = pt.part_id + ':' + assignedT.map(t => t.global_id).slice().sort((a, b) => a - b).join(',');
        if (verifiedSets.has(setKey)) return;

        const T = _solvePoseJS(qFaces, assignedT);
        if (!T) return;

        // Verify signed surface gaps and area/extent compatibility for each pair
        const R = [T[0].slice(0, 3), T[1].slice(0, 3), T[2].slice(0, 3)];
        const tVec = [T[0][3], T[1][3], T[2][3]];
        let sumGap = 0, minGap = 1e9, maxGap = -1e9;
        for (let i = 0; i < k; i++) {
          const qi = qFaces[i], ti = assignedT[i];
          const ratio = Math.min(qi.area, ti.area) / Math.max(qi.area, ti.area, 1e-3);
          if (k === 1 && ratio < 0.65) return;
          if (k >= 2 && (ti.area < qi.area * 0.65 || ratio < 0.15)) return;

          let g = 0.01;
          if (ti.surface_type === 'PLANE' && ti.normal) {
            const pq = _vAdd(_matVec3(R, qi.origin || qi.centroid), tVec);
            g = _vDot(_vSub(pq, ti.origin || ti.centroid), ti.normal);
          } else if (ti.surface_type === 'CYLINDER' || ti.surface_type === 'SPHERE') {
            g = ti.sense < 0 ? (ti.radius - qi.radius) : (qi.radius - ti.radius);
          }
          if (g < -0.15 || g > maxClear + 0.2) return;
          // Trimmed face sample-point coverage check with scale-adaptive sample spacing
          if (qi.sample_points && ti.sample_points && qi.sample_points.length > 0 && ti.sample_points.length > 0) {
            const tSpan = Math.max(
              ti.bbox_max[0] - ti.bbox_min[0],
              ti.bbox_max[1] - ti.bbox_min[1],
              ti.bbox_max[2] - ti.bbox_min[2],
              10.0
            );
            const tol2 = Math.max(25.0, (tSpan * 0.26) ** 2);
            let hitQ = 0, hitT = 0;
            const qPtsW = qi.sample_points.map(p => _vAdd(_matVec3(R, p), tVec));
            for (const pw of qPtsW) {
              let bestD2 = 1e18;
              for (const tp of ti.sample_points) {
                const d2 = (pw[0] - tp[0]) ** 2 + (pw[1] - tp[1]) ** 2 + (pw[2] - tp[2]) ** 2;
                if (d2 < bestD2) bestD2 = d2;
              }
              if (bestD2 <= tol2) hitQ++;
            }
            for (const tp of ti.sample_points) {
              let bestD2 = 1e18;
              for (const pw of qPtsW) {
                const d2 = (pw[0] - tp[0]) ** 2 + (pw[1] - tp[1]) ** 2 + (pw[2] - tp[2]) ** 2;
                if (d2 < bestD2) bestD2 = d2;
              }
              if (bestD2 <= tol2) hitT++;
            }
            const cov = Math.max(hitQ / qPtsW.length, hitT / ti.sample_points.length);
            if (cov < 0.45) return;
          }
          sumGap += g;
          minGap = Math.min(minGap, g);
          maxGap = Math.max(maxGap, g);
        }

        // Compute opposing clearance or radial clearance
        let opp = null;
        for (let i = 0; i < k; i++) {
          const qi = qFaces[i], ti = assignedT[i];
          if (qi.surface_type === 'CYLINDER' || qi.surface_type === 'SPHERE') {
            const rDiff = ti.sense < 0 ? (ti.radius - qi.radius) : (qi.radius - ti.radius);
            opp = opp === null ? rDiff : Math.max(opp, rDiff);
          }
          for (let j = i + 1; j < k; j++) {
            const qj = qFaces[j], tj = assignedT[j];
            if (qi.surface_type === 'PLANE' && qj.surface_type === 'PLANE' && _vDot(qi.normal, qj.normal) < -0.995) {
              const dq = _vDot(_vSub(qj.origin || qj.centroid, qi.origin || qi.centroid), qi.normal);
              const dt = _vDot(_vSub(tj.origin || tj.centroid, ti.origin || ti.centroid), ti.normal);
              opp = opp === null ? (dt + dq) : Math.max(opp, dt + dq);
            }
          }
        }

        verifiedSets.add(setKey);
        const meanGap = sumGap / k;
        const gapInd = opp !== null ? opp : meanGap;
        const fit = minGap < -0.01 ? 'INTERFERENCE_FIT' : (gapInd <= exactTol && maxGap <= exactTol + 0.05 ? 'EXACT_FIT' : 'SLIDING_CLEARANCE_FIT');

        results.push({
          match_id: results.length,
          part_ids: [pt.part_id],
          part_names: [pt.name],
          query_face_ids: qFaces.map(f => f.local_face_id),
          target_global_face_ids: assignedT.map(f => f.global_id),
          target_local_face_ids: assignedT.map(f => f.local_face_id),
          target_part_ids: assignedT.map(f => f.part_id),
          fit_type: fit,
          kinematic_pair: k >= 3 ? 'PRISMATIC_SLIDE' : (qFaces[0].surface_type === 'CYLINDER' ? 'CYLINDRICAL' : 'PLANAR'),
          dof_translation: k >= 3 ? 1 : 2,
          dof_rotation: k >= 2 ? 0 : 1,
          mean_gap_mm: Math.max(0, meanGap),
          min_gap_mm: minGap,
          max_gap_mm: maxGap,
          opposing_clearance_mm: opp,
          normal_error_deg: 0.0,
          overlap_score: 1.0,
          score: 0.96,
          transform_4x4: T,
        });
        return;
      }

      const qCur = qFaces[depth];
      for (const tCand of candsPerQ[depth]) {
        if (usedIds.has(tCand.global_id)) continue;
        let ok = true;
        for (let p = 0; p < depth; p++) {
          if (!_pairwiseCompatJS(qFaces[p], qCur, assignedT[p], tCand, maxClear)) {
            ok = false;
            break;
          }
        }
        if (!ok) continue;
        usedIds.add(tCand.global_id);
        assignedT.push(tCand);
        backtrack(depth + 1);
        assignedT.pop();
        usedIds.delete(tCand.global_id);
      }
    }

    backtrack(0);
  });

  return results;
}

async function executeMatchRequest(part, assembly, faces, name, resetCam, forceRender = false) {
  if (currentAbortCtrl) {
    currentAbortCtrl.abort();
  }
  currentAbortCtrl = new AbortController();
  const exactTol = parseFloat(document.getElementById('exactTolInput').value) || 0.08;
  const maxClear = parseFloat(document.getElementById('maxClearInput').value) || 2.5;
  const runBtn = document.getElementById('runBtn');
  runBtn.textContent = '⏳ Matching...';
  try {
    let usedStaticFallback = false;
    try {
      const res = await fetch('/api/match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          part,
          assembly,
          faces,
          exact_tol: exactTol,
          max_clearance: maxClear,
          name,
          force_render: forceRender,
        }),
        signal: currentAbortCtrl.signal,
      });
      if (res.ok) {
        currentData = await res.json();
        currentData._clientRendered = false;
      } else {
        usedStaticFallback = true;
      }
    } catch (fetchErr) {
      if (fetchErr.name === 'AbortError') return;
      usedStaticFallback = true;
    }

    if (usedStaticFallback) {
      const scName = name !== 'custom' ? name : (scenarios[document.getElementById('scenarioSelect').selectedIndex]?.name || 'chuck_key_3slot');
      if (!currentData || currentData.scenario !== scName) {
        const sRes = await fetch(`/data/${scName}.json`, { signal: currentAbortCtrl.signal });
        currentData = await sRes.json();
      }
      currentData.selected_face_ids = faces;
      currentData.matches = clientSideMatch(currentData, faces, exactTol, maxClear);
      currentData._clientRendered = true;
    }

    selectedFaceIds = new Set(currentData.selected_face_ids);
    renderSidebarFaces();
    renderMatchesTable();
    buildQueryScene(resetCam);
    buildAssemblyScene(resetCam);
    syncViewportSizes();

    // Update modal image if open
    if (document.getElementById('renderModal').style.display === 'flex') {
      updateRenderModalImage();
    }

    const nFound = (currentData.matches || []).length;
    runBtn.textContent = `✓ Matched (${nFound})`;
    setTimeout(() => { runBtn.textContent = '▶ Run Auto-Mate'; }, 1000);
  } catch (err) {
    if (err.name !== 'AbortError') console.error(err);
    runBtn.textContent = '▶ Run Auto-Mate';
  }
}

async function boot() {
  initThree();
  let res = await fetch('/api/scenarios').catch(() => null);
  if (!res || !res.ok) {
    res = await fetch('/data/scenarios.json');
  }
  scenarios = await res.json();
  const sel = document.getElementById('scenarioSelect');
  scenarios.forEach((sc, idx) => {
    const opt = document.createElement('option');
    opt.value = idx;
    opt.textContent = `${idx + 1}. ${sc.name} (expected: ${sc.expected_matches} ${sc.expected_fit})`;
    sel.appendChild(opt);
  });
  sel.onchange = () => loadScenarioByIndex(parseInt(sel.value, 10));
  if (scenarios.length > 0) await loadScenarioByIndex(0);
}

boot();
</script>
</body>
</html>
"""


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        import sys
        _, exc, _ = sys.exc_info()
        if isinstance(exc, _CLIENT_DISCONNECT_ERRORS):
            return
        super().handle_error(request, client_address)


class AutoMateUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _safe_write(self, raw: bytes, content_type: str, status: int = 200) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(raw)
        except _CLIENT_DISCONNECT_ERRORS:
            # Browser tab refreshed or aborted an in-flight request; ignore quietly
            return

    def _send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data).encode("utf-8")
        self._safe_write(raw, "application/json; charset=utf-8", status=status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._safe_write(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8", status=200)
            return

        if parsed.path == "/api/scenarios":
            manifest_path = PROJECT_ROOT / "scenarios" / "manifest.json"
            if manifest_path.exists():
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                data = []
            self._send_json(data)
            return

        if parsed.path.startswith("/outputs/"):
            rel = parsed.path.lstrip("/")
            fpath = PROJECT_ROOT / rel
            if fpath.exists() and fpath.is_file():
                raw = fpath.read_bytes()
                mime, _ = mimetypes.guess_type(str(fpath))
                self._safe_write(raw, mime or "application/octet-stream", status=200)
                return

        try:
            self.send_error(404, "Not Found")
        except _CLIENT_DISCONNECT_ERRORS:
            return

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/match":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                payload = _run_scenario_payload(
                    part_path=body["part"],
                    assembly_path=body["assembly"],
                    selected_faces=[int(x) for x in body["faces"]],
                    exact_tol=float(body.get("exact_tol", 0.08)),
                    max_clearance=float(body.get("max_clearance", 2.5)),
                    scenario_name=str(body.get("name", "custom")),
                    force_render=bool(body.get("force_render", False)),
                )
            except _CLIENT_DISCONNECT_ERRORS:
                return
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"error": str(exc)}, status=500)
                return

            self._send_json(payload)
            return

        try:
            self.send_error(404, "Not Found")
        except _CLIENT_DISCONNECT_ERRORS:
            return


def run_ui_server(host: str = "127.0.0.1", port: int = 8050) -> None:
    server = QuietThreadingHTTPServer((host, port), AutoMateUIHandler)
    print(f"Auto-Mating CAD Studio UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
