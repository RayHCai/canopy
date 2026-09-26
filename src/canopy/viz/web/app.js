// Canopy viewer. Renders what canopy.viz.viewer.ViewerSession reports.
//
// Python owns the simulation; this page owns only presentation. Each animation
// frame it hands Python the elapsed wall-clock time through the pywebview bridge
// and draws the interpolated poses that come back. Nothing here decides where a
// drone goes.
//
// Frames: Canopy's world is Z-up (x east, y north); three.js is Y-up. Every
// world coordinate crosses over through toThree() and nowhere else.

import * as THREE from './vendor/three.module.min.js';

const SKY_TOP = new THREE.Color('#b8c9da');
const HORIZON = new THREE.Color('#eef1f4');
const GROUND = new THREE.Color('#fbfbfc');
const ACCENTS = ['#2f7d5b', '#3b6fd8', '#d9822b', '#b0489e', '#2a9bb0'];

/** Scale applied to the drone model. The true-size 0.26 m scout is a speck at
 *  orbit-viewing distance; this keeps it legible without looking like a toy. */
const DRONE_SCALE = 3.5;

const toThree = (p, out = new THREE.Vector3()) => out.set(p[0], p[2], -p[1]);
const damp = (rate, dt) => 1 - Math.exp(-rate * dt);

// ---------------------------------------------------------------------------
// Renderer, sky, ground, light
// ---------------------------------------------------------------------------
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
// Neutral rather than ACES: ACES greys out near-whites, and the sandbox is white.
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 1.12;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
// Neighbours extend to about +/-45 m from lot centre; fog must start well past
// that or the scene the swarm is mapping fades before the camera reaches it.
scene.fog = new THREE.Fog(HORIZON, 90, 260);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.05, 900);

scene.add(
  new THREE.Mesh(
    new THREE.SphereGeometry(600, 32, 16),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
      uniforms: { top: { value: SKY_TOP }, horizon: { value: HORIZON } },
      vertexShader: /* glsl */ `
        varying vec3 vDir;
        void main() {
          vDir = normalize(position);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 top;
        uniform vec3 horizon;
        varying vec3 vDir;
        void main() {
          // Start the gradient just below the horizon so the fogged ground
          // meets the sky without a seam.
          float h = clamp(vDir.y + 0.015, 0.0, 1.0);
          gl_FragColor = vec4(mix(horizon, top, pow(h, 0.6)), 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    }),
  ),
);

const groundMat = new THREE.MeshStandardMaterial({ color: GROUND, roughness: 0.95, metalness: 0 });
addGrid(groundMat);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(800, 800), groundMat);
ground.rotation.x = -Math.PI / 2;
// Slightly below z=0: the generated property's lawn sits at the world's true
// ground and must draw on top of this sandbox floor, not fight it for a pixel.
ground.position.y = -0.03;
ground.receiveShadow = true;
scene.add(ground);

// A whisper of a 1 m grid: enough to read distance and motion, faded by the fog
// so it never becomes the tiled floor it replaced.
//
// It is shaded into the ground rather than drawn as a GridHelper. Hundreds of
// GL lines spanning the screen are pathologically slow to rasterize on some
// GPUs (ANGLE/D3D11 on Intel Arc: ~165 ms a frame looking down over the swarm,
// against ~7 ms without them), whereas this costs a few ALU ops per ground
// pixel. Lines fade out where cells shrink toward a pixel, so the distance
// resolves to flat ground instead of shimmering.
function addGrid(material) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.gridColor = { value: new THREE.Color(0x8a96a3) };
    shader.uniforms.gridOpacity = { value: 0.16 };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec2 vGrid;')
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\nvGrid = (modelMatrix * vec4(transformed, 1.0)).xz;',
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        '#include <common>\nvarying vec2 vGrid;\nuniform vec3 gridColor;\nuniform float gridOpacity;',
      )
      .replace(
        '#include <color_fragment>',
        /* glsl */ `#include <color_fragment>
        {
          vec2 w = fwidth(vGrid);
          vec2 px = abs(fract(vGrid - 0.5) - 0.5) / w; // distance to a line, in pixels
          float line = 1.0 - min(min(px.x, px.y), 1.0);
          float fade = 1.0 - smoothstep(0.15, 0.6, max(w.x, w.y));
          diffuseColor.rgb = mix(diffuseColor.rgb, gridColor, gridOpacity * line * fade);
        }`,
      );
  };
}

// Procedural surface detail for the generated property's lawns and paving.
//
// The meshes are flat-coloured by class, which is what the reveal needs, but a
// 30 x 40 m lawn in one flat green reads as a placeholder. This modulates the
// vertex colour per pixel instead of adding geometry or textures: mottled
// patches and tufts on grass, a fine grain on asphalt and concrete. The
// pattern is keyed to world position, so it stays put as the camera moves and
// lines up across neighbouring lots.
//
// Only brightness changes on a grey (unrevealed) surface -- the dry-grass tint
// is weighted by the base colour's saturation -- so the reveal still reads as
// grey turning to colour. The finest detail fades out where it would shrink
// below a pixel, the same trick the grid uses, so distant lawns don't shimmer.
const SURFACE_GLSL = /* glsl */ `
  varying vec2 vSurf;
  float sdHash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }
  float sdNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(sdHash(i), sdHash(i + vec2(1.0, 0.0)), u.x),
               mix(sdHash(i + vec2(0.0, 1.0)), sdHash(i + vec2(1.0, 1.0)), u.x), u.y);
  }
  float sdFbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int k = 0; k < 4; k++) {
      v += a * sdNoise(p);
      p = p * 2.03 + vec2(17.1, 9.7);
      a *= 0.5;
    }
    return v / 0.9375;
  }`;

/** Which classes get surface detail, and the shader define selecting it. */
const SURFACE_KIND = { GROUND: 'SURFACE_GRASS', DRIVEWAY: 'SURFACE_PAVING' };

function addSurfaceDetail(material, cls) {
  const kind = SURFACE_KIND[cls];
  if (!kind) return;
  // A define, not a closure flag: three.js keys its program cache on defines,
  // and would otherwise hand the paving material the grass program.
  material.defines = { ...material.defines, [kind]: '' };
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec2 vSurf;')
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\nvSurf = (modelMatrix * vec4(transformed, 1.0)).xz;',
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${SURFACE_GLSL}`)
      .replace(
        '#include <color_fragment>',
        /* glsl */ `#include <color_fragment>
        {
          vec2 p = vSurf;
          float mpp = max(fwidth(p).x, fwidth(p).y); // metres per pixel
          float fine = 1.0 - smoothstep(0.03, 0.12, mpp);
          #ifdef SURFACE_GRASS
            float patches = sdFbm(p * 0.22);
            float tufts = sdNoise(p * 3.3) + 0.5 * sdNoise(p * 7.9 + 3.1);
            float stripe = smoothstep(-0.25, 0.25, sin(p.x * 3.14159 / 1.6));
            float stripes = 1.0 - smoothstep(0.08, 0.3, mpp);
            float lum = 0.84 + 0.30 * patches
                      + fine * 0.10 * (tufts / 1.5 - 0.5)
                      + stripes * 0.045 * (stripe - 0.5);
            vec3 c = diffuseColor.rgb * lum;
            float sat = max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
            float dry = smoothstep(0.55, 0.85, sdFbm(p * 0.06 + 31.0));
            c = mix(c, c * vec3(1.18, 1.06, 0.72), dry * 0.55 * clamp(sat * 8.0, 0.0, 1.0));
            diffuseColor.rgb = c;
          #endif
          #ifdef SURFACE_PAVING
            float blotch = sdFbm(p * 0.35);
            float grit = sdHash(floor(p * 14.0));
            diffuseColor.rgb *= 0.92 + 0.12 * blotch + fine * 0.08 * (grit - 0.5);
          #endif
        }`,
      );
  };
}

scene.add(new THREE.HemisphereLight(0xe4ecf5, 0xf1ede6, 1.25));

const sun = new THREE.DirectionalLight(0xffffff, 2.4);
sun.position.set(14, 24, 9);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
sun.shadow.camera.left = -24;
sun.shadow.camera.right = 24;
sun.shadow.camera.top = 24;
sun.shadow.camera.bottom = -24;
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 90;
sun.shadow.bias = -0.0003;
sun.shadow.normalBias = 0.02;
sun.shadow.radius = 5;
scene.add(sun, sun.target);

// ---------------------------------------------------------------------------
// Drone model
// ---------------------------------------------------------------------------
// The mesh is the authored canopy_scout, which Python reads from the asset
// library and sends once (canopy.viz.viewer.drone_model). Its geometry is built
// once into a kit that every drone shares; only the accent material is per drone.

/** Material whose faces are the propellers, split into four spinning rotors. */
const ROTOR_MATERIAL = 'prop_dark';
/** Rotor speeds, rad/s: in flight, and idling on the pad before take-off. */
const ROTOR_SPIN_AIRBORNE = 38;
const ROTOR_SPIN_IDLE = 9;
/** Material recoloured per drone, so each matches its pad, orbit and trail. */
const ACCENT_MATERIAL = 'drone_accent';

const shared = {
  disc: new THREE.MeshBasicMaterial({ color: 0x9aa4ae, transparent: true, opacity: 0.16, depthWrite: false }),
};
/** Materials and geometry that outlive a scene rebuild; disposeTree skips them. */
const keep = new Set(Object.values(shared));
let droneKit = null;

function shadowed(mesh) {
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/** Flat-shaded geometry for one run of triangles. The scout is authored with one
 *  normal per face, so un-indexing before computing normals reproduces it. */
function flatGeometry(positions, indices) {
  const indexed = new THREE.BufferGeometry();
  indexed.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  indexed.setIndex(indices);
  const flat = indexed.toNonIndexed();
  indexed.dispose();
  flat.computeVertexNormals();
  keep.add(flat);
  return flat;
}

/** Split the propeller triangles into one rotor per quadrant, each recentred on
 *  its own hub so spinning the hub spins the blades about their shaft. */
function splitRotors(geometry) {
  const pos = geometry.getAttribute('position').array;
  const quads = new Map();
  for (let t = 0; t < pos.length; t += 9) {
    const cx = pos[t] + pos[t + 3] + pos[t + 6];
    const cz = pos[t + 2] + pos[t + 5] + pos[t + 8];
    const key = `${Math.sign(cx)},${Math.sign(cz)}`;
    if (!quads.has(key)) quads.set(key, []);
    quads.get(key).push(...pos.subarray(t, t + 9));
  }
  const rotors = [...quads.values()].map((flat) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
    g.computeBoundingBox();
    const centre = g.boundingBox.getCenter(new THREE.Vector3());
    g.translate(-centre.x, -centre.y, -centre.z);
    g.computeVertexNormals();
    g.computeBoundingSphere();
    const disc = new THREE.CircleGeometry(g.boundingSphere.radius, 32);
    keep.add(g).add(disc);
    return { geometry: g, disc, centre };
  });
  // Neighbours counter-rotate, as on a real quad: order the hubs around the body.
  rotors.sort((a, b) => Math.atan2(a.centre.z, a.centre.x) - Math.atan2(b.centre.z, b.centre.x));
  geometry.dispose();
  keep.delete(geometry);
  return rotors;
}

/** A model's body-frame vertices (Z-up, from Python) as three.js positions. */
function modelPositions(model) {
  const src = model.vertices;
  const positions = new Float32Array(src.length);
  const p = new THREE.Vector3();
  for (let i = 0; i < src.length; i += 3) {
    toThree([src[i], src[i + 1], src[i + 2]], p).toArray(positions, i);
  }
  return positions;
}

/** One material per authored MTL group; kept across scene rebuilds. */
function modelMaterial(group) {
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color().setRGB(...group.color, THREE.SRGBColorSpace),
    roughness: 0.5,
    metalness: 0.1,
    transparent: group.opacity < 1,
    opacity: group.opacity,
  });
  keep.add(material);
  return material;
}

function makeDroneKit(model) {
  const positions = modelPositions(model);
  const parts = [];
  let rotors = [];
  let rotorMat = null;
  for (const group of model.groups) {
    const material = modelMaterial(group);
    const geometry = flatGeometry(positions, group.indices);
    if (group.material === ROTOR_MATERIAL) {
      rotors = splitRotors(geometry);
      rotorMat = material;
    } else {
      parts.push({ geometry, material, accent: group.material === ACCENT_MATERIAL });
    }
  }
  return { parts, rotors, rotorMat };
}

/** The battery drawn at each suggested site: its parts, shared by every copy,
 *  and its bounding box in the site frame (+X out of the wall), which is what
 *  the highlight outline wraps. */
function makeBatteryKit(model) {
  const positions = modelPositions(model);
  const parts = model.groups.map((group) => ({
    geometry: flatGeometry(positions, group.indices),
    material: modelMaterial(group),
  }));
  const bounds = new THREE.Box3();
  for (const part of parts) {
    part.geometry.computeBoundingBox();
    bounds.union(part.geometry.boundingBox);
  }
  return { parts, bounds };
}

function makeDrone(accent) {
  const root = new THREE.Group();
  const body = new THREE.Group();
  body.scale.setScalar(DRONE_SCALE);
  // The model's origin is its skid contact point: parked, it stands on the pad.
  body.position.y = PAD_HEIGHT;
  root.add(body);

  const accentMat = new THREE.MeshStandardMaterial({
    color: accent, emissive: accent, emissiveIntensity: 0.2, roughness: 0.4,
  });
  for (const part of droneKit.parts) {
    body.add(shadowed(new THREE.Mesh(part.geometry, part.accent ? accentMat : part.material)));
  }

  // One blur material per drone, not the shared one: each drone's rotors
  // spool down on their own when it lands.
  const discMat = shared.disc.clone();
  const rotors = droneKit.rotors.map((r, k) => {
    const hub = new THREE.Group();
    hub.position.copy(r.centre);
    hub.add(shadowed(new THREE.Mesh(r.geometry, droneKit.rotorMat)));
    const disc = new THREE.Mesh(r.disc, discMat);
    disc.rotation.x = -Math.PI / 2;
    hub.add(disc);
    body.add(hub);
    return { hub, disc, dir: k % 2 ? 1 : -1 };
  });

  return { root, body, rotors, accentMat };
}

// ---------------------------------------------------------------------------
// Scene contents. ``worldGroup`` holds the generated property and the lot
// boundary, rebuilt only when the seed changes; ``layout`` holds the swarm
// (pads, trails, drones, goal lines, frontier markers), rebuilt every reset.
// ---------------------------------------------------------------------------
const worldGroup = new THREE.Group();
scene.add(worldGroup);
const layout = new THREE.Group();
scene.add(layout);

const FRONTIER_CAPACITY = 128;

function makeFrontierPoints() {
  const geometry = new THREE.BufferGeometry();
  const attr = new THREE.BufferAttribute(new Float32Array(FRONTIER_CAPACITY * 3), 3);
  attr.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', attr);
  geometry.setDrawRange(0, 0);
  const points = new THREE.Points(
    geometry,
    new THREE.PointsMaterial({ color: 0xd9822b, size: 0.35, sizeAttenuation: true }),
  );
  points.frustumCulled = false;
  keep.add(geometry).add(points.material);
  return points;
}

const state = {
  epoch: -1,
  drones: new Map(), // id -> { model, pos, yaw, tilt, born, trail, goalLine }
  pads: new THREE.Group(),
  trails: new THREE.Group(),
  frontierPoints: makeFrontierPoints(),
  detections: new THREE.Group(),
  boxes: new Map(), // track id -> outline box
  show: { pads: true, trails: true, frontiers: true, detections: true },
};

function disposeTree(obj) {
  obj.traverse((o) => {
    if (o.geometry && !keep.has(o.geometry)) o.geometry.dispose();
    if (o.material && !keep.has(o.material)) o.material.dispose();
  });
}

const PAD_RADIUS = 0.55;
const PAD_HEIGHT = 0.03;
const PULSE_PERIOD_MS = 2600;

/** The pad's top face: a helipad "H" inside a ring lettered with the drone's
 *  number, so a pad reads as "drone N starts here" from any angle. */
function padTexture(accent, number) {
  const size = 512;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const mid = size / 2;

  g.fillStyle = '#fbfcfd';
  g.beginPath();
  g.arc(mid, mid, mid, 0, Math.PI * 2);
  g.fill();

  g.strokeStyle = accent;
  g.lineWidth = 16;
  g.beginPath();
  g.arc(mid, mid, mid - 14, 0, Math.PI * 2);
  g.stroke();
  g.lineWidth = 4;
  g.globalAlpha = 0.45;
  g.beginPath();
  g.arc(mid, mid, mid - 92, 0, Math.PI * 2);
  g.stroke();
  g.globalAlpha = 1;

  // Lettering around the band, repeated so it reads from every side.
  const text = `LAUNCH \u00b7 DRONE ${number} \u00b7 `.repeat(2);
  g.fillStyle = accent;
  g.font = '600 30px "Segoe UI", system-ui, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  const r = mid - 54;
  const chars = [...text];
  const widths = chars.map((ch) => g.measureText(ch).width + 3);
  const total = widths.reduce((a, w) => a + w, 0);
  // Spread the letters evenly over the full circle.
  const stretch = (Math.PI * 2 * r) / total;
  let angle = -Math.PI / 2;
  chars.forEach((ch, i) => {
    const half = (widths[i] * stretch) / 2 / r;
    angle += half;
    g.save();
    g.translate(mid + Math.cos(angle) * r, mid + Math.sin(angle) * r);
    g.rotate(angle + Math.PI / 2);
    g.fillText(ch, 0, 0);
    g.restore();
    angle += half;
  });

  g.font = '800 190px "Segoe UI", system-ui, sans-serif';
  g.fillText('H', mid, mid + 8);

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return tex;
}

function makePad(pos, accent, index) {
  const pad = new THREE.Group();
  toThree(pos, pad.position);

  const plinth = new THREE.Mesh(
    new THREE.CylinderGeometry(PAD_RADIUS, PAD_RADIUS + 0.03, PAD_HEIGHT, 64),
    new THREE.MeshStandardMaterial({ color: 0xf2f4f6, roughness: 0.6 }),
  );
  plinth.position.y = PAD_HEIGHT / 2;
  plinth.castShadow = true;
  plinth.receiveShadow = true;

  const face = new THREE.Mesh(
    new THREE.CircleGeometry(PAD_RADIUS, 64),
    new THREE.MeshStandardMaterial({ map: padTexture(accent, index + 1), roughness: 0.55 }),
  );
  face.rotation.x = -Math.PI / 2;
  face.position.y = PAD_HEIGHT + 0.001;
  face.receiveShadow = true;

  // A slow ripple across the ground marks the spot even from far off.
  const pulse = new THREE.Mesh(
    new THREE.RingGeometry(0.94, 1, 64),
    new THREE.MeshBasicMaterial({ color: accent, transparent: true, opacity: 0, depthWrite: false }),
  );
  pulse.rotation.x = -Math.PI / 2;
  pulse.position.y = 0.003;
  pulse.userData.phase = index * 0.23;

  pad.add(plinth, face, pulse);
  pad.userData.pulse = pulse;
  return pad;
}

function updatePads(now) {
  if (!state.pads.visible) return;
  for (const pad of state.pads.children) {
    const pulse = pad.userData.pulse;
    const t = (now / PULSE_PERIOD_MS + pulse.userData.phase) % 1;
    const eased = 1 - (1 - t) ** 3;
    pulse.scale.setScalar(PAD_RADIUS + eased * 0.4);
    pulse.material.opacity = 0.45 * (1 - t);
  }
}

// ---------------------------------------------------------------------------
// Flight trails: everywhere each drone has been, as one growing polyline
// ---------------------------------------------------------------------------
/** Minimum spacing between recorded trail points, metres. */
const TRAIL_STEP_M = 0.05;

function makeTrail(accent, start) {
  const capacity = 4096;
  const geometry = new THREE.BufferGeometry();
  const attr = new THREE.BufferAttribute(new Float32Array(capacity * 3), 3);
  attr.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', attr);
  geometry.setDrawRange(0, 0);
  const line = new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color: accent, transparent: true, opacity: 0.45, depthWrite: false }),
  );
  // The bounds grow every frame; culling against stale ones would hide the line.
  line.frustumCulled = false;
  const trail = { line, count: 0, last: new THREE.Vector3() };
  pushTrail(trail, start);
  return trail;
}

function pushTrail(trail, p) {
  if (trail.count > 0 && trail.last.distanceToSquared(p) < TRAIL_STEP_M ** 2) return;
  const geometry = trail.line.geometry;
  let attr = geometry.getAttribute('position');
  if (trail.count === attr.count) {
    // Full: double the buffer, so growth costs one copy per doubling.
    const grown = new Float32Array(attr.array.length * 2);
    grown.set(attr.array);
    attr = new THREE.BufferAttribute(grown, 3);
    attr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('position', attr);
  } else {
    // Upload only the new vertex, not the whole history.
    attr.clearUpdateRanges();
    attr.addUpdateRange(trail.count * 3, 3);
  }
  attr.setXYZ(trail.count, p.x, p.y, p.z);
  attr.needsUpdate = true;
  trail.count += 1;
  trail.last.copy(p);
  geometry.setDrawRange(0, trail.count);
}

/** A thin line from a drone to its current goal, hidden while there is none. */
function makeGoalLine(accent) {
  const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const line = new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color: accent, transparent: true, opacity: 0.35, depthWrite: false }),
  );
  line.frustumCulled = false;
  line.visible = false;
  return line;
}

/** The surveyed lot boundary: a thin dashed rectangle on the ground. */
function makeLotBoundary(lot) {
  const [lo, hi] = lot;
  const corners = [
    [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
    [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]], [lo[0], lo[1], lo[2]],
  ].map((p) => toThree(p));
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(corners),
    new THREE.LineDashedMaterial({
      color: 0x3b6fd8, dashSize: 0.6, gapSize: 0.4, transparent: true, opacity: 0.5, depthWrite: false,
    }),
  );
  line.position.y = 0.01;
  line.computeLineDistances();
  return line;
}

// ---------------------------------------------------------------------------
// The generated property: grayscale until mapped, true colour triangle by
// triangle as the swarm reveals it. Background objects (the neighbours) are
// always drawn in true colour -- they were never the swarm's job to survey.
// ---------------------------------------------------------------------------
const GRAY_LUMINANCE_SCALE = 0.6;
const property = { seed: null, objects: [] }; // { mesh, geometry, colorAttr, offset, count, background, trueColor }

function decodeF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function decodeU32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Uint32Array(bytes.buffer);
}

function decodeI32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Int32Array(bytes.buffer);
}

/** Class-colour luminance (sRGB 0..1), dimmed: the grayscale a surveyed
 *  object starts in before any of its triangles have been mapped. */
function grayOf(color) {
  const lum = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
  return lum * GRAY_LUMINANCE_SCALE;
}

function disposeProperty() {
  for (const obj of property.objects) {
    obj.mesh.geometry.dispose();
    obj.mesh.material.dispose();
    worldGroup.remove(obj.mesh);
  }
  property.objects = [];
}

function buildProperty(desc) {
  disposeProperty();
  property.seed = desc.seed;
  for (const obj of desc.objects) {
    const srcPos = decodeF32(obj.positions);
    const indices = decodeU32(obj.indices);
    const positions = new Float32Array(srcPos.length);
    for (let i = 0; i < srcPos.length; i += 3) {
      // Z-up world -> Y-up three.js: (x, y, z) -> (x, z, -y).
      positions[i] = srcPos[i];
      positions[i + 1] = srcPos[i + 2];
      positions[i + 2] = -srcPos[i + 1];
    }
    const indexed = new THREE.BufferGeometry();
    indexed.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    // The axis swap above is a rotation (-90 degrees about x, determinant +1),
    // not a mirror, so winding carries over unchanged. Reordering the indices
    // here turns every face inside out, and back-face culling then hides a
    // wall from outside while still drawing it from within.
    indexed.setIndex(new THREE.BufferAttribute(indices, 1));
    const geometry = indexed.toNonIndexed();
    indexed.dispose();
    geometry.computeVertexNormals();

    const trueColor = new THREE.Color().setRGB(...obj.color, THREE.SRGBColorSpace);
    const gray = grayOf(obj.color);
    const startColor = obj.background
      ? trueColor
      : new THREE.Color().setRGB(gray, gray, gray, THREE.SRGBColorSpace);
    const count = geometry.getAttribute('position').count;
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) startColor.toArray(colors, i * 3);
    const colorAttr = new THREE.BufferAttribute(colors, 3);
    colorAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('color', colorAttr);

    const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0 });
    addSurfaceDetail(material, obj.cls);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    worldGroup.add(mesh);
    property.objects.push({
      mesh, colorAttr, offset: obj.tri_offset, count: count / 3,
      background: obj.background, trueColor, grayColor: startColor.clone(),
    });
  }
}

/** Repaint every surveyed (non-background) object back to grayscale, for a
 *  new mission over the same property (a drone-count change). */
function repaintGray() {
  for (const obj of property.objects) {
    if (obj.background) continue;
    const arr = obj.colorAttr.array;
    for (let i = 0; i < arr.length; i += 3) obj.grayColor.toArray(arr, i);
    obj.colorAttr.needsUpdate = true;
  }
}

/** Find the object owning global triangle id ``triId`` by its ascending
 *  ``tri_offset``, then paint that triangle's 3 vertices its true colour. */
function revealTriangles(ids) {
  if (!ids.length) return;
  const offsets = property.objects.map((o) => o.offset);
  const touched = new Set();
  for (const id of ids) {
    // Last offset <= id: a standard upper-bound binary search.
    let lo = 0;
    let hi = offsets.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (offsets[mid] <= id) lo = mid; else hi = mid - 1;
    }
    const obj = property.objects[lo];
    const local = id - obj.offset;
    if (local < 0 || local >= obj.count) continue; // background or out of range
    const base = local * 9;
    const arr = obj.colorAttr.array;
    for (let v = 0; v < 3; v++) {
      obj.trueColor.toArray(arr, base + v * 3);
    }
    touched.add(obj);
  }
  for (const obj of touched) obj.colorAttr.needsUpdate = true;
}

let lotLine = null;
function setLotBoundary(lot) {
  if (lotLine) {
    worldGroup.remove(lotLine);
    lotLine.geometry.dispose();
    lotLine.material.dispose();
  }
  lotLine = makeLotBoundary(lot);
  worldGroup.add(lotLine);
}

function buildScene(desc) {
  for (const child of [...layout.children]) {
    layout.remove(child);
    disposeTree(child);
  }
  state.drones.clear();
  state.epoch = desc.epoch;
  latest = null; // a frame from the old swarm must not move the new one
  state.pads = new THREE.Group();
  state.trails = new THREE.Group();
  state.pads.visible = state.show.pads;
  state.trails.visible = state.show.trails;
  layout.add(state.pads, state.trails);
  layout.add(state.frontierPoints);
  state.detections.clear();
  state.boxes.clear();
  state.detections.visible = state.show.detections;
  layout.add(state.detections);
  sites.group = new THREE.Group();
  layout.add(sites.group);
  clearSites();

  const now = performance.now();
  desc.pads.forEach((pad, i) => {
    const accent = ACCENTS[i % ACCENTS.length];
    state.pads.add(makePad(pad, accent, i));
    const model = makeDrone(accent);
    toThree(pad, model.root.position);
    model.root.scale.setScalar(0.001);
    layout.add(model.root);
    const pos = model.root.position.clone();
    const trail = makeTrail(accent, pos);
    state.trails.add(trail.line);
    const goalLine = makeGoalLine(accent);
    layout.add(goalLine);
    state.drones.set(i, {
      model, pos, yaw: 0, spin: 0, tilt: new THREE.Vector2(), born: now + i * 90, trail, goalLine,
    });
  });

  setLotBoundary(desc.lot);
  ui.sync(desc);
}

// ---------------------------------------------------------------------------
// Per-frame drone update
// ---------------------------------------------------------------------------
const _v = new THREE.Vector3();
const _tilt = new THREE.Vector2();
const easeOutBack = (t) => 1 + 2.2 * (t - 1) ** 3 + 1.2 * (t - 1) ** 2;

function updateDrones(frame, dt, now) {
  for (const [id, d] of state.drones) {
    const { root, rotors } = d.model;

    const age = Math.min(Math.max((now - d.born) / 550, 0), 1);
    root.scale.setScalar(Math.max(easeOutBack(age), 0.001));

    const report = frame?.drones[id];
    if (report) {
      // Python already interpolates between ticks; this only absorbs the jitter
      // of the bridge's round-trip so motion stays silky.
      toThree(report.pos, _v);
      d.pos.lerp(_v, damp(28, dt));

      let dy = report.yaw - d.yaw;
      dy = Math.atan2(Math.sin(dy), Math.cos(dy));
      d.yaw += dy * damp(14, dt);

      // Lean into the direction of travel, as a real quad must to move. The
      // reported velocity steps at the 20 Hz tick, so the lean is smoothed
      // rather than derived from its (spiky) rate of change.
      const hx = 0.06 * report.vel[0];
      const hy = 0.06 * report.vel[1];
      const c = Math.cos(d.yaw);
      const s = Math.sin(d.yaw);
      const pitch = THREE.MathUtils.clamp(hx * c + hy * s, -0.35, 0.35);
      const roll = THREE.MathUtils.clamp(-hx * s + hy * c, -0.35, 0.35);
      d.tilt.lerp(_tilt.set(pitch, roll), damp(5, dt));

      if (report.goal) {
        const points = d.goalLine.geometry.attributes.position;
        points.setXYZ(0, d.pos.x, d.pos.y, d.pos.z);
        toThree(report.goal, _v);
        points.setXYZ(1, _v.x, _v.y, _v.z);
        points.needsUpdate = true;
        d.goalLine.visible = true;
      } else {
        d.goalLine.visible = false;
      }
    }

    root.position.copy(d.pos);
    // Recorded even while hidden, so switching trails on shows the full history.
    if (report) pushTrail(d.trail, d.pos);
    root.rotation.set(-d.tilt.y, d.yaw, -d.tilt.x, 'YXZ');

    // A drone that has flown home and touched down powers off; the rotors
    // spool down rather than stop dead, as a real motor would.
    const airborne = d.pos.y > 0.05;
    const target = report?.landed ? 0 : airborne ? ROTOR_SPIN_AIRBORNE : ROTOR_SPIN_IDLE;
    d.spin += (target - d.spin) * damp(target > d.spin ? 6 : 1.5, dt);
    const blur = d.spin / ROTOR_SPIN_AIRBORNE;
    for (const r of rotors) {
      r.hub.rotation.y += r.dir * d.spin * dt;
      r.disc.material.opacity = 0.18 * blur;
    }
  }
}

function updateFrontiers(frame) {
  const points = state.frontierPoints;
  points.visible = state.show.frontiers;
  if (!frame || !state.show.frontiers) return;
  const n = Math.min(frame.frontiers.length, FRONTIER_CAPACITY);
  const attr = points.geometry.getAttribute('position');
  for (let i = 0; i < n; i++) {
    toThree(frame.frontiers[i], _v);
    attr.setXYZ(i, _v.x, _v.y, _v.z);
  }
  attr.needsUpdate = true;
  points.geometry.setDrawRange(0, n);
}

// ---------------------------------------------------------------------------
// Detections: a thin outline around each object perception has classified.
// Every box shares one unit-cube edge geometry, scaled and turned per object,
// and is keyed by track id, so an outline tightens in place as more of its
// object is seen. Outlines are depth-tested like the property itself: drawn
// through walls, every bush on the far side of the house would clutter the
// view of the near one.
// ---------------------------------------------------------------------------
const BOX_EDGES = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
keep.add(BOX_EDGES);
const boxMaterials = new Map(); // 'r,g,b' -> material

function makeOutline(color) {
  const key = color.join(',');
  let material = boxMaterials.get(key);
  if (!material) {
    material = new THREE.LineBasicMaterial({ color: new THREE.Color().setRGB(...color, THREE.SRGBColorSpace) });
    keep.add(material);
    boxMaterials.set(key, material);
  }
  return new THREE.LineSegments(BOX_EDGES, material);
}

function updateDetections(frame) {
  state.detections.visible = state.show.detections;
  if (!frame) return;
  const seen = new Set();
  for (const d of frame.detections) {
    let box = state.boxes.get(d.id);
    if (!box) {
      box = makeOutline(d.color);
      state.boxes.set(d.id, box);
      state.detections.add(box);
    }
    toThree(d.center, box.position);
    // Size is (length along yaw, width, height). Height is three.js local y
    // and width local z, and a yaw about world +Z is the same turn about
    // three.js +Y.
    box.scale.set(d.size[0], d.size[2], d.size[1]);
    box.rotation.y = d.yaw;
    seen.add(d.id);
  }
  for (const [id, box] of state.boxes) {
    if (seen.has(id)) continue;
    state.detections.remove(box);
    state.boxes.delete(id);
  }
}

// ---------------------------------------------------------------------------
// Battery sites. Once mapping is complete, Python suggests the best places on
// the house's walls for the battery (canopy.site) and every frame carries
// them. Each is drawn with the battery model inside an outline like the
// detection boxes -- light yellow when it meets every placement rule, orange
// when it breaks one -- that breathes in and out, strongest on the site the
// carousel is showing. A dashed line runs from the meter to each battery.
// ---------------------------------------------------------------------------
/** One full fade in and out of the highlight outline. */
const SITE_PULSE_MS = 1800;
/** Gap between the battery and its outline, metres, so the lines never z-fight. */
const SITE_OUTLINE_MARGIN = 0.06;
/** Where the camera sits for a site, in the site's frame: out from the wall,
 *  along it (so the meter beside it is in shot), and up. Metres. */
const SITE_VIEW = { out: 6.24, along: 2.16, up: 3.0 };
const SITE_GLIDE_MS = 1500;
/** Delay between the batteries popping in, so the eye follows them in rank order. */
const SITE_STAGGER_MS = 160;

let batteryKit = null;

const sites = {
  epoch: -1, // the mission these sites were drawn for; -1 until they arrive
  group: new THREE.Group(),
  items: [], // per site: { root, body, skins, outline, conduit, data, born }
  battery: null, // [width, depth, height] of the sited box, metres
  index: 0,
};

function clearSites() {
  sites.epoch = -1;
  sites.items = [];
  sites.index = 0;
  carousel.hide();
}

/** The outline's box, in the site frame, from the model's own bounds. */
function siteOutline(color) {
  const material = new THREE.LineBasicMaterial({
    color: new THREE.Color().setRGB(...color, THREE.SRGBColorSpace),
    transparent: true,
    opacity: 0,
    depthWrite: false,
  });
  const outline = new THREE.LineSegments(BOX_EDGES, material);
  const size = batteryKit.bounds.getSize(new THREE.Vector3()).addScalar(2 * SITE_OUTLINE_MARGIN);
  batteryKit.bounds.getCenter(outline.position);
  outline.scale.copy(size);
  return outline;
}

/** Meter to battery top: the conduit run the site implies. */
function siteConduit(s, color) {
  const top = [s.pos[0], s.pos[1], sites.battery[2]];
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([toThree(s.meter), toThree(top)]),
    new THREE.LineDashedMaterial({
      color: new THREE.Color().setRGB(...color, THREE.SRGBColorSpace),
      dashSize: 0.12,
      gapSize: 0.08,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    }),
  );
  line.computeLineDistances();
  return line;
}

function buildSites(site, now) {
  sites.epoch = state.epoch;
  if (!site.sites.length) {
    toast.show(`No battery site suggested: ${site.message ?? 'nothing met the rules'}`);
    return;
  }
  sites.battery = site.battery;
  site.sites.forEach((s, i) => {
    const root = new THREE.Group();
    toThree(s.pos, root.position);
    // A yaw about world +Z is the same turn about three.js +Y.
    root.rotation.y = s.yaw;
    const body = new THREE.Group();
    body.scale.setScalar(0.001);
    // Without the model (it failed to load) the dashed conduit still marks the site.
    const outline = batteryKit ? siteOutline(s.color) : null;
    // Each copy fades on its own beat, so it gets its own materials; the
    // model's authored opacity is kept as the ceiling of the fade.
    const skins = [];
    if (batteryKit) {
      for (const part of batteryKit.parts) {
        const skin = part.material.clone();
        skin.transparent = true;
        skin.userData.opacity = part.material.opacity;
        skins.push(skin);
        body.add(shadowed(new THREE.Mesh(part.geometry, skin)));
      }
      body.add(outline);
    }
    root.add(body);
    const conduit = siteConduit(s, s.color);
    sites.group.add(root, conduit);
    sites.items.push({ root, body, skins, outline, conduit, data: s, born: now + i * SITE_STAGGER_MS });
  });
  hint.dismiss();
  carousel.show(sites.items.map((it) => it.data));
  showSite(0);
}

/** Where to stand to look at site ``s``, and what to look at. */
function siteView(s) {
  const [, depth, height] = sites.battery;
  const n = [Math.cos(s.yaw), Math.sin(s.yaw)];
  const t = [-n[1], n[0]];
  const front = [s.pos[0] + (n[0] * depth) / 2, s.pos[1] + (n[1] * depth) / 2, height / 2];
  // Aim between the battery and the meter, so the distance between them reads.
  const target = front.map((c, k) => c + (s.meter[k] - c) * 0.25);
  // Stand on the meter's side, so it is in shot rather than behind the battery.
  const side = Math.sign((s.meter[0] - s.pos[0]) * t[0] + (s.meter[1] - s.pos[1]) * t[1]) || 1;
  const eye = [
    front[0] + n[0] * SITE_VIEW.out + t[0] * SITE_VIEW.along * side,
    front[1] + n[1] * SITE_VIEW.out + t[1] * SITE_VIEW.along * side,
    SITE_VIEW.up,
  ];
  return { eye: toThree(eye), target: toThree(target) };
}

function showSite(index) {
  const n = sites.items.length;
  if (!n) return;
  sites.index = ((index % n) + n) % n;
  const { eye, target } = siteView(sites.items[sites.index].data);
  fly.glideTo(eye, target, SITE_GLIDE_MS);
  carousel.render(sites.index);
}

function updateSites(frame, now) {
  if (frame?.site && sites.epoch !== state.epoch) buildSites(frame.site, now);
  const wave = 0.5 - 0.5 * Math.cos((now / SITE_PULSE_MS) * Math.PI * 2);
  sites.items.forEach((it, i) => {
    const age = Math.min(Math.max((now - it.born) / 650, 0), 1);
    it.body.scale.setScalar(Math.max(easeOutBack(age), 0.001));
    const active = i === sites.index;
    // Battery and outline fade in and out of transparency together; the site
    // on screen swings fully.
    const opacity = (active ? 0.15 + 0.85 * wave : 0.08 + 0.3 * wave) * age;
    if (it.outline) it.outline.material.opacity = opacity;
    for (const skin of it.skins) skin.opacity = skin.userData.opacity * opacity;
    it.conduit.material.opacity = (active ? 0.85 : 0.3) * age;
  });
}

// ---------------------------------------------------------------------------
// Camera: WASD / arrows to move, Space/Shift up/down, drag to look,
// middle-drag to pan, wheel to push in and out. Everything eases.
// ---------------------------------------------------------------------------
class FlyCamera {
  constructor(cam, dom) {
    this.cam = cam;
    this.dom = dom;
    // Framed to take in the whole 30 x 40 m lot plus its neighbours (out to
    // about +/-45 m), from the front-left and high enough to read as a map.
    this.pos = new THREE.Vector3(-28, 27, 32);
    this.vel = new THREE.Vector3();
    this.keys = new Set();
    this.drag = null;
    this.glide = null; // an eased flight to a viewpoint, see glideTo()
    this.lookAt(new THREE.Vector3(0, 0, 0));
    this.yaw = this.goalYaw;
    this.pitch = this.goalPitch;

    dom.addEventListener('pointerdown', (e) => this.onDown(e));
    dom.addEventListener('pointermove', (e) => this.onMove(e));
    dom.addEventListener('pointerup', (e) => this.onUp(e));
    dom.addEventListener('pointercancel', (e) => this.onUp(e));
    dom.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    dom.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('keydown', (e) => this.onKey(e, true));
    window.addEventListener('keyup', (e) => this.onKey(e, false));
    window.addEventListener('blur', () => this.keys.clear());
  }

  lookAt(target) {
    const d = target.clone().sub(this.pos).normalize();
    const yaw = Math.atan2(-d.x, -d.z);
    // Dragging winds the yaw past +/-pi; turn the short way round to the target.
    const turn = this.yaw === undefined ? 0 : Math.atan2(Math.sin(yaw - this.yaw), Math.cos(yaw - this.yaw));
    this.goalYaw = this.yaw === undefined ? yaw : this.yaw + turn;
    this.goalPitch = Math.asin(THREE.MathUtils.clamp(d.y, -1, 1));
  }

  /** Fly to ``pos`` over ``ms``, easing in and out, turning to keep ``target``
   *  in view the whole way. Any manual input takes the camera straight back. */
  glideTo(pos, target, ms) {
    this.glide = { from: this.pos.clone(), to: pos.clone(), target: target.clone(), t: 0, ms };
    this.vel.set(0, 0, 0);
  }

  onKey(e, down) {
    // Arrow keys belong to the slider while it has focus.
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLButtonElement) return;
    if (!MOVE_KEYS.has(e.code)) return;
    e.preventDefault();
    this.glide = null;
    if (down) this.keys.add(e.code);
    else this.keys.delete(e.code);
    hint.dismiss();
  }

  onDown(e) {
    this.glide = null;
    this.dom.focus();
    this.drag = { id: e.pointerId, x: e.clientX, y: e.clientY, pan: e.button === 1 };
    this.dom.setPointerCapture(e.pointerId);
    this.dom.classList.add('looking');
    hint.dismiss();
  }

  onMove(e) {
    if (!this.drag || e.pointerId !== this.drag.id) return;
    const dx = e.clientX - this.drag.x;
    const dy = e.clientY - this.drag.y;
    this.drag.x = e.clientX;
    this.drag.y = e.clientY;
    if (this.drag.pan) {
      const k = 0.0025 * Math.max(this.pos.y, 2);
      this.pos.addScaledVector(this.right(), -dx * k).addScaledVector(this.upVec(), dy * k);
    } else {
      this.goalYaw -= dx * 0.0032;
      this.goalPitch = THREE.MathUtils.clamp(this.goalPitch - dy * 0.0032, -1.45, 1.45);
    }
  }

  onUp(e) {
    if (!this.drag || e.pointerId !== this.drag.id) return;
    this.drag = null;
    this.dom.classList.remove('looking');
  }

  onWheel(e) {
    e.preventDefault();
    this.glide = null;
    const step = THREE.MathUtils.clamp(-e.deltaY, -240, 240) * 0.06;
    this.vel.addScaledVector(this.forward(true), step);
  }

  forward(full = false) {
    const cp = full ? Math.cos(this.pitch) : 1;
    return new THREE.Vector3(-Math.sin(this.yaw) * cp, full ? Math.sin(this.pitch) : 0, -Math.cos(this.yaw) * cp);
  }

  right() {
    return new THREE.Vector3(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
  }

  upVec() {
    return new THREE.Vector3().crossVectors(this.right(), this.forward(true));
  }

  axis(pos, neg) {
    const has = (codes) => codes.some((c) => this.keys.has(c));
    return (has(pos) ? 1 : 0) - (has(neg) ? 1 : 0);
  }

  update(dt) {
    if (this.glide) {
      const g = this.glide;
      g.t = Math.min(g.t + (dt * 1000) / g.ms, 1);
      const e = g.t < 0.5 ? 4 * g.t ** 3 : 1 - (-2 * g.t + 2) ** 3 / 2; // ease in-out cubic
      this.pos.lerpVectors(g.from, g.to, e);
      this.lookAt(g.target);
      if (g.t === 1) this.glide = null;
    }

    const wish = new THREE.Vector3()
      .addScaledVector(this.forward(), this.axis(['KeyW', 'ArrowUp'], ['KeyS', 'ArrowDown']))
      .addScaledVector(this.right(), this.axis(['KeyD', 'ArrowRight'], ['KeyA', 'ArrowLeft']))
      .add(new THREE.Vector3(0, this.axis(['Space'], ['ShiftLeft', 'ShiftRight']), 0));
    if (wish.lengthSq() > 0) wish.normalize();
    wish.multiplyScalar(8);

    // Keys steer toward a target velocity; with no input the camera glides to
    // a stop rather than halting dead.
    const keyed = wish.lengthSq() > 0;
    this.vel.lerp(wish, damp(keyed ? 7 : 4.5, dt));
    this.pos.addScaledVector(this.vel, dt);
    if (this.pos.y < 0.35) {
      this.pos.y = 0.35;
      this.vel.y = Math.max(this.vel.y, 0);
    }

    this.yaw += (this.goalYaw - this.yaw) * damp(16, dt);
    this.pitch += (this.goalPitch - this.pitch) * damp(16, dt);
    this.cam.position.copy(this.pos);
    this.cam.rotation.set(this.pitch, this.yaw, 0, 'YXZ');
  }
}

const MOVE_KEYS = new Set([
  'KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space',
  'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ShiftLeft', 'ShiftRight',
]);

// ---------------------------------------------------------------------------
// Chrome: hint, toast, gear + settings panel
// ---------------------------------------------------------------------------
const hint = {
  el: document.getElementById('hint'),
  timer: setTimeout(() => hint.dismiss(), 7000),
  dismiss() {
    clearTimeout(this.timer);
    this.el.classList.add('gone');
  },
};

const toast = {
  el: document.getElementById('toast'),
  timer: 0,
  show(text) {
    this.el.textContent = text;
    this.el.classList.add('show');
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.el.classList.remove('show'), 4200);
  },
};

/** The battery-site carousel at the bottom centre. It never advances on its
 *  own: the arrows (or the arrow keys while one has focus) step through the
 *  sites, and the camera glides to each. */
const carousel = {
  el: document.getElementById('sites'),
  prev: document.getElementById('site-prev'),
  next: document.getElementById('site-next'),
  count: document.getElementById('site-count'),
  card: document.getElementById('site-card'),
  title: document.getElementById('site-title'),
  meta: document.getElementById('site-meta'),
  warnings: document.getElementById('site-warnings'),
  sites: [],

  init() {
    this.prev.addEventListener('click', () => showSite(sites.index - 1));
    this.next.addEventListener('click', () => showSite(sites.index + 1));
    this.el.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      showSite(sites.index + (e.key === 'ArrowLeft' ? -1 : 1));
    });
  },

  show(list) {
    this.sites = list;
    const single = list.length < 2;
    this.prev.disabled = single;
    this.next.disabled = single;
    this.el.hidden = false;
    requestAnimationFrame(() => this.el.classList.add('open'));
  },

  hide() {
    this.sites = [];
    this.el.classList.remove('open');
    this.el.hidden = true;
  },

  render(index) {
    const s = this.sites[index];
    const flagged = s.warnings.length > 0;
    this.count.textContent = `${index + 1} / ${this.sites.length}`;
    this.title.textContent = `Battery site ${s.rank}`;
    const toMeter = s.breakdown.meter_distance;
    this.meta.textContent = typeof toMeter === 'number'
      ? `${toMeter.toFixed(2)} m (${(toMeter / 0.3048).toFixed(1)} ft) from the meter`
      : '';
    this.card.classList.toggle('flagged', flagged);
    this.card.style.setProperty('--site', `rgb(${s.color.map((c) => Math.round(c * 255)).join(' ')})`);
    this.warnings.replaceChildren(
      ...(flagged ? s.warnings : ['Meets every placement rule']).map((text) => {
        const li = document.createElement('li');
        li.textContent = text;
        return li;
      }),
    );
    // Restart the card's entrance so each step reads as a new slide.
    this.card.classList.remove('enter');
    void this.card.offsetWidth;
    this.card.classList.add('enter');
  },
};

/** Labelled stops under the time-speed slider. */
const SPEED_TICKS = [0.5, 1, 2, 5, 10];
/** Slider positions per decade-ish of speed; the input's max. */
const SPEED_STEPS = 1000;

const errorText = (err) => (err && (err.message || err.toString())) || 'Unknown error';

const ui = {
  gear: document.getElementById('gear'),
  playpause: document.getElementById('playpause'),
  restartBtn: document.getElementById('restart'),
  panel: document.getElementById('settings'),
  slider: document.getElementById('drones'),
  count: document.getElementById('drones-out'),
  ticks: document.getElementById('drone-ticks'),
  speed: document.getElementById('speed'),
  speedOut: document.getElementById('speed-out'),
  speedTicks: document.getElementById('speed-ticks'),
  speedRange: null, // [min, max] from the session, set on the first sync
  speedApplied: null,
  speedPending: null,
  newScene: document.getElementById('new-scene'),
  showPads: document.getElementById('show-pads'),
  showTrails: document.getElementById('show-trails'),
  showFrontiers: document.getElementById('show-frontiers'),
  showDetections: document.getElementById('show-detections'),
  hud: document.getElementById('hud'),
  applied: null,
  pending: null,
  debounce: 0,

  init() {
    this.gear.addEventListener('click', () => this.toggle());
    this.playpause.addEventListener('click', () => this.setPaused(!paused));
    this.restartBtn.addEventListener('click', () => this.restart());
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen()) this.toggle(false);
    });
    this.slider.addEventListener('input', () => {
      this.showCount(Number(this.slider.value), true);
      clearTimeout(this.debounce);
      this.debounce = setTimeout(() => this.applyCount(), 160);
    });
    this.speed.addEventListener('input', () => {
      this.showSpeed(this.speedAt(Number(this.speed.value)), true);
      this.applySpeed();
    });
    this.newScene.addEventListener('click', () => this.regenerate());
    this.showPads.addEventListener('change', () => {
      state.show.pads = this.showPads.checked;
      state.pads.visible = state.show.pads;
    });
    this.showTrails.addEventListener('change', () => {
      state.show.trails = this.showTrails.checked;
      state.trails.visible = state.show.trails;
    });
    this.showFrontiers.addEventListener('change', () => {
      state.show.frontiers = this.showFrontiers.checked;
    });
    this.showDetections.addEventListener('change', () => {
      state.show.detections = this.showDetections.checked;
      state.detections.visible = state.show.detections;
    });
  },

  /** Fetch and (re)build the property mesh only when the seed actually
   *  changed; a drone-count change reuses it but must forget past reveals. */
  async syncWorld(desc) {
    if (property.seed !== desc.seed) {
      buildProperty(await api().world());
    } else if (property.objects.length) {
      repaintGray();
    }
  },

  hudText(frame) {
    if (!frame) return '';
    const pct = (x) => `${Math.round(x * 100)}%`;
    const pausedTag = paused ? 'Paused · ' : '';
    if (frame.phase === 'done') return `${pausedTag}Done`;
    const label = frame.phase.charAt(0).toUpperCase() + frame.phase.slice(1);
    return `${pausedTag}${label} · Ground band ${pct(frame.coverage.ground_band)} · Surfaces ${pct(frame.coverage.total)}`;
  },

  // Pausing is purely presentational bookkeeping: the loop keeps polling so
  // a scene rebuilt while paused still gets its first pose, but it hands
  // Python zero elapsed time, so the simulation clock stands still.
  setPaused(on) {
    paused = on;
    this.playpause.setAttribute('aria-pressed', String(on));
    this.playpause.setAttribute('aria-label', on ? 'Resume' : 'Pause');
    this.hud.textContent = this.hudText(latest);
  },

  isOpen() {
    return this.panel.classList.contains('open');
  },

  toggle(open = !this.isOpen()) {
    this.panel.classList.toggle('open', open);
    this.panel.toggleAttribute('inert', !open);
    this.panel.setAttribute('aria-hidden', String(!open));
    this.gear.setAttribute('aria-expanded', String(open));
    if (!open) canvas.focus();
  },

  showCount(n, bump) {
    const max = Number(this.slider.max);
    const min = Number(this.slider.min);
    this.count.textContent = String(n);
    this.slider.style.setProperty('--fill', `${((n - min) / (max - min || 1)) * 100}%`);
    [...this.ticks.children].forEach((t, i) => t.classList.toggle('on', i + min <= n));
    if (bump) {
      this.count.classList.remove('bump');
      void this.count.offsetWidth; // restart the transition
      this.count.classList.add('bump');
      setTimeout(() => this.count.classList.remove('bump'), 180);
    }
  },

  // The slider position is a log-scale exponent: position p of SPEED_STEPS
  // maps to min * (max / min)^p, so each doubling of speed is the same drag.
  speedAt(pos) {
    const [min, max] = this.speedRange;
    const raw = min * (max / min) ** (pos / SPEED_STEPS);
    // One decimal is as fine as anyone can read; snap near 1x so real time is
    // easy to land on.
    const v = Math.abs(raw - 1) < 0.06 ? 1 : Math.round(raw * 10) / 10;
    return THREE.MathUtils.clamp(v, min, max);
  },

  speedPos(v) {
    const [min, max] = this.speedRange;
    return (Math.log(v / min) / Math.log(max / min)) * SPEED_STEPS;
  },

  showSpeed(v, bump) {
    this.speedOut.textContent = `${v}\u00d7`;
    const fill = this.speedPos(v) / SPEED_STEPS;
    this.speed.style.setProperty('--fill', `${fill * 100}%`);
    for (const t of this.speedTicks.children) t.classList.toggle('on', Number(t.dataset.v) <= v);
    if (bump) {
      this.speedOut.classList.remove('bump');
      void this.speedOut.offsetWidth; // restart the transition
      this.speedOut.classList.add('bump');
      setTimeout(() => this.speedOut.classList.remove('bump'), 180);
    }
  },

  syncSpeed(desc) {
    if (!this.speedRange) {
      this.speedRange = [desc.min_time_scale, desc.max_time_scale];
      this.speed.max = String(SPEED_STEPS);
      this.speedTicks.replaceChildren(
        ...SPEED_TICKS.filter((v) => v >= desc.min_time_scale && v <= desc.max_time_scale).map((v) => {
          const s = document.createElement('span');
          s.textContent = `${v}\u00d7`;
          s.dataset.v = String(v);
          s.style.setProperty('--at', String(this.speedPos(v) / SPEED_STEPS));
          return s;
        }),
      );
    }
    this.speedApplied = desc.time_scale;
    if (this.speedPending === null) {
      this.speed.value = String(Math.round(this.speedPos(desc.time_scale)));
      this.showSpeed(desc.time_scale, false);
    }
  },

  // As with the drone count, the last value the user settled on wins. Nothing
  // is rebuilt, so there is no debounce: the change should feel immediate.
  async applySpeed() {
    const want = this.speedAt(Number(this.speed.value));
    if (this.speedPending !== null || want === this.speedApplied) return;
    this.speedPending = want;
    try {
      this.speedApplied = (await api().set_time_scale(want)).time_scale;
    } catch (err) {
      toast.show(errorText(err));
    } finally {
      this.speedPending = null;
    }
    if (this.speedAt(Number(this.speed.value)) !== this.speedApplied) this.applySpeed();
  },

  sync(desc) {
    this.syncSpeed(desc);
    if (this.ticks.children.length !== desc.max_drones) {
      this.slider.max = String(desc.max_drones);
      this.ticks.replaceChildren(
        ...Array.from({ length: desc.max_drones }, (_, i) => {
          const s = document.createElement('span');
          s.textContent = String(i + 1);
          return s;
        }),
      );
    }
    this.applied = desc.drones;
    if (this.pending === null) {
      this.slider.value = String(desc.drones);
      this.showCount(desc.drones, false);
    }
  },

  // The last value the user settled on wins, however fast they drag.
  async applyCount() {
    const want = Number(this.slider.value);
    if (this.pending !== null || want === this.applied) return;
    this.pending = want;
    try {
      const desc = await api().set_drone_count(want);
      await this.syncWorld(desc);
      buildScene(desc);
    } catch (err) {
      toast.show(errorText(err));
    } finally {
      this.pending = null;
    }
    if (Number(this.slider.value) !== this.applied) this.applyCount();
  },

  // Same property and swarm, time back to zero. The world is not refetched:
  // syncWorld sees the unchanged seed and only repaints it grayscale, and
  // buildScene drops the old epoch's drones, trails, boxes and sites.
  async restart() {
    const btn = this.restartBtn;
    btn.disabled = true;
    btn.classList.remove('spin');
    void btn.offsetWidth; // restart the transition
    btn.classList.add('spin');
    try {
      const desc = await api().restart();
      await this.syncWorld(desc);
      buildScene(desc);
    } catch (err) {
      toast.show(errorText(err));
    } finally {
      setTimeout(() => {
        btn.disabled = false;
        btn.classList.remove('spin');
      }, 650);
    }
  },

  async regenerate() {
    const btn = this.newScene;
    btn.disabled = true;
    btn.classList.remove('spin');
    void btn.offsetWidth;
    btn.classList.add('spin');
    try {
      const desc = await api().new_scene();
      await this.syncWorld(desc);
      buildScene(desc);
    } catch (err) {
      toast.show(errorText(err));
    } finally {
      setTimeout(() => {
        btn.disabled = false;
        btn.classList.remove('spin');
      }, 700);
    }
  },
};

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
const api = () => window.pywebview.api;
const fly = new FlyCamera(camera, canvas);
let latest = null;
let pendingDt = 0;
let paused = false;
let inflight = false;
let last = performance.now();

function requestFrame() {
  if (inflight || state.epoch < 0) return;
  inflight = true;
  const send = pendingDt;
  pendingDt = 0;
  api()
    .frame(send)
    .then((frame) => {
      // A frame from a swarm that no longer exists is dropped; its revealed
      // ids are lost too, which is fine because the new epoch repaints
      // everything grayscale and starts fresh.
      if (frame.epoch !== state.epoch) return;
      latest = frame;
      revealTriangles(decodeI32(frame.revealed));
      ui.hud.textContent = ui.hudText(frame);
    })
    .catch((err) => toast.show(errorText(err)))
    .finally(() => {
      inflight = false;
    });
}

function tick(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  // The camera always runs on wall time; only the simulation freezes.
  const simDt = paused ? 0 : dt;
  pendingDt += simDt;
  requestFrame();
  fly.update(dt);
  updateDrones(latest, simDt, now);
  updateFrontiers(latest);
  updateDetections(latest);
  updateSites(latest, now);
  updatePads(now);
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight, false);
});

let started = false;
async function start() {
  if (started) return;
  started = true;
  ui.init();
  carousel.init();
  try {
    droneKit = makeDroneKit(await api().drone_model());
    batteryKit = makeBatteryKit(await api().battery_model());
    const desc = await api().scene();
    await ui.syncWorld(desc);
    buildScene(desc);
  } catch (err) {
    toast.show(errorText(err));
  }
  canvas.classList.add('ready');
  canvas.focus();
}

if (window.pywebview?.api) start();
else window.addEventListener('pywebviewready', start);

// Without the bridge (the page opened in a plain browser) there is no
// simulation to show; say so rather than render an empty world forever.
setTimeout(() => {
  if (!started) {
    canvas.classList.add('ready');
    toast.show('No simulation attached. Launch the viewer with `canopy-view`.');
  }
}, 2500);

requestAnimationFrame(tick);
