"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { buildAtlasArray, makeAtlasMaterial } from "./atlasMaterial";
import { token } from "@/lib/theme";
import type { SpaceData } from "@/lib/space";

/**
 * The 3D embedding space, driven by three.js DIRECTLY.
 *
 * WHY NOT REACT-THREE-FIBER
 * R3F 9.7 mounts its canvas but never initialises its root under React 19.1
 * here: onCreated never fires, useFrame never runs, the drawing buffer stays at
 * the default 300x150, and NO error is thrown — a silently black canvas. WebGL2
 * itself is verified fully working (ANGLE/Metal, 2048 array layers). Shader
 * compilation, strict-mode double-mount, resize debounce and the scene contents
 * were all ruled out: a bare probe and a plain box fail identically.
 *
 * The scene is not complicated enough to need a reconciler. One InstancedMesh,
 * one material, one control. Owning the lifecycle directly removes a dependency
 * that was failing silently, and a silent failure in the signature moment of
 * the project is not a dependency worth keeping.
 *
 * PRODUCT TEXTURES ARE NEVER TINTED (§12.4). Only the ground, the LOD points
 * and the neighbour lines read theme tokens.
 */
export function SceneRaw({
  data,
  selected,
  neighbours,
  onSelect,
  onHover,
  onReady,
}: {
  data: SpaceData;
  selected: number;
  neighbours: { index: number; weight: number }[];
  onSelect: (i: number) => void;
  onHover: (i: number) => void;
  onReady?: () => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<any>(null);

  // ── build once ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const el = host.current;
    if (!el) return;

    const cfg = data.manifest.variants[data.variant];
    const n = cfg.n_instances;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // §12.5 budget
    renderer.setSize(el.clientWidth, el.clientHeight, false);
    el.appendChild(renderer.domElement);
    Object.assign(renderer.domElement.style, {
      width: "100%", height: "100%", display: "block", outline: "none",
    });

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      55, el.clientWidth / el.clientHeight, 0.1, 900,
    );
    camera.position.set(0, 0, 110);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.5;
    controls.minDistance = 3;
    controls.maxDistance = 340;

    // ── geometry: one InstancedMesh, one draw call ──────────────────────────
    const geo = new THREE.PlaneGeometry(1.15, 1.15);
    const tile = new Float32Array(n);
    const layer = new Float32Array(n);
    const colour = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      tile[i] = i % cfg.tiles_per_sheet;
      layer[i] = Math.floor(i / cfg.tiles_per_sheet);
      colour[i * 3] = data.colours[i * 3] / 255;
      colour[i * 3 + 1] = data.colours[i * 3 + 1] / 255;
      colour[i * 3 + 2] = data.colours[i * 3 + 2] / 255;
    }
    geo.setAttribute("aTile", new THREE.InstancedBufferAttribute(tile, 1));
    geo.setAttribute("aLayer", new THREE.InstancedBufferAttribute(layer, 1));
    geo.setAttribute("aColour", new THREE.InstancedBufferAttribute(colour, 3));

    const blank = new THREE.DataArrayTexture(
      new Uint8Array(4 * cfg.n_sheets), 1, 1, cfg.n_sheets,
    );
    blank.needsUpdate = true;
    const material = makeAtlasMaterial(blank, cfg.tiles_per_side);

    const mesh = new THREE.InstancedMesh(geo, material, n);
    mesh.frustumCulled = false;
    const m4 = new THREE.Matrix4();
    for (let i = 0; i < n; i++) {
      m4.setPosition(
        data.positions[i * 3], data.positions[i * 3 + 1], data.positions[i * 3 + 2],
      );
      mesh.setMatrixAt(i, m4);
    }
    mesh.instanceMatrix.needsUpdate = true;
    scene.add(mesh);

    const lines = new THREE.Group();
    scene.add(lines);

    // ── theme binding: chrome only, never the garments ──────────────────────
    const applyTheme = () => {
      scene.background = new THREE.Color(token("--scene-bg", "#080807"));
      material.uniforms.uSignal.value = new THREE.Color(token("--signal", "#0CF9E6"));
      lines.children.forEach((l: any) => {
        l.material.color = new THREE.Color(token("--signal", "#0CF9E6"));
      });
    };
    applyTheme();
    window.addEventListener("dhawq-theme-change", applyTheme);

    // ── progressive texture streaming: points first, photographs after ──────
    Promise.all(
      data.manifest.sheets[data.variant].map(
        (f) => new Promise<HTMLImageElement>((res, rej) => {
          const img = new Image();
          img.onload = () => res(img);
          img.onerror = rej;
          img.src = `/static/${f}`;
        }),
      ),
    ).then((imgs) => {
      const tex = buildAtlasArray(imgs, cfg.sheet_px);
      tex.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());
      material.uniforms.uAtlas.value = tex;
    }).catch(() => { /* stays on measured dominant colours */ });

    // ── picking ─────────────────────────────────────────────────────────────
    const ray = new THREE.Raycaster();
    const ptr = new THREE.Vector2();
    let lastHover = -1;

    const pick = (e: PointerEvent) => {
      const r = renderer.domElement.getBoundingClientRect();
      ptr.x = ((e.clientX - r.left) / r.width) * 2 - 1;
      ptr.y = -((e.clientY - r.top) / r.height) * 2 + 1;
      ray.setFromCamera(ptr, camera);
      const hit = ray.intersectObject(mesh, false)[0];
      return hit?.instanceId ?? -1;
    };

    let moveRaf = 0;
    const onMove = (e: PointerEvent) => {
      if (moveRaf) return;                      // throttle: raycast is not free
      moveRaf = requestAnimationFrame(() => {
        moveRaf = 0;
        const id = pick(e);
        if (id !== lastHover) { lastHover = id; onHover(id); }
      });
    };
    const onClick = (e: PointerEvent) => {
      const id = pick(e);
      if (id >= 0) onSelect(id);
    };
    renderer.domElement.addEventListener("pointermove", onMove);
    renderer.domElement.addEventListener("click", onClick as any);

    // ── flyTo ───────────────────────────────────────────────────────────────
    let fly: { from: THREE.Vector3; to: THREE.Vector3; look: THREE.Vector3; t: number } | null = null;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // ── loop ────────────────────────────────────────────────────────────────
    let raf = 0;
    const clock = new THREE.Clock();
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const dt = clock.getDelta();

      if (fly) {
        fly.t = Math.min(1, fly.t + dt / (reduced ? 0.01 : 0.9));
        const e = 1 - Math.pow(1 - fly.t, 3);            // cubic ease-out
        camera.position.lerpVectors(fly.from, fly.to, e);
        controls.target.lerp(fly.look, e);
        if (fly.t >= 1) fly = null;
      }
      controls.update();
      renderer.render(scene, camera);
    };
    tick();
    onReady?.();

    // ── resize ──────────────────────────────────────────────────────────────
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth, h = el.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    ro.observe(el);

    api.current = {
      material, lines, scene, camera, controls, data,
      flyTo(i: number) {
        const p = new THREE.Vector3(
          data.positions[i * 3], data.positions[i * 3 + 1], data.positions[i * 3 + 2],
        );
        fly = {
          from: camera.position.clone(),
          to: p.clone().add(new THREE.Vector3(0, 0, 7)),
          look: p, t: 0,
        };
      },
    };

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("dhawq-theme-change", applyTheme);
      renderer.domElement.removeEventListener("pointermove", onMove);
      renderer.domElement.removeEventListener("click", onClick as any);
      controls.dispose();
      geo.dispose();
      material.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, [data]);

  // ── selection: highlight + flyTo ───────────────────────────────────────────
  useEffect(() => {
    const a = api.current;
    if (!a) return;
    a.material.uniforms.uSelected.value = selected;
    if (selected >= 0) a.flyTo(selected);
  }, [selected]);

  // ── neighbour lines, opacity proportional to similarity ────────────────────
  useEffect(() => {
    const a = api.current;
    if (!a) return;
    while (a.lines.children.length) {
      const l = a.lines.children.pop();
      l.geometry.dispose(); l.material.dispose();
    }
    if (selected < 0) return;
    const p = a.data.positions;
    const from = new THREE.Vector3(p[selected * 3], p[selected * 3 + 1], p[selected * 3 + 2]);
    const colour = new THREE.Color(token("--signal", "#0CF9E6"));
    for (const nb of neighbours) {
      const to = new THREE.Vector3(p[nb.index * 3], p[nb.index * 3 + 1], p[nb.index * 3 + 2]);
      a.lines.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([from, to]),
        new THREE.LineBasicMaterial({
          color: colour, transparent: true, opacity: 0.15 + nb.weight * 0.75,
        }),
      ));
    }
  }, [selected, neighbours]);

  return <div ref={host} style={{ inlineSize: "100%", blockSize: "100%" }} />;
}
