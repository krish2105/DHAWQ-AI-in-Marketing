"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { buildAtlasArray, makeAtlasMaterial } from "./atlasMaterial";
import { token } from "@/lib/theme";
import type { SpaceData } from "@/lib/space";

const PLANE = 1.15;

function Cloud({
  data,
  selected,
  onSelect,
  onHover,
}: {
  data: SpaceData;
  selected: number;
  onSelect: (i: number) => void;
  onHover: (i: number) => void;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const matRef = useRef<THREE.ShaderMaterial | null>(null);
  const [atlasReady, setAtlasReady] = useState(false);
  const { gl } = useThree();

  const cfg = data.manifest.variants[data.variant];
  const n = cfg.n_instances;

  const geometry = useMemo(() => {
    const g = new THREE.PlaneGeometry(PLANE, PLANE);
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
    g.setAttribute("aTile", new THREE.InstancedBufferAttribute(tile, 1));
    g.setAttribute("aLayer", new THREE.InstancedBufferAttribute(layer, 1));
    g.setAttribute("aColour", new THREE.InstancedBufferAttribute(colour, 3));
    return g;
  }, [data, n, cfg]);

  // Placeholder 1x1 array so the scene renders IMMEDIATELY as dominant-colour
  // points; the real sheets stream in and swap. Never a blank canvas with a
  // spinner (§12.5).
  const material = useMemo(() => {
    const blank = new THREE.DataArrayTexture(
      new Uint8Array(4 * cfg.n_sheets), 1, 1, cfg.n_sheets,
    );
    blank.needsUpdate = true;
    const m = makeAtlasMaterial(blank, cfg.tiles_per_side);
    matRef.current = m;
    return m;
  }, [cfg]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const m = new THREE.Matrix4();
    for (let i = 0; i < n; i++) {
      m.setPosition(
        data.positions[i * 3],
        data.positions[i * 3 + 1],
        data.positions[i * 3 + 2],
      );
      mesh.setMatrixAt(i, m);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [data, n]);

  // Progressive texture load.
  useEffect(() => {
    let cancelled = false;
    const sheets = data.manifest.sheets[data.variant];
    Promise.all(
      sheets.map(
        (f) =>
          new Promise<HTMLImageElement>((res, rej) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => res(img);
            img.onerror = rej;
            img.src = `/static/${f}`;
          }),
      ),
    )
      .then((imgs) => {
        if (cancelled || !matRef.current) return;
        const tex = buildAtlasArray(imgs, cfg.sheet_px);
        tex.anisotropy = Math.min(4, gl.capabilities.getMaxAnisotropy());
        matRef.current.uniforms.uAtlas.value = tex;
        setAtlasReady(true);
      })
      .catch(() => setAtlasReady(false));
    return () => {
      cancelled = true;
    };
  }, [data, cfg, gl]);

  useEffect(() => {
    if (matRef.current) matRef.current.uniforms.uSelected.value = selected;
  }, [selected]);

  // The scene rebinds on theme change — but only its own chrome.
  useEffect(() => {
    const sync = () => {
      if (matRef.current) {
        matRef.current.uniforms.uSignal.value = new THREE.Color(
          token("--signal", "#0CF9E6"),
        );
      }
    };
    sync();
    window.addEventListener("dhawq-theme-change", sync);
    return () => window.removeEventListener("dhawq-theme-change", sync);
  }, []);

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, material, n]}
      frustumCulled={false}
      onClick={(e) => {
        e.stopPropagation();
        if (e.instanceId != null) onSelect(e.instanceId);
      }}
      onPointerMove={(e) => {
        e.stopPropagation();
        if (e.instanceId != null) onHover(e.instanceId);
      }}
      onPointerOut={() => onHover(-1)}
    />
  );
}

/** flyTo — the moment (§12.5). Eased, damped, never a jump cut. */
function FlyTo({ target }: { target: THREE.Vector3 | null }) {
  const { camera } = useThree();
  const from = useRef(new THREE.Vector3());
  const t = useRef(1);

  useEffect(() => {
    if (!target) return;
    from.current.copy(camera.position);
    t.current = 0;
  }, [target, camera]);

  useFrame((_, dt) => {
    if (!target || t.current >= 1) return;
    t.current = Math.min(1, t.current + dt / 0.9);
    const e = 1 - Math.pow(1 - t.current, 3); // cubic ease-out
    const dest = target.clone().add(new THREE.Vector3(0, 0, 9));
    camera.position.lerpVectors(from.current, dest, e);
    camera.lookAt(target);
  });
  return null;
}

function Neighbours({
  data,
  selected,
  neighbours,
}: {
  data: SpaceData;
  selected: number;
  neighbours: { index: number; weight: number }[];
}) {
  const [colour, setColour] = useState("#0CF9E6");
  useEffect(() => {
    const sync = () => setColour(token("--signal", "#0CF9E6"));
    sync();
    window.addEventListener("dhawq-theme-change", sync);
    return () => window.removeEventListener("dhawq-theme-change", sync);
  }, []);

  if (selected < 0 || !neighbours.length) return null;
  const a = new THREE.Vector3(
    data.positions[selected * 3],
    data.positions[selected * 3 + 1],
    data.positions[selected * 3 + 2],
  );

  return (
    <group>
      {neighbours.map((nb) => {
        const b = new THREE.Vector3(
          data.positions[nb.index * 3],
          data.positions[nb.index * 3 + 1],
          data.positions[nb.index * 3 + 2],
        );
        const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
        return (
          <primitive
            key={nb.index}
            object={
              new THREE.Line(
                geo,
                new THREE.LineBasicMaterial({
                  color: colour,
                  transparent: true,
                  // Opacity proportional to similarity (§12.5).
                  opacity: 0.15 + nb.weight * 0.75,
                }),
              )
            }
          />
        );
      })}
    </group>
  );
}

export function Scene({
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
  const [bg, setBg] = useState("#080807");
  useEffect(() => {
    const sync = () => setBg(token("--scene-bg", "#080807"));
    sync();
    window.addEventListener("dhawq-theme-change", sync);
    return () => window.removeEventListener("dhawq-theme-change", sync);
  }, []);

  const target =
    selected >= 0
      ? new THREE.Vector3(
          data.positions[selected * 3],
          data.positions[selected * 3 + 1],
          data.positions[selected * 3 + 2],
        )
      : null;

  return (
    <Canvas
      // dpr capped per §12.5 mobile budget.
      dpr={[1, 1.5]}
      camera={{ position: [0, 0, 95], fov: 55, far: 800 }}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      resize={{ debounce: 0, scroll: false }}
      onCreated={() => onReady?.()}
      style={{ background: bg, width: "100%", height: "100%", display: "block" }}
    >
      <color attach="background" args={[bg]} />
      <ambientLight intensity={1} />
      <Cloud data={data} selected={selected} onSelect={onSelect} onHover={onHover} />
      <Neighbours data={data} selected={selected} neighbours={neighbours} />
      <FlyTo target={target} />
      {/* No bloom. It would wash out product colour, which IS the content. */}
      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.5}
        zoomSpeed={0.8}
        maxDistance={320}
        minDistance={4}
      />
    </Canvas>
  );
}
