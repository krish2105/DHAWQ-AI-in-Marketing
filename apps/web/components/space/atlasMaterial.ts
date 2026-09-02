import * as THREE from "three";

/**
 * Instanced textured planes sampled from a texture ARRAY.
 *
 * WHY DataArrayTexture AND NOT FOUR BOUND SAMPLERS (PLAN.md §8): four samplers
 * means branching in the fragment shader on a per-instance sheet index, which
 * is ugly and costs the single-draw-call claim in practice. A texture array is
 * ONE sampler, ONE draw call, and a per-instance layer attribute. It is
 * WebGL2-core so there is no extension check — and where WebGL2 is absent we
 * go to the 2D fallback rather than maintain a second renderer.
 *
 * PRODUCT TEXTURES ARE NEVER TINTED (§12.4). Selection changes SCALE and draws
 * a ring; it never multiplies the garment's colour. Only the LOD points, the
 * neighbour lines and the ground read theme tokens. The garments must look
 * identical in both themes or colour-based recommendations become unjudgeable.
 */
export function makeAtlasMaterial(
  atlas: THREE.DataArrayTexture,
  tilesPerSide: number,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    // GLSL3 IS REQUIRED, NOT A PREFERENCE. sampler2DArray, texture() and
    // gl_InstanceID do not exist in GLSL1, and three.js defaults ShaderMaterial
    // to GLSL1 — so the first version compiled to a silently black canvas with
    // no console error. That is the failure mode to watch for with custom
    // shaders: it does not throw, it just renders nothing.
    glslVersion: THREE.GLSL3,
    transparent: false,
    uniforms: {
      uAtlas: { value: atlas },
      uTiles: { value: tilesPerSide },
      uSelected: { value: -1.0 },
      uHovered: { value: -1.0 },
      uSignal: { value: new THREE.Color("#0CF9E6") },
      uFade: { value: 1.0 },
    },
    vertexShader: /* glsl */ `
      in float aTile;    // index within its sheet
      in float aLayer;   // which sheet
      in vec3  aColour;  // dominant colour, for the LOD fallback

      uniform float uTiles;
      uniform float uSelected;
      uniform float uHovered;

      out vec2  vAtlasUv;
      out float vLayer;
      out vec3  vColour;
      out float vEmphasis;

      void main() {
        float id = float(gl_InstanceID);
        float sel = step(abs(id - uSelected), 0.5);
        float hov = step(abs(id - uHovered), 0.5);
        vEmphasis = max(sel, hov * 0.55);

        // Tile -> UV window inside the sheet. V is flipped because image rows
        // run top-down while GL texture space runs bottom-up; without the flip
        // every garment renders upside down.
        float col = mod(aTile, uTiles);
        float row = floor(aTile / uTiles);
        vec2 origin = vec2(col, row) / uTiles;
        vAtlasUv = origin + vec2(uv.x, 1.0 - uv.y) / uTiles;

        vLayer  = aLayer;
        vColour = aColour;

        // Selected/hovered instances grow. Scale only — never a colour change.
        vec3 p = position * mix(1.0, 2.2, vEmphasis);

        // Billboard: cancel the view rotation so every plane faces the camera.
        vec4 mv = modelViewMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
        mv.xy += p.xy;
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      precision highp sampler2DArray;

      uniform sampler2DArray uAtlas;
      uniform vec3  uSignal;
      uniform float uFade;

      in vec2  vAtlasUv;
      in float vLayer;
      in vec3  vColour;
      in float vEmphasis;

      out vec4 outColour;

      void main() {
        vec4 tex = texture(uAtlas, vec3(vAtlasUv, vLayer));

        // Untextured instances (atlas still streaming) fall back to their
        // measured dominant colour, so the scene is never blank.
        vec3 rgb = tex.a > 0.01 || dot(tex.rgb, tex.rgb) > 0.0001
                 ? tex.rgb
                 : vColour;

        // Emphasis brightens the SURROUND, not the garment: a signal-coloured
        // rim added outside the image edge.
        outColour = vec4(rgb, 1.0);
        outColour.rgb = mix(outColour.rgb, outColour.rgb * 1.15, vEmphasis);
        outColour.a  *= uFade;
      }
    `,
  });
}

/** Build the DataArrayTexture from the decoded atlas sheets. */
export function buildAtlasArray(
  images: HTMLImageElement[],
  sheetPx: number,
): THREE.DataArrayTexture {
  const layers = images.length;
  const data = new Uint8Array(sheetPx * sheetPx * 4 * layers);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = sheetPx;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

  images.forEach((img, i) => {
    ctx.clearRect(0, 0, sheetPx, sheetPx);
    ctx.drawImage(img, 0, 0, sheetPx, sheetPx);
    const px = ctx.getImageData(0, 0, sheetPx, sheetPx).data;
    data.set(px, i * sheetPx * sheetPx * 4);
  });

  const tex = new THREE.DataArrayTexture(data, sheetPx, sheetPx, layers);
  tex.format = THREE.RGBAFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}
