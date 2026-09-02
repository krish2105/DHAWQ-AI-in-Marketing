export type SpaceManifest = {
  n: number;
  extent: { min: number[]; max: number[] };
  signal: { hex: string; hue_degrees: number; chosen_candidate: string };
  variants: Record<string, { tile_px: number; sheet_px: number; tiles_per_sheet: number; tiles_per_side: number; n_instances: number; n_sheets: number }>;
  sheets: Record<string, string[]>;
};

export type SpaceData = {
  manifest: SpaceManifest;
  positions: Float32Array;
  colours: Uint8Array;
  ids: string[];
  meta: Record<string, { prod_name?: string; product_type_name?: string; colour_group_name?: string; index_group_name?: string }>;
  variant: "desktop" | "mobile";
};

/**
 * Load the frozen scene artefacts.
 *
 * THE CANONICAL INDEX IS THE INVARIANT (PLAN.md §8). positions[i*3..], colours
 * row i, atlas tile i and ids[i] are all the SAME article. Get it wrong and
 * every garment shows the wrong photograph — a bug that looks like a rendering
 * problem and is very hard to spot when 13,548 garments all look plausible.
 * So it is asserted here at load, loudly, in the browser.
 */
export async function loadSpace(mobile: boolean): Promise<SpaceData> {
  const variant = mobile ? "mobile" : "desktop";
  const [manifest, posBuf, colBuf, ids, meta] = await Promise.all([
    fetch("/static/space.json").then((r) => r.json()),
    fetch("/static/positions.bin").then((r) => r.arrayBuffer()),
    fetch(`/static/colours_${variant}.bin`).then((r) => r.arrayBuffer()),
    fetch("/static/article_ids.json").then((r) => r.json()),
    fetch("/static/meta.json").then((r) => r.json()),
  ]);

  const positions = new Float32Array(posBuf);
  const colours = new Uint8Array(colBuf);
  const n = manifest.variants[variant].n_instances;

  if (positions.length / 3 < n) {
    throw new Error(
      `canonical index violated: ${positions.length / 3} positions for ${n} instances`,
    );
  }
  if (colours.length / 3 !== n) {
    throw new Error(
      `canonical index violated: ${colours.length / 3} colours for ${n} instances`,
    );
  }
  if (ids.length < n) {
    throw new Error(`canonical index violated: ${ids.length} ids for ${n} instances`);
  }

  return { manifest, positions, colours, ids, meta, variant };
}
