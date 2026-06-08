import React, { useMemo, useRef, useEffect, useState } from "react";
import Globe from "react-globe.gl";
import type { VariableForecast } from "../types/forecast";
import { VARIABLE_INFO } from "../config/variables";

interface GlobeViewProps {
  varName: string;
  varData: VariableForecast;
  validTime: string;
  stepIndex?: number;
  /** All forecasts for click-to-inspect */
  allForecasts?: Record<string, VariableForecast>;
}

/** Interpolate between two RGB colors */
function lerpColor(
  c1: [number, number, number],
  c2: [number, number, number],
  t: number
): [number, number, number] {
  return [
    Math.round(c1[0] + (c2[0] - c1[0]) * t),
    Math.round(c1[1] + (c2[1] - c1[1]) * t),
    Math.round(c1[2] + (c2[2] - c1[2]) * t),
  ];
}

/** Color scales as RGB stop arrays */
const COLOR_SCALES: Record<string, [number, [number, number, number]][]> = {
  RdBu: [
    [0, [5, 10, 172]],
    [0.35, [106, 137, 247]],
    [0.5, [245, 245, 245]],
    [0.65, [244, 109, 67]],
    [1, [178, 24, 43]],
  ],
  Viridis: [
    [0, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.5, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1, [253, 231, 37]],
  ],
  Blues: [
    [0, [247, 251, 255]],
    [0.25, [198, 219, 239]],
    [0.5, [107, 174, 214]],
    [0.75, [33, 113, 181]],
    [1, [8, 48, 107]],
  ],
  Turbo: [
    [0, [48, 18, 59]],
    [0.2, [67, 127, 225]],
    [0.4, [25, 206, 153]],
    [0.6, [175, 222, 53]],
    [0.8, [253, 149, 32]],
    [1, [122, 4, 3]],
  ],
};

function valueToColor(
  val: number,
  vmin: number,
  vmax: number,
  cmap: string,
  reversed: boolean
): string {
  let t = (val - vmin) / (vmax - vmin || 1);
  t = Math.max(0, Math.min(1, t));
  if (reversed) t = 1 - t;

  const scaleName = cmap.replace("_r", "");
  const scale = COLOR_SCALES[scaleName] || COLOR_SCALES.Turbo;

  // Find the two stops to interpolate between
  let i = 0;
  while (i < scale.length - 1 && scale[i + 1][0] < t) i++;
  if (i >= scale.length - 1) {
    const c = scale[scale.length - 1][1];
    return `rgba(${c[0]},${c[1]},${c[2]},0.85)`;
  }

  const [t0, c0] = scale[i];
  const [t1, c1] = scale[i + 1];
  const localT = (t - t0) / (t1 - t0 || 1);
  const c = lerpColor(c0, c1, localT);
  return `rgba(${c[0]},${c[1]},${c[2]},0.85)`;
}

interface GlobePoint {
  lat: number;
  lng: number;
  val: number;
  color: string;
  size: number;
}

/** Find nearest grid value for a click lat/lon */
function findNearestValue(
  lat: number,
  lon: number,
  varData: VariableForecast,
  stepIndex?: number,
  scale = 1,
  offset = 0
): number | null {
  const gridToUse =
    varData.grids && varData.grids.length > 0
      ? varData.grids[stepIndex !== undefined ? Math.min(stepIndex, varData.grids.length - 1) : varData.grids.length - 1]
      : varData.grid;
  if (!gridToUse || !varData.grid_lat || !varData.grid_lon) return null;

  let bestR = 0, bestC = 0, bestDist = Infinity;
  for (let r = 0; r < varData.grid_lat.length; r++) {
    const dLat = Math.abs(varData.grid_lat[r] - lat);
    if (dLat > bestDist) continue;
    for (let c = 0; c < varData.grid_lon.length; c++) {
      let gLon = varData.grid_lon[c];
      if (gLon > 180) gLon -= 360;
      const dist = dLat + Math.abs(gLon - lon);
      if (dist < bestDist) {
        bestDist = dist;
        bestR = r;
        bestC = c;
      }
    }
  }
  if (bestR < gridToUse.length && bestC < (gridToUse[0]?.length || 0)) {
    return gridToUse[bestR][bestC] * scale + offset;
  }
  return null;
}

export default function GlobeView({
  varName,
  varData,
  validTime,
  stepIndex,
  allForecasts,
}: GlobeViewProps) {
  const globeRef = useRef<any>(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const [inspectPoint, setInspectPoint] = useState<{ lat: number; lng: number } | null>(null);

  const varInfo = VARIABLE_INFO[varName] || {
    name: varName,
    unit: "?",
    offset: 0,
    scale: 1,
    cmap: "Turbo",
    range: [0, 1] as [number, number],
  };

  const reversed = varInfo.cmap.endsWith("_r");

  const points: GlobePoint[] = useMemo(() => {
    if (!varData.grid_lat || !varData.grid_lon) return [];

    // Pick grid
    let gridToUse: number[][] | undefined;
    if (varData.grids && varData.grids.length > 0) {
      const idx =
        stepIndex !== undefined
          ? Math.min(stepIndex, varData.grids.length - 1)
          : varData.grids.length - 1;
      gridToUse = varData.grids[idx];
    } else {
      gridToUse = varData.grid;
    }
    if (!gridToUse) return [];

    const latArr = varData.grid_lat;
    const lonArr = varData.grid_lon;
    // Use ALL available points for dense coverage
    const pts: GlobePoint[] = [];
    for (let r = 0; r < gridToUse.length; r++) {
      for (let c = 0; c < (gridToUse[r]?.length || 0); c++) {
        const rawLon = lonArr[c];
        const lng = rawLon > 180 ? rawLon - 360 : rawLon;
        const raw = gridToUse[r][c];
        const val = raw * varInfo.scale + varInfo.offset;
        pts.push({
          lat: latArr[r],
          lng,
          val,
          color: valueToColor(val, varInfo.range[0], varInfo.range[1], varInfo.cmap, reversed),
          size: 0.6,
        });
      }
    }
    return pts;
  }, [varData, varInfo, stepIndex, reversed]);

  // Auto-rotate
  useEffect(() => {
    if (globeRef.current) {
      const controls = globeRef.current.controls();
      if (controls) {
        controls.autoRotate = autoRotate;
        controls.autoRotateSpeed = 0.5;
      }
    }
  }, [autoRotate]);

  // Set initial view
  useEffect(() => {
    if (globeRef.current) {
      globeRef.current.pointOfView({ lat: 30, lng: 0, altitude: 2.2 }, 0);
    }
  }, []);

  if (points.length === 0) {
    return (
      <div className="bg-dark-800 rounded-xl p-6 mb-4">
        <h3 className="text-lg font-semibold text-dark-200 mb-2">
          {varInfo.name}
        </h3>
        <p className="text-dark-400">Grid data not available for globe view.</p>
      </div>
    );
  }

  return (
    <div className="bg-dark-800 rounded-xl overflow-hidden mb-4 relative">
      {/* Header overlay */}
      <div className="absolute top-3 left-4 z-10 flex items-center gap-3">
        <h3 className="text-sm font-medium text-dark-200 bg-dark-900/80 backdrop-blur-sm px-3 py-1.5 rounded-lg border border-dark-600">
          🌍 {varInfo.name} — {validTime}
        </h3>
        <button
          onClick={() => setAutoRotate((r) => !r)}
          className={`text-xs px-2.5 py-1.5 rounded-lg border backdrop-blur-sm transition-colors ${
            autoRotate
              ? "bg-blue-600/20 text-blue-300 border-blue-500/40"
              : "bg-dark-900/80 text-dark-400 border-dark-600"
          }`}
        >
          {autoRotate ? "🔄 Rotating" : "⏸ Paused"}
        </button>
      </div>

      {/* Color scale legend */}
      <div className="absolute bottom-4 right-4 z-10 bg-dark-900/80 backdrop-blur-sm rounded-lg border border-dark-600 p-2">
        <div className="text-[10px] text-dark-400 mb-1 text-center">
          {varInfo.name} ({varInfo.unit})
        </div>
        <div
          className="w-3 h-32 rounded-sm mx-auto"
          style={{
            background: `linear-gradient(to bottom, ${valueToColor(
              varInfo.range[1],
              varInfo.range[0],
              varInfo.range[1],
              varInfo.cmap,
              reversed
            )}, ${valueToColor(
              (varInfo.range[0] + varInfo.range[1]) / 2,
              varInfo.range[0],
              varInfo.range[1],
              varInfo.cmap,
              reversed
            )}, ${valueToColor(
              varInfo.range[0],
              varInfo.range[0],
              varInfo.range[1],
              varInfo.cmap,
              reversed
            )})`,
          }}
        />
        <div className="flex flex-col justify-between h-32 absolute right-9 top-6 text-[10px] text-dark-400">
          <span>{varInfo.range[1]}</span>
          <span>{((varInfo.range[0] + varInfo.range[1]) / 2).toFixed(0)}</span>
          <span>{varInfo.range[0]}</span>
        </div>
      </div>

      {/* Click-to-inspect panel */}
      {inspectPoint && allForecasts && (
        <div className="absolute bottom-4 left-4 z-10 bg-dark-900/90 backdrop-blur-sm rounded-xl border border-dark-600 p-3 max-w-sm">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-medium text-dark-200">
              📍 {inspectPoint.lat.toFixed(1)}°N, {inspectPoint.lng.toFixed(1)}°E
            </h4>
            <button
              onClick={() => setInspectPoint(null)}
              className="text-dark-400 hover:text-dark-200 text-xs px-1.5 py-0.5 rounded bg-dark-800 border border-dark-600"
            >
              ✕
            </button>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {Object.entries(allForecasts).map(([vn, vd]) => {
              const vi = VARIABLE_INFO[vn];
              if (!vi) return null;
              const val = findNearestValue(inspectPoint.lat, inspectPoint.lng, vd, stepIndex, vi.scale, vi.offset);
              if (val === null) return null;
              return (
                <div
                  key={vn}
                  className={`rounded-lg p-2 border ${
                    vn === varName
                      ? "bg-blue-600/15 border-blue-500/40"
                      : "bg-dark-800 border-dark-700"
                  }`}
                >
                  <div className="text-[9px] text-dark-400 truncate">{vi.name}</div>
                  <div className="text-xs font-mono font-semibold text-dark-100">
                    {val.toFixed(1)} <span className="text-[9px] text-dark-400">{vi.unit}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div style={{ height: 560, background: "#0a0e1a" }}>
        <Globe
          ref={globeRef}
          globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
          bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
          backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"
          pointsData={points}
          pointLat="lat"
          pointLng="lng"
          pointColor="color"
          pointAltitude={() => 0.005}
          pointRadius="size"
          pointsMerge={true}
          atmosphereColor="#3a7ecf"
          atmosphereAltitude={0.15}
          width={undefined}
          height={560}
          animateIn={true}
          onGlobeClick={({ lat, lng }: { lat: number; lng: number }) => {
            setInspectPoint({ lat, lng });
          }}
        />
      </div>
    </div>
  );
}
