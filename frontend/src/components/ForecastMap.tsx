import React, { useMemo, useState, useCallback } from "react";
import Plot from "react-plotly.js";
import type { VariableForecast } from "../types/forecast";
import {
  VARIABLE_INFO,
  getPlotlyColorscale,
  isReversedColorscale,
} from "../config/variables";

type Projection = "natural earth" | "equirectangular" | "robinson" | "mollweide";

const PROJECTIONS: { id: Projection; label: string; icon: string }[] = [
  { id: "natural earth", label: "Natural Earth", icon: "🗺️" },
  { id: "robinson", label: "Robinson", icon: "🌐" },
  { id: "mollweide", label: "Mollweide", icon: "🥚" },
  { id: "equirectangular", label: "Flat", icon: "▭" },
];

const REGION_PRESETS: { label: string; icon: string; center: [number, number]; range?: { lon: [number, number]; lat: [number, number] } }[] = [
  { label: "World", icon: "🌍", center: [0, 20], range: { lon: [-180, 180], lat: [-90, 90] } },
  { label: "Europe", icon: "🇪🇺", center: [15, 50], range: { lon: [-15, 45], lat: [35, 72] } },
  { label: "N. America", icon: "🌎", center: [-95, 40], range: { lon: [-140, -50], lat: [15, 72] } },
  { label: "Asia", icon: "🌏", center: [100, 35], range: { lon: [55, 155], lat: [5, 60] } },
  { label: "S. America", icon: "🌎", center: [-60, -15], range: { lon: [-85, -30], lat: [-58, 15] } },
  { label: "Africa", icon: "🌍", center: [20, 5], range: { lon: [-25, 55], lat: [-40, 40] } },
  { label: "Australia", icon: "🌏", center: [135, -28], range: { lon: [110, 160], lat: [-45, -8] } },
];

interface ForecastMapProps {
  varName: string;
  varData: VariableForecast;
  validTime: string;
  stepIndex?: number;
  /** All forecasts for click-to-inspect */
  allForecasts?: Record<string, VariableForecast>;
  /** Callback to switch displayed variable */
  onVarSwitch?: (varName: string) => void;
  /** All available variable names for quick-switch */
  availableVars?: string[];
}

function downsampleGrid(
  grid: number[][],
  latArr: number[],
  lonArr: number[],
  scale: number,
  offset: number,
  maxPoints = 50000
) {
  const nRows = grid.length;
  const nCols = grid[0]?.length || 0;
  const totalPoints = nRows * nCols;
  const step = Math.max(1, Math.ceil(Math.sqrt(totalPoints / maxPoints)));

  const lats: number[] = [];
  const lons: number[] = [];
  const vals: number[] = [];

  for (let r = 0; r < nRows; r += step) {
    for (let c = 0; c < nCols; c += step) {
      const rawLon = lonArr[c];
      const lon = rawLon > 180 ? rawLon - 360 : rawLon;
      lats.push(latArr[r]);
      lons.push(lon);
      vals.push(grid[r][c] * scale + offset);
    }
  }

  return { lats, lons, vals };
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

  // Find nearest lat/lon indices
  let bestR = 0, bestC = 0, bestDist = Infinity;
  for (let r = 0; r < varData.grid_lat.length; r++) {
    const dLat = Math.abs(varData.grid_lat[r] - lat);
    if (dLat > bestDist) continue;
    for (let c = 0; c < varData.grid_lon.length; c++) {
      let gLon = varData.grid_lon[c];
      if (gLon > 180) gLon -= 360;
      const dLon = Math.abs(gLon - lon);
      const dist = dLat + dLon;
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

export default function ForecastMap({
  varName,
  varData,
  validTime,
  stepIndex,
  allForecasts,
  onVarSwitch,
  availableVars,
}: ForecastMapProps) {
  const [projection, setProjection] = useState<Projection>("natural earth");
  const [region, setRegion] = useState(0); // index into REGION_PRESETS
  const [inspectPoint, setInspectPoint] = useState<{ lat: number; lon: number } | null>(null);

  const varInfo = VARIABLE_INFO[varName] || {
    name: varName,
    unit: "?",
    offset: 0,
    scale: 1,
    cmap: "Turbo",
    range: [0, 1] as [number, number],
  };

  const geoData = useMemo(() => {
    if (!varData.grid_lat || !varData.grid_lon) return null;

    let gridToUse: number[][] | undefined;
    if (varData.grids && varData.grids.length > 0) {
      const idx = stepIndex !== undefined
        ? Math.min(stepIndex, varData.grids.length - 1)
        : varData.grids.length - 1;
      gridToUse = varData.grids[idx];
    } else {
      gridToUse = varData.grid;
    }
    if (!gridToUse) return null;

    return downsampleGrid(gridToUse, varData.grid_lat, varData.grid_lon, varInfo.scale, varInfo.offset);
  }, [varData, varInfo.scale, varInfo.offset, stepIndex]);

  // Click handler for inspect
  const handlePlotClick = useCallback((event: any) => {
    if (event?.points?.[0]) {
      const pt = event.points[0];
      setInspectPoint({ lat: pt.lat, lon: pt.lon });
    }
  }, []);

  // Build inspect data for all variables at clicked point
  const inspectData = useMemo(() => {
    if (!inspectPoint || !allForecasts) return null;
    const data: { varName: string; name: string; value: number; unit: string }[] = [];
    for (const [vn, vd] of Object.entries(allForecasts)) {
      const vi = VARIABLE_INFO[vn];
      if (!vi) continue;
      const val = findNearestValue(inspectPoint.lat, inspectPoint.lon, vd, stepIndex, vi.scale, vi.offset);
      if (val !== null) {
        data.push({ varName: vn, name: vi.name, value: val, unit: vi.unit });
      }
    }
    return data;
  }, [inspectPoint, allForecasts, stepIndex]);

  if (!geoData) {
    return (
      <div className="bg-dark-800 rounded-xl p-6 mb-4">
        <h3 className="text-lg font-semibold text-dark-200 mb-2">{varInfo.name}</h3>
        <p className="text-dark-400">Grid data not available.</p>
      </div>
    );
  }

  const plotlyColorscale = getPlotlyColorscale(varInfo.cmap);
  const reversed = isReversedColorscale(varInfo.cmap);
  const markerSize = 2.2;
  const currentRegion = REGION_PRESETS[region];

  return (
    <div className="bg-dark-800 rounded-xl overflow-hidden mb-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-1 px-4 pt-3 pb-1">
        {/* Variable quick-switch */}
        {availableVars && availableVars.length > 1 && onVarSwitch && (
          <div className="flex items-center gap-1 mr-3 pr-3 border-r border-dark-600">
            <span className="text-xs text-dark-400">Var:</span>
            <select
              value={varName}
              onChange={(e) => onVarSwitch(e.target.value)}
              className="bg-dark-700 text-dark-200 text-xs px-2 py-1 rounded border border-dark-600 focus:border-blue-500 focus:outline-none"
            >
              {availableVars.map((v) => (
                <option key={v} value={v}>
                  {VARIABLE_INFO[v]?.name || v}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Projection */}
        <span className="text-xs text-dark-400 mr-1">Proj:</span>
        {PROJECTIONS.map((p) => (
          <button
            key={p.id}
            onClick={() => setProjection(p.id)}
            className={`px-2 py-1 text-xs rounded-md transition-colors ${
              projection === p.id
                ? "bg-blue-600/30 text-blue-300 border border-blue-500/40"
                : "bg-dark-700 text-dark-400 border border-dark-600 hover:text-dark-200"
            }`}
            title={p.label}
          >
            {p.icon} {p.label}
          </button>
        ))}

        {/* Region zoom presets */}
        <div className="flex items-center gap-1 ml-3 pl-3 border-l border-dark-600">
          <span className="text-xs text-dark-400 mr-1">Region:</span>
          {REGION_PRESETS.map((r, i) => (
            <button
              key={r.label}
              onClick={() => setRegion(i)}
              className={`px-2 py-1 text-xs rounded-md transition-colors ${
                region === i
                  ? "bg-emerald-600/30 text-emerald-300 border border-emerald-500/40"
                  : "bg-dark-700 text-dark-400 border border-dark-600 hover:text-dark-200"
              }`}
              title={r.label}
            >
              {r.icon}
            </button>
          ))}
        </div>
      </div>

      <Plot
        data={[
          {
            type: "scattergeo" as const,
            mode: "markers" as const,
            lat: geoData.lats,
            lon: geoData.lons,
            marker: {
              color: geoData.vals,
              colorscale: plotlyColorscale,
              reversescale: reversed,
              cmin: varInfo.range[0],
              cmax: varInfo.range[1],
              size: markerSize,
              symbol: "square",
              opacity: 0.85,
              colorbar: {
                title: {
                  text: `${varInfo.name} (${varInfo.unit})`,
                  side: "right" as const,
                  font: { color: "#94a3b8", size: 12 },
                },
                thickness: 14,
                len: 0.8,
                tickfont: { color: "#94a3b8", size: 10 },
                bgcolor: "rgba(15,23,42,0.7)",
                bordercolor: "rgba(148,163,184,0.2)",
                borderwidth: 1,
              },
            },
            hovertemplate:
              `<b>${varInfo.name}</b><br>Lat: %{lat:.1f}°<br>Lon: %{lon:.1f}°<br>` +
              `Value: %{marker.color:.2f} ${varInfo.unit}<br><i>Click to inspect all vars</i><extra></extra>`,
          },
        ]}
        layout={{
          title: {
            text: `🌍 ${varInfo.name} — Valid: ${validTime}`,
            font: { size: 15, color: "#e2e8f0" },
            y: 0.97,
          },
          geo: {
            projection: { type: projection as string },
            center: { lon: currentRegion.center[0], lat: currentRegion.center[1] },
            ...(currentRegion.range && region !== 0
              ? {
                  lonaxis: { range: currentRegion.range.lon },
                  lataxis: { range: currentRegion.range.lat },
                }
              : {
                  lonaxis: { range: [-180, 180] },
                  lataxis: { range: [-90, 90] },
                }),
            showland: true,
            landcolor: "rgba(30,41,59,0.3)",
            showocean: true,
            oceancolor: "rgba(15,23,42,0.5)",
            showcoastlines: true,
            coastlinecolor: "rgba(148,163,184,0.5)",
            coastlinewidth: 0.8,
            showcountries: true,
            countrycolor: "rgba(148,163,184,0.25)",
            countrywidth: 0.5,
            showlakes: true,
            lakecolor: "rgba(15,23,42,0.4)",
            showframe: false,
            bgcolor: "#0f172a",
            resolution: 50,
          },
          height: 520,
          margin: { l: 0, r: 0, t: 40, b: 0 },
          paper_bgcolor: "#1e293b",
          font: { color: "#e2e8f0" },
          dragmode: "pan",
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ["lasso2d", "select2d"],
          scrollZoom: true,
        }}
        style={{ width: "100%" }}
        onClick={handlePlotClick}
      />

      {/* Click-to-inspect panel */}
      {inspectPoint && inspectData && inspectData.length > 0 && (
        <div className="mx-4 mb-4 bg-dark-900 border border-dark-600 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-dark-200">
              📍 Location: {inspectPoint.lat.toFixed(1)}°N, {inspectPoint.lon.toFixed(1)}°E
            </h4>
            <button
              onClick={() => setInspectPoint(null)}
              className="text-dark-400 hover:text-dark-200 text-xs px-2 py-1 rounded bg-dark-800 border border-dark-600"
            >
              ✕ Close
            </button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
            {inspectData.map((d) => (
              <div
                key={d.varName}
                className={`rounded-lg p-2.5 border transition-colors cursor-pointer ${
                  d.varName === varName
                    ? "bg-blue-600/15 border-blue-500/40"
                    : "bg-dark-800 border-dark-700 hover:border-dark-500"
                }`}
                onClick={() => onVarSwitch?.(d.varName)}
              >
                <div className="text-[10px] text-dark-400 truncate">{d.name}</div>
                <div className="text-sm font-mono font-semibold text-dark-100">
                  {d.value.toFixed(1)} <span className="text-xs text-dark-400">{d.unit}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
