/** Variable metadata ported from the Streamlit app */

export interface VariableInfo {
  name: string;
  unit: string;
  offset: number;
  scale: number;
  cmap: string;
  range: [number, number];
}

export const VARIABLE_INFO: Record<string, VariableInfo> = {
  t2m: {
    name: "2m Temperature",
    unit: "°C",
    offset: -273.15,
    scale: 1,
    cmap: "RdBu_r",
    range: [-50, 50],
  },
  t850: {
    name: "Temperature at 850 hPa",
    unit: "°C",
    offset: -273.15,
    scale: 1,
    cmap: "RdBu_r",
    range: [-50, 40],
  },
  z500: {
    name: "Geopotential Height 500 hPa",
    unit: "dam",
    offset: 0,
    scale: 1 / 98.0665,
    cmap: "Viridis",
    range: [470, 600],
  },
  z1000: {
    name: "Geopotential Height 1000 hPa",
    unit: "dam",
    offset: 0,
    scale: 1 / 98.0665,
    cmap: "Viridis",
    range: [-50, 50],
  },
  z300: {
    name: "Geopotential Height 300 hPa",
    unit: "dam",
    offset: 0,
    scale: 1 / 98.0665,
    cmap: "Viridis",
    range: [800, 990],
  },
  z700: {
    name: "Geopotential Height 700 hPa",
    unit: "dam",
    offset: 0,
    scale: 1 / 98.0665,
    cmap: "Viridis",
    range: [230, 330],
  },
  tcwv: {
    name: "Total Column Water Vapor",
    unit: "kg/m²",
    offset: 0,
    scale: 1,
    cmap: "Blues",
    range: [0, 70],
  },
  msl: {
    name: "Mean Sea Level Pressure",
    unit: "hPa",
    offset: 0,
    scale: 0.01,
    cmap: "RdBu_r",
    range: [960, 1050],
  },
  sp: {
    name: "Surface Pressure",
    unit: "hPa",
    offset: 0,
    scale: 0.01,
    cmap: "RdBu_r",
    range: [500, 1050],
  },
  u10m: {
    name: "10m U-Wind",
    unit: "m/s",
    offset: 0,
    scale: 1,
    cmap: "RdBu_r",
    range: [-30, 30],
  },
  v10m: {
    name: "10m V-Wind",
    unit: "m/s",
    offset: 0,
    scale: 1,
    cmap: "RdBu_r",
    range: [-30, 30],
  },
};

/**
 * Supported display variables per model (intersection of model output and VARIABLE_INFO).
 * Model keys match the uppercase names returned by the backend (e.g. "DLWP", "FCN3").
 */
export const MODEL_VARIABLES: Record<string, string[]> = {
  DLWP: ["t2m", "t850", "z500", "z1000", "z700", "z300", "tcwv"],
  FCN3: ["t2m", "t850", "z500", "z1000", "z300", "z700", "tcwv", "msl", "u10m", "v10m"],
};

/** Get display variables available for a model. Falls back to all VARIABLE_INFO keys. */
export function getModelVariables(model: string): string[] {
  return MODEL_VARIABLES[model] || Object.keys(VARIABLE_INFO);
}

/** Plotly colorscale mappings (Plotly uses different names than matplotlib) */
export function getPlotlyColorscale(cmap: string): string {
  const mapping: Record<string, string> = {
    RdBu_r: "RdBu",
    Viridis: "Viridis",
    Blues: "Blues",
    Plasma: "Plasma",
    Turbo: "Turbo",
  };
  return mapping[cmap] || cmap;
}

/** Whether we need to reverse the colorscale (RdBu_r is reversed RdBu) */
export function isReversedColorscale(cmap: string): boolean {
  return cmap.endsWith("_r");
}
