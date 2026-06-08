import React from "react";
import { VARIABLE_INFO, getModelVariables } from "../config/variables";
import type { Endpoint } from "../types/forecast";

interface SidebarProps {
  endpoints: Endpoint[];
  selectedEndpoint: string;
  selectedModel: string;
  onEndpointChange: (name: string) => void;
  onRefreshEndpoints: () => void;
  date: string;
  onDateChange: (date: string) => void;
  cycle: number;
  onCycleChange: (cycle: number) => void;
  leadTime: number;
  onLeadTimeChange: (hours: number) => void;
  selectedVars: string[];
  onVarsChange: (vars: string[]) => void;
  loading: boolean;
}

const CYCLES = [0, 6, 12, 18];

export default function Sidebar({
  endpoints,
  selectedEndpoint,
  selectedModel,
  onEndpointChange,
  onRefreshEndpoints,
  date,
  onDateChange,
  cycle,
  onCycleChange,
  leadTime,
  onLeadTimeChange,
  selectedVars,
  onVarsChange,
  loading,
}: SidebarProps) {
  const toggleVar = (v: string) => {
    if (selectedVars.includes(v)) {
      onVarsChange(selectedVars.filter((x) => x !== v));
    } else {
      onVarsChange([...selectedVars, v]);
    }
  };

  return (
    <aside className="w-72 min-w-[288px] bg-dark-900 border-r border-dark-700 p-5 flex flex-col gap-5 overflow-y-auto">
      <div>
        <h2 className="text-sm font-semibold text-dark-300 uppercase tracking-wider mb-3">
          ⚙️ Forecast Settings
        </h2>

        {/* Endpoint count badge */}
        {endpoints.length > 0 ? (
          <div className="bg-emerald-900/30 border border-emerald-700/50 rounded-lg px-3 py-2 mb-3">
            <span className="text-emerald-400 text-sm font-medium">
              🟢 {endpoints.length} endpoint{endpoints.length !== 1 ? "s" : ""} running
            </span>
          </div>
        ) : (
          <div className="bg-amber-900/30 border border-amber-700/50 rounded-lg px-3 py-2 mb-3">
            <span className="text-amber-400 text-sm">
              ⚠️ No running earth2-* endpoints found
            </span>
          </div>
        )}

        {/* Endpoint selector */}
        <label className="block text-sm text-dark-300 mb-1">
          🔀 Model Endpoint
        </label>
        <select
          className="w-full bg-dark-800 border border-dark-600 rounded-lg px-3 py-2 text-sm text-dark-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={selectedEndpoint}
          onChange={(e) => onEndpointChange(e.target.value)}
          disabled={loading}
        >
          {endpoints.length === 0 && (
            <option value="">No endpoints available</option>
          )}
          {endpoints.map((ep) => (
            <option key={ep.endpoint_name} value={ep.endpoint_name}>
              {ep.endpoint_name} ({ep.model})
            </option>
          ))}
        </select>

        <button
          onClick={onRefreshEndpoints}
          disabled={loading}
          className="mt-2 w-full bg-dark-800 hover:bg-dark-700 border border-dark-600 rounded-lg px-3 py-1.5 text-sm text-dark-300 transition-colors disabled:opacity-50"
        >
          🔄 Refresh Endpoints
        </button>
      </div>

      {/* Date picker */}
      <div>
        <label className="block text-sm text-dark-300 mb-1">
          📅 Initial Condition Date
        </label>
        <input
          type="date"
          className="w-full bg-dark-800 border border-dark-600 rounded-lg px-3 py-2 text-sm text-dark-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={date}
          onChange={(e) => onDateChange(e.target.value)}
          disabled={loading}
        />
      </div>

      {/* GFS cycle */}
      <div>
        <label className="block text-sm text-dark-300 mb-1">
          🕐 GFS Cycle (UTC)
        </label>
        <div className="grid grid-cols-4 gap-1">
          {CYCLES.map((c) => (
            <button
              key={c}
              onClick={() => onCycleChange(c)}
              disabled={loading}
              className={`px-2 py-1.5 text-sm rounded-lg border transition-colors ${
                cycle === c
                  ? "bg-blue-600 border-blue-500 text-white"
                  : "bg-dark-800 border-dark-600 text-dark-300 hover:bg-dark-700"
              } disabled:opacity-50`}
            >
              {String(c).padStart(2, "0")}Z
            </button>
          ))}
        </div>
      </div>

      {/* Lead time slider */}
      <div>
        <label className="block text-sm text-dark-300 mb-1">
          ⏱️ Lead Time: <span className="text-dark-100 font-medium">{leadTime}h</span>
          <span className="text-dark-400 text-xs ml-1">({(leadTime / 24).toFixed(1)}d)</span>
        </label>
        <input
          type="range"
          min={6}
          max={240}
          step={6}
          value={leadTime}
          onChange={(e) => onLeadTimeChange(Number(e.target.value))}
          disabled={loading}
          className="w-full accent-blue-500"
        />
        <div className="flex justify-between text-xs text-dark-500">
          <span>6h</span>
          <span>120h</span>
          <span>240h</span>
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-dark-700" />

      {/* Variable selection */}
      <div>
        <h3 className="text-sm font-semibold text-dark-300 uppercase tracking-wider mb-2">
          🎨 Variables
        </h3>
        <div className="space-y-1">
          {Object.entries(VARIABLE_INFO)
            .filter(([key]) => getModelVariables(selectedModel).includes(key))
            .map(([key, info]) => (
            <label
              key={key}
              className="flex items-center gap-2 px-2 py-1 rounded-lg hover:bg-dark-800 cursor-pointer transition-colors"
            >
              <input
                type="checkbox"
                checked={selectedVars.includes(key)}
                onChange={() => toggleVar(key)}
                disabled={loading}
                className="rounded border-dark-600 bg-dark-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
              />
              <span className="text-sm text-dark-200">
                <span className="font-mono text-dark-400">{key}</span>
                {" — "}
                {info.name}
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Warning */}
      <div className="mt-auto border-t border-dark-700 pt-3">
        <p className="text-xs text-amber-400/80">
          ⚠️ Don't forget to delete the endpoint when done!
        </p>
        {selectedEndpoint && (
          <code className="block mt-1 text-xs text-dark-400 bg-dark-800 rounded px-2 py-1 break-all">
            python deploy.py --delete --endpoint-name {selectedEndpoint}
          </code>
        )}
      </div>
    </aside>
  );
}
