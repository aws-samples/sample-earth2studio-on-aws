import React, { useState, useEffect, useRef, useCallback } from "react";
import type { ForecastResponse } from "../types/forecast";
import ForecastMap from "./ForecastMap";
import GlobeView from "./GlobeView";
import StatsCard from "./StatsCard";

interface ResultTabsProps {
  result: ForecastResponse;
  selectedVars: string[];
}

type TabId = "map" | "globe" | "stats" | "json";

const TABS: { id: TabId; label: string; icon: string }[] = [
  { id: "map", label: "Map View", icon: "🗺️" },
  { id: "globe", label: "3D Globe", icon: "🌍" },
  { id: "stats", label: "Statistics", icon: "📊" },
  { id: "json", label: "Raw JSON", icon: "📝" },
];

/** Determine how many timestep frames are available from grids data */
function getNumSteps(result: ForecastResponse, selectedVars: string[]): number {
  const forecasts = result.forecasts || {};
  for (const varName of selectedVars) {
    const varData = forecasts[varName];
    if (varData?.grids && varData.grids.length > 1) {
      return varData.grids.length;
    }
  }
  return 1; // No multi-step grids, just a single frame
}

export default function ResultTabs({ result, selectedVars }: ResultTabsProps) {
  const [activeTab, setActiveTab] = useState<TabId>("map");
  const [displayVar, setDisplayVar] = useState(selectedVars[0] || "t2m");

  const forecasts = result.forecasts || {};
  const validTimes = result.valid_times || [];

  // Keep displayVar in sync if selectedVars changes
  React.useEffect(() => {
    if (!selectedVars.includes(displayVar) && selectedVars.length > 0) {
      setDisplayVar(selectedVars[0]);
    }
  }, [selectedVars, displayVar]);

  // ---------- Animation state ----------
  const numSteps = getNumSteps(result, selectedVars);
  const hasAnimation = numSteps > 1;

  const [stepIndex, setStepIndex] = useState(numSteps - 1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(800); // ms per frame
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Reset step when result changes
  useEffect(() => {
    setStepIndex(numSteps - 1);
    setPlaying(false);
  }, [result, numSteps]);

  // Animation loop
  useEffect(() => {
    if (playing && hasAnimation) {
      timerRef.current = setInterval(() => {
        setStepIndex((prev) => (prev + 1) % numSteps);
      }, speed);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing, hasAnimation, numSteps, speed]);

  const togglePlay = useCallback(() => setPlaying((p) => !p), []);
  const stepBack = useCallback(() => {
    setPlaying(false);
    setStepIndex((prev) => (prev - 1 + numSteps) % numSteps);
  }, [numSteps]);
  const stepForward = useCallback(() => {
    setPlaying(false);
    setStepIndex((prev) => (prev + 1) % numSteps);
  }, [numSteps]);
  const goToStart = useCallback(() => {
    setPlaying(false);
    setStepIndex(0);
  }, []);
  const goToEnd = useCallback(() => {
    setPlaying(false);
    setStepIndex(numSteps - 1);
  }, [numSteps]);

  // Current valid time for the step
  const currentValidTime =
    validTimes.length > 0
      ? validTimes[Math.min(stepIndex, validTimes.length - 1)] || validTimes[validTimes.length - 1]
      : "N/A";
  const validTo =
    validTimes.length > 0 ? validTimes[validTimes.length - 1] : "N/A";

  // Build display JSON (hide large grid data)
  const displayResult = JSON.parse(JSON.stringify(result, (_, v) => v));
  if (displayResult.forecasts) {
    for (const vd of Object.values(displayResult.forecasts) as Record<string, unknown>[]) {
      if (vd.grid && Array.isArray(vd.grid)) {
        const g = vd.grid as unknown[][];
        vd.grid = `[${g.length}×${g[0]?.length || 0} array — hidden]`;
      }
      if (vd.grids && Array.isArray(vd.grids)) {
        const gs = vd.grids as unknown[][][];
        vd.grids = `[${gs.length} steps × ${gs[0]?.length || 0}×${gs[0]?.[0]?.length || 0} — hidden]`;
      }
      if (vd.grid_lat && Array.isArray(vd.grid_lat)) {
        vd.grid_lat = `[${(vd.grid_lat as unknown[]).length} values — hidden]`;
      }
      if (vd.grid_lon && Array.isArray(vd.grid_lon)) {
        vd.grid_lon = `[${(vd.grid_lon as unknown[]).length} values — hidden]`;
      }
    }
  }

  return (
    <div>
      {/* Tab bar */}
      <div className="flex border-b border-dark-700 mb-4">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2.5 text-sm font-medium transition-colors border-b-2 ${
              activeTab === tab.id
                ? "border-blue-500 text-blue-400"
                : "border-transparent text-dark-400 hover:text-dark-200 hover:border-dark-500"
            }`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "map" && (
        <div>
          {/* ========== Animation Player ========== */}
          {hasAnimation && (
            <div className="bg-dark-900 border border-dark-700 rounded-xl p-4 mb-4">
              {/* Header row */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-dark-200">
                    ⏩ Forecast Animation
                  </span>
                  <span className="text-xs bg-blue-600/20 text-blue-300 px-2 py-0.5 rounded-full border border-blue-500/30">
                    {numSteps} steps
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-dark-400">Speed:</span>
                  {[
                    { label: "0.5×", ms: 1600 },
                    { label: "1×", ms: 800 },
                    { label: "2×", ms: 400 },
                    { label: "4×", ms: 200 },
                  ].map((s) => (
                    <button
                      key={s.label}
                      onClick={() => setSpeed(s.ms)}
                      className={`px-2 py-0.5 text-xs rounded transition-colors ${
                        speed === s.ms
                          ? "bg-blue-600/30 text-blue-300 border border-blue-500/40"
                          : "bg-dark-800 text-dark-400 border border-dark-600 hover:text-dark-200"
                      }`}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Transport controls */}
              <div className="flex items-center gap-2 mb-3">
                <button
                  onClick={goToStart}
                  className="p-1.5 rounded-lg bg-dark-800 text-dark-300 hover:text-dark-100 hover:bg-dark-700 border border-dark-600 transition-colors"
                  title="Go to start"
                >
                  ⏮
                </button>
                <button
                  onClick={stepBack}
                  className="p-1.5 rounded-lg bg-dark-800 text-dark-300 hover:text-dark-100 hover:bg-dark-700 border border-dark-600 transition-colors"
                  title="Previous step"
                >
                  ⏪
                </button>
                <button
                  onClick={togglePlay}
                  className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors border ${
                    playing
                      ? "bg-amber-600/20 text-amber-300 border-amber-500/40 hover:bg-amber-600/30"
                      : "bg-emerald-600/20 text-emerald-300 border-emerald-500/40 hover:bg-emerald-600/30"
                  }`}
                  title={playing ? "Pause" : "Play"}
                >
                  {playing ? "⏸ Pause" : "▶ Play"}
                </button>
                <button
                  onClick={stepForward}
                  className="p-1.5 rounded-lg bg-dark-800 text-dark-300 hover:text-dark-100 hover:bg-dark-700 border border-dark-600 transition-colors"
                  title="Next step"
                >
                  ⏩
                </button>
                <button
                  onClick={goToEnd}
                  className="p-1.5 rounded-lg bg-dark-800 text-dark-300 hover:text-dark-100 hover:bg-dark-700 border border-dark-600 transition-colors"
                  title="Go to end"
                >
                  ⏭
                </button>

                {/* Current time display */}
                <div className="ml-3 flex-1 text-right">
                  <span className="text-xs text-dark-400">Step </span>
                  <span className="text-sm font-mono font-medium text-dark-200">
                    {stepIndex + 1}/{numSteps}
                  </span>
                  <span className="text-xs text-dark-400 ml-2">Valid: </span>
                  <span className="text-sm font-mono font-medium text-blue-300">
                    {currentValidTime}
                  </span>
                </div>
              </div>

              {/* Timeline slider */}
              <div className="flex items-center gap-3">
                <span className="text-xs text-dark-400 font-mono w-14 text-right shrink-0">
                  {validTimes[0] ? validTimes[0].slice(11, 16) : "T+0"}
                </span>
                <input
                  type="range"
                  min={0}
                  max={numSteps - 1}
                  value={stepIndex}
                  onChange={(e) => {
                    setPlaying(false);
                    setStepIndex(parseInt(e.target.value));
                  }}
                  className="flex-1 h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer
                    [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4
                    [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full
                    [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:shadow-lg
                    [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-blue-300
                    [&::-webkit-slider-thumb]:cursor-grab"
                />
                <span className="text-xs text-dark-400 font-mono w-14 shrink-0">
                  {validTimes[validTimes.length - 1]
                    ? validTimes[validTimes.length - 1].slice(11, 16)
                    : `T+${result.lead_time_hours || 24}`}
                </span>
              </div>

              {/* Step markers */}
              <div className="flex justify-between mt-1 px-[3.75rem]">
                {validTimes.map((_, i) => (
                  <div
                    key={i}
                    className={`w-1.5 h-1.5 rounded-full transition-colors cursor-pointer ${
                      i === stepIndex
                        ? "bg-blue-400"
                        : i < stepIndex
                        ? "bg-dark-500"
                        : "bg-dark-700"
                    }`}
                    onClick={() => {
                      setPlaying(false);
                      setStepIndex(i);
                    }}
                    title={validTimes[i]}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Single map with variable quick-switch + click-to-inspect */}
          {forecasts[displayVar] ? (
            <ForecastMap
              key={displayVar}
              varName={displayVar}
              varData={forecasts[displayVar]}
              validTime={hasAnimation ? currentValidTime : validTo}
              stepIndex={hasAnimation ? stepIndex : undefined}
              allForecasts={forecasts}
              onVarSwitch={setDisplayVar}
              availableVars={selectedVars.filter((v) => forecasts[v])}
            />
          ) : (
            <div className="bg-dark-800 rounded-xl p-4 mb-4 text-amber-400 text-sm">
              ⚠️ Variable <code>{displayVar}</code> not in forecast output
            </div>
          )}
        </div>
      )}

      {activeTab === "globe" && (
        <div>
          {/* Animation player shared with map view */}
          {hasAnimation && (
            <div className="bg-dark-900 border border-dark-700 rounded-xl p-4 mb-4">
              <div className="flex items-center gap-2 mb-3">
                <button onClick={goToStart} className="p-1.5 rounded-lg bg-dark-800 text-dark-300 hover:text-dark-100 hover:bg-dark-700 border border-dark-600 transition-colors" title="Go to start">⏮</button>
                <button onClick={stepBack} className="p-1.5 rounded-lg bg-dark-800 text-dark-300 hover:text-dark-100 hover:bg-dark-700 border border-dark-600 transition-colors" title="Previous step">⏪</button>
                <button onClick={togglePlay} className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors border ${playing ? "bg-amber-600/20 text-amber-300 border-amber-500/40" : "bg-emerald-600/20 text-emerald-300 border-emerald-500/40"}`}>{playing ? "⏸ Pause" : "▶ Play"}</button>
                <button onClick={stepForward} className="p-1.5 rounded-lg bg-dark-800 text-dark-300 hover:text-dark-100 hover:bg-dark-700 border border-dark-600 transition-colors" title="Next step">⏩</button>
                <button onClick={goToEnd} className="p-1.5 rounded-lg bg-dark-800 text-dark-300 hover:text-dark-100 hover:bg-dark-700 border border-dark-600 transition-colors" title="Go to end">⏭</button>
                <div className="ml-3 flex-1 text-right">
                  <span className="text-xs text-dark-400">Step </span>
                  <span className="text-sm font-mono font-medium text-dark-200">{stepIndex + 1}/{numSteps}</span>
                  <span className="text-xs text-dark-400 ml-2">Valid: </span>
                  <span className="text-sm font-mono font-medium text-blue-300">{currentValidTime}</span>
                </div>
              </div>
              <input type="range" min={0} max={numSteps - 1} value={stepIndex} onChange={(e) => { setPlaying(false); setStepIndex(parseInt(e.target.value)); }} className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500" />
            </div>
          )}

          {/* Globe views */}
          {selectedVars.map((varName) => {
            if (!forecasts[varName]) {
              return (
                <div key={varName} className="bg-dark-800 rounded-xl p-4 mb-4 text-amber-400 text-sm">
                  ⚠️ Variable <code>{varName}</code> not in forecast output
                </div>
              );
            }
            return (
              <GlobeView
                key={varName}
                varName={varName}
                varData={forecasts[varName]}
                validTime={hasAnimation ? currentValidTime : validTo}
                stepIndex={hasAnimation ? stepIndex : undefined}
                allForecasts={forecasts}
              />
            );
          })}
        </div>
      )}

      {activeTab === "stats" && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(forecasts).map(([varName, varData]) => (
            <StatsCard key={varName} varName={varName} varData={varData} />
          ))}
        </div>
      )}

      {activeTab === "json" && (
        <div className="bg-dark-800 rounded-xl p-4 overflow-auto max-h-[600px]">
          <pre className="text-xs text-dark-300 font-mono whitespace-pre-wrap">
            {JSON.stringify(displayResult, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
