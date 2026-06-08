import React from "react";
import type { VariableForecast } from "../types/forecast";
import { VARIABLE_INFO } from "../config/variables";

interface StatsCardProps {
  varName: string;
  varData: VariableForecast;
}

export default function StatsCard({ varName, varData }: StatsCardProps) {
  const varInfo = VARIABLE_INFO[varName] || {
    name: varName,
    unit: "?",
    offset: 0,
    scale: 1,
  };

  const { offset, scale, unit } = varInfo;

  const vmin = varData.min * scale + offset;
  const vmax = varData.max * scale + offset;
  const vmean = varData.mean * scale + offset;
  const vstd = varData.std * scale;

  const stats = [
    { label: "Min", value: vmin, color: "text-blue-400" },
    { label: "Mean", value: vmean, color: "text-emerald-400" },
    { label: "Max", value: vmax, color: "text-red-400" },
    { label: "Std", value: vstd, color: "text-amber-400" },
  ];

  return (
    <div className="bg-dark-800 border border-dark-700 rounded-xl p-4">
      <h3 className="text-sm font-semibold text-dark-200 mb-1">
        {varInfo.name}
      </h3>
      <p className="text-xs text-dark-500 font-mono mb-3">{varName}</p>

      <div className="space-y-2">
        {stats.map((s) => (
          <div key={s.label} className="flex justify-between items-center">
            <span className="text-xs text-dark-400">{s.label}</span>
            <span className={`text-sm font-mono ${s.color}`}>
              {s.value.toFixed(2)} {unit}
            </span>
          </div>
        ))}
      </div>

      {varData.shape && (
        <p className="text-xs text-dark-500 mt-3 pt-2 border-t border-dark-700">
          Shape: {varData.shape.join(" × ")}
        </p>
      )}
    </div>
  );
}
