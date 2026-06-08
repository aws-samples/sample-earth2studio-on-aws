import React, { useState } from "react";

export default function EducationalGuide() {
  const [open, setOpen] = useState(false);

  return (
    <div className="bg-dark-800 border border-dark-700 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3 text-left hover:bg-dark-750 transition-colors"
      >
        <span className="text-sm font-semibold text-dark-200">
          📖 How to Use This App &amp; Read Weather Maps
        </span>
        <span
          className={`text-dark-400 transition-transform ${
            open ? "rotate-180" : ""
          }`}
        >
          ▼
        </span>
      </button>

      {open && (
        <div className="px-5 pb-5 text-sm text-dark-300 space-y-4 leading-relaxed">
          {/* What Is This */}
          <section>
            <h3 className="text-base font-semibold text-dark-100 mb-1">
              What Is This?
            </h3>
            <p>
              This app runs <strong>AI weather prediction models</strong> on a GPU in the cloud
              (Amazon SageMaker). Instead of traditional physics-based weather simulations
              (which take hours on supercomputers), these AI models produce{" "}
              <strong>global weather forecasts in seconds</strong> using deep learning.
            </p>
          </section>

          {/* How to Run a Forecast */}
          <section>
            <h3 className="text-base font-semibold text-dark-100 mb-1">
              How to Run a Forecast
            </h3>
            <ol className="list-decimal list-inside space-y-1">
              <li>
                <strong>Pick a Date</strong> in the sidebar — this is the "initial condition"
                (the starting weather state). Use a date that's at least{" "}
                <strong>6 hours in the past</strong>.
              </li>
              <li>
                <strong>Pick a GFS Cycle</strong> — GFS runs 4 times daily: 00, 06, 12, 18 UTC.
              </li>
              <li>
                <strong>Set Lead Time</strong> — How many hours into the future to forecast
                (6h to 240h / 10 days).
              </li>
              <li>
                <strong>Select Variables</strong> — Choose which weather quantities to visualize.
              </li>
              <li>
                Click <strong>🚀 Run Forecast</strong> and wait ~15 seconds.
              </li>
            </ol>
          </section>

          {/* Variable descriptions */}
          <section>
            <h3 className="text-base font-semibold text-dark-100 mb-2">
              Understanding the Weather Variables
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr className="border-b border-dark-600">
                    <th className="text-left py-2 px-2 text-dark-300">Variable</th>
                    <th className="text-left py-2 px-2 text-dark-300">Full Name</th>
                    <th className="text-left py-2 px-2 text-dark-300">What It Shows</th>
                  </tr>
                </thead>
                <tbody className="text-dark-400">
                  <tr className="border-b border-dark-700/50">
                    <td className="py-1.5 px-2 font-mono font-semibold text-dark-200">t2m</td>
                    <td className="py-1.5 px-2">2-meter Temperature</td>
                    <td className="py-1.5 px-2">Air temp at 2m above ground. 🔴 Red = warm, 🔵 Blue = cold.</td>
                  </tr>
                  <tr className="border-b border-dark-700/50">
                    <td className="py-1.5 px-2 font-mono font-semibold text-dark-200">z500</td>
                    <td className="py-1.5 px-2">500 hPa Geopotential Height</td>
                    <td className="py-1.5 px-2">The most important map in meteorology. Troughs = storms, Ridges = fair weather.</td>
                  </tr>
                  <tr className="border-b border-dark-700/50">
                    <td className="py-1.5 px-2 font-mono font-semibold text-dark-200">tcwv</td>
                    <td className="py-1.5 px-2">Total Column Water Vapor</td>
                    <td className="py-1.5 px-2">Total moisture in the atmosphere. Dark blue = moist (tropical), white = dry.</td>
                  </tr>
                  <tr className="border-b border-dark-700/50">
                    <td className="py-1.5 px-2 font-mono font-semibold text-dark-200">msl</td>
                    <td className="py-1.5 px-2">Mean Sea Level Pressure</td>
                    <td className="py-1.5 px-2">Low centers (&lt;1000 hPa) = cyclones/storms. High (&gt;1020 hPa) = fair weather.</td>
                  </tr>
                  <tr className="border-b border-dark-700/50">
                    <td className="py-1.5 px-2 font-mono font-semibold text-dark-200">u10m/v10m</td>
                    <td className="py-1.5 px-2">10m Wind Components</td>
                    <td className="py-1.5 px-2">U = east-west, V = north-south. Red = positive, blue = negative.</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* Reading the maps */}
          <section>
            <h3 className="text-base font-semibold text-dark-100 mb-1">
              Reading the Maps
            </h3>
            <ul className="list-disc list-inside space-y-1">
              <li><strong>X-axis</strong> = Longitude (-180° to 180°, where 0° = Greenwich, UK)</li>
              <li><strong>Y-axis</strong> = Latitude (-90° to 90°, where 0° = Equator)</li>
              <li><strong>Colors</strong> = Value of the weather variable (see colorbar)</li>
              <li><strong>Hover</strong> over any point to see exact lat/lon and value</li>
            </ul>
          </section>

          {/* Tips */}
          <section>
            <h3 className="text-base font-semibold text-dark-100 mb-1">
              Tips for Interpretation
            </h3>
            <ul className="list-disc list-inside space-y-1">
              <li>🌀 <strong>Cyclones</strong> appear as circular low-pressure patterns</li>
              <li>🌊 <strong>Fronts</strong> show up as sharp temperature gradients</li>
              <li>☁️ <strong>Storm tracks</strong> follow the 500 hPa troughs moving eastward</li>
              <li>🌴 <strong>Tropics</strong> are warm (t2m &gt; 25°C) with high moisture (tcwv &gt; 40 kg/m²)</li>
              <li>❄️ <strong>Polar regions</strong> show very low temperatures and low moisture</li>
            </ul>
          </section>
        </div>
      )}
    </div>
  );
}
