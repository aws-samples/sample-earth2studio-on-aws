import React, { useState, useEffect, useCallback } from "react";
import { useAuth } from "./auth/AuthContext";
import LoginPage from "./components/LoginPage";
import Sidebar from "./components/Sidebar";
import ResultTabs from "./components/ResultTabs";
import EducationalGuide from "./components/EducationalGuide";
import {
  listEndpoints,
  runForecast,
  checkEndpointStatus,
  pollForResult,
} from "./api/forecast";
import type { Endpoint, ForecastResponse } from "./types/forecast";
import { getModelVariables } from "./config/variables";

/** Yesterday's date in YYYY-MM-DD format */
function getDefaultDate(): string {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return d.toISOString().slice(0, 10);
}

export default function App() {
  const { isAuthenticated, isLoading: authLoading, userEmail, signOut } =
    useAuth();

  // ---------- Sidebar state ----------
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [selectedEndpoint, setSelectedEndpoint] = useState("");
  const [date, setDate] = useState(getDefaultDate);
  const [cycle, setCycle] = useState(12);
  const [leadTime, setLeadTime] = useState(24);
  const [selectedVars, setSelectedVars] = useState(["t2m", "z500", "tcwv"]);

  // ---------- App state ----------
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ForecastResponse | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [endpointsLoading, setEndpointsLoading] = useState(false);

  // Derive model name from selected endpoint
  const selectedModel =
    endpoints.find((ep) => ep.endpoint_name === selectedEndpoint)?.model || "";

  // Reset selected vars when endpoint changes to only include supported vars
  const handleEndpointChange = useCallback(
    (name: string) => {
      setSelectedEndpoint(name);
      const model = endpoints.find((ep) => ep.endpoint_name === name)?.model || "";
      const supported = getModelVariables(model);
      setSelectedVars((prev) => {
        const filtered = prev.filter((v) => supported.includes(v));
        return filtered.length > 0 ? filtered : supported.filter((v) => ["t2m", "z500"].includes(v));
      });
    },
    [endpoints]
  );

  // ---------- Fetch endpoints on mount (when authenticated) ----------
  const fetchEndpoints = useCallback(async () => {
    setEndpointsLoading(true);
    try {
      const data = await listEndpoints();
      setEndpoints(data.endpoints);
      if (data.endpoints.length > 0 && !selectedEndpoint) {
        setSelectedEndpoint(data.endpoints[0].endpoint_name);
      }
    } catch (err) {
      console.error("Failed to fetch endpoints:", err);
    } finally {
      setEndpointsLoading(false);
    }
  }, [selectedEndpoint]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchEndpoints();
    }
  }, [isAuthenticated]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- Show loading spinner while checking auth ----------
  if (authLoading) {
    return (
      <div className="min-h-screen bg-dark-950 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block w-8 h-8 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mb-4" />
          <p className="text-dark-400 text-sm">Loading...</p>
        </div>
      </div>
    );
  }

  // ---------- Show login page if not authenticated ----------
  if (!isAuthenticated) {
    return <LoginPage />;
  }

  // ---------- Handlers ----------
  const handleRunForecast = async () => {
    if (!selectedEndpoint) {
      setError("Please select an endpoint first");
      return;
    }
    if (selectedVars.length === 0) {
      setError("Please select at least one variable");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setStatusMsg(null);

    const dateStr = `${date}T${String(cycle).padStart(2, "0")}:00:00`;

    try {
      const res = await runForecast({
        endpoint_name: selectedEndpoint,
        date: dateStr,
        lead_time_hours: leadTime,
        variables: selectedVars,
        return_grid: true,
      });

      if (res.status === "pending" && res.request_id) {
        setStatusMsg("Forecast is running... polling for result");
        const finalRes = await pollForResult(res.request_id, res.output_location || "");
        setResult(finalRes);
        setStatusMsg(null);
      } else if (res.status === "error") {
        setError(res.error || "Unknown error from SageMaker");
      } else {
        setResult(res);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Forecast request failed");
    } finally {
      setLoading(false);
    }
  };

  const handleCheckEndpoint = async () => {
    if (!selectedEndpoint) return;
    setStatusMsg(null);
    try {
      const res = await checkEndpointStatus(selectedEndpoint);
      setStatusMsg(
        res.status === "InService"
          ? `✅ ${selectedEndpoint} is InService`
          : `⚠️ ${selectedEndpoint} status: ${res.status}`
      );
    } catch (err) {
      setStatusMsg(
        `❌ ${err instanceof Error ? err.message : "Check failed"}`
      );
    }
  };

  // ---------- Render ----------
  const dateStr = `${date}T${String(cycle).padStart(2, "0")}:00:00`;

  return (
    <div className="flex h-screen bg-dark-950">
      {/* Sidebar */}
      <Sidebar
        endpoints={endpoints}
        selectedEndpoint={selectedEndpoint}
        selectedModel={selectedModel}
        onEndpointChange={handleEndpointChange}
        onRefreshEndpoints={fetchEndpoints}
        date={date}
        onDateChange={setDate}
        cycle={cycle}
        onCycleChange={setCycle}
        leadTime={leadTime}
        onLeadTimeChange={setLeadTime}
        selectedVars={selectedVars}
        onVarsChange={setSelectedVars}
        loading={loading || endpointsLoading}
      />

      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-6">
        {/* Header with user info + sign out */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-dark-50">
              🌤️ Earth2Studio Weather Forecast
            </h1>
            <p className="text-sm text-dark-400 mt-1">
              AI-powered global weather prediction using NVIDIA Earth2Studio on
              Amazon SageMaker
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-dark-400">{userEmail}</span>
            <button
              onClick={signOut}
              className="bg-dark-800 hover:bg-dark-700 border border-dark-600 text-dark-300 px-3 py-1.5 rounded-lg text-xs transition-colors"
            >
              Sign Out
            </button>
          </div>
        </div>

        {/* Action bar */}
        <div className="bg-dark-900 rounded-xl border border-dark-700 p-4 mb-6">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex-1 text-sm text-dark-300">
              <span className="font-mono bg-dark-800 px-2 py-0.5 rounded text-dark-200">
                {dateStr}
              </span>
              {" | Lead: "}
              <span className="text-dark-200 font-medium">{leadTime}h</span>
              {" | Vars: "}
              <span className="text-dark-200">
                {selectedVars.join(", ")}
              </span>
            </div>

            <button
              onClick={handleRunForecast}
              disabled={loading || !selectedEndpoint}
              className="bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 disabled:opacity-50 text-white px-5 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
            >
              {loading ? (
                <>
                  <span className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Running...
                </>
              ) : (
                "🚀 Run Forecast"
              )}
            </button>

            <button
              onClick={handleCheckEndpoint}
              disabled={loading || !selectedEndpoint}
              className="bg-dark-800 hover:bg-dark-700 disabled:opacity-50 border border-dark-600 text-dark-200 px-4 py-2 rounded-lg text-sm transition-colors"
            >
              🔍 Check Endpoint
            </button>
          </div>

          {statusMsg && (
            <div className="mt-3 text-sm text-dark-300">{statusMsg}</div>
          )}
        </div>

        {/* Error display */}
        {error && (
          <div className="bg-red-900/30 border border-red-700/50 rounded-xl p-4 mb-6">
            <p className="text-red-400 text-sm font-medium">❌ {error}</p>
          </div>
        )}

        {/* Success banner */}
        {result && result.status !== "error" && (
          <div className="bg-emerald-900/30 border border-emerald-700/50 rounded-xl p-4 mb-6">
            <p className="text-emerald-400 text-sm font-medium">
              ✅ Forecast complete! Model:{" "}
              <span className="font-bold">{result.model}</span> | Init:{" "}
              <span className="font-bold">{result.init_time}</span> | Steps:{" "}
              <span className="font-bold">
                {result.n_steps} × {result.step_hours}h
              </span>
              {result.full_res_s3 && (
                <span className="ml-2 text-emerald-300">
                  | 📦 Full-res archived ({result.full_res_size_mb} MB)
                </span>
              )}
            </p>
            {result.full_res_s3 && (
              <p className="text-emerald-600 text-xs mt-1 font-mono">
                {result.full_res_s3}
              </p>
            )}
          </div>
        )}

        {/* Result tabs */}
        {result && result.forecasts && (
          <ResultTabs result={result} selectedVars={selectedVars} />
        )}

        {/* Educational guide */}
        <div className="mt-8">
          <EducationalGuide />
        </div>
      </main>
    </div>
  );
}
