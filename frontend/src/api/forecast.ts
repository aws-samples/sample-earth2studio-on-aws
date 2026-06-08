/** API client for the Earth2 Forecast backend (with Cognito auth) */

import type {
  EndpointsResponse,
  ForecastRequest,
  ForecastResponse,
  EndpointStatus,
} from "../types/forecast";
import { getIdToken } from "../auth/cognito";
import { DEV_MODE } from "../config/auth";

const API_BASE = "/api";

async function getAuthHeaders(): Promise<Record<string, string>> {
  // In dev mode, no Cognito token needed (Flask backend has no auth)
  if (DEV_MODE) {
    return { "Content-Type": "application/json" };
  }
  try {
    const token = await getIdToken();
    return {
      "Content-Type": "application/json",
      Authorization: token,
    };
  } catch {
    return { "Content-Type": "application/json" };
  }
}

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const authHeaders = await getAuthHeaders();
  const res = await fetch(url, {
    ...options,
    headers: {
      ...authHeaders,
      ...(options?.headers || {}),
    },
  });

  if (res.status === 401) {
    throw new Error("Unauthorized — please sign in again");
  }

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data as T;
}

/** GET /api/endpoints — list active SageMaker endpoints */
export async function listEndpoints(): Promise<EndpointsResponse> {
  return fetchJSON<EndpointsResponse>(`${API_BASE}/endpoints`);
}

/** POST /api/forecast — run a forecast */
export async function runForecast(
  request: ForecastRequest
): Promise<ForecastResponse> {
  return fetchJSON<ForecastResponse>(`${API_BASE}/forecast`, {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/** GET /api/status/{endpoint_name} — check endpoint status */
export async function checkEndpointStatus(
  endpointName: string
): Promise<EndpointStatus> {
  return fetchJSON<EndpointStatus>(
    `${API_BASE}/status/${encodeURIComponent(endpointName)}`
  );
}

/** Poll for a pending forecast result */
export async function pollForResult(
  requestId: string,
  outputLocation: string,
  maxAttempts = 60,
  intervalMs = 5000
): Promise<ForecastResponse> {
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((r) => setTimeout(r, intervalMs));
    try {
      const params = new URLSearchParams({ output_location: outputLocation });
      const res = await fetchJSON<ForecastResponse>(
        `${API_BASE}/forecast/${requestId}?${params}`
      );
      if (res.status !== "pending") return res;
    } catch {
      // continue polling
    }
  }
  throw new Error("Forecast timed out after polling");
}
