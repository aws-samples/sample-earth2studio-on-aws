/** TypeScript interfaces for the Earth2 Forecast API */

export interface Endpoint {
  endpoint_name: string;
  model: string;
  status: string;
  created: string;
}

export interface EndpointsResponse {
  endpoints: Endpoint[];
}

export interface ForecastRequest {
  endpoint_name: string;
  date: string;
  lead_time_hours: number;
  variables: string[];
  return_grid: boolean;
}

export interface VariableForecast {
  shape: number[];
  min: number;
  max: number;
  mean: number;
  std: number;
  unit: string;
  /** Last-step grid (backward compat) */
  grid?: number[][];
  /** Multi-step grids: one 2D array per forecast timestep */
  grids?: number[][][];
  grid_lat?: number[];
  grid_lon?: number[];
}

export interface ForecastResponse {
  status: string;
  model?: string;
  init_time?: string;
  lead_time_hours?: number;
  step_hours?: number;
  n_steps?: number;
  valid_times?: string[];
  forecasts?: Record<string, VariableForecast>;
  // Pending response fields
  request_id?: string;
  output_location?: string;
  message?: string;
  // Full-resolution data archive
  full_res_s3?: string;
  full_res_size_mb?: number;
  // Error fields
  error?: string;
  traceback?: string;
}

export interface EndpointStatus {
  endpoint_name: string;
  status: string;
  error?: string;
}
