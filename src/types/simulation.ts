export type SimulationStatus = "idle" | "loading" | "ready" | "error";

export type ScenarioKey = "custom" | "baseCase" | "downside" | "upside";

export type ScenarioComparisonMetricKey =
  | "expectedReturn"
  | "downsideProbability"
  | "var95"
  | "cvar95";

export interface SimulationRequest {
  inflation: number;
  interestRate: number;
  horizon: number;
  pathCount: number;
}

export interface LiveStateMeta {
  market_data_as_of?: string;
  model_input_as_of?: string;
  live_fetched_at?: string;
  cache_status?: "fresh" | "cached";
  cache_age_seconds?: number;
  cache_ttl_seconds?: number;
}

export interface SimulationResponse {
  paths: number[][][];
  assets: string[];
  return_scaler_mean: number[];
  return_scaler_scale: number[];
  start_prices?: number[];
  horizon?: number;
  path_count?: number;
  model_horizon?: number;
  live_state?: LiveStateMeta;
}

export interface SimulationBundle {
  custom: SimulationResponse;
  baseCase: SimulationResponse;
  downside: SimulationResponse;
  upside: SimulationResponse;
}

export interface ScenarioPreset {
  regime: string;
  label: string;
  description: string;
  whyItMatters: string;
  values: SimulationRequest;
}

export interface ScenarioPathStats {
  key: ScenarioKey;
  label: string;
  asset: string;
  startPrice: number;
  horizon: number;
  pathCount: number;
  pricePaths: number[][];
  meanPath: number[];
  medianPath: number[];
  p5Path: number[];
  p95Path: number[];
  terminalValues: number[];
  terminalReturns: number[];
  expectedReturn: number;
  downsideProbability: number;
  confidenceLow: number;
  confidenceHigh: number;
  stdDeviation: number;
  skewness: number;
  kurtosis: number;
  var95: number;
  var99: number;
  cvar95: number;
  cvar99: number;
  sharpeRatio: number;
  maxDrawdown: number;
}

export interface ScenarioComparisonRow {
  key: ScenarioKey;
  label: string;
  expectedReturn: number;
  downsideProbability: number;
  var95: number;
  cvar95: number;
  medianTerminal: number;
  meanTerminal: number;
}

export interface ForecastChartPoint {
  step: number;
  activeP5: number;
  activeP95: number;
  activeBandBase: number;
  activeBandSize: number;
  customMedian: number;
  baseCaseMedian: number;
  downsideMedian: number;
  upsideMedian: number;
  [key: string]: number;
}

export interface TerminalDistributionPoint {
  binStart: number;
  binEnd: number;
  center: number;
  label: string;
  count: number;
}

export interface RiskCurvePoint {
  step: number;
  belowSpotProbability: number;
  stopLossProbability: number;
  expectedDrawdown: number;
}

export interface RecommendationBand {
  lower: number;
  upper: number;
  detail: string;
}

export interface TailAssessment {
  label: "Contained" | "Elevated" | "Fragile";
  detail: string;
}

export interface ScenarioAnalytics {
  asset: string;
  scenarioLabel: string;
  startPrice: number;
  horizon: number;
  pathCount: number;
  expectedReturn: number;
  downsideProbability: number;
  confidenceLow: number;
  confidenceHigh: number;
  baselineReturnDelta: number;
  baselineDownsideDelta: number;
  stdDeviation: number;
  skewness: number;
  kurtosis: number;
  var95: number;
  var99: number;
  cvar95: number;
  cvar99: number;
  sharpeRatio: number;
  maxDrawdown: number;
  signal: "Bullish" | "Neutral" | "Bearish";
  confidenceLabel: "High" | "Medium" | "Low";
  confidenceScore: number;
  tailAssessment: TailAssessment;
  positionBand: RecommendationBand;
  stopBand: RecommendationBand;
  hedgeTrigger: string;
  keyDrivers: string[];
  suggestedAction: string;
  modelNote: string;
  scenarios: Record<ScenarioKey, ScenarioPathStats>;
  comparisonRows: ScenarioComparisonRow[];
}
