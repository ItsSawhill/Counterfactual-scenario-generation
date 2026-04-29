import type {
  ForecastChartPoint,
  RecommendationBand,
  RiskCurvePoint,
  ScenarioAnalytics,
  ScenarioComparisonMetricKey,
  ScenarioComparisonRow,
  ScenarioKey,
  ScenarioPathStats,
  ScenarioPreset,
  SimulationBundle,
  SimulationRequest,
  SimulationResponse,
  TailAssessment,
  TerminalDistributionPoint,
} from "../types/simulation";

const CURRENCY_FORMATTER = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const PERCENT_FORMATTER = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const NUMBER_FORMATTER = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
});

const ASSET_DISPLAY_NAMES: Record<string, string> = {
  SPY: "S&P 500",
  QQQ: "Nasdaq 100",
  GC: "Gold Futures",
  CL: "Crude Oil Futures",
  ZN: "10Y Treasury Futures",
  GLD: "Gold ETF",
};

const SCENARIO_DISPLAY_NAMES: Record<ScenarioKey, string> = {
  custom: "Custom",
  baseCase: "Base Case",
  downside: "Sticky Inflation",
  upside: "Disinflation / Easing",
};

const DEFAULT_START_PRICES: Record<string, number> = {
  SPY: 450,
  QQQ: 380,
  GLD: 190,
};

export const SCENARIO_ORDER: ScenarioKey[] = [
  "custom",
  "baseCase",
  "downside",
  "upside",
];

export const SCENARIO_PRESETS: ScenarioPreset[] = [
  {
    regime: "Preset Match",
    label: "Base Case Macro Regime",
    description:
      "Balanced inflation, restrictive policy, and neutral dispersion.",
    whyItMatters:
      "Use this as the anchor case when you want to judge whether the custom path is adding real edge or just noise.",
    values: {
      inflation: 0.03,
      interestRate: 0.05,
      horizon: 30,
      pathCount: 120,
    },
  },
  {
    regime: "Inflation Shock",
    label: "Sticky Inflation",
    description: "Higher inflation with policy held tighter for longer.",
    whyItMatters:
      "This is the defensive macro test. If your thesis breaks here, the trade probably needs smaller size or a hedge.",
    values: {
      inflation: 0.05,
      interestRate: 0.06,
      horizon: 30,
      pathCount: 144,
    },
  },
  {
    regime: "Easing Shock",
    label: "Disinflation / Easing Regime",
    description: "Cooling CPI with easier policy and narrower terminal risk.",
    whyItMatters:
      "This is the constructive macro release valve and helps you see how much upside depends on policy easing.",
    values: {
      inflation: 0.02,
      interestRate: 0.035,
      horizon: 30,
      pathCount: 120,
    },
  },
];

export const BASELINE_SCENARIO = SCENARIO_PRESETS[0];

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(sortedValues: number[], quantile: number): number {
  if (sortedValues.length === 0) {
    return 0;
  }

  const index = (sortedValues.length - 1) * quantile;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);

  if (lower === upper) {
    return sortedValues[lower];
  }

  const weight = index - lower;
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

function standardDeviation(values: number[]): number {
  if (values.length <= 1) {
    return 0;
  }

  const mean = average(values);
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    (values.length - 1);

  return Math.sqrt(variance);
}

function skewness(values: number[]): number {
  if (values.length < 3) {
    return 0;
  }

  const mean = average(values);
  const sigma = standardDeviation(values);

  if (sigma === 0) {
    return 0;
  }

  const thirdMoment =
    values.reduce((sum, value) => sum + (value - mean) ** 3, 0) /
    values.length;

  return thirdMoment / sigma ** 3;
}

function kurtosis(values: number[]): number {
  if (values.length < 4) {
    return 0;
  }

  const mean = average(values);
  const sigma = standardDeviation(values);

  if (sigma === 0) {
    return 0;
  }

  const fourthMoment =
    values.reduce((sum, value) => sum + (value - mean) ** 4, 0) /
    values.length;

  return fourthMoment / sigma ** 4 - 3;
}

function averageOfWorstTail(
  sortedValues: number[],
  cutoffQuantile: number,
): number {
  if (sortedValues.length === 0) {
    return 0;
  }

  const threshold = percentile(sortedValues, cutoffQuantile);
  const tail = sortedValues.filter((value) => value <= threshold);
  return average(tail.length > 0 ? tail : [threshold]);
}

function maxDrawdown(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  let peak = values[0];
  let worstDrawdown = 0;

  values.forEach((value) => {
    peak = Math.max(peak, value);
    const drawdown = peak === 0 ? 0 : (peak - value) / peak;
    worstDrawdown = Math.max(worstDrawdown, drawdown);
  });

  return worstDrawdown;
}

function approximatelyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) < 0.000001;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function buildTailAssessment(
  customScenario: ScenarioPathStats,
  downsideScenario: ScenarioPathStats,
): TailAssessment {
  if (
    customScenario.var95 > 0.085 ||
    customScenario.cvar95 > 0.11 ||
    customScenario.downsideProbability > 0.55
  ) {
    return {
      label: "Fragile",
      detail:
        "Left-tail losses are wide enough that the custom path needs active hedging and smaller gross exposure.",
    };
  }

  if (
    customScenario.var95 > 0.06 ||
    customScenario.cvar95 > 0.08 ||
    downsideScenario.downsideProbability > 0.5
  ) {
    return {
      label: "Elevated",
      detail:
        "Tail losses are still investable, but the Sticky Inflation path can widen drawdown quickly if the edge fades.",
    };
  }

  return {
    label: "Contained",
    detail:
      "Tail risk is comparatively orderly, so sizing can lean on the custom versus base-case spread rather than on pure defense.",
  };
}

function buildPositionBand(
  customScenario: ScenarioPathStats,
  confidenceScore: number,
  baselineReturnDelta: number,
): RecommendationBand {
  const upper = clamp(
    0.03 +
      confidenceScore * 0.09 +
      clamp(customScenario.expectedReturn, -0.02, 0.1) * 0.55 -
      customScenario.var95 * 0.3 -
      customScenario.downsideProbability * 0.025,
    0.02,
    0.16,
  );
  const lowerShare = baselineReturnDelta >= 0 ? 0.58 : 0.36;
  const lower = clamp(
    upper * lowerShare,
    0.01,
    Math.max(upper - 0.015, 0.01),
  );

  return {
    lower,
    upper,
    detail:
      baselineReturnDelta >= 0
        ? "Use the upper half only while the custom path still outruns the base case."
        : "Stay in the lower half until the custom path retakes a positive edge versus the base case.",
  };
}

function buildStopBand(
  customScenario: ScenarioPathStats,
  signal: ScenarioAnalytics["signal"],
): RecommendationBand {
  const lower = clamp(
    Math.max(customScenario.var95 * 0.55, customScenario.maxDrawdown * 0.65, 0.035),
    0.03,
    0.09,
  );
  const upper = clamp(
    Math.max(customScenario.var95 * 0.95, customScenario.maxDrawdown * 0.85, lower + 0.015),
    0.05,
    0.14,
  );

  return {
    lower,
    upper,
    detail:
      signal === "Bullish"
        ? "Keep the stop band wide enough to survive the typical diffusion drawdown, not just the median path."
        : "Use the tighter end of the band unless the custom edge re-accelerates against Sticky Inflation.",
  };
}

function buildPricePaths(
  response: SimulationResponse,
  assetIndex: number,
): {
  asset: string;
  startPrice: number;
  horizon: number;
  pathCount: number;
  pricePaths: number[][];
} | null {
  const asset = response.assets[assetIndex];
  const rawPaths = response.paths;

  if (!asset || rawPaths.length === 0) {
    return null;
  }

  const startPrice = response.start_prices?.[assetIndex] ?? DEFAULT_START_PRICES[asset] ?? 100;
  const returnMean = response.return_scaler_mean[assetIndex] ?? 0;
  const returnScale = response.return_scaler_scale[assetIndex] ?? 1;

  const pricePaths = rawPaths.map((path) => {
    let runningPrice = startPrice;

    return path.map((step) => {
      const latentReturn = step[assetIndex] ?? 0;
      const logReturn = latentReturn * returnScale + returnMean;
      runningPrice *= Math.exp(logReturn);
      return runningPrice;
    });
  });

  return {
    asset,
    startPrice,
    horizon: response.horizon ?? pricePaths[0]?.length ?? 0,
    pathCount: response.path_count ?? rawPaths.length,
    pricePaths,
  };
}

function buildScenarioPathStats(
  key: ScenarioKey,
  label: string,
  source: {
    asset: string;
    startPrice: number;
    horizon: number;
    pathCount: number;
    pricePaths: number[][];
  },
): ScenarioPathStats {
  const timeSeriesStats = Array.from({ length: source.horizon }, (_, stepIndex) => {
    const stepValues = source.pricePaths
      .map((path) => path[stepIndex])
      .filter(Number.isFinite)
      .sort((left, right) => left - right);

    return {
      mean: average(stepValues),
      median: percentile(stepValues, 0.5),
      p5: percentile(stepValues, 0.05),
      p95: percentile(stepValues, 0.95),
    };
  });

  const terminalValues = source.pricePaths
    .map((path) => path[path.length - 1])
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  const terminalReturns = terminalValues
    .map((value) => value / source.startPrice - 1)
    .sort((left, right) => left - right);
  const expectedReturn = average(terminalReturns);
  const downsideProbability =
    terminalReturns.filter((value) => value < 0).length / terminalReturns.length;
  const confidenceLow = percentile(terminalValues, 0.05);
  const confidenceHigh = percentile(terminalValues, 0.95);
  const stdDeviation = standardDeviation(terminalReturns);

  return {
    key,
    label,
    asset: source.asset,
    startPrice: source.startPrice,
    horizon: source.horizon,
    pathCount: source.pathCount,
    pricePaths: source.pricePaths,
    meanPath: timeSeriesStats.map((point) => point.mean),
    medianPath: timeSeriesStats.map((point) => point.median),
    p5Path: timeSeriesStats.map((point) => point.p5),
    p95Path: timeSeriesStats.map((point) => point.p95),
    terminalValues,
    terminalReturns,
    expectedReturn,
    downsideProbability,
    confidenceLow,
    confidenceHigh,
    stdDeviation,
    skewness: skewness(terminalReturns),
    kurtosis: kurtosis(terminalReturns),
    var95: Math.max(0, -percentile(terminalReturns, 0.05)),
    var99: Math.max(0, -percentile(terminalReturns, 0.01)),
    cvar95: Math.max(0, -averageOfWorstTail(terminalReturns, 0.05)),
    cvar99: Math.max(0, -averageOfWorstTail(terminalReturns, 0.01)),
    sharpeRatio:
      stdDeviation === 0
        ? 0
        : (expectedReturn / stdDeviation) * Math.sqrt(252 / source.horizon),
    maxDrawdown: maxDrawdown(timeSeriesStats.map((point) => point.median)),
  };
}

export function formatCurrency(value: number): string {
  return CURRENCY_FORMATTER.format(value);
}

export function formatPercent(value: number): string {
  return PERCENT_FORMATTER.format(value);
}

export function formatSignedPercent(value: number): string {
  if (approximatelyEqual(value, 0)) {
    return "0.0%";
  }

  return `${value > 0 ? "+" : ""}${formatPercent(value)}`;
}

export function formatNumber(value: number): string {
  return NUMBER_FORMATTER.format(value);
}

export function formatAssetLabel(asset: string): string {
  return ASSET_DISPLAY_NAMES[asset] ?? asset;
}

export function formatScenarioLabel(key: ScenarioKey): string {
  return SCENARIO_DISPLAY_NAMES[key];
}

export function findMatchingPreset(
  params: SimulationRequest,
): ScenarioPreset | null {
  const matchingPreset = SCENARIO_PRESETS.find(({ values }) => {
    return (
      approximatelyEqual(params.inflation, values.inflation) &&
      approximatelyEqual(params.interestRate, values.interestRate) &&
      params.horizon === values.horizon &&
      params.pathCount === values.pathCount
    );
  });

  return matchingPreset ?? null;
}

export function describeRegime(params: SimulationRequest): string {
  return findMatchingPreset(params)?.label ?? "Custom";
}

export function buildMacroWarnings(params: SimulationRequest): string[] {
  const warnings: string[] = [];

  if (params.interestRate + 0.005 < params.inflation) {
    warnings.push(
      "Policy rate sits below inflation, implying a negative real-rate regime.",
    );
  }

  if (params.interestRate - params.inflation > 0.025) {
    warnings.push(
      "Rates exceed inflation by more than 250 bps, which implies a strongly restrictive setup.",
    );
  }

  if (params.pathCount < 96) {
    warnings.push("Tail metrics are less stable below 96 generated paths.");
  }

  if (params.horizon >= 28) {
    warnings.push(
      "This checkpoint is trained on a 30-day horizon, so the longest settings sit near the model boundary.",
    );
  }

  return warnings;
}

export function buildScenarioSuite(
  params: SimulationRequest,
  modelHorizon: number,
): {
  baseCase: SimulationRequest;
  downside: SimulationRequest;
  upside: SimulationRequest;
} {
  const shared = {
    horizon: Math.min(params.horizon, modelHorizon),
    pathCount: params.pathCount,
  };

  return {
    baseCase: {
      ...BASELINE_SCENARIO.values,
      ...shared,
    },
    downside: {
      ...params,
      ...shared,
      inflation: clamp(params.inflation + 0.01, 0.01, 0.09),
      interestRate: clamp(params.interestRate + 0.01, 0.01, 0.09),
    },
    upside: {
      ...params,
      ...shared,
      inflation: clamp(params.inflation - 0.01, 0.0, 0.09),
      interestRate: clamp(params.interestRate - 0.01, 0.0, 0.09),
    },
  };
}

export function buildScenarioAnalytics(
  bundle: SimulationBundle,
  params: SimulationRequest,
  assetIndex: number,
): ScenarioAnalytics | null {
  const rawScenarioMap = {
    custom: buildPricePaths(bundle.custom, assetIndex),
    baseCase: buildPricePaths(bundle.baseCase, assetIndex),
    downside: buildPricePaths(bundle.downside, assetIndex),
    upside: buildPricePaths(bundle.upside, assetIndex),
  };

  if (
    !rawScenarioMap.custom ||
    !rawScenarioMap.baseCase ||
    !rawScenarioMap.downside ||
    !rawScenarioMap.upside
  ) {
    return null;
  }

  const scenarios = {
    custom: buildScenarioPathStats(
      "custom",
      formatScenarioLabel("custom"),
      rawScenarioMap.custom,
    ),
    baseCase: buildScenarioPathStats(
      "baseCase",
      formatScenarioLabel("baseCase"),
      rawScenarioMap.baseCase,
    ),
    downside: buildScenarioPathStats(
      "downside",
      formatScenarioLabel("downside"),
      rawScenarioMap.downside,
    ),
    upside: buildScenarioPathStats(
      "upside",
      formatScenarioLabel("upside"),
      rawScenarioMap.upside,
    ),
  };

  const customScenario = scenarios.custom;
  const baseCaseScenario = scenarios.baseCase;
  const downsideScenario = scenarios.downside;
  const upsideScenario = scenarios.upside;
  const scenarioLabel = describeRegime(params);
  const baselineReturnDelta =
    customScenario.expectedReturn - baseCaseScenario.expectedReturn;
  const baselineDownsideDelta =
    customScenario.downsideProbability - baseCaseScenario.downsideProbability;

  let signal: ScenarioAnalytics["signal"] = "Neutral";

  if (
    customScenario.expectedReturn > 0.06 &&
    customScenario.downsideProbability < 0.35 &&
    baselineReturnDelta >= 0
  ) {
    signal = "Bullish";
  } else if (
    customScenario.expectedReturn < -0.01 ||
    customScenario.downsideProbability > 0.55 ||
    downsideScenario.expectedReturn < -0.03
  ) {
    signal = "Bearish";
  }

  const confidenceScore = clamp(
    0.55 +
      clamp(Math.abs(baselineReturnDelta) / 0.06, 0, 0.18) +
      clamp(Math.abs(customScenario.expectedReturn) / 0.12, 0, 0.2) -
      clamp(customScenario.stdDeviation / 0.22, 0, 0.18) -
      clamp(downsideScenario.var95 / 0.18, 0, 0.18),
    0,
    1,
  );

  let confidenceLabel: ScenarioAnalytics["confidenceLabel"] = "Medium";

  if (confidenceScore >= 0.72) {
    confidenceLabel = "High";
  } else if (confidenceScore <= 0.46) {
    confidenceLabel = "Low";
  }

  const tailAssessment = buildTailAssessment(customScenario, downsideScenario);
  const positionBand = buildPositionBand(
    customScenario,
    confidenceScore,
    baselineReturnDelta,
  );
  const stopBand = buildStopBand(customScenario, signal);

  const keyDrivers = [
    baselineReturnDelta >= 0
      ? `Custom expected return is ${formatSignedPercent(
          baselineReturnDelta,
        )} versus the base case.`
      : `Custom expected return trails the base case by ${formatPercent(
          Math.abs(baselineReturnDelta),
        )}.`,
    `Sticky Inflation lands near ${formatPercent(
      downsideScenario.expectedReturn,
    )} with ${formatPercent(downsideScenario.downsideProbability)} downside probability.`,
    `Disinflation / Easing lifts expected return toward ${formatPercent(
      upsideScenario.expectedReturn,
    )} with CVaR(95) near ${formatPercent(upsideScenario.cvar95)}.`,
  ];

  const suggestedAction =
    signal === "Bullish"
      ? "Add risk in stages, keep the stop band adaptive, and let the Sticky Inflation path set the gross ceiling rather than the median custom case."
      : signal === "Bearish"
        ? "Stay defensive, keep hedges on, and wait for the custom path to reclaim a cleaner edge over the base case."
        : "Stay balanced, lean on the lower half of the exposure band, and wait for wider separation between the custom and Sticky Inflation paths.";

  const hedgeTrigger =
    baselineReturnDelta >= 0
      ? `Trim or hedge once the custom edge versus Base Case turns negative, or if Sticky Inflation CVaR widens beyond ${formatPercent(
          downsideScenario.cvar95,
        )}.`
      : `Keep a partial hedge on until the custom path retakes a positive edge and Sticky Inflation downside probability moves back toward ${formatPercent(
          0.45,
        )}.`;

  const comparisonRows: ScenarioComparisonRow[] = SCENARIO_ORDER.map((key) => {
    const scenario = scenarios[key];
    return {
      key,
      label: scenario.label,
      expectedReturn: scenario.expectedReturn,
      downsideProbability: scenario.downsideProbability,
      var95: scenario.var95,
      cvar95: scenario.cvar95,
      medianTerminal: percentile(scenario.terminalValues, 0.5),
      meanTerminal: average(scenario.terminalValues),
    };
  });

  return {
    asset: customScenario.asset,
    scenarioLabel,
    startPrice: customScenario.startPrice,
    horizon: customScenario.horizon,
    pathCount: customScenario.pathCount,
    expectedReturn: customScenario.expectedReturn,
    downsideProbability: customScenario.downsideProbability,
    confidenceLow: customScenario.confidenceLow,
    confidenceHigh: customScenario.confidenceHigh,
    baselineReturnDelta,
    baselineDownsideDelta,
    stdDeviation: customScenario.stdDeviation,
    skewness: customScenario.skewness,
    kurtosis: customScenario.kurtosis,
    var95: customScenario.var95,
    var99: customScenario.var99,
    cvar95: customScenario.cvar95,
    cvar99: customScenario.cvar99,
    sharpeRatio: customScenario.sharpeRatio,
    maxDrawdown: customScenario.maxDrawdown,
    signal,
    confidenceLabel,
    confidenceScore,
    tailAssessment,
    positionBand,
    stopBand,
    hedgeTrigger,
    keyDrivers,
    suggestedAction,
    modelNote: `This view compares ${customScenario.pathCount} DDPM-generated paths across Custom, Base Case, Sticky Inflation, and Disinflation / Easing scenarios over ${customScenario.horizon} days. Tail statistics and scenario comparison are computed from the diffusion model's generated path distribution rather than from a single terminal estimate.`,
    scenarios,
    comparisonRows,
  };
}

export function buildForecastChartData(
  analytics: ScenarioAnalytics,
  focusKey: ScenarioKey,
  sampleCount = 4,
): ForecastChartPoint[] {
  const activeScenario = analytics.scenarios[focusKey];
  const visibleSampleKeys = activeScenario.pricePaths
    .slice(0, Math.min(activeScenario.pricePaths.length, sampleCount))
    .map((_, index) => `sample_${index}`);

  return Array.from({ length: analytics.horizon }, (_, stepIndex) => {
    const point: ForecastChartPoint = {
      step: stepIndex + 1,
      activeP5: activeScenario.p5Path[stepIndex],
      activeP95: activeScenario.p95Path[stepIndex],
      activeBandBase: activeScenario.p5Path[stepIndex],
      activeBandSize:
        activeScenario.p95Path[stepIndex] - activeScenario.p5Path[stepIndex],
      customMedian: analytics.scenarios.custom.medianPath[stepIndex],
      baseCaseMedian: analytics.scenarios.baseCase.medianPath[stepIndex],
      downsideMedian: analytics.scenarios.downside.medianPath[stepIndex],
      upsideMedian: analytics.scenarios.upside.medianPath[stepIndex],
    };

    visibleSampleKeys.forEach((sampleKey, sampleIndex) => {
      point[sampleKey] = activeScenario.pricePaths[sampleIndex][stepIndex];
    });

    return point;
  });
}

export function buildTerminalDistributionData(
  analytics: ScenarioAnalytics,
  focusKey: ScenarioKey,
  binCount = 16,
): TerminalDistributionPoint[] {
  const values = analytics.scenarios[focusKey].terminalReturns;

  if (values.length === 0) {
    return [];
  }

  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);

  if (approximatelyEqual(minValue, maxValue)) {
    return [
      {
        binStart: minValue,
        binEnd: maxValue,
        center: minValue,
        label: formatPercent(minValue),
        count: values.length,
      },
    ];
  }

  const width = (maxValue - minValue) / binCount;
  const counts = Array.from({ length: binCount }, () => 0);

  values.forEach((value) => {
    const normalized = (value - minValue) / width;
    const rawIndex = Math.floor(normalized);
    const index = Math.min(Math.max(rawIndex, 0), binCount - 1);
    counts[index] += 1;
  });

  return counts.map((count, index) => {
    const binStart = minValue + index * width;
    const binEnd = binStart + width;
    const center = binStart + width / 2;

    return {
      binStart,
      binEnd,
      center,
      label: `${formatPercent(binStart)} to ${formatPercent(binEnd)}`,
      count,
    };
  });
}

export function buildRiskCurveData(
  analytics: ScenarioAnalytics,
  focusKey: ScenarioKey,
  stopLossPercent: number,
): RiskCurvePoint[] {
  const activeScenario = analytics.scenarios[focusKey];
  const stopLossPrice = activeScenario.startPrice * (1 - stopLossPercent);

  return Array.from({ length: activeScenario.horizon }, (_, stepIndex) => {
    let belowSpotCount = 0;
    let stopLossCount = 0;
    let drawdownSum = 0;

    activeScenario.pricePaths.forEach((path) => {
      const pathSlice = path.slice(0, stepIndex + 1);
      const currentPrice = pathSlice[pathSlice.length - 1];
      const runningPeak = Math.max(...pathSlice);
      const worstSeenPrice = Math.min(...pathSlice);

      if (currentPrice < activeScenario.startPrice) {
        belowSpotCount += 1;
      }

      if (worstSeenPrice <= stopLossPrice) {
        stopLossCount += 1;
      }

      drawdownSum += runningPeak === 0 ? 0 : (runningPeak - currentPrice) / runningPeak;
    });

    return {
      step: stepIndex + 1,
      belowSpotProbability: belowSpotCount / activeScenario.pricePaths.length,
      stopLossProbability: stopLossCount / activeScenario.pricePaths.length,
      expectedDrawdown: drawdownSum / activeScenario.pricePaths.length,
    };
  });
}

export function getComparisonMetricLabel(
  metricKey: ScenarioComparisonMetricKey,
): string {
  return {
    expectedReturn: "Expected Return",
    downsideProbability: "Downside Probability",
    var95: "VaR (95%)",
    cvar95: "CVaR (95%)",
  }[metricKey];
}

export function getScenarioMetricValue(
  row: ScenarioComparisonRow,
  metricKey: ScenarioComparisonMetricKey,
): number {
  return row[metricKey];
}

export function formatScenarioMetricValue(
  metricKey: ScenarioComparisonMetricKey,
  value: number,
): string {
  return metricKey === "expectedReturn"
    ? formatSignedPercent(value)
    : formatPercent(value);
}
