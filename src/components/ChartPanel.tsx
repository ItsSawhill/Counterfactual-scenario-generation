import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  Cell,
  ComposedChart,
  Label,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  buildForecastChartData,
  buildRiskCurveData,
  buildTerminalDistributionData,
  formatAssetLabel,
  formatCurrency,
  formatPercent,
  formatScenarioLabel,
  formatScenarioMetricValue,
  formatSignedPercent,
  getComparisonMetricLabel,
  getScenarioMetricValue,
  SCENARIO_ORDER,
} from "../lib/analytics";
import type {
  ForecastChartPoint,
  RiskCurvePoint,
  ScenarioAnalytics,
  ScenarioComparisonMetricKey,
  ScenarioKey,
  SimulationStatus,
  TerminalDistributionPoint,
} from "../types/simulation";

interface ChartPanelProps {
  analytics: ScenarioAnalytics | null;
  assets: string[];
  selectedAsset: number;
  onAssetChange: (nextIndex: number) => void;
  status: SimulationStatus;
  error: string | null;
}

interface ForecastTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: ForecastChartPoint }>;
  focusKey: ScenarioKey;
}

interface DistributionTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: TerminalDistributionPoint }>;
  totalCount: number;
}

interface RiskTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: RiskCurvePoint }>;
}

interface ComparisonTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: { label: string; metricValue: number } }>;
  metricKey: ScenarioComparisonMetricKey;
}

const SCENARIO_STYLES: Record<
  ScenarioKey,
  { stroke: string; fill: string; bar: string }
> = {
  custom: {
    stroke: "#4de2c5",
    fill: "rgba(77, 226, 197, 0.28)",
    bar: "#4de2c5",
  },
  baseCase: {
    stroke: "#92a5b8",
    fill: "rgba(146, 165, 184, 0.18)",
    bar: "#92a5b8",
  },
  downside: {
    stroke: "#ff7c7c",
    fill: "rgba(255, 124, 124, 0.22)",
    bar: "#ff7c7c",
  },
  upside: {
    stroke: "#74b7ff",
    fill: "rgba(116, 183, 255, 0.24)",
    bar: "#74b7ff",
  },
};

const FORECAST_DOMAIN_MIN_PADDING_PCT: Record<string, number> = {
  SPY: 0.006,
  QQQ: 0.008,
  GC: 0.005,
  CL: 0.012,
  ZN: 0.0025,
};

const FORECAST_DOMAIN_MAX_PADDING_PCT: Record<string, number> = {
  SPY: 0.02,
  QQQ: 0.024,
  GC: 0.016,
  CL: 0.032,
  ZN: 0.01,
};

const COMPARISON_METRIC_OPTIONS: ScenarioComparisonMetricKey[] = [
  "expectedReturn",
  "downsideProbability",
  "var95",
  "cvar95",
];

const SCENARIO_MEDIAN_KEYS: Record<ScenarioKey, keyof ForecastChartPoint> = {
  custom: "customMedian",
  baseCase: "baseCaseMedian",
  downside: "downsideMedian",
  upside: "upsideMedian",
};

function ForecastTooltip({
  active,
  payload,
  focusKey,
}: ForecastTooltipProps) {
  if (!active || !payload?.length) {
    return null;
  }

  const point = payload[0].payload;

  return (
    <div className="chart-tooltip">
      <p className="tooltip-title">Day {point.step}</p>
      <div className="tooltip-row">
        <span className="tooltip-label">{formatScenarioLabel(focusKey)}</span>
        <span className="tooltip-value">{formatCurrency(point[`${focusKey}Median`] ?? point.customMedian)}</span>
      </div>
      <div className="tooltip-row">
        <span className="tooltip-label">Base Case</span>
        <span className="tooltip-value">
          {formatCurrency(point.baseCaseMedian)}
        </span>
      </div>
      <div className="tooltip-row">
        <span className="tooltip-label">Sticky Inflation</span>
        <span className="tooltip-value">
          {formatCurrency(point.downsideMedian)}
        </span>
      </div>
      <div className="tooltip-row">
        <span className="tooltip-label">Disinflation / Easing</span>
        <span className="tooltip-value">
          {formatCurrency(point.upsideMedian)}
        </span>
      </div>
    </div>
  );
}

function DistributionTooltip({
  active,
  payload,
  totalCount,
}: DistributionTooltipProps) {
  if (!active || !payload?.length) {
    return null;
  }

  const point = payload[0].payload;
  const bucketShare = totalCount > 0 ? point.count / totalCount : 0;

  return (
    <div className="chart-tooltip">
      <p className="tooltip-title">{point.label}</p>
      <div className="tooltip-row">
        <span className="tooltip-label">Paths</span>
        <span className="tooltip-value">{point.count}</span>
      </div>
      <div className="tooltip-row">
        <span className="tooltip-label">Share</span>
        <span className="tooltip-value">{formatPercent(bucketShare)}</span>
      </div>
    </div>
  );
}

function RiskTooltip({ active, payload }: RiskTooltipProps) {
  if (!active || !payload?.length) {
    return null;
  }

  const point = payload[0].payload;

  return (
    <div className="chart-tooltip">
      <p className="tooltip-title">Day {point.step}</p>
      <div className="tooltip-row">
        <span className="tooltip-label">Below spot</span>
        <span className="tooltip-value">
          {formatPercent(point.belowSpotProbability)}
        </span>
      </div>
      <div className="tooltip-row">
        <span className="tooltip-label">Stop-loss breach</span>
        <span className="tooltip-value">
          {formatPercent(point.stopLossProbability)}
        </span>
      </div>
      <div className="tooltip-row">
        <span className="tooltip-label">Expected drawdown</span>
        <span className="tooltip-value">
          {formatPercent(point.expectedDrawdown)}
        </span>
      </div>
    </div>
  );
}

function ComparisonTooltip({
  active,
  payload,
  metricKey,
}: ComparisonTooltipProps) {
  if (!active || !payload?.length) {
    return null;
  }

  const point = payload[0].payload;

  return (
    <div className="chart-tooltip">
      <p className="tooltip-title">{point.label}</p>
      <div className="tooltip-row">
        <span className="tooltip-label">{getComparisonMetricLabel(metricKey)}</span>
        <span className="tooltip-value">
          {formatScenarioMetricValue(metricKey, point.metricValue)}
        </span>
      </div>
    </div>
  );
}

function medianFromSorted(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  const middle = Math.floor(values.length / 2);
  return values.length % 2 === 0
    ? (values[middle - 1] + values[middle]) / 2
    : values[middle];
}

function ChartPanel({
  analytics,
  assets,
  selectedAsset,
  onAssetChange,
  status,
  error,
}: ChartPanelProps) {
  const [focusedScenario, setFocusedScenario] = useState<ScenarioKey>("custom");
  const [comparisonMetric, setComparisonMetric] =
    useState<ScenarioComparisonMetricKey>("expectedReturn");
  const [stopLossPercent, setStopLossPercent] = useState(0.08);
  const [lensStep, setLensStep] = useState<number | null>(null);
  const [lensPinned, setLensPinned] = useState(false);
  const [zoomRange, setZoomRange] = useState<{
    startIndex: number;
    endIndex: number;
  } | null>(null);

  useEffect(() => {
    setFocusedScenario("custom");
    setLensStep(null);
    setLensPinned(false);
    setZoomRange(null);
  }, [analytics?.asset, analytics?.horizon]);

  const activeScenario = useMemo(
    () => analytics?.scenarios[focusedScenario] ?? null,
    [analytics, focusedScenario],
  );
  const focusedMedianKey = SCENARIO_MEDIAN_KEYS[focusedScenario];
  const forecastData = useMemo(
    () => (analytics ? buildForecastChartData(analytics, focusedScenario) : []),
    [analytics, focusedScenario],
  );
  const riskCurveData = useMemo(
    () =>
      analytics
        ? buildRiskCurveData(analytics, focusedScenario, stopLossPercent)
        : [],
    [analytics, focusedScenario, stopLossPercent],
  );
  const distributionData = useMemo(
    () =>
      analytics ? buildTerminalDistributionData(analytics, focusedScenario) : [],
    [analytics, focusedScenario],
  );
  const sampleKeys = useMemo(
    () =>
      activeScenario
        ? activeScenario.pricePaths
            .slice(0, Math.min(activeScenario.pricePaths.length, 4))
            .map((_, index) => `sample_${index}`)
        : [],
    [activeScenario],
  );

  const zoomStartIndex = zoomRange?.startIndex ?? 0;
  const zoomEndIndex =
    zoomRange?.endIndex ?? Math.max(forecastData.length - 1, 0);
  const visibleForecastData = useMemo(
    () => forecastData.slice(zoomStartIndex, zoomEndIndex + 1),
    [forecastData, zoomEndIndex, zoomStartIndex],
  );
  const forecastDomain = useMemo(() => {
    if (!analytics || visibleForecastData.length === 0) {
      return undefined;
    }

    const allValues = visibleForecastData.flatMap((point) => {
      const values = [
        point.activeP5,
        point.activeP95,
        point.customMedian,
        point.baseCaseMedian,
        point.downsideMedian,
        point.upsideMedian,
      ];
      sampleKeys.forEach((sampleKey) => {
        const sampleValue = point[sampleKey];
        if (sampleValue !== undefined) {
          values.push(sampleValue);
        }
      });
      return values;
    });
    allValues.push(analytics.startPrice);

    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);
    const valueRange = Math.max(maxValue - minValue, analytics.startPrice * 0.002);
    const rawBuffer = valueRange * 0.06;
    const floorBuffer =
      analytics.startPrice *
      (FORECAST_DOMAIN_MIN_PADDING_PCT[analytics.asset] ?? 0.006);
    const ceilingBuffer =
      analytics.startPrice *
      (FORECAST_DOMAIN_MAX_PADDING_PCT[analytics.asset] ?? 0.02);
    const buffer = Math.min(Math.max(rawBuffer, floorBuffer), ceilingBuffer);

    return [minValue - buffer, maxValue + buffer] as [number, number];
  }, [analytics, sampleKeys, stopLossPercent, visibleForecastData]);

  const comparisonChartData = useMemo(
    () =>
      analytics
        ? analytics.comparisonRows.map((row) => ({
            ...row,
            metricValue: getScenarioMetricValue(row, comparisonMetric),
          }))
        : [],
    [analytics, comparisonMetric],
  );

  const stepTicks = useMemo(
    () =>
      analytics
        ? Array.from(
            new Set([
              1,
              ...Array.from(
                { length: Math.floor((analytics.horizon - 1) / 5) + 1 },
                (_, index) => 1 + index * 5,
              ),
              analytics.horizon,
            ]),
          ).filter((tick) => tick <= analytics.horizon)
        : undefined,
    [analytics],
  );

  const stopLossPrice = useMemo(
    () => (analytics ? analytics.startPrice * (1 - stopLossPercent) : 0),
    [analytics, stopLossPercent],
  );
  const lensIndex =
    lensStep === null
      ? -1
      : forecastData.findIndex((point) => point.step === lensStep);
  const lensWindowRadius = 3;
  const lensData = useMemo(
    () =>
      lensIndex === -1
        ? []
        : forecastData.slice(
            Math.max(0, lensIndex - lensWindowRadius),
            Math.min(forecastData.length, lensIndex + lensWindowRadius + 1),
          ),
    [forecastData, lensIndex],
  );
  const lensFocusPoint =
    lensIndex >= 0 ? forecastData[lensIndex] : null;
  const lensFocusPrice =
    lensFocusPoint && typeof lensFocusPoint[focusedMedianKey] === "number"
      ? (lensFocusPoint[focusedMedianKey] as number)
      : null;
  const lensPreviousPoint =
    lensIndex > 0 ? forecastData[lensIndex - 1] : null;
  const lensPreviousPrice =
    lensPreviousPoint && typeof lensPreviousPoint[focusedMedianKey] === "number"
      ? (lensPreviousPoint[focusedMedianKey] as number)
      : analytics?.startPrice ?? null;
  const lensCumulativeReturn =
    lensFocusPrice !== null && analytics
      ? lensFocusPrice / analytics.startPrice - 1
      : null;
  const lensDailyReturn =
    lensFocusPrice !== null &&
    lensPreviousPrice !== null &&
    lensPreviousPrice !== 0
      ? lensFocusPrice / lensPreviousPrice - 1
      : null;
  const medianTerminalReturn = useMemo(
    () => medianFromSorted(activeScenario?.terminalReturns ?? []),
    [activeScenario],
  );
  const distributionDomain = useMemo(() => {
    if (distributionData.length === 0) {
      return undefined;
    }

    return [
      distributionData[0].binStart,
      distributionData[distributionData.length - 1].binEnd,
    ] as [number, number];
  }, [distributionData]);
  const distributionMarkers = useMemo(
    () => [
      {
        key: "mean",
        label: "Mean",
        value: activeScenario?.expectedReturn ?? 0,
        stroke: "#f4d69c",
      },
      {
        key: "median",
        label: "Median",
        value: medianTerminalReturn,
        stroke: "#4de2c5",
      },
      {
        key: "var95",
        label: "VaR (95%)",
        value: -(activeScenario?.var95 ?? 0),
        stroke: "#ff8a8a",
      },
      {
        key: "cvar95",
        label: "CVaR (95%)",
        value: -(activeScenario?.cvar95 ?? 0),
        stroke: "#b873ff",
      },
    ],
    [activeScenario, medianTerminalReturn],
  );
  const horizonRiskSnapshot =
    riskCurveData.length > 0 ? riskCurveData[riskCurveData.length - 1] : null;
  const lensDomain = useMemo(() => {
    if (lensData.length === 0) {
      return undefined;
    }

    const allValues = lensData.flatMap((point) =>
      focusedScenario === "baseCase"
        ? [point.baseCaseMedian]
        : [point[focusedMedianKey] as number, point.baseCaseMedian],
    );

    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);
    const buffer = Math.max((maxValue - minValue) * 0.06, maxValue * 0.002);

    return [minValue - buffer, maxValue + buffer] as [number, number];
  }, [focusedMedianKey, focusedScenario, lensData]);
  const showStopLossReference = useMemo(() => {
    if (!forecastDomain) {
      return false;
    }

    const [domainMin, domainMax] = forecastDomain;
    return stopLossPrice >= domainMin && stopLossPrice <= domainMax;
  }, [forecastDomain, stopLossPrice]);

  const handleChartHover = useCallback((
    event: { activeLabel?: string | number } | undefined,
  ) => {
    if (lensPinned) {
      return;
    }

    const activeLabel = event?.activeLabel;
    const nextStep =
      typeof activeLabel === "number"
        ? activeLabel
        : activeLabel !== undefined
          ? Number(activeLabel)
          : Number.NaN;

    if (!Number.isNaN(nextStep)) {
      setLensStep((current) => (current === nextStep ? current : nextStep));
    }
  }, [lensPinned]);

  const handleChartClick = useCallback((
    event: { activeLabel?: string | number } | undefined,
  ) => {
    const activeLabel = event?.activeLabel;
    const nextStep =
      typeof activeLabel === "number"
        ? activeLabel
        : activeLabel !== undefined
          ? Number(activeLabel)
          : Number.NaN;

    if (Number.isNaN(nextStep)) {
      return;
    }

    if (lensPinned && lensStep === nextStep) {
      setLensPinned(false);
      return;
    }

    setLensPinned(true);
    setLensStep(nextStep);
  }, [lensPinned, lensStep]);

  const handleChartLeave = useCallback(() => {
    if (!lensPinned) {
      setLensStep(null);
    }
  }, [lensPinned]);

  const handleZoomChange = useCallback(
    (next: { startIndex?: number; endIndex?: number } | undefined) => {
      if (next?.startIndex === undefined || next?.endIndex === undefined) {
        return;
      }

      const startIndex = next.startIndex;
      const endIndex = next.endIndex;

      setZoomRange((current) => {
        if (
          current?.startIndex === startIndex &&
          current?.endIndex === endIndex
        ) {
          return current;
        }

        return {
          startIndex,
          endIndex,
        };
      });
    },
    [],
  );

  return (
    <div className="surface-stack">
      <div className="chart-header">
        <div>
          <p className="panel-kicker">Decision Surface</p>
          <h2 className="panel-title">
            {analytics
              ? `${formatAssetLabel(analytics.asset)} Diffusion Forecast`
              : "Conditional Diffusion Forecast"}
          </h2>
          <p className="panel-copy">
            Move between the custom, base, Sticky Inflation, and
            Disinflation / Easing paths, then read the terminal distribution
            and breach risk before sizing the trade.
          </p>
        </div>

        <div className="chart-toolbar chart-toolbar-stack">
          <label className="select-field" htmlFor="asset-select">
            <span className="status-label">Asset</span>
            <select
              id="asset-select"
              value={selectedAsset}
              onChange={(event) => onAssetChange(Number(event.target.value))}
              disabled={assets.length === 0}
            >
              {assets.map((asset, index) => (
                <option key={asset} value={index}>
                  {formatAssetLabel(asset)}
                </option>
              ))}
            </select>
          </label>

          <div className="focus-toggle-group">
            <span className="status-label">Scenario Focus</span>
            <div className="scenario-toggle-row">
              {SCENARIO_ORDER.map((scenarioKey) => (
                <button
                  key={scenarioKey}
                  type="button"
                  className={`scenario-toggle ${
                    focusedScenario === scenarioKey
                      ? "scenario-toggle-active"
                      : ""
                  }`}
                  onClick={() => setFocusedScenario(scenarioKey)}
                >
                  {formatScenarioLabel(scenarioKey)}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {analytics && activeScenario && (
        <div className="metric-strip metric-strip-compact">
          <div className="metric-chip">
            <span className="metric-label">{activeScenario.label} Return</span>
            <strong className="metric-value">
              {formatPercent(activeScenario.expectedReturn)}
            </strong>
          </div>

          <div className="metric-chip">
            <span className="metric-label">Downside Probability</span>
            <strong className="metric-value">
              {formatPercent(activeScenario.downsideProbability)}
            </strong>
          </div>

          <div className="metric-chip">
            <span className="metric-label">90% Interval</span>
            <strong className="metric-value">
              {formatCurrency(activeScenario.confidenceLow)} to{" "}
              {formatCurrency(activeScenario.confidenceHigh)}
            </strong>
          </div>

          <div className="metric-chip">
            <span className="metric-label">Custom Vs Base</span>
            <strong className="metric-value">
              {formatSignedPercent(analytics.baselineReturnDelta)}
            </strong>
          </div>
        </div>
      )}

      {analytics && activeScenario && (
        <>
          <div className="chart-card chart-card-hero">
            <div className="chart-card-header">
              <div>
                <h3>Conditional Diffusion Forecast</h3>
                <p className="panel-copy">
                  The selected scenario controls the confidence band, sparse
                  sample paths, and stop-loss reference line.
                </p>
              </div>

              <div className="chart-actions">
                <span className="status-label">
                  Viewing days{" "}
                  {zoomRange
                    ? `${forecastData[zoomStartIndex]?.step ?? 1} to ${
                        forecastData[zoomEndIndex]?.step ?? analytics.horizon
                      }`
                    : `1 to ${analytics.horizon}`}
                </span>
                <div className="chart-action-row">
                  <button
                    type="button"
                    className="secondary-button secondary-button-compact"
                    onClick={() => setZoomRange(null)}
                  >
                    Reset Zoom
                  </button>
                  {lensPinned && (
                    <button
                      type="button"
                      className="secondary-button secondary-button-compact"
                      onClick={() => {
                        setLensPinned(false);
                        setLensStep(null);
                      }}
                    >
                      Unpin Lens
                    </button>
                  )}
                </div>
              </div>
            </div>

            <div className="zoom-helper-bar">
              <div className="zoom-helper-copy">
                <span className="zoom-helper-title">Path lens</span>
                <span className="zoom-helper-note">
                  Hover the forecast to inspect a local slice without changing
                  the full chart. Click once to pin the focus.
                </span>
              </div>
              <span className="zoom-helper-range">
                {lensStep !== null
                  ? `${lensPinned ? "Pinned" : "Live"} focus near day ${lensStep}`
                  : "Hover the path to activate the lens"}
              </span>
            </div>

            <div className="chart-legend chart-legend-strong">
              <span className="legend-item">
                <span className="legend-swatch legend-swatch-median" />
                Custom
              </span>
              <span className="legend-item">
                <span className="legend-swatch legend-swatch-baseline" />
                Base Case
              </span>
              <span className="legend-item">
                <span className="legend-swatch legend-swatch-downside" />
                Sticky Inflation
              </span>
              <span className="legend-item">
                <span className="legend-swatch legend-swatch-upside" />
                Disinflation / Easing
              </span>
              <span className="legend-item">
                <span className="legend-swatch legend-swatch-band" />
                Active 90% band
              </span>
              <span className="legend-item">
                <span className="legend-swatch legend-swatch-spot" />
                Current and stop-loss
              </span>
            </div>

            <div className="chart-frame chart-frame-hero">
              {lensData.length > 0 && (
                <div className="chart-lens-overlay">
                  <div className="chart-lens-header">
                    <div className="chart-lens-meta">
                      <span className="chart-lens-title">
                        {lensPinned ? "Pinned Lens" : "Live Lens"}
                      </span>
                      <span className="chart-lens-day">
                        Day {lensStep}
                      </span>
                    </div>
                    <div className="chart-lens-quote">
                      <strong>
                        {lensFocusPrice !== null
                          ? formatCurrency(lensFocusPrice)
                          : "N/A"}
                      </strong>
                      <div className="chart-lens-change-stack">
                        <span className="chart-lens-change-label">
                          Daily
                        </span>
                        <span
                          className={`chart-lens-change ${
                            (lensDailyReturn ?? 0) >= 0
                              ? "chart-lens-change-positive"
                              : "chart-lens-change-negative"
                          }`}
                        >
                          {lensDailyReturn !== null
                            ? formatSignedPercent(lensDailyReturn)
                            : "N/A"}
                        </span>
                      </div>
                      <div className="chart-lens-change-stack">
                        <span className="chart-lens-change-label">
                          Since Day 1
                        </span>
                        <span
                          className={`chart-lens-change ${
                            (lensCumulativeReturn ?? 0) >= 0
                              ? "chart-lens-change-positive"
                              : "chart-lens-change-negative"
                          }`}
                        >
                          {lensCumulativeReturn !== null
                            ? formatSignedPercent(lensCumulativeReturn)
                            : "N/A"}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="chart-lens-body">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={lensData}
                        margin={{ top: 8, right: 10, left: 2, bottom: 4 }}
                      >
                        <defs>
                          <linearGradient
                            id="scenarioBandInset"
                            x1="0"
                            x2="0"
                            y1="0"
                            y2="1"
                          >
                            <stop
                              offset="0%"
                              stopColor={SCENARIO_STYLES[focusedScenario].bar}
                              stopOpacity={0.4}
                            />
                            <stop
                              offset="100%"
                              stopColor={SCENARIO_STYLES[focusedScenario].bar}
                              stopOpacity={0.02}
                            />
                          </linearGradient>
                        </defs>

                        <CartesianGrid stroke="rgba(130, 149, 165, 0.1)" />
                        <ReferenceLine
                          x={lensStep ?? undefined}
                          stroke="rgba(244, 214, 156, 0.65)"
                          strokeDasharray="2 3"
                        />
                        <ReferenceLine
                          y={analytics.startPrice}
                          stroke="rgba(154, 199, 255, 0.28)"
                          strokeDasharray="3 3"
                          label={{
                            value: "Spot",
                            position: "right",
                            fill: "#9ac7ff",
                            fontSize: 10,
                          }}
                        />
                        <ReferenceLine
                          y={stopLossPrice}
                          stroke="rgba(255, 124, 124, 0.24)"
                          strokeDasharray="3 3"
                        />
                        {lensFocusPrice !== null && (
                          <ReferenceLine
                            y={lensFocusPrice}
                            stroke="rgba(244, 214, 156, 0.4)"
                            strokeDasharray="2 3"
                            label={{
                              value: formatCurrency(lensFocusPrice),
                              position: "right",
                              fill: "#f4d69c",
                              fontSize: 10,
                            }}
                          />
                        )}
                        <XAxis
                          axisLine={false}
                          dataKey="step"
                          tickLine={false}
                          type="number"
                          domain={[
                            lensData[0]?.step ?? 1,
                            lensData[lensData.length - 1]?.step ?? analytics.horizon,
                          ]}
                          allowDecimals={false}
                          tick={{ fill: "#7f93a8", fontSize: 10 }}
                          height={18}
                          tickFormatter={(value: number) => `D${value}`}
                        />
                        <YAxis
                          axisLine={false}
                          tickLine={false}
                          tickFormatter={(value: number) => formatCurrency(value)}
                          tick={{ fill: "#7f93a8", fontSize: 10 }}
                          width={64}
                          domain={lensDomain}
                        />
                        <Area
                          type="monotone"
                          dataKey={focusedMedianKey}
                          fill="url(#scenarioBandInset)"
                          stroke={SCENARIO_STYLES[focusedScenario].stroke}
                          strokeWidth={2.4}
                          dot={false}
                          activeDot={false}
                          isAnimationActive={false}
                        />
                        {focusedScenario !== "baseCase" && (
                          <Line
                            type="monotone"
                            dataKey="baseCaseMedian"
                            stroke={SCENARIO_STYLES.baseCase.stroke}
                            strokeWidth={1.4}
                            strokeDasharray="4 3"
                            dot={false}
                            isAnimationActive={false}
                          />
                        )}
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              <ResponsiveContainer width="100%" height={540}>
                <AreaChart
                  syncId="diffusion-surface"
                  data={forecastData}
                  margin={{ top: 10, right: 18, left: 10, bottom: 8 }}
                  onMouseMove={handleChartHover}
                  onMouseLeave={handleChartLeave}
                  onClick={handleChartClick}
                >
                  <defs>
                    <linearGradient
                      id="scenarioBand"
                      x1="0"
                      x2="0"
                      y1="0"
                      y2="1"
                    >
                      <stop
                        offset="0%"
                        stopColor={SCENARIO_STYLES[focusedScenario].fill}
                        stopOpacity={0.85}
                      />
                      <stop
                        offset="100%"
                        stopColor={SCENARIO_STYLES[focusedScenario].fill}
                        stopOpacity={0.18}
                      />
                    </linearGradient>
                  </defs>

                  <CartesianGrid stroke="rgba(130, 149, 165, 0.14)" />
                  <ReferenceLine
                    y={analytics.startPrice}
                    stroke="rgba(154, 199, 255, 0.35)"
                    strokeDasharray="4 4"
                    ifOverflow="hidden"
                    label={{
                      value: "Current price",
                      position: "insideTopRight",
                      fill: "#7f93a8",
                      fontSize: 11,
                    }}
                  />
                  {showStopLossReference && (
                    <ReferenceLine
                      y={stopLossPrice}
                      stroke="rgba(255, 124, 124, 0.42)"
                      strokeDasharray="3 3"
                      ifOverflow="hidden"
                      label={{
                        value: "Stop loss",
                        position: "insideBottomRight",
                        fill: "#ff9d9d",
                        fontSize: 11,
                      }}
                    />
                  )}
                  <XAxis
                    axisLine={false}
                    dataKey="step"
                    tickLine={false}
                    type="number"
                    domain={[
                      forecastData[zoomStartIndex]?.step ?? 1,
                      forecastData[zoomEndIndex]?.step ?? analytics.horizon,
                    ]}
                    ticks={stepTicks}
                    allowDecimals={false}
                    tick={{ fill: "#7f93a8", fontSize: 12 }}
                    height={34}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#7f93a8", fontSize: 12 }}
                    tickFormatter={(value: number) => formatCurrency(value)}
                    width={86}
                    domain={forecastDomain}
                  >
                    <Label
                      value="Simulated price (USD)"
                      angle={-90}
                      position="insideLeft"
                      fill="#7f93a8"
                      fontSize={12}
                      dx={-8}
                    />
                  </YAxis>
                  <Tooltip
                    content={<ForecastTooltip focusKey={focusedScenario} />}
                  />

                  <Area
                    dataKey="activeBandBase"
                    fill="transparent"
                    stackId="distribution"
                    stroke="transparent"
                    isAnimationActive={false}
                  />
                  <Area
                    dataKey="activeBandSize"
                    fill="url(#scenarioBand)"
                    stackId="distribution"
                    stroke="transparent"
                    isAnimationActive={false}
                  />

                  {sampleKeys.map((sampleKey) => (
                    <Line
                      key={sampleKey}
                      type="monotone"
                      dataKey={sampleKey}
                      stroke={SCENARIO_STYLES[focusedScenario].stroke}
                      strokeOpacity={0.18}
                      strokeWidth={1}
                      dot={false}
                      isAnimationActive={false}
                    />
                  ))}

                  <Line
                    type="monotone"
                    dataKey="customMedian"
                    stroke={SCENARIO_STYLES.custom.stroke}
                    strokeWidth={3}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="baseCaseMedian"
                    stroke={SCENARIO_STYLES.baseCase.stroke}
                    strokeWidth={2}
                    strokeDasharray="5 4"
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="downsideMedian"
                    stroke={SCENARIO_STYLES.downside.stroke}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="upsideMedian"
                    stroke={SCENARIO_STYLES.upside.stroke}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Brush
                    dataKey="step"
                    height={34}
                    stroke="#4de2c5"
                    fill="rgba(77, 226, 197, 0.12)"
                    travellerWidth={14}
                    startIndex={zoomStartIndex}
                    endIndex={zoomEndIndex}
                    onChange={handleZoomChange}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="chart-grid-secondary">
            <div className="chart-card">
              <div className="chart-card-header">
                <div>
                  <h3>Terminal Outcome Distribution</h3>
                  <p className="panel-copy">
                    Read the terminal return histogram before trusting the
                    single path median.
                  </p>
                </div>
              </div>

              <div className="metric-inline-strip">
                <div className="metric-inline-chip">
                  <span className="metric-label">Mean</span>
                  <strong className="metric-value">
                    {formatPercent(activeScenario.expectedReturn)}
                  </strong>
                </div>
                <div className="metric-inline-chip">
                  <span className="metric-label">Median</span>
                  <strong className="metric-value">
                    {formatPercent(
                      activeScenario.terminalValues.length === 0
                        ? 0
                        : activeScenario.terminalValues[
                            Math.floor(activeScenario.terminalValues.length / 2)
                          ] /
                            activeScenario.startPrice -
                            1,
                    )}
                  </strong>
                </div>
                <div className="metric-inline-chip">
                  <span className="metric-label">VaR (95%)</span>
                  <strong className="metric-value">
                    {formatPercent(activeScenario.var95)}
                  </strong>
                </div>
                <div className="metric-inline-chip">
                  <span className="metric-label">CVaR (95%)</span>
                  <strong className="metric-value">
                    {formatPercent(activeScenario.cvar95)}
                  </strong>
                </div>
              </div>

              <div className="distribution-marker-strip">
                {distributionMarkers.map((marker) => (
                  <span key={marker.key} className="distribution-marker-chip">
                    <span
                      className="distribution-marker-line"
                      style={{ background: marker.stroke }}
                    />
                    {marker.label} {formatSignedPercent(marker.value)}
                  </span>
                ))}
              </div>

              <div className="chart-frame chart-frame-secondary">
                <ResponsiveContainer width="100%" height={250}>
                  <ComposedChart
                    data={distributionData}
                    margin={{ top: 10, right: 12, left: 0, bottom: 8 }}
                  >
                    <CartesianGrid stroke="rgba(130, 149, 165, 0.12)" />
                    <XAxis
                      axisLine={false}
                      dataKey="center"
                      tickLine={false}
                      type="number"
                      domain={distributionDomain}
                      tick={{ fill: "#7f93a8", fontSize: 11 }}
                      tickFormatter={(value: number) => formatSignedPercent(value)}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#7f93a8", fontSize: 11 }}
                      allowDecimals={false}
                    />
                    <Tooltip
                      content={
                        <DistributionTooltip totalCount={activeScenario.pathCount} />
                      }
                    />
                    <ReferenceLine
                      x={0}
                      stroke="rgba(146, 165, 184, 0.32)"
                      strokeDasharray="3 3"
                      ifOverflow="extendDomain"
                    />
                    {distributionMarkers.map((marker) => (
                      <ReferenceLine
                        key={marker.key}
                        x={marker.value}
                        stroke={marker.stroke}
                        strokeDasharray="4 3"
                        ifOverflow="extendDomain"
                      />
                    ))}
                    <Bar
                      dataKey="count"
                      radius={[4, 4, 0, 0]}
                      isAnimationActive={false}
                    >
                      {distributionData.map((point) => {
                        const isLossBucket = point.center < 0;
                        return (
                          <Cell
                            key={`${point.binStart}-${point.binEnd}`}
                            fill={
                              isLossBucket
                                ? "rgba(255, 124, 124, 0.68)"
                                : SCENARIO_STYLES[focusedScenario].bar
                            }
                          />
                        );
                      })}
                    </Bar>
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="chart-card">
              <div className="chart-card-header">
                <div>
                  <h3>Breach And Drawdown Risk</h3>
                  <p className="panel-copy">
                    Track stop-loss breach probability and expected drawdown
                    through the horizon.
                  </p>
                </div>

                <label className="numeric-field stoploss-field" htmlFor="stop-loss-input">
                  <input
                    id="stop-loss-input"
                    type="number"
                    min="2"
                    max="30"
                    step="1"
                    value={(stopLossPercent * 100).toFixed(0)}
                    onChange={(event) => {
                      const nextValue = Number(event.target.value);
                      if (Number.isNaN(nextValue)) {
                        return;
                      }

                      setStopLossPercent(Math.min(0.3, Math.max(0.02, nextValue / 100)));
                    }}
                  />
                  <span>% stop loss</span>
                </label>
              </div>

              <div className="risk-threshold-strip">
                <div className="risk-threshold-chip">
                  <span className="metric-label">Stop Level</span>
                  <strong className="metric-value">
                    {formatCurrency(stopLossPrice)}
                  </strong>
                </div>
                <div className="risk-threshold-chip">
                  <span className="metric-label">Breach By Horizon</span>
                  <strong className="metric-value">
                    {formatPercent(horizonRiskSnapshot?.stopLossProbability ?? 0)}
                  </strong>
                </div>
                <div className="risk-threshold-chip">
                  <span className="metric-label">Expected Drawdown</span>
                  <strong className="metric-value">
                    {formatPercent(horizonRiskSnapshot?.expectedDrawdown ?? 0)}
                  </strong>
                </div>
              </div>

              <div className="chart-frame chart-frame-secondary">
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart
                    syncId="diffusion-surface"
                    data={riskCurveData}
                    margin={{ top: 10, right: 10, left: 0, bottom: 10 }}
                    onMouseMove={handleChartHover}
                    onMouseLeave={handleChartLeave}
                  >
                    <CartesianGrid stroke="rgba(130, 149, 165, 0.12)" />
                    <XAxis
                      axisLine={false}
                      tickLine={false}
                      dataKey="step"
                      tick={{ fill: "#7f93a8", fontSize: 11 }}
                      allowDecimals={false}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#7f93a8", fontSize: 11 }}
                      tickFormatter={(value: number) => formatPercent(value)}
                    />
                    <Tooltip content={<RiskTooltip />} />
                    {lensStep !== null && (
                      <ReferenceLine
                        x={lensStep}
                        stroke="rgba(244, 214, 156, 0.38)"
                        strokeDasharray="3 3"
                      />
                    )}
                    <Line
                      type="monotone"
                      dataKey="belowSpotProbability"
                      stroke="#4de2c5"
                      strokeWidth={2.2}
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="stopLossProbability"
                      stroke="#ff7c7c"
                      strokeWidth={2.2}
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="expectedDrawdown"
                      stroke="#74b7ff"
                      strokeWidth={2.2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="chart-card chart-card-wide">
              <div className="chart-card-header">
                <div>
                  <h3>Scenario Comparison</h3>
                  <p className="panel-copy">
                    Compare the scenario ladder on one decision metric at a
                    time instead of overlaying everything on one axis.
                  </p>
                </div>

                <label className="select-field select-field-compact" htmlFor="comparison-metric">
                  <span className="status-label">Metric</span>
                  <select
                    id="comparison-metric"
                    value={comparisonMetric}
                    onChange={(event) =>
                      setComparisonMetric(
                        event.target.value as ScenarioComparisonMetricKey,
                      )
                    }
                  >
                    {COMPARISON_METRIC_OPTIONS.map((metricKey) => (
                      <option key={metricKey} value={metricKey}>
                        {getComparisonMetricLabel(metricKey)}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="chart-frame chart-frame-secondary">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={comparisonChartData}
                    margin={{ top: 12, right: 12, left: 0, bottom: 10 }}
                  >
                    <CartesianGrid stroke="rgba(130, 149, 165, 0.12)" />
                    <XAxis
                      axisLine={false}
                      tickLine={false}
                      dataKey="label"
                      tick={{ fill: "#7f93a8", fontSize: 11 }}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#7f93a8", fontSize: 11 }}
                      tickFormatter={(value: number) =>
                        formatScenarioMetricValue(comparisonMetric, value)
                      }
                    />
                    <Tooltip
                      content={
                        <ComparisonTooltip metricKey={comparisonMetric} />
                      }
                    />
                    <Bar
                      dataKey="metricValue"
                      radius={[6, 6, 0, 0]}
                      isAnimationActive={false}
                    >
                      {comparisonChartData.map((row) => (
                        <Cell
                          key={row.key}
                          fill={SCENARIO_STYLES[row.key].bar}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      )}

      {status === "loading" && !analytics ? (
        <div className="state-card">
          <p className="state-title">Running scenario suite</p>
          <p className="state-copy">
            Building the custom, base, Sticky Inflation, and
            Disinflation / Easing surfaces from the current live market state.
          </p>
        </div>
      ) : status === "error" ? (
        <div className="state-card state-card-error">
          <p className="state-title">Simulation unavailable</p>
          <p className="state-copy">{error}</p>
        </div>
      ) : !analytics ? (
        <div className="state-card">
          <p className="state-title">No run yet</p>
          <p className="state-copy">
            Generate a scenario suite to render the full forecast,
            distribution, breach-risk, and comparison canvas.
          </p>
        </div>
      ) : null}
    </div>
  );
}

export default ChartPanel;
