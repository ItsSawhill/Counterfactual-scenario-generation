import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import "./App.css";
import ErrorBoundary from "./components/ErrorBoundary";
import Sidebar from "./components/Sidebar";
import ChartPanel from "./components/ChartPanel";
import Insights from "./components/Insights";
import {
  BASELINE_SCENARIO,
  buildScenarioAnalytics,
  buildScenarioSuite,
  describeRegime,
  formatAssetLabel,
} from "./lib/analytics";
import type {
  LiveStateMeta,
  SimulationBundle,
  SimulationRequest,
  SimulationResponse,
  SimulationStatus,
} from "./types/simulation";

const API_BASE_URL = "http://127.0.0.1:8000";
const FALLBACK_MODEL_HORIZON = 30;

function parseDateLabel(value?: string): Date | null {
  if (!value) {
    return null;
  }

  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    const [year, month, day] = value.split("-").map(Number);
    return new Date(year, month - 1, day);
  }

  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function formatDateLabel(value: string | undefined, fallback: string): string {
  const parsed = parseDateLabel(value);

  if (!parsed) {
    return fallback;
  }

  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(parsed);
}

function formatTimeLabel(value: string | undefined, fallback: string): string {
  const parsed = parseDateLabel(value);

  if (!parsed) {
    return fallback;
  }

  return new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
  }).format(parsed);
}

function formatCacheAge(seconds: number | undefined): string {
  if (seconds === undefined || !Number.isFinite(seconds)) {
    return "Cache age unavailable";
  }

  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
}

function App() {
  const [request, setRequest] = useState<SimulationRequest>(BASELINE_SCENARIO.values);
  const [selectedAsset, setSelectedAsset] = useState(0);
  const [simulationBundle, setSimulationBundle] =
    useState<SimulationBundle | null>(null);
  const [status, setStatus] = useState<SimulationStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [liveStateMeta, setLiveStateMeta] = useState<LiveStateMeta | null>(null);
  const [isRefreshingLiveState, setIsRefreshingLiveState] = useState(false);
  const [lastRunDateLabel, setLastRunDateLabel] = useState("Pending first run");
  const [lastRunTimeLabel, setLastRunTimeLabel] = useState("Awaiting simulation");

  const modelHorizon =
    simulationBundle?.custom.model_horizon ?? FALLBACK_MODEL_HORIZON;

  useEffect(() => {
    let isCancelled = false;

    const runSimulation = (params: SimulationRequest) =>
      axios.post<SimulationResponse>(`${API_BASE_URL}/simulate`, {
        inflation: params.inflation,
        interest_rate: params.interestRate,
        horizon: params.horizon,
        path_count: params.pathCount,
      });

    const boundedRequest = {
      ...request,
      horizon: Math.min(request.horizon, modelHorizon),
    };
    const suite = buildScenarioSuite(boundedRequest, modelHorizon);

    setStatus("loading");
    setError(null);

    Promise.all([
      runSimulation(boundedRequest),
      runSimulation(suite.baseCase),
      runSimulation(suite.downside),
      runSimulation(suite.upside),
    ])
      .then(([custom, baseCase, downside, upside]) => {
        if (isCancelled) {
          return;
        }

        setSimulationBundle({
          custom: custom.data,
          baseCase: baseCase.data,
          downside: downside.data,
          upside: upside.data,
        });
        setLiveStateMeta(custom.data.live_state ?? null);
        setSelectedAsset((currentIndex) => {
          const maxIndex = Math.max(custom.data.assets.length - 1, 0);
          return Math.min(currentIndex, maxIndex);
        });
        setStatus("ready");
        setLastRunDateLabel(
          new Intl.DateTimeFormat("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
          }).format(new Date()),
        );
        setLastRunTimeLabel(
          new Intl.DateTimeFormat("en-US", {
            hour: "numeric",
            minute: "2-digit",
          }).format(new Date()),
        );
      })
      .catch((requestError) => {
        if (isCancelled) {
          return;
        }

        console.error("SIMULATION SUITE FAILED", requestError);
        setStatus("error");
        setError(
          "The simulation service did not respond. Check that FastAPI is running on http://127.0.0.1:8000.",
        );
      });

    return () => {
      isCancelled = true;
    };
  }, [request, modelHorizon]);

  const analytics = useMemo(
    () =>
      simulationBundle
        ? buildScenarioAnalytics(simulationBundle, request, selectedAsset)
        : null,
    [simulationBundle, request, selectedAsset],
  );
  const activeAsset = simulationBundle?.custom.assets[selectedAsset] ?? "SPY";
  const scenarioLabel = describeRegime(request);
  const marketDataDateLabel = formatDateLabel(
    liveStateMeta?.market_data_as_of,
    "Pending first pull",
  );
  const modelInputDateLabel = formatDateLabel(
    liveStateMeta?.model_input_as_of,
    "Pending feature panel",
  );
  const liveFetchedTimeLabel = formatTimeLabel(
    liveStateMeta?.live_fetched_at,
    "Awaiting refresh",
  );
  const liveStateTone =
    liveStateMeta?.cache_status === "fresh" ? "status-live" : "status-neutral";
  const liveStateLabel =
    liveStateMeta?.cache_status === "fresh"
      ? "Live"
      : liveStateMeta?.cache_status === "cached"
        ? "Cached"
        : "Awaiting";
  const cacheAgeLabel = liveStateMeta
    ? `${formatCacheAge(liveStateMeta.cache_age_seconds)} age`
    : "Pull the latest market and macro state on demand.";
  const cacheTtlLabel =
    liveStateMeta?.cache_ttl_seconds !== undefined
      ? `${formatCacheAge(liveStateMeta.cache_ttl_seconds)} TTL`
      : "TTL unavailable";
  const statusTone =
    status === "loading"
      ? "status-live"
      : status === "ready"
        ? "status-positive"
        : status === "error"
          ? "status-negative"
          : "status-neutral";

  const handleRefreshLiveState = useCallback(() => {
    if (status === "loading" || isRefreshingLiveState) {
      return;
    }

    setIsRefreshingLiveState(true);
    setError(null);

    axios
      .post<LiveStateMeta>(`${API_BASE_URL}/refresh-live-state`)
      .then((response) => {
        setLiveStateMeta(response.data);
        setRequest((current) => ({ ...current }));
      })
      .catch((refreshError) => {
        console.error("LIVE STATE REFRESH FAILED", refreshError);
        setStatus("error");
        setError(
          "The live market state could not be refreshed. Check that FastAPI is running and that the data sources are reachable.",
        );
      })
      .finally(() => {
        setIsRefreshingLiveState(false);
      });
  }, [isRefreshingLiveState, status]);

  return (
    <div className="app-shell">
      <header className="workspace-header">
        <div className="header-copy">
          <div className="header-kicker-row">
            <p className="eyebrow">Quant Scenario Research</p>
            <span className={`status-pill ${statusTone}`}>
              {status === "loading"
                ? "Running"
                : status === "ready"
                  ? "Ready"
                  : status === "error"
                    ? "Error"
                    : "Idle"}
            </span>
          </div>
          <h1>Diffusion Scenario Lab</h1>
          <p className="lede">
            Scenario comparison, tail risk, and base-case overlays in one
            decision surface.
          </p>

          <div className="live-state-bar">
            <div className="live-state-copy">
              <div className="live-state-topline">
                <span className="status-label">Market Data As Of</span>
                <span className={`status-pill ${liveStateTone}`}>
                  {liveStateLabel}
                </span>
              </div>
              <strong className="live-state-value">{marketDataDateLabel}</strong>
              <div className="live-state-details">
                <span>Feature panel through {modelInputDateLabel}</span>
                <span>Fetched {liveFetchedTimeLabel}</span>
                <span>
                  {cacheAgeLabel} / {cacheTtlLabel}
                </span>
              </div>
            </div>

            <button
              type="button"
              className="secondary-button header-refresh-button"
              onClick={handleRefreshLiveState}
              disabled={status === "loading" || isRefreshingLiveState}
            >
              {isRefreshingLiveState ? "Refreshing..." : "Refresh Live Data"}
            </button>
          </div>
        </div>

        <div className="header-summary-grid">
          <div className="summary-card">
            <span className="status-label">Asset</span>
            <strong className="summary-value">
              {formatAssetLabel(activeAsset)}
            </strong>
          </div>
          <div className="summary-card">
            <span className="status-label">Scenario</span>
            <strong className="summary-value">{scenarioLabel}</strong>
          </div>
          <div className="summary-card">
            <span className="status-label">Horizon</span>
            <strong className="summary-value">{request.horizon} Days</strong>
          </div>
          <div className="summary-card">
            <span className="status-label">Last Run</span>
            <strong className="summary-value">{lastRunTimeLabel}</strong>
            <span className="summary-subtle">{lastRunDateLabel}</span>
          </div>
        </div>
      </header>

      <main className="workspace-grid">
        <aside className="panel sidebar-panel">
          <Sidebar
            initialValues={request}
            onGenerate={setRequest}
            isLoading={status === "loading"}
            modelHorizon={modelHorizon}
          />
        </aside>

        <section className="panel surface-panel">
          <ErrorBoundary
            title="Forecast surface unavailable"
            message="The chart surface hit a runtime error. Refresh the app or rerun the scenario to recover without losing the rest of the workspace."
          >
            <ChartPanel
              analytics={analytics}
              assets={simulationBundle?.custom.assets ?? []}
              selectedAsset={selectedAsset}
              onAssetChange={setSelectedAsset}
              status={status}
              error={error}
            />
          </ErrorBoundary>
        </section>

        <aside className="panel insights-panel">
          <ErrorBoundary
            title="Insights unavailable"
            message="The decision support panel hit a runtime error. The rest of the scenario workspace is still available."
          >
            <Insights analytics={analytics} status={status} />
          </ErrorBoundary>
        </aside>
      </main>
    </div>
  );
}

export default App;
