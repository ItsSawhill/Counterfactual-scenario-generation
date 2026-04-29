import {
  formatNumber,
  formatPercent,
  formatSignedPercent,
} from "../lib/analytics";
import type {
  ScenarioAnalytics,
  SimulationStatus,
} from "../types/simulation";

interface InsightsProps {
  analytics: ScenarioAnalytics | null;
  status: SimulationStatus;
}

function Insights({ analytics, status }: InsightsProps) {
  const downsideScenario = analytics?.scenarios.downside;
  const upsideScenario = analytics?.scenarios.upside;
  const tailTone =
    analytics?.tailAssessment.label === "Contained"
      ? "status-positive"
      : analytics?.tailAssessment.label === "Fragile"
        ? "status-negative"
        : "status-neutral";

  return (
    <div className="insight-stack">
      <div className="panel-header">
        <p className="panel-kicker">Decision Support</p>
        <h2 className="panel-title">Readout</h2>
        <p className="panel-copy">
          Translate the scenario ladder into a stance, then use the Sticky
          Inflation case to size risk instead of relying on the median path
          alone.
        </p>
      </div>

      {analytics ? (
        <>
          <div className="insight-card">
            <div className="section-header">
              <div className="signal-header-copy">
                <h3>Scenario Signal</h3>
                <p className="panel-copy signal-header-note">
                  Translate the simulation into a portfolio stance.
                </p>
              </div>
              <span
                className={`status-pill ${
                  analytics.signal === "Bullish"
                    ? "status-positive"
                    : analytics.signal === "Bearish"
                      ? "status-negative"
                      : "status-neutral"
                }`}
              >
                {analytics.signal}
              </span>
            </div>

            <div className="decision-grid">
              <div className="decision-card decision-card-compact">
                <span className="insight-label decision-label">Confidence</span>
                <span className="decision-value decision-value-stack">
                  <span className="decision-primary">
                    {analytics.confidenceLabel}
                  </span>
                  <span className="decision-subtle">
                    ({Math.round(analytics.confidenceScore * 100)}%)
                  </span>
                </span>
              </div>

              <div className="decision-card decision-card-compact">
                <span className="insight-label decision-label">Tail Risk</span>
                <div className="tail-risk-stack">
                  <span className={`status-pill ${tailTone}`}>
                    {analytics.tailAssessment.label}
                  </span>
                  <span className="decision-subtle decision-subtle-wrap">
                    {analytics.tailAssessment.detail}
                  </span>
                </div>
              </div>
            </div>

            <div className="recommendation-stack">
              <div className="recommendation-card">
                <span className="insight-label recommendation-label">
                  Exposure Band
                </span>
                <strong className="recommendation-value">
                  {formatPercent(analytics.positionBand.lower)} to{" "}
                  {formatPercent(analytics.positionBand.upper)}
                </strong>
                <span className="recommendation-copy">
                  {analytics.positionBand.detail}
                </span>
              </div>

              <div className="recommendation-card">
                <span className="insight-label recommendation-label">
                  Stop Band
                </span>
                <strong className="recommendation-value">
                  {formatPercent(analytics.stopBand.lower)} to{" "}
                  {formatPercent(analytics.stopBand.upper)} below spot
                </strong>
                <span className="recommendation-copy">
                  {analytics.stopBand.detail}
                </span>
              </div>

              <div className="recommendation-card">
                <span className="insight-label recommendation-label">
                  Hedge Trigger
                </span>
                <span className="recommendation-copy recommendation-copy-strong">
                  {analytics.hedgeTrigger}
                </span>
              </div>
            </div>

            <div className="decision-card decision-card-wide decision-card-emphasis">
              <span className="insight-label decision-label">
                Suggested Action
              </span>
              <span className="decision-value decision-value-wrap">
                {analytics.suggestedAction}
              </span>
            </div>

            <div className="driver-list">
              {analytics.keyDrivers.map((driver) => (
                <p key={driver} className="driver-item">
                  {driver}
                </p>
              ))}
            </div>
          </div>

          <div className="insight-card">
            <div className="section-header">
              <h3>Risk Snapshot</h3>
              <span className="status-label">Tail-aware metrics</span>
            </div>

            <div className="insight-metrics insight-metrics-dense">
              <div className="insight-metric">
                <span className="insight-label">VaR (95%)</span>
                <strong className="insight-value">
                  {formatPercent(analytics.var95)}
                </strong>
              </div>
              <div className="insight-metric">
                <span className="insight-label">CVaR (95%)</span>
                <strong className="insight-value">
                  {formatPercent(analytics.cvar95)}
                </strong>
              </div>
              <div className="insight-metric">
                <span className="insight-label">VaR (99%)</span>
                <strong className="insight-value">
                  {formatPercent(analytics.var99)}
                </strong>
              </div>
              <div className="insight-metric">
                <span className="insight-label">Std dev</span>
                <strong className="insight-value">
                  {formatPercent(analytics.stdDeviation)}
                </strong>
              </div>
              <div className="insight-metric">
                <span className="insight-label">Sharpe Ratio</span>
                <strong className="insight-value">
                  {formatNumber(analytics.sharpeRatio)}
                </strong>
              </div>
              <div className="insight-metric">
                <span className="insight-label">Max Drawdown</span>
                <strong className="insight-value">
                  {formatPercent(analytics.maxDrawdown)}
                </strong>
              </div>
              <div className="insight-metric">
                <span className="insight-label">Skew</span>
                <strong className="insight-value">
                  {formatNumber(analytics.skewness)}
                </strong>
              </div>
              <div className="insight-metric">
                <span className="insight-label">Kurtosis</span>
                <strong className="insight-value">
                  {formatNumber(analytics.kurtosis)}
                </strong>
              </div>
            </div>
          </div>

          <div className="insight-card">
            <div className="card-title-block">
              <h3>Scenario Comparison</h3>
              <span className="status-label">
                Base, Sticky Inflation, And Disinflation
              </span>
            </div>

            <div className="comparison-grid">
              <div className="comparison-card">
                <span className="insight-label">VS Base Case Return</span>
                <strong className="insight-value">
                  {formatSignedPercent(analytics.baselineReturnDelta)}
                </strong>
              </div>
              <div className="comparison-card">
                <span className="insight-label">VS Base Loss Risk</span>
                <strong className="insight-value">
                  {formatSignedPercent(-analytics.baselineDownsideDelta)}
                </strong>
              </div>
              <div className="comparison-card">
                <span className="insight-label">Sticky Inflation Return</span>
                <strong className="insight-value">
                  {downsideScenario
                    ? formatPercent(downsideScenario.expectedReturn)
                    : "N/A"}
                </strong>
              </div>
              <div className="comparison-card">
                <span className="insight-label">
                  Disinflation / Easing Return
                </span>
                <strong className="insight-value">
                  {upsideScenario
                    ? formatPercent(upsideScenario.expectedReturn)
                    : "N/A"}
                </strong>
              </div>
            </div>
          </div>

          <div className="note-block">
            <p className="state-title">Diffusion Model Context</p>
            <p className="state-copy">{analytics.modelNote}</p>
          </div>
        </>
      ) : (
        <div className="note-block">
          <p className="state-title">
            {status === "loading" ? "Running scenario suite" : "Awaiting simulation"}
          </p>
          <p className="state-copy">
            Once the suite completes, this panel will turn the diffusion run
            into a signal, a risk view, and a comparison against the base case.
          </p>
        </div>
      )}
    </div>
  );
}

export default Insights;
