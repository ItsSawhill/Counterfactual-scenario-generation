import { useEffect, useState } from "react";
import {
  buildMacroWarnings,
  describeRegime,
  findMatchingPreset,
  formatPercent,
  SCENARIO_PRESETS,
} from "../lib/analytics";
import type { SimulationRequest } from "../types/simulation";

interface SidebarProps {
  initialValues: SimulationRequest;
  onGenerate: (values: SimulationRequest) => void;
  isLoading: boolean;
  modelHorizon: number;
}

function Sidebar({
  initialValues,
  onGenerate,
  isLoading,
  modelHorizon,
}: SidebarProps) {
  const [draft, setDraft] = useState<SimulationRequest>(initialValues);

  useEffect(() => {
    setDraft(initialValues);
  }, [initialValues]);

  const updateField = <K extends keyof SimulationRequest>(
    field: K,
    value: SimulationRequest[K],
  ) => {
    setDraft((current) => ({
      ...current,
      [field]: value,
    }));
  };

  const updatePercentInput = (
    field: "inflation" | "interestRate",
    rawValue: string,
  ) => {
    const nextValue = Number(rawValue);

    if (Number.isNaN(nextValue)) {
      return;
    }

    updateField(field, nextValue / 100);
  };

  const scenarioLabel = describeRegime(draft);
  const activePreset = findMatchingPreset(draft);
  const warnings = buildMacroWarnings(draft);

  return (
    <div className="sidebar-stack">
      <div className="panel-header">
        <p className="panel-kicker">Scenario Builder</p>
        <h2 className="panel-title">Inputs</h2>
        <p className="panel-copy">
          Tighten the macro assumptions, preserve unit precision, and validate
          whether the scenario still belongs to a preset regime.
        </p>
      </div>

      <div className="scenario-banner">
        <div className="card-title-block">
          <h3>Scenario</h3>
          <span className="status-label">
            {activePreset ? activePreset.regime : "User-defined"}
          </span>
        </div>
        <strong className="scenario-banner-value">{scenarioLabel}</strong>
        <p className="scenario-banner-copy">
          {activePreset
            ? activePreset.whyItMatters
            : "The current inputs no longer line up with a preset regime, so the run is treated as a custom macro view."}
        </p>
      </div>

      {warnings.length > 0 && (
        <div className="callout-card">
          <div className="card-title-block">
            <h3>Macro Consistency</h3>
            <span className="status-label">Guardrails</span>
          </div>
          <div className="warning-list">
            {warnings.slice(0, 2).map((warning) => (
              <p key={warning} className="warning-item">
                {warning}
              </p>
            ))}
          </div>
        </div>
      )}

      <div className="control-stack">
        <div className="input-control">
          <div className="control-meta">
            <div className="control-header">
              <label className="control-label" htmlFor="inflation">
                Inflation
              </label>
              <span className="control-value">
                {formatPercent(draft.inflation)}
              </span>
            </div>
            <label className="numeric-field" htmlFor="inflation-input">
              <input
                id="inflation-input"
                type="number"
                min="1.0"
                max="8.0"
                step="0.1"
                value={(draft.inflation * 100).toFixed(1)}
                onChange={(event) =>
                  updatePercentInput("inflation", event.target.value)
                }
              />
              <span>% YoY</span>
            </label>
          </div>
          <input
            id="inflation"
            className="slider"
            type="range"
            min="0.01"
            max="0.08"
            step="0.0025"
            value={draft.inflation}
            onChange={(event) =>
              updateField("inflation", Number(event.target.value))
            }
          />
          <p className="control-help">
            Annualized CPI assumption driving the shock regime.
          </p>
        </div>

        <div className="input-control">
          <div className="control-meta">
            <div className="control-header">
              <label className="control-label" htmlFor="interest-rate">
                Policy rate
              </label>
              <span className="control-value">
                {formatPercent(draft.interestRate)}
              </span>
            </div>
            <label className="numeric-field" htmlFor="interest-rate-input">
              <input
                id="interest-rate-input"
                type="number"
                min="1.0"
                max="8.0"
                step="0.1"
                value={(draft.interestRate * 100).toFixed(1)}
                onChange={(event) =>
                  updatePercentInput("interestRate", event.target.value)
                }
              />
              <span>% Fed Funds</span>
            </label>
          </div>
          <input
            id="interest-rate"
            className="slider"
            type="range"
            min="0.01"
            max="0.08"
            step="0.0025"
            value={draft.interestRate}
            onChange={(event) =>
              updateField("interestRate", Number(event.target.value))
            }
          />
          <p className="control-help">
            Macro policy anchor used to condition the diffusion run.
          </p>
        </div>

        <div className="input-control">
          <div className="control-meta">
            <div className="control-header">
              <label className="control-label" htmlFor="horizon">
                Horizon
              </label>
              <span className="control-value">{draft.horizon} days</span>
            </div>
            <label className="numeric-field" htmlFor="horizon-input">
              <input
                id="horizon-input"
                type="number"
                min="10"
                max={modelHorizon}
                step="1"
                value={draft.horizon}
                onChange={(event) =>
                  updateField(
                    "horizon",
                    Math.min(modelHorizon, Math.max(10, Number(event.target.value))),
                  )
                }
              />
              <span>Days</span>
            </label>
          </div>
          <input
            id="horizon"
            className="slider"
            type="range"
            min="10"
            max={modelHorizon}
            step="1"
            value={draft.horizon}
            onChange={(event) =>
              updateField("horizon", Number(event.target.value))
            }
          />
          <p className="control-help">
            Forecast window. This checkpoint supports up to {modelHorizon} days.
          </p>
        </div>

        <div className="input-control">
          <div className="control-meta">
            <div className="control-header">
              <label className="control-label" htmlFor="path-count">
                Path count
              </label>
              <span className="control-value">{draft.pathCount}</span>
            </div>
            <label className="numeric-field" htmlFor="path-count-input">
              <input
                id="path-count-input"
                type="number"
                min="72"
                max="192"
                step="24"
                value={draft.pathCount}
                onChange={(event) =>
                  updateField(
                    "pathCount",
                    Math.min(192, Math.max(72, Number(event.target.value))),
                  )
                }
              />
              <span>Paths</span>
            </label>
          </div>
          <input
            id="path-count"
            className="slider"
            type="range"
            min="72"
            max="192"
            step="24"
            value={draft.pathCount}
            onChange={(event) =>
              updateField("pathCount", Number(event.target.value))
            }
          />
          <p className="control-help">
            Higher path counts improve VaR, CVaR, and sensitivity stability.
          </p>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-header">
          <h3>Presets</h3>
        </div>

        <div className="preset-grid">
          {SCENARIO_PRESETS.map((preset) => {
            const isActive = preset.label === scenarioLabel;

            return (
              <button
                key={preset.label}
                type="button"
                className={`preset-card ${isActive ? "preset-card-active" : ""}`}
                onClick={() => setDraft(preset.values)}
              >
                <span className="preset-regime">{preset.regime}</span>
                <span className="preset-title">{preset.label}</span>
                <span className="preset-copy">{preset.description}</span>
                <div className="preset-detail-row">
                  <span className="preset-detail-chip">
                    CPI {formatPercent(preset.values.inflation)}
                  </span>
                  <span className="preset-detail-chip">
                    Rate {formatPercent(preset.values.interestRate)}
                  </span>
                  <span className="preset-detail-chip">
                    {preset.values.horizon}D
                  </span>
                  <span className="preset-detail-chip">
                    {preset.values.pathCount} paths
                  </span>
                </div>
                <span className="preset-context">{preset.whyItMatters}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="button-row">
        <button
          type="button"
          className="secondary-button"
          onClick={() => setDraft(initialValues)}
        >
          Reset
        </button>
        <button
          type="button"
          className="primary-button primary-button-emphasis"
          onClick={() => onGenerate(draft)}
          disabled={isLoading}
        >
          {isLoading ? "Running Scenario..." : "Run Scenario"}
        </button>
      </div>
    </div>
  );
}

export default Sidebar;
