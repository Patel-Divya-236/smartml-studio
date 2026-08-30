import { useEffect, useState } from 'react';
import { api, type ShapGlobalResult, type ShapLocalResult } from '../api/client';
import { RankedBarChart } from '../components/charts';
import {
  Alert,
  Badge,
  Button,
  Card,
  DataTable,
  EmptyState,
  Loading,
  Page,
  Select,
  Tabs,
} from '../components/ui';
import { NextStep } from '../components/NextStep';
import { usePipeline } from '../store/pipeline';

type TabId = 'global' | 'local';

/**
 * SHAP explanations, with a plain-English narration of each.
 *
 * Every number here is computed by SHAP from the trained model. The narration buttons
 * send those already-computed numbers to a language model and get prose back — the model
 * never produces an attribution, a ranking, or a metric.
 *
 * The contribution units matter and are shown: SHAP values are only readable as
 * probability for some model and explainer combinations. Presenting a log-odds
 * contribution as a percentage would mislead, so the units travel with the numbers.
 */
export function ExplainabilityStep() {
  const { markComplete } = usePipeline();

  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState('');
  const [tab, setTab] = useState<TabId>('global');
  const [llmAvailable, setLlmAvailable] = useState(false);

  const [global, setGlobal] = useState<ShapGlobalResult | null>(null);
  const [local, setLocal] = useState<ShapLocalResult | null>(null);
  const [sampleIndex, setSampleIndex] = useState(0);

  const [globalNarrative, setGlobalNarrative] = useState<string | null>(null);
  const [localNarrative, setLocalNarrative] = useState<string | null>(null);
  const [narrating, setNarrating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.exportSummary(), api.llmStatus()])
      .then(([summary, status]) => {
        setModels(summary.models);
        setModel(summary.models[0] ?? '');
        setLlmAvailable(status.available);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not load models.'));
  }, []);

  useEffect(() => {
    if (!model) return;
    setGlobal(null);
    setGlobalNarrative(null);
    setError(null);
    api
      .shapGlobal(model)
      .then((result) => {
        setGlobal(result);
        // Completion only. Navigating here would fire the moment SHAP finishes
        // computing — before the user has looked at anything.
        markComplete('explainability');
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'SHAP failed.'));
    // markComplete is stable on the store; re-running on model change is intended.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  useEffect(() => {
    if (!model || tab !== 'local') return;
    setLocal(null);
    setLocalNarrative(null);
    api
      .shapLocal(model, sampleIndex)
      .then(setLocal)
      .catch((err) => setError(err instanceof Error ? err.message : 'SHAP failed.'));
  }, [model, sampleIndex, tab]);

  async function narrate(kind: TabId) {
    setNarrating(true);
    try {
      const result =
        kind === 'global'
          ? await api.narrateGlobal(model)
          : await api.narrateLocal(model, sampleIndex);
      if (kind === 'global') setGlobalNarrative(result.narrative);
      else setLocalNarrative(result.narrative);
    } catch {
      /* narration is additive — the plot above is unaffected */
    } finally {
      setNarrating(false);
    }
  }

  if (error && !global) {
    return (
      <Page title="Explainability">
        <Alert tone="danger">{error}</Alert>
      </Page>
    );
  }

  if (!models.length) {
    return (
      <Page title="Explainability">
        <Loading />
      </Page>
    );
  }

  return (
    <Page
      title="Explainability"
      subtitle="Which features drive this model overall, and why it made one particular prediction."
      actions={
        <div style={{ minWidth: 220 }}>
          <Select value={model} options={models} onChange={setModel} aria-label="Model" />
        </div>
      }
    >
      <div className="stack">
        <Tabs<TabId>
          active={tab}
          onChange={setTab}
          tabs={[
            { id: 'global', label: 'Global — what drives the model' },
            { id: 'local', label: 'Local — one prediction' },
          ]}
        />

        {global?.available && (
          <NextStep from="explainability" note="Explore both tabs before moving on." />
        )}

        {tab === 'global' && (
          <>
            {!global ? (
              <Loading label="Computing SHAP values…" />
            ) : !global.available ? (
              <EmptyState
                title="No usable attributions"
                description="SHAP could not reduce this model's output to one value per feature."
              />
            ) : (
              <div className="stack">
                <div className="row" style={{ flexWrap: 'wrap' }}>
                  <Badge tone="info">{global.explainer_type} explainer</Badge>
                  <Badge tone="neutral">units: {global.output_space}</Badge>
                  {global.is_subset && (
                    <Badge tone="warning">evaluated on a sample of rows for speed</Badge>
                  )}
                </div>

                <Card
                  title="Mean absolute SHAP per feature"
                  hint="How much each feature moves the model's output on average, direction ignored."
                >
                  <RankedBarChart
                    data={global.importances}
                    xKey="value"
                    yKey="feature"
                    height={Math.max(260, global.importances.length * 26)}
                  />
                </Card>

                <Card
                  title="What this says, in plain English"
                  hint="The chart above encodes impact as bar length. This describes the same ranking in words."
                  actions={
                    llmAvailable && !globalNarrative ? (
                      <Button size="sm" variant="secondary" loading={narrating} onClick={() => narrate('global')}>
                        Explain what drives this model
                      </Button>
                    ) : undefined
                  }
                >
                  {globalNarrative ? (
                    <>
                      <p className="small prose">{globalNarrative}</p>
                      <p className="xs muted" style={{ marginTop: 'var(--space-3)' }}>
                        Rankings are the mean absolute SHAP values plotted above, computed
                        locally. The language model only rewrites them as prose.
                      </p>
                    </>
                  ) : (
                    <p className="small muted">
                      {llmAvailable
                        ? 'Press the button to generate a description of this ranking.'
                        : 'No language model is configured, so narration is unavailable. The chart above is unaffected.'}
                    </p>
                  )}
                </Card>
              </div>
            )}
          </>
        )}

        {tab === 'local' && (
          <div className="stack">
            <Card title="Choose a test row">
              <div className="row">
                <input
                  className="slider"
                  type="range"
                  min={0}
                  max={Math.max(0, local?.max_index ?? 0)}
                  value={sampleIndex}
                  onChange={(event) => setSampleIndex(Number(event.target.value))}
                  aria-label="Test sample index"
                />
                <span className="strong num" style={{ minWidth: 48, textAlign: 'right' }}>
                  #{sampleIndex}
                </span>
              </div>
            </Card>

            {!local ? (
              <Loading label="Computing attributions…" />
            ) : !local.available ? (
              <EmptyState title="No usable attributions for this row" />
            ) : (
              <>
                <div className="row" style={{ flexWrap: 'wrap' }}>
                  <Badge tone="primary">predicted: {String(local.predicted)}</Badge>
                  <Badge tone={String(local.predicted) === String(local.actual) ? 'success' : 'danger'}>
                    actual: {String(local.actual)}
                  </Badge>
                  <Badge tone="neutral">units: {local.output_space}</Badge>
                  {local.base_value !== null && (
                    <Badge tone="neutral">baseline: {local.base_value.toFixed(4)}</Badge>
                  )}
                </div>

                <div className="grid grid-2">
                  <Card
                    title="Feature contributions"
                    hint="Positive pushes the prediction up, negative pushes it down."
                  >
                    <RankedBarChart
                      data={local.contributions.slice(0, 15)}
                      xKey="contribution"
                      yKey="feature"
                      height={Math.max(240, Math.min(15, local.contributions.length) * 26)}
                      diverging
                    />
                  </Card>

                  <Card title="Values behind this row" className="card-flush">
                    <DataTable
                      maxHeight={360}
                      columns={[
                        { key: 'feature', header: 'Feature' },
                        { key: 'value', header: 'Value', numeric: true },
                        { key: 'contribution', header: 'Contribution', numeric: true },
                      ]}
                      rows={local.contributions.slice(0, 25)}
                    />
                  </Card>
                </div>

                <Card
                  title="Explain this prediction in plain English"
                  actions={
                    llmAvailable && !localNarrative ? (
                      <Button size="sm" variant="secondary" loading={narrating} onClick={() => narrate('local')}>
                        Generate explanation
                      </Button>
                    ) : undefined
                  }
                >
                  {localNarrative ? (
                    <>
                      <p className="small prose">{localNarrative}</p>
                      <p className="xs muted" style={{ marginTop: 'var(--space-3)' }}>
                        Generated from the SHAP values shown above. The prediction and those
                        attributions are computed locally — the language model only rewrites
                        them as prose.
                      </p>
                    </>
                  ) : (
                    <p className="small muted">
                      {llmAvailable
                        ? 'Turns the numbers above into two short paragraphs, for readers who do not read SHAP plots.'
                        : 'No language model is configured, so narration is unavailable.'}
                    </p>
                  )}
                </Card>
              </>
            )}
          </div>
        )}
      </div>
    </Page>
  );
}
