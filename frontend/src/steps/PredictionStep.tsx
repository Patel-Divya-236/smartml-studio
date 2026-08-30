import { useEffect, useState } from 'react';
import { api, downloadFile, type PredictResult } from '../api/client';
import {
  Alert,
  Badge,
  Button,
  Card,
  DataTable,
  Loading,
  Page,
  Select,
  Stat,
} from '../components/ui';
import { NextStep } from '../components/NextStep';
import { usePipeline } from '../store/pipeline';

const VOTING = ['majority', 'weighted', 'average'];

/**
 * Predict with a single model, or combine several into a voting ensemble.
 *
 * Predictions run over the held-out test set, so the accuracy shown here is the same
 * held-out number reported on the comparison page — not a fresh, optimistic score.
 */
export function PredictionStep() {
  const { markComplete } = usePipeline();

  const [models, setModels] = useState<string[]>([]);
  const [mode, setMode] = useState<'single' | 'ensemble'>('single');
  const [model, setModel] = useState('');
  const [ensemble, setEnsemble] = useState<Set<string>>(new Set());
  const [voting, setVoting] = useState('majority');
  const [result, setResult] = useState<PredictResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .exportSummary()
      .then((summary) => {
        setModels(summary.models);
        setModel(summary.models[0] ?? '');
        setEnsemble(new Set(summary.models.slice(0, 2)));
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not load models.'));
  }, []);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const response = await api.predict(
        mode === 'single'
          ? { mode: 'single', model }
          : { mode: 'ensemble', ensemble_models: [...ensemble], voting },
      );
      setResult(response);
      markComplete('prediction', response.completed_steps);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed.');
    } finally {
      setBusy(false);
    }
  }

  function toggle(name: string) {
    setEnsemble((current) => {
      const updated = new Set(current);
      if (updated.has(name)) updated.delete(name);
      else updated.add(name);
      return updated;
    });
  }

  if (!models.length && !error) {
    return (
      <Page title="Prediction">
        <Loading />
      </Page>
    );
  }

  const previewColumns = result?.preview[0] ? Object.keys(result.preview[0]) : [];

  return (
    <Page
      title="Prediction"
      subtitle="Generate predictions on the held-out test set using one model or a voting ensemble."
    >
      <div className="stack">
        {error && <Alert tone="danger">{error}</Alert>}

        <div className="split">
          <div className="stack">
            <Card title="Strategy">
              <div className="row" style={{ marginBottom: 'var(--space-4)' }}>
                <Button
                  size="sm"
                  variant={mode === 'single' ? 'primary' : 'secondary'}
                  onClick={() => setMode('single')}
                >
                  Single model
                </Button>
                <Button
                  size="sm"
                  variant={mode === 'ensemble' ? 'primary' : 'secondary'}
                  onClick={() => setMode('ensemble')}
                >
                  Ensemble
                </Button>
              </div>

              {mode === 'single' ? (
                <div className="field" style={{ maxWidth: 300 }}>
                  <label className="label">Model</label>
                  <Select value={model} options={models} onChange={setModel} />
                </div>
              ) : (
                <div className="stack">
                  <div className="grid grid-3">
                    {models.map((name) => (
                      <label key={name} className="row" style={{ cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          className="checkbox"
                          checked={ensemble.has(name)}
                          onChange={() => toggle(name)}
                        />
                        <span className="small">{name}</span>
                      </label>
                    ))}
                  </div>
                  <div className="field" style={{ maxWidth: 220 }}>
                    <label className="label">Voting</label>
                    <Select value={voting} options={VOTING} onChange={setVoting} />
                  </div>
                  {ensemble.size < 2 && (
                    <Alert tone="warning">Pick at least two models for an ensemble.</Alert>
                  )}
                </div>
              )}
            </Card>

            {result && (
              <Card
                title="Predictions"
                hint={`${result.count.toLocaleString()} test rows`}
                className="card-flush"
                actions={
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => downloadFile('/artifacts/predictions.csv', 'predictions.csv')}
                  >
                    Download CSV
                  </Button>
                }
              >
                <DataTable
                  maxHeight={420}
                  columns={previewColumns.map((column) => ({
                    key: column,
                    header: column,
                    numeric: column === 'row' || column === 'error',
                    render:
                      column === 'correct'
                        ? (row) => (
                            <Badge tone={row.correct ? 'success' : 'danger'}>
                              {row.correct ? 'correct' : 'wrong'}
                            </Badge>
                          )
                        : undefined,
                  }))}
                  rows={result.preview}
                />
              </Card>
            )}
          </div>

          <div className="stack">
            <Card title="Run">
              <Button
                variant="action"
                block
                loading={busy}
                disabled={mode === 'ensemble' ? ensemble.size < 2 : !model}
                onClick={run}
              >
                Generate predictions
              </Button>
            </Card>

            {result && <NextStep from="prediction" note="Predictions are saved for export." />}

            {result && (
              <Card title="Result">
                <div className="stack">
                  <Stat
                    value={result.accuracy !== null ? `${(result.accuracy * 100).toFixed(2)}%` : '—'}
                    label={result.accuracy !== null ? 'Accuracy on test set' : 'Regression run'}
                  />
                  <div>
                    <div className="label">Strategy</div>
                    <p className="small secondary">{result.strategy}</p>
                  </div>
                </div>
              </Card>
            )}
          </div>
        </div>
      </div>
    </Page>
  );
}
