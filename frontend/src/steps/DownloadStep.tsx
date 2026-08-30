import { useCallback, useEffect, useState } from 'react';
import { api, downloadFile, type ExportSummary, type ReportPreview } from '../api/client';
import { Markdown } from '../components/Markdown';
import {
  Alert,
  Badge,
  Button,
  Card,
  EmptyState,
  Loading,
  Page,
  Switch,
} from '../components/ui';

/**
 * Export the run: the evaluation report, the prediction CSV, and each trained model.
 *
 * The report is shown before it is offered as a file. Downloading a document you have
 * not read, opening it, and finding it is not what you wanted is a worse loop than
 * reading it in place and then deciding.
 */
export function DownloadStep() {
  const [summary, setSummary] = useState<ExportSummary | null>(null);
  const [llmAvailable, setLlmAvailable] = useState(false);
  const [narrate, setNarrate] = useState(false);

  const [preview, setPreview] = useState<ReportPreview | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadPreview = useCallback(async (withNarration: boolean) => {
    setPreviewing(true);
    setError(null);
    try {
      setPreview(await api.reportPreview(withNarration));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not build the report.');
      setPreview(null);
    } finally {
      setPreviewing(false);
    }
  }, []);

  useEffect(() => {
    Promise.all([api.exportSummary(), api.llmStatus()])
      .then(([exportSummary, status]) => {
        setSummary(exportSummary);
        setLlmAvailable(status.available);
        // The plain report is instant; the narrated one costs an API call, so it is
        // opt-in rather than the default on arrival.
        if (exportSummary.has_comparison) void loadPreview(false);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not load exports.'));
  }, [loadPreview]);

  async function toggleNarration(next: boolean) {
    setNarrate(next);
    await loadPreview(next && llmAvailable);
  }

  async function download(path: string, filename: string, key: string) {
    setBusy(key);
    setError(null);
    try {
      await downloadFile(path, filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed.');
    } finally {
      setBusy(null);
    }
  }

  if (error && !summary) {
    return (
      <Page title="Download">
        <Alert tone="danger">{error}</Alert>
      </Page>
    );
  }

  if (!summary) {
    return (
      <Page title="Download">
        <Loading />
      </Page>
    );
  }

  return (
    <Page
      title="Download"
      subtitle="Read the report, then export it along with the predictions and any trained model."
      actions={
        <Button
          variant="action"
          loading={busy === 'report'}
          disabled={!summary.has_comparison}
          onClick={() =>
            download(
              `/artifacts/report?narrate=${narrate && llmAvailable}`,
              'smartml_report.md',
              'report',
            )
          }
        >
          Download report
        </Button>
      }
    >
      <div className="stack">
        {error && <Alert tone="danger">{error}</Alert>}

        <Card title="Run summary">
          <div className="grid grid-4">
            <div>
              <div className="label">Dataset</div>
              <p className="small strong truncate">{summary.dataset_name ?? '—'}</p>
            </div>
            <div>
              <div className="label">Target</div>
              <p className="small strong">{summary.target_column ?? '—'}</p>
            </div>
            <div>
              <div className="label">Task</div>
              <p className="small strong">{summary.problem_type ?? '—'}</p>
            </div>
            <div>
              <div className="label">Models trained</div>
              <p className="small strong">{summary.models.length}</p>
            </div>
          </div>
          {summary.prediction_strategy && (
            <p className="small muted" style={{ marginTop: 'var(--space-4)' }}>
              Prediction strategy: {summary.prediction_strategy}
            </p>
          )}
        </Card>

        <Card
          title="Evaluation report"
          hint="Exactly what the downloaded file contains."
          actions={
            <div className="row">
              <Switch
                checked={narrate && llmAvailable}
                onChange={toggleNarration}
                label={
                  llmAvailable
                    ? 'AI executive summary'
                    : 'Summary unavailable (no model configured)'
                }
              />
              {previewing && <span className="spinner" />}
            </div>
          }
        >
          {!summary.has_comparison ? (
            <EmptyState
              title="No report yet"
              description="Train at least one model — the report is built from the comparison metrics."
            />
          ) : preview ? (
            <>
              {preview.llm_error && (
                <div style={{ marginBottom: 'var(--space-4)' }}>
                  <Alert tone="warning">
                    The executive summary could not be generated ({preview.llm_error}).
                    The rest of the report is complete and unaffected.
                  </Alert>
                </div>
              )}
              {preview.narrated && (
                <div style={{ marginBottom: 'var(--space-4)' }}>
                  <Badge tone="primary">includes an AI-written summary</Badge>
                </div>
              )}
              <div className="report-frame">
                <Markdown source={preview.markdown} />
              </div>
            </>
          ) : (
            <Loading label="Building report…" />
          )}
        </Card>

        <div className="grid grid-2">
          <Card title="Predictions" hint="The held-out test rows with actual and predicted values.">
            <Button
              variant="secondary"
              loading={busy === 'csv'}
              disabled={!summary.has_predictions}
              onClick={() => download('/artifacts/predictions.csv', 'predictions.csv', 'csv')}
            >
              Download CSV
            </Button>
            {!summary.has_predictions && (
              <p className="xs muted" style={{ marginTop: 'var(--space-3)' }}>
                Run the Prediction step first.
              </p>
            )}
          </Card>

          <Card title="Trained models" hint="Each estimator is exported as a pickle.">
            {summary.models.length === 0 ? (
              <p className="small muted">No trained models.</p>
            ) : (
              <div className="stack-sm">
                {summary.models.map((model) => (
                  <div key={model} className="row-between">
                    <span className="small strong">{model}</span>
                    <Button
                      size="sm"
                      variant="secondary"
                      loading={busy === model}
                      onClick={() =>
                        download(
                          `/artifacts/model/${encodeURIComponent(model)}`,
                          `${model.toLowerCase().replace(/\s+/g, '_')}.pkl`,
                          model,
                        )
                      }
                    >
                      Download .pkl
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>
    </Page>
  );
}
