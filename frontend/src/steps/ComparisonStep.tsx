import { useEffect, useState } from 'react';
import {
  api,
  type ComparisonResult,
  type DiagnosticsResult,
  type ImportanceItem,
} from '../api/client';
import {
  ConfusionMatrix,
  CurveChart,
  RankedBarChart,
  ScatterPlot,
} from '../components/charts';
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
import { usePipeline } from '../store/pipeline';

type TabId = 'metrics' | 'diagnostics' | 'importance';

export function ComparisonStep() {
  const { completeAndGo } = usePipeline();

  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [model, setModel] = useState('');
  const [tab, setTab] = useState<TabId>('metrics');
  const [diagnostics, setDiagnostics] = useState<DiagnosticsResult | null>(null);
  const [importance, setImportance] = useState<ImportanceItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .comparison()
      .then((result) => {
        setComparison(result);
        setModel(result.best_model ?? String(result.rows[0]?.['Model Name'] ?? ''));
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not load metrics.'));
  }, []);

  useEffect(() => {
    if (!model) return;
    setDiagnostics(null);
    setImportance(null);
    api.diagnostics(model).then(setDiagnostics).catch(() => setDiagnostics(null));
    api
      .featureImportance(model)
      .then((result) => setImportance(result.available ? result.importances : []))
      .catch(() => setImportance([]));
  }, [model]);

  if (error) {
    return (
      <Page title="Model Comparison">
        <Alert tone="danger">{error}</Alert>
      </Page>
    );
  }

  if (!comparison) {
    return (
      <Page title="Model Comparison">
        <Loading label="Computing metrics…" />
      </Page>
    );
  }

  const modelNames = comparison.rows.map((row) => String(row['Model Name']));
  const isClassification = comparison.problem_type === 'Classification';

  return (
    <Page
      title="Model Comparison"
      subtitle="Every trained model scored on the held-out test set, side by side."
      actions={
        <Button variant="action" onClick={() => completeAndGo('comparison')}>
          Continue
        </Button>
      }
    >
      <div className="stack">
        {comparison.best_model && (
          <Alert tone="success">
            Best by {comparison.primary_metric}: <span className="strong">{comparison.best_model}</span>
          </Alert>
        )}

        <Tabs<TabId>
          active={tab}
          onChange={setTab}
          tabs={[
            { id: 'metrics', label: 'Metrics' },
            { id: 'diagnostics', label: 'Diagnostics' },
            { id: 'importance', label: 'Feature Importance' },
          ]}
        />

        {tab === 'metrics' && (
          <Card title="Comparison table" className="card-flush">
            <DataTable
              columns={comparison.columns.map((column) => ({
                key: column,
                header: column,
                numeric: column !== 'Model Name',
              }))}
              rows={comparison.rows}
              rowClassName={(row) =>
                row['Model Name'] === comparison.best_model ? 'is-best' : ''
              }
            />
          </Card>
        )}

        {tab !== 'metrics' && (
          <div className="row">
            <span className="label">Model</span>
            <div style={{ maxWidth: 260, width: '100%' }}>
              <Select value={model} options={modelNames} onChange={setModel} />
            </div>
          </div>
        )}

        {tab === 'diagnostics' && (
          <>
            {!diagnostics ? (
              <Loading label="Loading diagnostics…" />
            ) : isClassification ? (
              <div className="grid grid-2">
                <Card title="Confusion matrix" hint="Rows are actual classes, columns predicted.">
                  {diagnostics.confusion_matrix && diagnostics.labels ? (
                    <ConfusionMatrix
                      matrix={diagnostics.confusion_matrix}
                      labels={diagnostics.labels}
                    />
                  ) : (
                    <p className="small muted">Unavailable for this model.</p>
                  )}
                </Card>
                <Card title="ROC curve" hint="Binary targets only.">
                  {diagnostics.roc ? (
                    <CurveChart
                      data={diagnostics.roc.fpr.map((fpr, index) => ({
                        fpr: Number(fpr.toFixed(4)),
                        tpr: diagnostics.roc!.tpr[index],
                        reference: fpr,
                      }))}
                      xKey="fpr"
                      yKey="tpr"
                      diagonal
                    />
                  ) : (
                    <p className="small muted">
                      No probability scores, or the target has more than two classes.
                      Check ROC-AUC in the metrics table instead.
                    </p>
                  )}
                </Card>
              </div>
            ) : (
              <div className="grid grid-2">
                <Card title="Actual vs predicted">
                  {diagnostics.actual_vs_predicted ? (
                    <ScatterPlot
                      data={diagnostics.actual_vs_predicted.actual.map((actual, index) => ({
                        actual,
                        predicted: diagnostics.actual_vs_predicted!.predicted[index],
                      }))}
                      xKey="actual"
                      yKey="predicted"
                    />
                  ) : (
                    <p className="small muted">Unavailable.</p>
                  )}
                </Card>
                <Card title="Residuals" hint="Errors should scatter evenly around zero.">
                  {diagnostics.residuals ? (
                    <ScatterPlot
                      data={diagnostics.residuals.predicted.map((predicted, index) => ({
                        predicted,
                        residual: diagnostics.residuals!.residual[index],
                      }))}
                      xKey="predicted"
                      yKey="residual"
                      zeroLine
                    />
                  ) : (
                    <p className="small muted">Unavailable.</p>
                  )}
                </Card>
              </div>
            )}
          </>
        )}

        {tab === 'importance' && (
          <Card
            title={`Built-in importance — ${model}`}
            hint="From the estimator itself, not SHAP. Tree models expose split gain; linear models expose coefficient magnitude."
            actions={<Badge tone="neutral">{importance?.length ?? 0} features</Badge>}
          >
            {importance === null ? (
              <Loading />
            ) : importance.length === 0 ? (
              <EmptyState
                title="Not available for this model"
                description="This estimator exposes neither feature_importances_ nor coef_. The Explainability page computes SHAP values, which work for any model."
              />
            ) : (
              <RankedBarChart
                data={importance.slice(0, 20)}
                xKey="importance"
                yKey="feature"
                height={Math.max(240, Math.min(20, importance.length) * 26)}
              />
            )}
          </Card>
        )}
      </div>
    </Page>
  );
}
