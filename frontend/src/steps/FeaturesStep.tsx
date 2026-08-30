import { useState } from 'react';
import { api, type FeatureResult } from '../api/client';
import { Alert, Badge, Button, Card, DataTable, Page, Stat, Switch } from '../components/ui';
import { NextStep } from '../components/NextStep';
import { usePipeline } from '../store/pipeline';

/**
 * Opt-in feature engineering. Every step here is fitted on the training rows only.
 *
 * SelectKBest is the one worth naming: fitting it on the whole dataset selects features
 * using the test set's target values, which is direct label leakage rather than the
 * milder statistical kind.
 */
export function FeaturesStep() {
  const { markComplete } = usePipeline();

  const [lowVariance, setLowVariance] = useState(false);
  const [threshold, setThreshold] = useState(0.01);
  const [poly, setPoly] = useState(false);
  const [degree, setDegree] = useState(2);
  const [interactionOnly, setInteractionOnly] = useState(false);
  const [pca, setPca] = useState(false);
  const [components, setComponents] = useState(2);
  const [selectK, setSelectK] = useState(false);
  const [k, setK] = useState(5);

  const [result, setResult] = useState<FeatureResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const anyActive = lowVariance || poly || pca || selectK;

  async function apply() {
    setBusy(true);
    setError(null);
    try {
      const response = await api.features({
        low_variance_active: lowVariance,
        low_variance_threshold: threshold,
        poly_active: poly,
        poly_degree: degree,
        poly_interaction_only: interactionOnly,
        pca_active: pca,
        pca_components: components,
        select_k_best_active: selectK,
        select_k_best_k: k,
      });
      setResult(response);
      markComplete('features', response.completed_steps);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Feature engineering failed.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Page
      title="Feature Engineering"
      subtitle="Optional transformations applied after the split. Skipping this step entirely is a valid choice — the preprocessed features carry straight through."
    >
      <div className="stack">
        {error && <Alert tone="danger">{error}</Alert>}

        <Alert tone="info">
          Every step below is fitted on the training rows and replayed onto the test rows.
          Feature selection in particular must never see the test target.
        </Alert>

        <div className="grid grid-2">
          <Card
            title="Drop low-variance features"
            hint="Removes columns that barely change across rows and so carry almost no signal."
            actions={<Switch checked={lowVariance} onChange={setLowVariance} label="" />}
          >
            <div className="field">
              <label className="label">Variance threshold</label>
              <input
                className="input"
                type="number"
                step={0.001}
                min={0}
                value={threshold}
                disabled={!lowVariance}
                onChange={(event) => setThreshold(Number(event.target.value))}
              />
            </div>
          </Card>

          <Card
            title="Polynomial features"
            hint="Multiplies numeric columns together so a linear model can express curved relationships."
            actions={<Switch checked={poly} onChange={setPoly} label="" />}
          >
            <div className="grid grid-2">
              <div className="field">
                <label className="label">Degree</label>
                <input
                  className="input"
                  type="number"
                  min={2}
                  max={3}
                  value={degree}
                  disabled={!poly}
                  onChange={(event) => setDegree(Number(event.target.value))}
                />
              </div>
              <div className="field">
                <label className="label">Interactions only</label>
                <Switch
                  checked={interactionOnly}
                  onChange={setInteractionOnly}
                  label={interactionOnly ? 'Products only' : 'Include powers'}
                />
              </div>
            </div>
          </Card>

          <Card
            title="PCA"
            hint="Compresses correlated numeric columns into a smaller set of combined components."
            actions={<Switch checked={pca} onChange={setPca} label="" />}
          >
            <div className="field">
              <label className="label">Components</label>
              <input
                className="input"
                type="number"
                min={1}
                value={components}
                disabled={!pca}
                onChange={(event) => setComponents(Number(event.target.value))}
              />
            </div>
          </Card>

          <Card
            title="Select K best"
            hint="Keeps the K features most statistically associated with the target."
            actions={<Switch checked={selectK} onChange={setSelectK} label="" />}
          >
            <div className="field">
              <label className="label">K</label>
              <input
                className="input"
                type="number"
                min={1}
                value={k}
                disabled={!selectK}
                onChange={(event) => setK(Number(event.target.value))}
              />
            </div>
          </Card>
        </div>

        <div className="row">
          <Button variant="action" loading={busy} onClick={apply}>
            {anyActive ? 'Apply feature engineering' : 'Continue without changes'}
          </Button>
          {!anyActive && (
            <span className="small muted">
              No step selected — features pass through unchanged.
            </span>
          )}
        </div>

        {result && (
          <>
            <NextStep from="features" note="The engineered feature set is stored." />

            <div className="grid grid-3">
              <Stat value={result.features_before} label="Features before" />
              <Stat value={result.features_after} label="Features after" />
              <Stat
                value={
                  result.features_after === result.features_before
                    ? 'unchanged'
                    : `${result.features_after > result.features_before ? '+' : ''}${
                        result.features_after - result.features_before
                      }`
                }
                label="Net change"
              />
            </div>

            <Card title="Final feature set" hint={`${result.feature_names.length} model input columns`}>
              <div className="row" style={{ flexWrap: 'wrap', gap: 'var(--space-2)' }}>
                {result.feature_names.slice(0, 60).map((name) => (
                  <Badge key={name} tone="neutral">
                    {name}
                  </Badge>
                ))}
                {result.feature_names.length > 60 && (
                  <span className="xs muted">
                    +{result.feature_names.length - 60} more
                  </span>
                )}
              </div>
            </Card>

            <Card title="Transformed preview" className="card-flush">
              <DataTable
                maxHeight={320}
                columns={result.feature_names.slice(0, 12).map((name) => ({
                  key: name,
                  header: name,
                  numeric: true,
                }))}
                rows={result.preview}
              />
            </Card>
          </>
        )}
      </div>
    </Page>
  );
}
