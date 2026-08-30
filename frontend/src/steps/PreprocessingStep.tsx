import { useEffect, useMemo, useState } from 'react';
import { api, type ProfileResult, type Recommendation } from '../api/client';
import {
  Alert,
  Badge,
  Button,
  Card,
  Loading,
  Modal,
  Page,
  Select,
} from '../components/ui';
import { usePipeline } from '../store/pipeline';

const IMPUTERS = ['None', 'Mean', 'Median', 'Mode', 'KNN', 'Drop Column'];
const ENCODERS = ['None', 'One-Hot', 'Ordinal/Label'];
const SCALERS = ['None', 'Standard', 'MinMax', 'Robust', 'Log1p'];

const IMPUTER_BY_ACTION: Record<string, string> = {
  mean: 'Mean',
  median: 'Median',
  mode: 'Mode',
  knn: 'KNN',
  drop: 'Drop Column',
};
const ENCODER_BY_ACTION: Record<string, string> = {
  onehot: 'One-Hot',
  ordinal: 'Ordinal/Label',
  label: 'Ordinal/Label',
};
const SCALER_BY_ACTION: Record<string, string> = {
  standard: 'Standard',
  minmax: 'MinMax',
  robust: 'Robust',
  log1p: 'Log1p',
};

interface ColumnRow {
  column: string;
  isNumeric: boolean;
  isTarget: boolean;
  missing: number;
  cardinality: number;
}

/**
 * The split ratio and the per-column pipeline, in one table.
 *
 * The split happens before any transformer is fitted. That ordering is stated on screen
 * rather than left implicit, because it is the reason the metrics reported later are
 * trustworthy — fitting on the full dataset would leak test statistics into training.
 */
export function PreprocessingStep() {
  const { completeAndAdvance } = usePipeline();

  const [profile, setProfile] = useState<ProfileResult | null>(null);
  const [byColumn, setByColumn] = useState<Record<string, Recommendation[]>>({});
  const [llmAvailable, setLlmAvailable] = useState(false);

  const [testSize, setTestSize] = useState(20);
  const [impute, setImpute] = useState<Record<string, string>>({});
  const [encode, setEncode] = useState<Record<string, string>>({});
  const [scale, setScale] = useState<Record<string, string>>({});

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [explaining, setExplaining] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.profile(), api.preprocessingAdvice(), api.llmStatus()])
      .then(([profileResult, advice, status]) => {
        setProfile(profileResult);
        setByColumn(advice.by_column);
        setLlmAvailable(status.available);

        // Seed the selectors from the advisor's recommendations so the defaults on screen
        // are the recommended pipeline, not an empty form.
        const nextImpute: Record<string, string> = {};
        const nextEncode: Record<string, string> = {};
        const nextScale: Record<string, string> = {};
        for (const recs of Object.values(advice.by_column)) {
          for (const rec of recs) {
            const column = String(rec.metadata.column ?? '');
            const action = String(rec.metadata.action ?? '').toLowerCase();
            if (rec.category === 'imputation' && IMPUTER_BY_ACTION[action]) {
              nextImpute[column] = IMPUTER_BY_ACTION[action];
            } else if (rec.category === 'encoding' && ENCODER_BY_ACTION[action]) {
              nextEncode[column] = ENCODER_BY_ACTION[action];
            } else if (rec.category === 'scaling' && SCALER_BY_ACTION[action]) {
              nextScale[column] = SCALER_BY_ACTION[action];
            }
          }
        }
        setImpute(nextImpute);
        setEncode(nextEncode);
        setScale(nextScale);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not load advice.'));
  }, []);

  const rows: ColumnRow[] = useMemo(() => {
    if (!profile) return [];
    const raw = profile.profile;
    const dtypes: Record<string, string> = raw.dtypes ?? {};
    const numeric: string[] = raw.numeric_columns ?? [];
    const target = profile.summary.target_column;

    return Object.keys(dtypes).map((column) => ({
      column,
      isNumeric: numeric.includes(column),
      isTarget: column === target,
      missing: raw.missing_pct?.[column] ?? 0,
      cardinality: raw.cardinality?.[column] ?? 0,
    }));
  }, [profile]);

  async function apply() {
    setBusy(true);
    setError(null);
    try {
      const result = await api.preprocess({
        test_size: testSize / 100,
        impute: Object.fromEntries(Object.entries(impute).filter(([, v]) => v && v !== 'None')),
        encode: Object.fromEntries(Object.entries(encode).filter(([, v]) => v && v !== 'None')),
        scale: Object.fromEntries(Object.entries(scale).filter(([, v]) => v && v !== 'None')),
      });
      completeAndAdvance('preprocessing', result.completed_steps);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Preprocessing failed.');
    } finally {
      setBusy(false);
    }
  }

  if (error && !profile) {
    return (
      <Page title="Smart Preprocessing">
        <Alert tone="danger">{error}</Alert>
      </Page>
    );
  }

  if (!profile) {
    return (
      <Page title="Smart Preprocessing">
        <Loading label="Loading column recommendations…" />
      </Page>
    );
  }

  const steps = [
    { n: 1, title: 'Split train / test', detail: `${100 - testSize}% train, ${testSize}% test` },
    { n: 2, title: 'Handle missing values', detail: `${Object.keys(impute).length} column(s)` },
    { n: 3, title: 'Encode categoricals', detail: `${Object.keys(encode).length} column(s)` },
    { n: 4, title: 'Scale features', detail: `${Object.keys(scale).length} column(s)` },
  ];

  return (
    <Page
      title="Smart Preprocessing"
      subtitle="Configure the pipeline and apply it. Recommended options are pre-selected — change any of them before applying."
    >
      <div className="stack">
        {error && <Alert tone="danger">{error}</Alert>}

        <div className="split">
          <div className="stack">
            <Card
              title="Train / test split"
              hint="The dataset is split before any imputer, scaler, or encoder is fitted."
            >
              <div className="row-between">
                <span className="small secondary">Test set ratio</span>
                <span className="strong num">{testSize}%</span>
              </div>
              <input
                className="slider"
                type="range"
                min={10}
                max={50}
                step={5}
                value={testSize}
                onChange={(event) => setTestSize(Number(event.target.value))}
                aria-label="Test dataset split ratio"
                style={{ marginTop: 'var(--space-3)' }}
              />
              <p className="xs muted" style={{ marginTop: 'var(--space-3)' }}>
                Transformers learn column statistics from the rows they are fitted on. Fitting
                them on the whole dataset would carry test-set information into training and make
                every score reported later optimistic.
              </p>
            </Card>

            <Card title="Column pipeline configuration" className="card-flush">
              <div className="table-wrap" style={{ border: 'none', maxHeight: 520, overflowY: 'auto' }}>
                <table className="data">
                  <thead>
                    <tr>
                      <th>Column</th>
                      <th>Type</th>
                      <th className="num">Missing</th>
                      <th className="num">Distinct</th>
                      <th>Imputer</th>
                      <th>Encoder</th>
                      <th>Scaler</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr key={row.column} className={row.isTarget ? 'is-target' : undefined}>
                        <td className="strong">{row.column}</td>
                        <td>
                          <Badge tone={row.isTarget ? 'danger' : row.isNumeric ? 'info' : 'accent'}>
                            {row.isTarget ? 'Target' : row.isNumeric ? 'Numeric' : 'Categorical'}
                          </Badge>
                        </td>
                        <td className="num">{row.missing.toFixed(1)}%</td>
                        <td className="num">{row.cardinality.toLocaleString()}</td>
                        <td>
                          <Select
                            small
                            disabled={row.isTarget}
                            value={impute[row.column] ?? 'None'}
                            options={IMPUTERS}
                            onChange={(value) =>
                              setImpute((current) => ({ ...current, [row.column]: value }))
                            }
                            aria-label={`Imputer for ${row.column}`}
                          />
                        </td>
                        <td>
                          <Select
                            small
                            disabled={row.isTarget || row.isNumeric}
                            value={encode[row.column] ?? 'None'}
                            options={ENCODERS}
                            onChange={(value) =>
                              setEncode((current) => ({ ...current, [row.column]: value }))
                            }
                            aria-label={`Encoder for ${row.column}`}
                          />
                        </td>
                        <td>
                          <Select
                            small
                            disabled={row.isTarget || !row.isNumeric}
                            value={scale[row.column] ?? 'None'}
                            options={SCALERS}
                            onChange={(value) =>
                              setScale((current) => ({ ...current, [row.column]: value }))
                            }
                            aria-label={`Scaler for ${row.column}`}
                          />
                        </td>
                        <td>
                          {byColumn[row.column]?.length > 0 && (
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => setExplaining(row.column)}
                            >
                              Why?
                            </Button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          <div className="stack">
            <Card title="Pipeline steps">
              <ol className="steps">
                {steps.map((step) => (
                  <li key={step.n}>
                    <span className="step-n">{step.n}</span>
                    <div>
                      <div className="small strong">{step.title}</div>
                      <div className="xs muted">{step.detail}</div>
                    </div>
                  </li>
                ))}
              </ol>
              <Button
                variant="action"
                block
                loading={busy}
                onClick={apply}
                style={{ marginTop: 'var(--space-4)' }}
              >
                Apply preprocessing
              </Button>
            </Card>

            <Card title="Recommended split">
              <div className="row">
                <Badge tone="accent">20%</Badge>
                <span className="small secondary">Good balance for model evaluation</span>
              </div>
            </Card>
          </div>
        </div>
      </div>

      {explaining && (
        <ColumnExplanation
          column={explaining}
          recommendations={byColumn[explaining] ?? []}
          llmAvailable={llmAvailable}
          onClose={() => setExplaining(null)}
        />
      )}
    </Page>
  );
}

/** Explains every step recommended for one column, together. */
function ColumnExplanation({
  column,
  recommendations,
  llmAvailable,
  onClose,
}: {
  column: string;
  recommendations: Recommendation[];
  llmAvailable: boolean;
  onClose: () => void;
}) {
  const [narrative, setNarrative] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function explain() {
    setBusy(true);
    setError(null);
    try {
      const result = await api.explainColumn(column, recommendations);
      if (result.narrative) setNarrative(result.narrative);
      else setError(result.error ?? 'Explanation unavailable right now.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Explanation failed.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal title={`Why these defaults for ${column}?`} onClose={onClose}>
      <div className="stack">
        {recommendations.map((rec, index) => (
          <div key={index}>
            <div className="strong small">{rec.label}</div>
            <p className="small secondary">{rec.reason}</p>
            <p className="small prose" style={{ marginTop: 'var(--space-2)' }}>
              {rec.why_explanation}
            </p>
          </div>
        ))}

        {llmAvailable && !narrative && (
          <Button variant="secondary" loading={busy} onClick={explain}>
            Explain in plain English
          </Button>
        )}

        {narrative && (
          <div className="card card-compact" style={{ background: 'var(--surface-sunken)' }}>
            <p className="small prose">{narrative}</p>
            <p className="xs muted" style={{ marginTop: 'var(--space-3)' }}>
              These steps were chosen by the rule-based advisor. The language model only
              explains them.
            </p>
          </div>
        )}

        {error && <p className="xs muted">{error}</p>}
      </div>
    </Modal>
  );
}
