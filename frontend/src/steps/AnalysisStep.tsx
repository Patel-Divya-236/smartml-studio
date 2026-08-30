import { useEffect, useMemo, useState } from 'react';
import { api, type ProfileResult } from '../api/client';
import { CategoryBarChart, DonutChart } from '../components/charts';
import {
  Alert,
  Badge,
  Button,
  Card,
  DataTable,
  Loading,
  Page,
  Stat,
  Tabs,
} from '../components/ui';
import { usePipeline } from '../store/pipeline';

type TabId = 'overview' | 'missing' | 'columns' | 'outliers' | 'correlations' | 'target';

const TONE_BY_SEVERITY = {
  info: 'info',
  success: 'success',
  warning: 'warning',
  danger: 'danger',
} as const;

export function AnalysisStep() {
  const { completeAndGo } = usePipeline();
  const [result, setResult] = useState<ProfileResult | null>(null);
  const [tab, setTab] = useState<TabId>('overview');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .profile()
      .then(setResult)
      .catch((err) => setError(err instanceof Error ? err.message : 'Profiling failed.'));
  }, []);

  const profile = result?.profile ?? {};

  const missingRows = useMemo(() => {
    const pct: Record<string, number> = profile.missing_pct ?? {};
    const counts: Record<string, number> = profile.missing_values ?? {};
    return Object.keys(pct)
      .filter((column) => (counts[column] ?? 0) > 0)
      .map((column) => ({ column, count: counts[column], pct: pct[column] }))
      .sort((a, b) => b.pct - a.pct);
  }, [profile]);

  const columnRows = useMemo(() => {
    const dtypes: Record<string, string> = profile.dtypes ?? {};
    const cardinality: Record<string, number> = profile.cardinality ?? {};
    const numeric: string[] = profile.numeric_columns ?? [];
    return Object.keys(dtypes).map((column) => ({
      column,
      dtype: dtypes[column],
      kind: numeric.includes(column) ? 'Numeric' : 'Categorical',
      cardinality: cardinality[column] ?? 0,
      missing: profile.missing_pct?.[column] ?? 0,
    }));
  }, [profile]);

  const outlierRows = useMemo(() => {
    const outliers: Record<string, { count?: number }> = profile.outliers ?? {};
    const skew: Record<string, number> = profile.skewness ?? {};
    return Object.keys(outliers).map((column) => ({
      column,
      outliers: outliers[column]?.count ?? 0,
      skewness: skew[column] ?? 0,
    }));
  }, [profile]);

  const correlationRows = useMemo(() => {
    const matrix: Record<string, Record<string, number>> = profile.correlation_matrix ?? {};
    const seen = new Set<string>();
    const pairs: { pair: string; r: number }[] = [];
    for (const a of Object.keys(matrix)) {
      for (const b of Object.keys(matrix[a] ?? {})) {
        if (a === b) continue;
        const key = [a, b].sort().join('||');
        if (seen.has(key)) continue;
        seen.add(key);
        const r = matrix[a][b];
        if (typeof r === 'number' && Number.isFinite(r)) pairs.push({ pair: `${a} × ${b}`, r });
      }
    }
    return pairs.sort((x, y) => Math.abs(y.r) - Math.abs(x.r)).slice(0, 20);
  }, [profile]);

  const classBalance = useMemo(() => {
    const balance: Record<string, number> = profile.class_balance ?? {};
    return Object.entries(balance).map(([name, count]) => ({ name: String(name), count }));
  }, [profile]);

  if (error) {
    return (
      <Page title="Dataset Analysis">
        <Alert tone="danger">{error}</Alert>
      </Page>
    );
  }

  if (!result) {
    return (
      <Page title="Dataset Analysis">
        <Loading label="Profiling dataset…" />
      </Page>
    );
  }

  const { summary, observations } = result;

  return (
    <Page
      title="Dataset Analysis"
      subtitle="Comprehensive analysis of your dataset to understand its characteristics and data quality."
      actions={
        <Button variant="action" onClick={() => completeAndGo('analysis')}>
          Continue
        </Button>
      }
    >
      <div className="stack">
        <div className="grid grid-5">
          <Stat value={summary.rows.toLocaleString()} label="Total Rows" />
          <Stat value={summary.columns} label="Total Columns" />
          <Stat
            value={summary.duplicates.toLocaleString()}
            label="Duplicate Rows"
            sub={`${summary.duplicate_pct}% of total`}
          />
          <Stat
            value={summary.missing_total.toLocaleString()}
            label="Missing Values"
            sub="Across all columns"
          />
          <Stat value={`${summary.memory_mb} MB`} label="Memory Usage" sub="Estimated" />
        </div>

        <Tabs<TabId>
          active={tab}
          onChange={setTab}
          tabs={[
            { id: 'overview', label: 'Overview' },
            { id: 'missing', label: 'Missing Values', count: missingRows.length },
            { id: 'columns', label: 'Columns & Cardinality', count: columnRows.length },
            { id: 'outliers', label: 'Outliers & Skewness', count: outlierRows.length },
            { id: 'correlations', label: 'Correlations', count: correlationRows.length },
            { id: 'target', label: 'Target Variable' },
          ]}
        />

        {tab === 'overview' && (
          <div className="grid grid-2">
            <Card title="Key Observations">
              <div className="stack-sm">
                {observations.map((observation, index) => (
                  <div key={index} className="row" style={{ alignItems: 'flex-start' }}>
                    <span
                      className="alert-dot"
                      style={{ background: `var(--${TONE_BY_SEVERITY[observation.severity]})` }}
                    />
                    <span className="small">{observation.text}</span>
                  </div>
                ))}
              </div>
            </Card>

            <Card title="Problem Type Detection">
              <div className="row">
                <span className="strong" style={{ fontSize: 'var(--text-lg)' }}>
                  {summary.problem_type}
                </span>
                <Badge tone="success">Detected</Badge>
              </div>
              <p className="small muted" style={{ marginTop: 'var(--space-2)' }}>
                Target: <span className="strong">{summary.target_column}</span>
              </p>
              {summary.n_classes > 0 && (
                <div style={{ marginTop: 'var(--space-4)' }}>
                  <Badge tone="primary">{summary.n_classes} unique classes</Badge>
                </div>
              )}
            </Card>
          </div>
        )}

        {tab === 'missing' && (
          <Card
            title="Missing values by column"
            hint={missingRows.length ? undefined : 'No column has a missing value.'}
            className="card-flush"
          >
            {missingRows.length > 0 && (
              <DataTable
                maxHeight={420}
                columns={[
                  { key: 'column', header: 'Column' },
                  { key: 'count', header: 'Missing', numeric: true },
                  {
                    key: 'pct',
                    header: '% of rows',
                    numeric: true,
                    render: (row) => (
                      <Badge tone={row.pct > 40 ? 'danger' : row.pct > 5 ? 'warning' : 'neutral'}>
                        {row.pct.toFixed(1)}%
                      </Badge>
                    ),
                  },
                ]}
                rows={missingRows}
              />
            )}
          </Card>
        )}

        {tab === 'columns' && (
          <Card title="Columns and cardinality" className="card-flush">
            <DataTable
              maxHeight={480}
              columns={[
                { key: 'column', header: 'Column' },
                {
                  key: 'kind',
                  header: 'Type',
                  render: (row) => (
                    <Badge tone={row.kind === 'Numeric' ? 'info' : 'accent'}>{row.kind}</Badge>
                  ),
                },
                { key: 'dtype', header: 'dtype' },
                { key: 'cardinality', header: 'Distinct', numeric: true },
                {
                  key: 'missing',
                  header: 'Missing',
                  numeric: true,
                  render: (row) => `${row.missing.toFixed(1)}%`,
                },
              ]}
              rows={columnRows}
            />
          </Card>
        )}

        {tab === 'outliers' && (
          <Card
            title="Outliers and skewness"
            hint="Outliers are counted by the IQR rule."
            className="card-flush"
          >
            <DataTable
              maxHeight={480}
              columns={[
                { key: 'column', header: 'Column' },
                { key: 'outliers', header: 'Outliers', numeric: true },
                {
                  key: 'skewness',
                  header: 'Skewness',
                  numeric: true,
                  render: (row) => (
                    <Badge tone={Math.abs(row.skewness) > 1 ? 'warning' : 'neutral'}>
                      {row.skewness.toFixed(2)}
                    </Badge>
                  ),
                },
              ]}
              rows={outlierRows}
            />
          </Card>
        )}

        {tab === 'correlations' && (
          <Card
            title="Strongest feature correlations"
            hint="Top 20 pairs by absolute Pearson r."
            className="card-flush"
          >
            <DataTable
              maxHeight={480}
              columns={[
                { key: 'pair', header: 'Feature pair' },
                {
                  key: 'r',
                  header: 'r',
                  numeric: true,
                  render: (row) => (
                    <Badge
                      tone={
                        Math.abs(row.r) > 0.8 ? 'danger' : Math.abs(row.r) > 0.5 ? 'warning' : 'neutral'
                      }
                    >
                      {row.r.toFixed(3)}
                    </Badge>
                  ),
                },
              ]}
              rows={correlationRows}
            />
          </Card>
        )}

        {tab === 'target' && (
          <div className="grid grid-2">
            <Card title={`Target: ${summary.target_column}`}>
              {classBalance.length > 0 ? (
                <DonutChart data={classBalance} nameKey="name" valueKey="count" />
              ) : (
                <DataTable
                  columns={[
                    { key: 'stat', header: 'Statistic' },
                    { key: 'value', header: 'Value', numeric: true },
                  ]}
                  rows={Object.entries(profile.target_summary ?? {}).map(([stat, value]) => ({
                    stat,
                    value,
                  }))}
                  emptyLabel="No target summary available."
                />
              )}
            </Card>
            {classBalance.length > 0 && (
              <Card title="Class counts">
                <CategoryBarChart data={classBalance} xKey="name" yKey="count" />
              </Card>
            )}
          </div>
        )}
      </div>
    </Page>
  );
}
