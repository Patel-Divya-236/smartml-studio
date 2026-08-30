import { useEffect, useMemo, useState } from 'react';
import {
  api,
  type DistributionResult,
  type ProfileResult,
  type Recommendation,
} from '../api/client';
import { CategoryBarChart, DonutChart, RankedBarChart } from '../components/charts';
import { RecommendationCard } from '../components/RecommendationCard';
import {
  Alert,
  Badge,
  Button,
  DataTable,
  EmptyState,
  Loading,
  Modal,
  Page,
  Tabs,
} from '../components/ui';
import { usePipeline } from '../store/pipeline';

type TabId = 'all' | 'distribution' | 'correlation' | 'target';

/**
 * Chart recommendations, each rendered from data the backend computes on demand.
 *
 * The Streamlit version drew every chart eagerly. Here a card states what the chart is
 * and why it was recommended, and the chart itself opens in a modal — nine plots do not
 * need to compete for attention before the user has chosen one.
 */
export function VisualizationStep() {
  const { completeAndGo } = usePipeline();
  const [recommendations, setRecommendations] = useState<Recommendation[] | null>(null);
  const [profile, setProfile] = useState<ProfileResult | null>(null);
  const [llmAvailable, setLlmAvailable] = useState(false);
  const [tab, setTab] = useState<TabId>('all');
  const [search, setSearch] = useState('');
  const [open, setOpen] = useState<Recommendation | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.visualizationAdvice(), api.profile(), api.llmStatus()])
      .then(([advice, profileResult, status]) => {
        setRecommendations(advice.recommendations);
        setProfile(profileResult);
        setLlmAvailable(status.available);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not load advice.'));
  }, []);

  const typeOf = (rec: Recommendation) => String(rec.metadata.chart_type ?? '');

  const matchesTab = (rec: Recommendation, which: TabId) => {
    const chartType = typeOf(rec);
    if (which === 'all') return true;
    if (which === 'target') return chartType.startsWith('target');
    if (which === 'correlation') return chartType.includes('correlation');
    return !chartType.startsWith('target') && !chartType.includes('correlation');
  };

  const filtered = useMemo(() => {
    if (!recommendations) return [];
    const term = search.trim().toLowerCase();
    return recommendations.filter(
      (rec) => matchesTab(rec, tab) && (!term || rec.label.toLowerCase().includes(term)),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recommendations, tab, search]);

  const counts = useMemo(() => {
    const all = recommendations ?? [];
    return {
      all: all.length,
      target: all.filter((rec) => matchesTab(rec, 'target')).length,
      correlation: all.filter((rec) => matchesTab(rec, 'correlation')).length,
      distribution: all.filter((rec) => matchesTab(rec, 'distribution')).length,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recommendations]);

  if (error) {
    return (
      <Page title="Smart Visualization">
        <Alert tone="danger">{error}</Alert>
      </Page>
    );
  }

  if (!recommendations || !profile) {
    return (
      <Page title="Smart Visualization">
        <Loading label="Selecting charts for this dataset…" />
      </Page>
    );
  }

  return (
    <Page
      title="Smart Visualization"
      subtitle="Charts chosen for this dataset's shape and task type. Each card explains why it was recommended."
      actions={
        <Button variant="action" onClick={() => completeAndGo('visualization')}>
          Continue
        </Button>
      }
    >
      <div className="stack">
        <div className="row-between">
          <input
            className="input"
            style={{ maxWidth: 320 }}
            placeholder="Search visualizations…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            aria-label="Search visualizations"
          />
          <Badge tone="accent">{counts.all} recommended</Badge>
        </div>

        <Tabs<TabId>
          active={tab}
          onChange={setTab}
          tabs={[
            { id: 'all', label: 'All', count: counts.all },
            { id: 'target', label: 'Target', count: counts.target },
            { id: 'distribution', label: 'Distribution', count: counts.distribution },
            { id: 'correlation', label: 'Correlation', count: counts.correlation },
          ]}
        />

        {filtered.length === 0 ? (
          <EmptyState title="No charts match" description="Try a different tab or clear the search." />
        ) : (
          <div className="grid grid-3">
            {filtered.map((rec, index) => (
              <RecommendationCard
                key={`${rec.label}-${index}`}
                recommendation={rec}
                llmAvailable={llmAvailable}
                footer={
                  <Button size="sm" variant="secondary" block onClick={() => setOpen(rec)}>
                    View chart
                  </Button>
                }
              />
            ))}
          </div>
        )}
      </div>

      {open && (
        <Modal title={open.label} onClose={() => setOpen(null)}>
          <ChartFor recommendation={open} profile={profile} />
          <p className="small prose" style={{ marginTop: 'var(--space-4)' }}>
            {open.why_explanation}
          </p>
        </Modal>
      )}
    </Page>
  );
}

/** Render the chart a recommendation describes. */
function ChartFor({
  recommendation,
  profile,
}: {
  recommendation: Recommendation;
  profile: ProfileResult;
}) {
  const chartType = String(recommendation.metadata.chart_type ?? '');
  const column =
    String(recommendation.metadata.column ?? '') ||
    (chartType.startsWith('target') ? String(profile.summary.target_column ?? '') : '');

  const isCorrelation = chartType.includes('correlation');
  const [distribution, setDistribution] = useState<DistributionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isCorrelation || !column) return;
    setDistribution(null);
    api
      .distribution(column)
      .then(setDistribution)
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not load the column.'));
  }, [column, isCorrelation]);

  if (isCorrelation) {
    const matrix: Record<string, Record<string, number>> = profile.profile.correlation_matrix ?? {};
    const seen = new Set<string>();
    const pairs: { name: string; r: number }[] = [];
    for (const a of Object.keys(matrix)) {
      for (const b of Object.keys(matrix[a] ?? {})) {
        if (a === b) continue;
        const key = [a, b].sort().join('||');
        if (seen.has(key)) continue;
        seen.add(key);
        const r = matrix[a][b];
        if (typeof r === 'number' && Number.isFinite(r)) pairs.push({ name: `${a} × ${b}`, r });
      }
    }
    const top = pairs.sort((x, y) => Math.abs(y.r) - Math.abs(x.r)).slice(0, 12);

    if (!top.length) {
      return <EmptyState title="No numeric pairs to correlate" />;
    }
    return (
      <>
        <RankedBarChart data={top} xKey="r" yKey="name" height={340} diverging />
        <p className="xs muted" style={{ marginTop: 'var(--space-3)' }}>
          Strongest feature pairs by absolute Pearson r. Copper bars are negative
          correlations.
        </p>
      </>
    );
  }

  if (error) return <Alert tone="danger">{error}</Alert>;
  if (!column) return <EmptyState title="This recommendation names no column" />;
  if (!distribution) return <Loading label="Loading distribution…" />;
  if (!distribution.data.length) return <EmptyState title="This column has no values to plot" />;

  // A free-text column has no categories to compare. Drawing bars of near-identical
  // height labelled with truncated sentences would look like a chart while telling the
  // reader nothing, so the counts are shown as a table with the reason stated.
  if (distribution.kind === 'free_text') {
    return (
      <div className="stack">
        <Alert tone="warning">
          <span className="strong">{distribution.column}</span> holds free text, not
          categories — {distribution.distinct?.toLocaleString()} distinct values across{' '}
          {distribution.rows?.toLocaleString()} rows, averaging{' '}
          {distribution.avg_label_length} characters. A frequency chart of it would be
          unreadable and would tell you nothing, so the most common values are listed
          instead.
        </Alert>
        <DataTable
          maxHeight={360}
          columns={[
            {
              key: 'name',
              header: 'Value',
              render: (row) => (
                <span title={String(row.name)}>
                  {String(row.name).length > 90
                    ? `${String(row.name).slice(0, 89)}…`
                    : String(row.name)}
                </span>
              ),
            },
            { key: 'count', header: 'Count', numeric: true },
            {
              key: 'share',
              header: 'Share',
              numeric: true,
              render: (row) => (row.share !== undefined ? `${row.share}%` : '—'),
            },
          ]}
          rows={distribution.data}
        />
        <p className="xs muted">
          A column like this is usually dropped before training, or turned into features
          (length, keyword flags) rather than used directly.
        </p>
      </div>
    );
  }

  const usePie = chartType.includes('pie') || chartType.includes('donut');

  return (
    <>
      {usePie ? (
        <DonutChart data={distribution.data} nameKey="name" valueKey="count" height={320} />
      ) : distribution.data.length > 8 ? (
        <RankedBarChart
          data={[...distribution.data].reverse()}
          xKey="count"
          yKey="name"
          height={Math.max(260, Math.min(25, distribution.data.length) * 24)}
        />
      ) : (
        <CategoryBarChart data={distribution.data} xKey="name" yKey="count" height={300} />
      )}

      <div className="row" style={{ marginTop: 'var(--space-3)', flexWrap: 'wrap' }}>
        <Badge tone="neutral">{distribution.kind === 'histogram' ? 'histogram' : 'value counts'}</Badge>
        {distribution.distinct !== undefined && (
          <Badge tone="neutral">{distribution.distinct.toLocaleString()} distinct</Badge>
        )}
        {distribution.truncated && <Badge tone="warning">showing the top 25</Badge>}
      </div>
    </>
  );
}
