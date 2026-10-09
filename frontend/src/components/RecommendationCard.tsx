import { useState, type ReactNode } from 'react';
import { api, type Recommendation } from '../api/client';
import { decidedBy, SourceBadge } from './PlannerNotice';
import { Badge, Button, ConfidenceStars } from './ui';

/**
 * The Recommend → Explain Why → User Decides pattern, in one component.
 *
 * The recommendation arrives already decided — by the AI planner after the validator
 * accepted it, or by a rule-based advisor — and the badge says which. This only presents
 * it. "Explain in plain English" is a separate request that expands the reasoning; it never
 * produces or revises a recommendation, and a failure leaves the static explanation on
 * screen rather than surfacing an error.
 */
export function RecommendationCard({
  recommendation,
  llmAvailable,
  selected,
  onToggle,
  footer,
}: {
  recommendation: Recommendation;
  llmAvailable: boolean;
  selected?: boolean;
  onToggle?: (next: boolean) => void;
  footer?: ReactNode;
}) {
  const [expanded, setExpanded] = useState(false);
  const [narrative, setNarrative] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function explain() {
    setBusy(true);
    setError(null);
    try {
      const result = await api.explain(recommendation);
      if (result.narrative) setNarrative(result.narrative);
      else setError(result.error ?? 'Explanation unavailable right now.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Explanation failed.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <article className="card card-compact stack-sm">
      <div className="row-between" style={{ alignItems: 'flex-start' }}>
        <div className="row" style={{ gap: 'var(--space-3)', alignItems: 'flex-start' }}>
          {onToggle && (
            <input
              type="checkbox"
              className="checkbox"
              checked={!!selected}
              onChange={(event) => onToggle(event.target.checked)}
              aria-label={`Select ${recommendation.label}`}
              style={{ marginTop: 3 }}
            />
          )}
          <div>
            <div className="strong">{recommendation.label}</div>
            <p className="small secondary" style={{ marginTop: 2 }}>
              {recommendation.reason}
            </p>
          </div>
        </div>
        <div className="stack-sm" style={{ alignItems: 'flex-end' }}>
          <ConfidenceStars
            rating={recommendation.star_rating}
            score={recommendation.confidence_score}
          />
          <div className="row" style={{ gap: 'var(--space-1)' }}>
            <SourceBadge recommendation={recommendation} />
            {recommendation.category && <Badge tone="neutral">{recommendation.category}</Badge>}
          </div>
        </div>
      </div>

      <div className="row" style={{ gap: 'var(--space-2)' }}>
        <Button size="sm" variant="ghost" onClick={() => setExpanded((open) => !open)}>
          {expanded ? 'Hide why' : 'Why?'}
        </Button>
        {llmAvailable && expanded && !narrative && (
          <Button size="sm" variant="ghost" loading={busy} onClick={explain}>
            Explain in plain English
          </Button>
        )}
      </div>

      {expanded && (
        <div className="stack-sm">
          <p className="small prose">{recommendation.why_explanation}</p>

          {narrative && (
            <div className="card card-compact" style={{ background: 'var(--surface-sunken)' }}>
              <p className="small prose">{narrative}</p>
              <p className="xs muted" style={{ marginTop: 'var(--space-3)' }}>
                {decidedBy([recommendation])}
              </p>
            </div>
          )}

          {error && <p className="xs muted">{error}</p>}
        </div>
      )}

      {footer}
    </article>
  );
}
