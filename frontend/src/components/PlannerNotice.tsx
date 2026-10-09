import { useState } from 'react';
import { sourceOf, type PlannerInfo, type Recommendation } from '../api/client';
import { Alert, Badge, Button } from './ui';

type Section = 'preprocessing' | 'features' | 'models';

const FALLBACK: Record<Section, string> = {
  preprocessing: "These are the rule-based advisor's recommendations.",
  features: 'Every technique starts switched off, which is a valid pipeline.',
  models: "These are the rule-based advisor's rankings.",
};

const COVERAGE: Record<Section, string> = {
  preprocessing: 'Cards marked AI came from the plan; Rule cards fill any column or step it left out.',
  features: 'Recommended techniques are switched on below; change anything before applying.',
  models: 'Models marked AI came from the plan; Rule models top up a short list.',
};

/**
 * Says where the recommendations on a page came from, and what the validator refused.
 *
 * The planner proposes and the deterministic validator decides. Showing the refusals —
 * not just the survivors — is what makes that visible: a reader can see the model suggest
 * scaling a text column and the validator turn it down, with the reason.
 */
export function PlannerNotice({
  planner,
  section,
}: {
  planner: PlannerInfo | null | undefined;
  section: Section;
}) {
  const [open, setOpen] = useState(false);
  if (!planner) return null;

  if (planner.status === 'unavailable') {
    return (
      <Alert tone="info">
        AI planning is off because no LLM API key is configured. {FALLBACK[section]}
      </Alert>
    );
  }

  if (planner.status === 'failed') {
    return (
      <Alert tone="warning">
        AI planning did not complete{planner.error ? `: ${planner.error}` : '.'} {FALLBACK[section]}
      </Alert>
    );
  }

  const rejected = planner.rejected;
  return (
    <Alert tone="success">
      <div className="stack-sm">
        <div className="row-between" style={{ alignItems: 'flex-start', gap: 'var(--space-3)' }}>
          <span>
            Planned by AI{planner.model ? ` (${planner.model})` : ''} in {planner.elapsed_seconds}s,
            then checked by the deterministic validator. {COVERAGE[section]}{' '}
            {rejected.length === 0
              ? 'The validator rejected nothing here.'
              : `The validator rejected ${rejected.length} suggestion${rejected.length === 1 ? '' : 's'}.`}
          </span>
          {rejected.length > 0 && (
            <Button size="sm" variant="ghost" onClick={() => setOpen((current) => !current)}>
              {open ? 'Hide rejected' : 'Show rejected'}
            </Button>
          )}
        </div>
        {open && (
          <ul className="small" style={{ margin: 0, paddingLeft: 'var(--space-5)' }}>
            {rejected.map((rejection, index) => (
              <li key={index} style={{ marginTop: 'var(--space-1)' }}>
                <span className="strong">{rejection.item}</span>: {rejection.reason}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Alert>
  );
}

/** "AI" for a validated planner recommendation, "Rule" for a rule-based one. */
export function SourceBadge({ recommendation }: { recommendation: Recommendation }) {
  return sourceOf(recommendation) === 'llm' ? (
    <Badge tone="primary">AI</Badge>
  ) : (
    <Badge tone="neutral">Rule</Badge>
  );
}

/** The footnote under a plain-English explanation, which depends on who decided. */
export function decidedBy(recommendations: Recommendation[]): string {
  const sources = new Set(recommendations.map(sourceOf));
  if (sources.size === 1 && sources.has('llm')) {
    return 'Proposed by the AI planner and accepted by the validator. This explanation is a separate request and cannot change what was recommended.';
  }
  if (sources.has('llm')) {
    return 'Proposed partly by the AI planner (accepted by the validator) and partly by the rule-based advisor. This explanation cannot change what was recommended.';
  }
  return 'The recommendation was decided by the rule-based advisor. The language model only explains it.';
}
