import { nextStepMeta, usePipeline } from '../store/pipeline';
import type { StepId } from '../api/client';
import { Button } from './ui';

/**
 * Explicit continue control for steps that render their own results.
 *
 * Auto-advance is right for steps whose output feeds the next screen, and wrong for
 * steps where the output *is* the point — predictions, SHAP, training timings. Those
 * mark themselves complete and offer this instead, so the result stays on screen until
 * the user is done with it.
 */
export function NextStep({ from, note }: { from: StepId; note?: string }) {
  const { goTo } = usePipeline();
  const next = nextStepMeta(from);
  if (!next) return null;

  return (
    <div className="card card-compact row-between">
      <div>
        <div className="small strong">Ready to continue</div>
        {note && <div className="xs muted">{note}</div>}
      </div>
      <Button variant="action" onClick={() => goTo(next.id)}>
        Continue to {next.label}
      </Button>
    </div>
  );
}
