import { useEffect, useState } from 'react';
import { api, type Recommendation } from '../api/client';
import { RecommendationCard } from '../components/RecommendationCard';
import { Alert, Badge, Button, Card, Loading, Page } from '../components/ui';
import { usePipeline } from '../store/pipeline';

/**
 * Model recommendations, ranked by rule confidence. The user selects what to train.
 *
 * Selection is stored client-side and sent with the training request, so nothing is
 * committed until the user moves on — the advisor recommends, it does not decide.
 */
export function ModelAdvisorStep() {
  const { completeAndAdvance } = usePipeline();
  const [recommendations, setRecommendations] = useState<Recommendation[] | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [llmAvailable, setLlmAvailable] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.modelAdvice(), api.llmStatus()])
      .then(([advice, status]) => {
        setRecommendations(advice.recommendations);
        setLlmAvailable(status.available);
        // Default to every recommendation checked: the advisor already filtered out
        // models that are invalid for this task type.
        setSelected(
          new Set(
            advice.recommendations
              .map((rec) => String(rec.metadata.model_name ?? ''))
              .filter(Boolean),
          ),
        );
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not load advice.'));
  }, []);

  function toggle(model: string, next: boolean) {
    setSelected((current) => {
      const updated = new Set(current);
      if (next) updated.add(model);
      else updated.delete(model);
      return updated;
    });
  }

  if (error) {
    return (
      <Page title="Smart Model Advisor">
        <Alert tone="danger">{error}</Alert>
      </Page>
    );
  }

  if (!recommendations) {
    return (
      <Page title="Smart Model Advisor">
        <Loading label="Ranking models for this dataset…" />
      </Page>
    );
  }

  return (
    <Page
      title="Smart Model Advisor"
      subtitle="Models ranked for this dataset's size, task type and feature mix. Uncheck any you do not want to train."
      actions={
        <Button
          variant="action"
          disabled={selected.size === 0}
          onClick={() => {
            usePipeline.setState({ meta: { ...usePipeline.getState().meta } });
            sessionStorage.setItem('smartml-selected-models', JSON.stringify([...selected]));
            completeAndAdvance('model-advisor');
          }}
        >
          Confirm {selected.size} model{selected.size === 1 ? '' : 's'}
        </Button>
      }
    >
      <div className="stack">
        <div className="row-between">
          <Badge tone="accent">{recommendations.length} recommended</Badge>
          <div className="row">
            <Button
              size="sm"
              variant="ghost"
              onClick={() =>
                setSelected(
                  new Set(
                    recommendations
                      .map((rec) => String(rec.metadata.model_name ?? ''))
                      .filter(Boolean),
                  ),
                )
              }
            >
              Select all
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setSelected(new Set())}>
              Clear
            </Button>
          </div>
        </div>

        {selected.size === 0 && (
          <Alert tone="warning">Select at least one model before continuing.</Alert>
        )}

        <div className="grid grid-2">
          {recommendations.map((rec, index) => {
            const model = String(rec.metadata.model_name ?? '');
            return (
              <RecommendationCard
                key={`${model}-${index}`}
                recommendation={rec}
                llmAvailable={llmAvailable}
                selected={selected.has(model)}
                onToggle={(next) => toggle(model, next)}
              />
            );
          })}
        </div>

        <Card title="What happens next">
          <p className="small prose">
            Each selected model is trained on the engineered training features and scored on the
            held-out test set. Training runs as a background job with live progress, and a model
            that fails to fit is reported rather than aborting the whole batch.
          </p>
        </Card>
      </div>
    </Page>
  );
}
