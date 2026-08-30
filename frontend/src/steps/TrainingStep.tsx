import { useEffect, useRef, useState } from 'react';
import { api, trainingSocketUrl, type TrainingJob } from '../api/client';
import {
  Alert,
  Badge,
  Button,
  Card,
  DataTable,
  Loading,
  Page,
  Progress,
} from '../components/ui';
import { NextStep } from '../components/NextStep';
import { usePipeline } from '../store/pipeline';

/**
 * Training with live progress.
 *
 * Progress arrives over a WebSocket, with polling as a fallback. That redundancy is
 * deliberate: a dropped socket during a long run is the most likely source of a
 * confusing bug, and the polling route returns the identical snapshot, so the UI can
 * recover without the two paths ever disagreeing.
 */
export function TrainingStep() {
  const { markComplete } = usePipeline();

  const [available, setAvailable] = useState<string[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    api
      .availableModels()
      .then(({ models }) => {
        setAvailable(models);
        // Carry the Model Advisor's selection across, keeping only models valid here.
        try {
          const stored = sessionStorage.getItem('smartml-selected-models');
          const parsed: string[] = stored ? JSON.parse(stored) : [];
          const valid = parsed.filter((model) => models.includes(model));
          setSelected(new Set(valid.length ? valid : models));
        } catch {
          setSelected(new Set(models));
        }
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not list models.'));

    // Navigating away must not leave a poller running against a dead component.
    return () => stopWatching();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Close the socket and stop the poller, if either is running. */
  function stopWatching() {
    socketRef.current?.close();
    socketRef.current = null;
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  function watch(jobId: string) {
    // Tear down any previous watcher first. Without this, starting a second run
    // orphans the first interval, which then polls a finished job forever -- and every
    // further click adds another.
    stopWatching();

    // Poll regardless of the socket. It is cheap, and it means a silent socket failure
    // degrades to a slower update rather than a UI that never finishes.
    pollRef.current = window.setInterval(async () => {
      try {
        const snapshot = await api.jobStatus(jobId);
        setJob(snapshot);
        if (snapshot.status === 'completed' || snapshot.status === 'failed') finish(snapshot);
      } catch {
        /* transient — the next tick retries */
      }
    }, 1000);

    try {
      const socket = new WebSocket(trainingSocketUrl(jobId));
      socketRef.current = socket;
      socket.onmessage = (event) => {
        const snapshot: TrainingJob = JSON.parse(event.data);
        setJob(snapshot);
        if (snapshot.status === 'completed' || snapshot.status === 'failed') finish(snapshot);
      };
      socket.onerror = () => socket.close();
    } catch {
      /* the poller above is the fallback */
    }
  }

  function finish(snapshot: TrainingJob) {
    stopWatching();
    if (snapshot.status === 'completed') {
      void api
        .trainingResults()
        .then((results) => markComplete('training', results.completed_steps));
    }
  }

  async function start() {
    if (starting || job?.status === 'queued' || job?.status === 'running') return;
    setStarting(true);
    setError(null);
    try {
      stopWatching();
      const created = await api.startTraining([...selected]);
      setJob(created);
      watch(created.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start training.');
    } finally {
      setStarting(false);
    }
  }

  function toggle(model: string) {
    setSelected((current) => {
      const updated = new Set(current);
      if (updated.has(model)) updated.delete(model);
      else updated.add(model);
      return updated;
    });
  }

  if (!available.length && !error) {
    return (
      <Page title="Model Training">
        <Loading />
      </Page>
    );
  }

  const running = job?.status === 'queued' || job?.status === 'running';
  const succeeded = job?.events.filter((event) => event.status === 'completed') ?? [];
  const failed = job?.events.filter((event) => event.status === 'failed') ?? [];

  return (
    <Page
      title="Model Training"
      subtitle="Each model is fitted on the training features and scored on the held-out test set."
    >
      <div className="stack">
        {error && <Alert tone="danger">{error}</Alert>}

        {job?.status === 'completed' && (
          <NextStep from="training" note="Trained models are held for the rest of the pipeline." />
        )}

        <div className="split">
          <div className="stack">
            <Card title="Models to train" hint={`${selected.size} of ${available.length} selected`}>
              <div className="grid grid-3">
                {available.map((model) => (
                  <label key={model} className="row" style={{ cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      className="checkbox"
                      checked={selected.has(model)}
                      disabled={running}
                      onChange={() => toggle(model)}
                    />
                    <span className="small">{model}</span>
                  </label>
                ))}
              </div>
            </Card>

            {job && (
              <Card
                title="Progress"
                actions={
                  <Badge
                    tone={
                      job.status === 'completed'
                        ? 'success'
                        : job.status === 'failed'
                          ? 'danger'
                          : 'info'
                    }
                  >
                    {job.status}
                  </Badge>
                }
              >
                <div className="stack-sm">
                  <div className="row-between">
                    <span className="small secondary">
                      {job.completed} of {job.total} models
                    </span>
                    <span className="small num muted">{job.elapsed}s elapsed</span>
                  </div>
                  <Progress value={job.completed} total={job.total} />
                </div>

                {job.events.length > 0 && (
                  <div style={{ marginTop: 'var(--space-4)' }}>
                    <DataTable
                      maxHeight={300}
                      columns={[
                        { key: 'model', header: 'Model' },
                        {
                          key: 'status',
                          header: 'Status',
                          render: (row) => (
                            <Badge tone={row.status === 'completed' ? 'success' : 'danger'}>
                              {row.status}
                            </Badge>
                          ),
                        },
                        {
                          key: 'fit_time',
                          header: 'Fit (s)',
                          numeric: true,
                          render: (row) =>
                            row.fit_time !== undefined ? row.fit_time.toFixed(3) : '—',
                        },
                        {
                          key: 'predict_time',
                          header: 'Predict (s)',
                          numeric: true,
                          render: (row) =>
                            row.predict_time !== undefined ? row.predict_time.toFixed(3) : '—',
                        },
                      ]}
                      rows={job.events}
                    />
                  </div>
                )}

                {failed.length > 0 && (
                  <div style={{ marginTop: 'var(--space-4)' }}>
                    <Alert tone="warning">
                      {failed.length} model{failed.length === 1 ? '' : 's'} failed to train and
                      {failed.length === 1 ? ' was' : ' were'} skipped. The rest completed
                      normally.
                      <ul className="small" style={{ margin: 'var(--space-2) 0 0', paddingLeft: 18 }}>
                        {failed.map((event) => (
                          <li key={event.model}>
                            <span className="strong">{event.model}</span>: {event.error}
                          </li>
                        ))}
                      </ul>
                    </Alert>
                  </div>
                )}

                {job.status === 'failed' && job.error && (
                  <div style={{ marginTop: 'var(--space-4)' }}>
                    <Alert tone="danger">{job.error}</Alert>
                  </div>
                )}
              </Card>
            )}
          </div>

          <div className="stack">
            <Card title="Run">
              <Button
                variant="action"
                block
                loading={starting || running}
                disabled={selected.size === 0}
                onClick={start}
              >
                {running ? 'Training…' : job ? 'Train again' : 'Start training'}
              </Button>
              <p className="xs muted" style={{ marginTop: 'var(--space-3)' }}>
                Progress streams over a WebSocket, with polling as a fallback. A model that
                fails is recorded and skipped rather than aborting the batch.
              </p>
            </Card>

            {job?.status === 'completed' && (
              <Card title="Result">
                <div className="stack-sm">
                  <div className="row-between">
                    <span className="small secondary">Trained</span>
                    <Badge tone="success">{succeeded.length}</Badge>
                  </div>
                  <div className="row-between">
                    <span className="small secondary">Failed</span>
                    <Badge tone={failed.length ? 'warning' : 'neutral'}>{failed.length}</Badge>
                  </div>
                  <div className="row-between">
                    <span className="small secondary">Total time</span>
                    <span className="small num">{job.elapsed}s</span>
                  </div>
                </div>
              </Card>
            )}
          </div>
        </div>
      </div>
    </Page>
  );
}
