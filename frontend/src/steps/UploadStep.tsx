import { useRef, useState } from 'react';
import {
  api,
  type ColumnSuggestion,
  type Observation,
  type UploadResult,
} from '../api/client';
import {
  Alert,
  Badge,
  Button,
  Card,
  DataTable,
  EmptyState,
  Page,
  Select,
  Stat,
} from '../components/ui';
import { usePipeline } from '../store/pipeline';

const PROBLEM_TYPES = ['Classification', 'Regression', 'Time Series'];

export function UploadStep() {
  const { completeAndAdvance, refreshState } = usePipeline();
  const inputRef = useRef<HTMLInputElement>(null);

  const [dataset, setDataset] = useState<UploadResult | null>(null);
  const [columns, setColumns] = useState<ColumnSuggestion[]>([]);
  const [target, setTarget] = useState('');
  const [problemType, setProblemType] = useState('');
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<Observation[]>([]);

  async function handleFile(file: File) {
    setBusy(true);
    setError(null);
    try {
      const result = await api.uploadDataset(file);
      setDataset(result);
      const suggestions = await api.columns();
      setColumns(suggestions.columns);

      // Pre-select the last column: overwhelmingly the convention for tabular targets,
      // and the user can change it before confirming.
      const last = suggestions.columns[suggestions.columns.length - 1];
      if (last) {
        setTarget(last.column);
        setProblemType(last.problem_type);
      }
      await refreshState();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed.');
      setDataset(null);
    } finally {
      setBusy(false);
    }
  }

  async function confirmTarget() {
    setBusy(true);
    setError(null);
    try {
      const result = await api.setTarget(target, problemType);
      setWarnings(result.warnings ?? []);
      await refreshState();

      // A serious problem with the target — an identifier or free text — is worth
      // stopping on. The target is still set, so continuing is one more click, but
      // walking into a meaningless training run unwarned is not the default.
      const blocking = (result.warnings ?? []).some((w) => w.severity === 'danger');
      if (!blocking) completeAndAdvance('upload', result.completed_steps);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not set the target column.');
    } finally {
      setBusy(false);
    }
  }

  const selected = columns.find((column) => column.column === target);

  return (
    <Page
      title="Dataset Upload"
      subtitle="Upload a tabular dataset to begin. SmartML Studio profiles it, detects the task type, and recommends every step from here on."
    >
      <div className="stack">
        {error && <Alert tone="danger">{error}</Alert>}

        <div
          className="dropzone"
          data-dragging={dragging}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const file = event.dataTransfer.files?.[0];
            if (file) void handleFile(file);
          }}
        >
          <div className="dropzone-icon" aria-hidden="true">↑</div>
          <div className="strong">Drag and drop your file here</div>
          <span className="small muted">or</span>
          <Button variant="primary" loading={busy} onClick={() => inputRef.current?.click()}>
            Choose file
          </Button>
          <span className="xs muted">CSV, XLSX, XLS · up to 200MB</span>
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void handleFile(file);
              event.target.value = '';
            }}
          />
        </div>

        {!dataset && (
          <Alert tone="info">
            Once uploaded, the dataset is profiled automatically and the task type is
            detected. You confirm the target column before anything else runs.
          </Alert>
        )}

        {dataset && (
          <>
            <div className="grid grid-4">
              <Stat value={dataset.rows.toLocaleString()} label="Rows" />
              <Stat value={dataset.columns} label="Columns" />
              <Stat value={`${dataset.memory_mb} MB`} label="In memory" sub="Estimated" />
              <Stat
                value={dataset.name.length > 18 ? `${dataset.name.slice(0, 17)}…` : dataset.name}
                label="File"
                sub="Uploaded just now"
              />
            </div>

            <Card
              title="Target column"
              hint="The column to predict. Everything downstream — task type, advisors, metrics — follows from this choice."
            >
              <div className="grid grid-2" style={{ alignItems: 'end' }}>
                <div className="field">
                  <label className="label">Column</label>
                  <Select
                    value={target}
                    options={columns.map((column) => column.column)}
                    onChange={(value) => {
                      setTarget(value);
                      setWarnings([]);
                      const match = columns.find((column) => column.column === value);
                      if (match) setProblemType(match.problem_type);
                    }}
                  />
                </div>
                <div className="field">
                  <label className="label">Task type</label>
                  <Select value={problemType} options={PROBLEM_TYPES} onChange={setProblemType} />
                </div>
              </div>

              {selected && (
                <div className="row" style={{ marginTop: 'var(--space-4)', flexWrap: 'wrap' }}>
                  <Badge tone="primary">detected: {selected.problem_type}</Badge>
                  <Badge tone="neutral">{selected.unique.toLocaleString()} unique</Badge>
                  <Badge tone={selected.missing_pct > 0 ? 'warning' : 'neutral'}>
                    {selected.missing_pct}% missing
                  </Badge>
                  <span className="small muted">{selected.rationale}</span>
                </div>
              )}

              {warnings.length > 0 && (
                <div className="stack-sm" style={{ marginTop: 'var(--space-4)' }}>
                  {warnings.map((warning, index) => (
                    <Alert key={index} tone={warning.severity === 'danger' ? 'danger' : 'warning'}>
                      {warning.text}
                    </Alert>
                  ))}
                </div>
              )}

              <div className="row" style={{ marginTop: 'var(--space-5)' }}>
                <Button variant="action" loading={busy} disabled={!target} onClick={confirmTarget}>
                  {warnings.some((w) => w.severity === 'danger')
                    ? 'Choose a different column'
                    : 'Confirm and analyse'}
                </Button>
                {warnings.some((w) => w.severity === 'danger') && (
                  <Button variant="ghost" onClick={() => completeAndAdvance('upload')}>
                    Continue anyway
                  </Button>
                )}
                {problemType && selected && problemType !== selected.problem_type && (
                  <span className="small muted">
                    Overriding the detected task type ({selected.problem_type}).
                  </span>
                )}
              </div>
            </Card>

            <Card title="Preview" hint={`First ${dataset.preview.length} rows`} className="card-flush">
              <DataTable
                maxHeight={340}
                columns={dataset.column_names.map((name) => ({
                  key: name,
                  header: name,
                  numeric: /int|float/.test(dataset.dtypes[name] ?? ''),
                }))}
                rows={dataset.preview}
              />
            </Card>
          </>
        )}

        {!dataset && !busy && (
          <EmptyState
            title="No dataset loaded"
            description="Everything on the other screens unlocks once a dataset is uploaded and a target column is confirmed."
          />
        )}
      </div>
    </Page>
  );
}
