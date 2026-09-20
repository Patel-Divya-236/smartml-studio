/**
 * Typed API client.
 *
 * Every request carries the session id the backend issued, so the server can hold this
 * browser's pipeline state. The id lives in sessionStorage rather than localStorage:
 * server sessions are process-local and expire, so a *tab* is the right lifetime — a
 * second tab getting its own pipeline is correct, and a restored id pointing at a
 * long-dead session would only produce confusing 400s.
 */

const BASE = import.meta.env.VITE_API_BASE ?? '/api';
const SESSION_KEY = 'smartml-session-id';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

function readSessionId(): string | null {
  try {
    return sessionStorage.getItem(SESSION_KEY);
  } catch {
    return null;
  }
}

function writeSessionId(id: string): void {
  try {
    sessionStorage.setItem(SESSION_KEY, id);
  } catch {
    /* storage blocked — the in-memory copy below still serves this page load */
  }
}

let sessionId: string | null = readSessionId();
let sessionPromise: Promise<string> | null = null;

async function createSession(): Promise<string> {
  // Uses the long budget because this is usually the first call of the page load, and so
  // the one that pays for a cold start.
  const response = await attempt(`${BASE}/session`, { method: 'POST' }, LONG_TIMEOUT_MS);
  if (!response.ok) throw new ApiError('Could not start a session.', response.status);
  const data = await response.json();
  sessionId = data.session_id;
  writeSessionId(data.session_id);
  return data.session_id;
}

/** Return the current session id, creating one if needed. Concurrent calls share one request. */
export async function ensureSession(): Promise<string> {
  if (sessionId) return sessionId;
  if (!sessionPromise) {
    sessionPromise = createSession().finally(() => {
      sessionPromise = null;
    });
  }
  return sessionPromise;
}

/** Discard the current session so the next request starts a fresh pipeline. */
export function resetSession(): void {
  sessionId = null;
  try {
    sessionStorage.removeItem(SESSION_KEY);
  } catch {
    /* nothing to clear */
  }
}

/** Default budget. Generous, because the server may be cold-starting. */
const DEFAULT_TIMEOUT_MS = 30_000;
/** For preprocessing and training, which do real work before replying. */
export const LONG_TIMEOUT_MS = 120_000;

/**
 * Upload budget, derived from the bytes actually being sent.
 *
 * A fixed cap cannot work here: measured throughput to the deployed API was ~0.33 MB/s, so
 * a 30MB file legitimately needs over a minute on the wire. A flat 120s aborted uploads
 * that were progressing normally and would have finished.
 *
 * The floor assumes a pessimistic 100 KB/s so a slow connection is not punished, plus a
 * minute of headroom for a cold start and server-side parsing. The ceiling exists so a
 * genuinely dead connection still fails rather than hanging forever.
 */
const UPLOAD_FLOOR_BYTES_PER_SEC = 100_000;
const UPLOAD_MAX_TIMEOUT_MS = 15 * 60_000;

function uploadTimeoutFor(bytes: number): number {
  const transfer = (bytes / UPLOAD_FLOOR_BYTES_PER_SEC) * 1000;
  return Math.min(UPLOAD_MAX_TIMEOUT_MS, Math.max(LONG_TIMEOUT_MS, transfer + 60_000));
}

/** Status used for failures that never reached the server, so they read like any other. */
const NETWORK_ERROR = 0;
const RETRY_STATUSES = [NETWORK_ERROR, 502, 503, 504];
const RETRY_DELAYS_MS = [1_000, 3_000];

const UNREACHABLE_MESSAGE = 'Could not reach the server. It may be starting up — retrying…';

/**
 * Set when the server rejects our session id, i.e. it restarted and lost this pipeline.
 * Read once by the store so the UI can say what happened instead of silently blanking.
 */
let sessionLost = false;

/** Whether the session was dropped server-side since this was last called. Clears the flag. */
export function takeSessionLost(): boolean {
  const lost = sessionLost;
  sessionLost = false;
  return lost;
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  formData?: FormData;
  raw?: boolean;
  timeoutMs?: number;
  headers?: Record<string, string>;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * One attempt. Network-level failures become ApiError(status 0) so callers never see the
 * browser's bare "Failed to fetch", which tells the user nothing about what to do.
 */
async function attempt(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError(
        `The server did not respond within ${Math.round(timeoutMs / 1000)}s.`,
        NETWORK_ERROR,
      );
    }
    throw new ApiError(UNREACHABLE_MESSAGE, NETWORK_ERROR);
  } finally {
    clearTimeout(timer);
  }
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const id = await ensureSession();
  const headers: Record<string, string> = { 'X-Session-Id': id, ...options.headers };
  let body: BodyInit | undefined;

  if (options.formData) {
    body = options.formData;
  } else if (options.body !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(options.body);
  }

  const method = options.method ?? (body ? 'POST' : 'GET');
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  // Only GETs are safe to repeat blind. A POST may have been applied server-side before the
  // connection dropped, so retrying one risks running the same work twice.
  const maxAttempts = method === 'GET' ? RETRY_DELAYS_MS.length + 1 : 1;

  let lastError: ApiError | null = null;

  for (let index = 0; index < maxAttempts; index += 1) {
    if (index > 0) await sleep(RETRY_DELAYS_MS[index - 1]);

    let response: Response;
    try {
      response = await attempt(`${BASE}${path}`, { method, headers, body }, timeoutMs);
    } catch (error) {
      lastError = error as ApiError;
      continue; // transport failure — always worth another go within the attempt budget
    }

    if (response.status === 400) {
      // The session expired or the server restarted. Start a new one so the next action can
      // recover, and record it so the UI can explain rather than silently resetting.
      sessionLost = true;
      resetSession();
    }

    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const payload = await response.json();
        if (typeof payload?.detail === 'string') detail = payload.detail;
      } catch {
        /* non-JSON error body — keep the generic message */
      }
      const failure = new ApiError(detail, response.status);
      if (RETRY_STATUSES.includes(response.status)) {
        lastError = failure;
        continue; // the server is up but not ready; a 4xx is never retried
      }
      throw failure;
    }

    if (options.raw) return (await response.text()) as unknown as T;
    return (await response.json()) as T;
  }

  throw lastError ?? new ApiError(UNREACHABLE_MESSAGE, NETWORK_ERROR);
}

/**
 * Formats that are already compressed containers. Gzipping one costs CPU and saves nothing.
 */
const INCOMPRESSIBLE = /\.(xlsx|xls|gz|zip)$/i;

/** Below this, the round trip dominates and compression is not worth the wait. */
const MIN_COMPRESS_BYTES = 256 * 1024;

/**
 * Gzip a file before upload, or return null to send it as-is.
 *
 * Uploads are network-bound: measured throughput to the deployed API was ~0.33 MB/s, so a
 * 4MB CSV spent ~12s on the wire against well under a second of parsing. CSV compresses
 * about 3x, which is the difference between a wait and a pause.
 *
 * Returns null whenever compression is unavailable, unhelpful or fails — the caller then
 * sends the original bytes and omits the header, which every server version accepts.
 */
async function compressForUpload(file: File): Promise<Blob | null> {
  if (file.size < MIN_COMPRESS_BYTES || INCOMPRESSIBLE.test(file.name)) return null;
  if (typeof CompressionStream === 'undefined') return null;

  try {
    const stream = file.stream().pipeThrough(new CompressionStream('gzip'));
    const packed = await new Response(stream).blob();
    // A file that does not actually shrink is not worth the extra decompression step.
    return packed.size < file.size ? packed : null;
  } catch {
    return null;
  }
}

/** Wake a sleeping backend before the first real call, so the UI can say what it is waiting for. */
export async function warmUp(): Promise<void> {
  await attempt(`${BASE}/health`, { method: 'GET' }, LONG_TIMEOUT_MS);
}

/** Open a download in the browser, forwarding the session header via a blob fetch. */
export async function downloadFile(path: string, filename: string): Promise<void> {
  const id = await ensureSession();
  const response = await attempt(
    `${BASE}${path}`,
    { headers: { 'X-Session-Id': id } },
    LONG_TIMEOUT_MS,
  );
  if (!response.ok) throw new ApiError('Download failed.', response.status);

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

/** WebSocket URL for a training job, derived from the API base so it follows the proxy. */
export function trainingSocketUrl(jobId: string): string {
  const absolute = new URL(`${BASE}/training/jobs/${jobId}/progress`, window.location.href);
  absolute.protocol = absolute.protocol === 'https:' ? 'wss:' : 'ws:';
  return absolute.toString();
}

export const api = {
  state: () => request<PipelineState>('/pipeline/state'),

  uploadDataset: async (file: File) => {
    const form = new FormData();
    const packed = await compressForUpload(file);

    if (packed) {
      // The server reads the filename to choose the CSV or Excel parser, so it is kept
      // as-is rather than gaining a .gz suffix.
      form.append('file', packed, file.name);
    } else {
      form.append('file', file);
    }

    return request<UploadResult>('/datasets', {
      formData: form,
      timeoutMs: uploadTimeoutFor(packed ? packed.size : file.size),
      headers: packed ? { 'X-Upload-Encoding': 'gzip' } : undefined,
    });
  },
  columns: () => request<{ columns: ColumnSuggestion[] }>('/datasets/columns'),
  setTarget: (target_column: string, problem_type?: string) =>
    request<TargetResult>('/datasets/target', { body: { target_column, problem_type } }),
  profile: () => request<ProfileResult>('/datasets/profile'),
  distribution: (column: string) =>
    request<DistributionResult>(`/datasets/distribution/${encodeURIComponent(column)}`),

  columnPair: (x: string, y: string) =>
    request<ColumnPairResult>(
      `/datasets/xy?x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}`,
    ),

  visualizationAdvice: () => request<{ recommendations: Recommendation[] }>('/advisors/visualization'),
  preprocessingAdvice: () =>
    request<{ recommendations: Recommendation[]; by_column: Record<string, Recommendation[]> }>(
      '/advisors/preprocessing',
    ),
  modelAdvice: () => request<{ recommendations: Recommendation[] }>('/advisors/model'),
  llmStatus: () => request<{ available: boolean; error: string | null }>('/advisors/llm-status'),
  explain: (rec: Recommendation) =>
    request<NarrationResult>('/advisors/explain', {
      body: {
        label: rec.label,
        category: rec.category,
        reason: rec.reason,
        why_explanation: rec.why_explanation,
        confidence_score: rec.confidence_score,
        metadata: rec.metadata,
      },
    }),
  explainColumn: (column: string, recommendations: Recommendation[]) =>
    request<NarrationResult>('/advisors/explain-column', {
      body: {
        column,
        recommendations: recommendations.map((rec) => ({
          label: rec.label,
          category: rec.category,
          reason: rec.reason,
          why_explanation: rec.why_explanation,
          confidence_score: rec.confidence_score,
          metadata: rec.metadata,
        })),
      },
    }),

  // The pipeline stages below fit transformers over the whole training set, so they get the
  // long budget: the default would abort work that was progressing normally.
  preprocess: (payload: PreprocessPayload) =>
    request<PreprocessResult>('/pipeline/preprocess', { body: payload, timeoutMs: LONG_TIMEOUT_MS }),
  features: (payload: FeaturePayload) =>
    request<FeatureResult>('/pipeline/features', { body: payload, timeoutMs: LONG_TIMEOUT_MS }),

  availableModels: () => request<{ models: string[]; problem_type: string }>('/training/available'),
  startTraining: (models: string[]) => request<TrainingJob>('/training/jobs', { body: { models } }),
  jobStatus: (jobId: string) => request<TrainingJob>(`/training/jobs/${jobId}`),
  trainingResults: () => request<TrainingResults>('/training/results'),

  comparison: () => request<ComparisonResult>('/evaluation/comparison'),
  featureImportance: (model: string) =>
    request<{ available: boolean; importances: ImportanceItem[] }>(
      `/evaluation/feature-importance/${encodeURIComponent(model)}`,
    ),
  diagnostics: (model: string) =>
    request<DiagnosticsResult>(`/evaluation/diagnostics/${encodeURIComponent(model)}`),
  // SHAP fits its own surrogate model before it can answer.
  shapGlobal: (model: string) =>
    request<ShapGlobalResult>(`/evaluation/shap/${encodeURIComponent(model)}/global`, {
      timeoutMs: LONG_TIMEOUT_MS,
    }),
  shapLocal: (model: string, index: number) =>
    request<ShapLocalResult>(`/evaluation/shap/${encodeURIComponent(model)}/local/${index}`, {
      timeoutMs: LONG_TIMEOUT_MS,
    }),
  narrateGlobal: (model: string) =>
    request<NarrationResult>('/evaluation/shap/narrate-global', { body: { model } }),
  narrateLocal: (model: string, sample_index: number) =>
    request<NarrationResult>('/evaluation/shap/narrate-local', { body: { model, sample_index } }),

  // Fits the hybrid ensemble before predicting.
  predict: (payload: PredictPayload) =>
    request<PredictResult>('/predictions', { body: payload, timeoutMs: LONG_TIMEOUT_MS }),
  exportSummary: () => request<ExportSummary>('/artifacts/summary'),
  reportPreview: (narrate: boolean) =>
    request<ReportPreview>(`/artifacts/report/preview?narrate=${narrate}`),
};

/* ── Types ─────────────────────────────────────────────────────────── */

export type StepId =
  | 'upload'
  | 'analysis'
  | 'visualization'
  | 'preprocessing'
  | 'features'
  | 'model-advisor'
  | 'training'
  | 'comparison'
  | 'prediction'
  | 'explainability'
  | 'download';

export type CompletedSteps = Record<StepId, boolean>;

export interface PipelineState {
  completed_steps: CompletedSteps;
  dataset_name: string | null;
  target_column: string | null;
  problem_type: string | null;
  test_size: number | null;
  feature_count: number | null;
  trained_model_names: string[];
}

export interface UploadResult {
  name: string;
  rows: number;
  columns: number;
  column_names: string[];
  dtypes: Record<string, string>;
  /** Text columns the server read as numeric because nearly every value parsed as a number. */
  coerced_numeric_columns: string[];
  memory_mb: number;
  preview: Record<string, unknown>[];
  completed_steps: CompletedSteps;
}

export interface ColumnSuggestion {
  column: string;
  dtype: string;
  unique: number;
  missing_pct: number;
  problem_type: string;
  confidence: number;
  rationale: string;
}

export interface TargetResult {
  target_column: string;
  problem_type: string;
  detected_problem_type: string;
  detection_confidence: number;
  detection_rationale: string;
  warnings: Observation[];
  completed_steps: CompletedSteps;
}

export interface Observation {
  severity: 'info' | 'success' | 'warning' | 'danger';
  text: string;
}

export interface ProfileSummary {
  rows: number;
  columns: number;
  duplicates: number;
  duplicate_pct: number;
  missing_total: number;
  missing_pct: number;
  memory_mb: number;
  problem_type: string | null;
  target_column: string | null;
  n_classes: number;
}

export interface ProfileResult {
  profile: Record<string, any>;
  summary: ProfileSummary;
  observations: Observation[];
  completed_steps: CompletedSteps;
}

export interface DistributionResult {
  column: string;
  kind: 'categories' | 'histogram' | 'free_text' | 'empty';
  distinct?: number;
  rows?: number;
  avg_label_length?: number;
  truncated?: boolean;
  data: { name: string; count: number; share?: number; midpoint?: number }[];
}

export interface ColumnPairResult {
  x: string;
  y: string;
  kind: 'points' | 'series' | 'empty';
  rows: number;
  sampled: boolean;
  data: { x: number | string; y: number }[];
}

export interface Recommendation {
  label: string;
  confidence_score: number;
  star_rating: number;
  reason: string;
  why_explanation: string;
  category: string;
  metadata: Record<string, any>;
}

export interface NarrationResult {
  narrative: string | null;
  available: boolean;
  error: string | null;
}

export interface PreprocessPayload {
  test_size: number;
  impute: Record<string, string>;
  encode: Record<string, string>;
  scale: Record<string, string>;
}

export interface PreprocessResult {
  train_shape: [number, number];
  test_shape: [number, number];
  feature_names: string[];
  dropped_columns: string[];
  classes: string[] | null;
  preview: Record<string, unknown>[];
  completed_steps: CompletedSteps;
}

export interface FeaturePayload {
  low_variance_active: boolean;
  low_variance_threshold: number;
  poly_active: boolean;
  poly_degree: number;
  poly_interaction_only: boolean;
  pca_active: boolean;
  pca_components: number;
  select_k_best_active: boolean;
  select_k_best_k: number;
}

export interface FeatureResult {
  train_shape: [number, number];
  test_shape: [number, number];
  feature_names: string[];
  features_before: number;
  features_after: number;
  preview: Record<string, unknown>[];
  completed_steps: CompletedSteps;
}

export interface TrainingEvent {
  model: string;
  status: 'completed' | 'failed';
  fit_time?: number;
  predict_time?: number;
  error?: string;
  completed: number;
  total: number;
}

export interface TrainingJob {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  completed: number;
  total: number;
  events: TrainingEvent[];
  error: string | null;
  elapsed: number;
}

export interface TrainingResults {
  models: { name: string; fit_time: number; predict_time: number; has_probabilities: boolean }[];
  failures: { name: string; error: string }[];
  completed_steps: CompletedSteps;
}

export interface ComparisonResult {
  columns: string[];
  rows: Record<string, number | string | null>[];
  primary_metric: string;
  best_model: string | null;
  problem_type: string;
  completed_steps: CompletedSteps;
}

export interface ImportanceItem {
  feature: string;
  importance: number;
}

export interface DiagnosticsResult {
  problem_type: string;
  labels?: string[];
  confusion_matrix?: number[][];
  roc?: { fpr: number[]; tpr: number[] } | null;
  actual_vs_predicted?: { actual: number[]; predicted: number[] };
  residuals?: { predicted: number[]; residual: number[] };
}

export interface ShapGlobalResult {
  available: boolean;
  explainer_type?: string;
  output_space?: string;
  is_subset?: boolean;
  importances: { feature: string; value: number; share: number }[];
}

export interface ShapLocalResult {
  available: boolean;
  output_space?: string;
  base_value: number | null;
  predicted: string | number | null;
  actual: string | number | null;
  max_index: number;
  contributions: { feature: string; value: number; contribution: number }[];
}

export interface PredictPayload {
  mode: 'single' | 'ensemble';
  model?: string;
  ensemble_models?: string[];
  voting?: string;
  weights?: number[] | null;
}

export interface PredictResult {
  strategy: string;
  count: number;
  accuracy: number | null;
  preview: Record<string, unknown>[];
  completed_steps: CompletedSteps;
}

export interface ReportPreview {
  markdown: string;
  narrated: boolean;
  llm_error: string | null;
}

export interface ExportSummary {
  dataset_name: string | null;
  target_column: string | null;
  problem_type: string | null;
  models: string[];
  has_predictions: boolean;
  has_comparison: boolean;
  prediction_strategy: string | null;
}
