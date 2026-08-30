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
  const response = await fetch(`${BASE}/session`, { method: 'POST' });
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

interface RequestOptions {
  method?: string;
  body?: unknown;
  formData?: FormData;
  raw?: boolean;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const id = await ensureSession();
  const headers: Record<string, string> = { 'X-Session-Id': id };
  let body: BodyInit | undefined;

  if (options.formData) {
    body = options.formData;
  } else if (options.body !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(options.body);
  }

  const response = await fetch(`${BASE}${path}`, {
    method: options.method ?? (body ? 'POST' : 'GET'),
    headers,
    body,
  });

  if (response.status === 400) {
    // The session expired server-side. Start a new one so the next action can recover,
    // rather than leaving the tab permanently broken.
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
    throw new ApiError(detail, response.status);
  }

  if (options.raw) return (await response.text()) as unknown as T;
  return (await response.json()) as T;
}

/** Open a download in the browser, forwarding the session header via a blob fetch. */
export async function downloadFile(path: string, filename: string): Promise<void> {
  const id = await ensureSession();
  const response = await fetch(`${BASE}${path}`, { headers: { 'X-Session-Id': id } });
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

  uploadDataset: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<UploadResult>('/datasets', { formData: form });
  },
  columns: () => request<{ columns: ColumnSuggestion[] }>('/datasets/columns'),
  setTarget: (target_column: string, problem_type?: string) =>
    request<TargetResult>('/datasets/target', { body: { target_column, problem_type } }),
  profile: () => request<ProfileResult>('/datasets/profile'),
  distribution: (column: string) =>
    request<DistributionResult>(`/datasets/distribution/${encodeURIComponent(column)}`),

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

  preprocess: (payload: PreprocessPayload) =>
    request<PreprocessResult>('/pipeline/preprocess', { body: payload }),
  features: (payload: FeaturePayload) =>
    request<FeatureResult>('/pipeline/features', { body: payload }),

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
  shapGlobal: (model: string) =>
    request<ShapGlobalResult>(`/evaluation/shap/${encodeURIComponent(model)}/global`),
  shapLocal: (model: string, index: number) =>
    request<ShapLocalResult>(`/evaluation/shap/${encodeURIComponent(model)}/local/${index}`),
  narrateGlobal: (model: string) =>
    request<NarrationResult>('/evaluation/shap/narrate-global', { body: { model } }),
  narrateLocal: (model: string, sample_index: number) =>
    request<NarrationResult>('/evaluation/shap/narrate-local', { body: { model, sample_index } }),

  predict: (payload: PredictPayload) => request<PredictResult>('/predictions', { body: payload }),
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
