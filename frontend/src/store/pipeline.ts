import { create } from 'zustand';
import { api, type CompletedSteps, type PipelineState, type StepId } from '../api/client';

/**
 * Pipeline navigation and completion state.
 *
 * The step order below is the dependency chain the Streamlit app expressed only through
 * scattered "please complete X first" warnings. Making it explicit lets the rail lock
 * steps whose prerequisites are unmet and grey out everything downstream of a change,
 * which is the same invariant the server's `reset_downstream` cascade enforces.
 */

export interface StepDefinition {
  id: StepId;
  label: string;
  group: string;
  /** Steps that must be complete before this one can be opened. */
  requires: StepId[];
}

export const STEP_GROUPS = [
  'Dataset',
  'Explore',
  'Prepare',
  'Build',
  'Predict',
  'Explain',
  'Export',
] as const;

export const STEPS: StepDefinition[] = [
  { id: 'upload', label: 'Upload', group: 'Dataset', requires: [] },
  { id: 'analysis', label: 'Analysis', group: 'Dataset', requires: ['upload'] },
  { id: 'visualization', label: 'Visualization', group: 'Explore', requires: ['analysis'] },
  { id: 'preprocessing', label: 'Preprocessing', group: 'Prepare', requires: ['analysis'] },
  { id: 'features', label: 'Feature Engineering', group: 'Prepare', requires: ['preprocessing'] },
  { id: 'model-advisor', label: 'Model Advisor', group: 'Build', requires: ['preprocessing'] },
  { id: 'training', label: 'Model Training', group: 'Build', requires: ['model-advisor'] },
  { id: 'comparison', label: 'Model Comparison', group: 'Build', requires: ['training'] },
  { id: 'prediction', label: 'Prediction', group: 'Predict', requires: ['training'] },
  { id: 'explainability', label: 'Explainability', group: 'Explain', requires: ['training'] },
  { id: 'download', label: 'Download', group: 'Export', requires: ['training'] },
];

const EMPTY_COMPLETION: CompletedSteps = {
  upload: false,
  analysis: false,
  visualization: false,
  preprocessing: false,
  features: false,
  'model-advisor': false,
  training: false,
  comparison: false,
  prediction: false,
  explainability: false,
  download: false,
};

interface PipelineStore {
  currentStep: StepId;
  completed: CompletedSteps;
  meta: Partial<PipelineState>;
  autoAdvance: boolean;

  goTo: (step: StepId) => void;
  setCompleted: (completed: CompletedSteps) => void;
  /** Mark the current step done and move to the next unlocked step, if auto-advance is on. */
  completeAndAdvance: (step: StepId, completed?: CompletedSteps) => void;
  /**
   * Mark a step done without navigating.
   *
   * Used by the steps whose output is the thing the user came to read — predictions,
   * SHAP attributions, training timings, the engineered feature set. Advancing off those
   * automatically shows the result for a frame and then throws it away.
   */
  markComplete: (step: StepId, completed?: CompletedSteps) => void;
  /**
   * Mark a step done and always move on, regardless of the auto-advance setting.
   *
   * This backs the explicit "Continue" buttons. A control the user pressed that is
   * labelled Continue has to continue — letting the toggle suppress it would make the
   * button silently do nothing.
   */
  completeAndGo: (step: StepId, completed?: CompletedSteps) => void;
  setAutoAdvance: (value: boolean) => void;
  refreshState: () => Promise<void>;
  reset: () => void;
}

export function isUnlocked(step: StepDefinition, completed: CompletedSteps): boolean {
  return step.requires.every((required) => completed[required]);
}

export function nextStepAfter(step: StepId): StepId | null {
  const index = STEPS.findIndex((candidate) => candidate.id === step);
  if (index === -1 || index === STEPS.length - 1) return null;
  return STEPS[index + 1].id;
}

export const usePipeline = create<PipelineStore>((set, get) => ({
  currentStep: 'upload',
  completed: EMPTY_COMPLETION,
  meta: {},
  autoAdvance: true,

  goTo: (step) => set({ currentStep: step }),

  setCompleted: (completed) => set({ completed }),

  completeAndAdvance: (step, completed) => {
    const merged = completed ?? { ...get().completed, [step]: true };
    const next = nextStepAfter(step);
    const shouldMove = get().autoAdvance && next !== null;

    set({
      completed: merged,
      currentStep: shouldMove && next ? next : get().currentStep,
    });
  },

  markComplete: (step, completed) =>
    set({ completed: completed ?? { ...get().completed, [step]: true } }),

  completeAndGo: (step, completed) => {
    const next = nextStepAfter(step);
    set({
      completed: completed ?? { ...get().completed, [step]: true },
      currentStep: next ?? get().currentStep,
    });
  },

  setAutoAdvance: (value) => set({ autoAdvance: value }),

  refreshState: async () => {
    try {
      const state = await api.state();
      set({ completed: state.completed_steps, meta: state });
    } catch {
      // A fresh or expired session simply has nothing complete yet; the upload step
      // is always reachable, so there is nothing to report to the user here.
      set({ completed: EMPTY_COMPLETION, meta: {} });
    }
  },

  reset: () => set({ currentStep: 'upload', completed: EMPTY_COMPLETION, meta: {} }),
}));

/** The step that follows `step`, with its display label, or null at the end. */
export function nextStepMeta(step: StepId): StepDefinition | null {
  const index = STEPS.findIndex((candidate) => candidate.id === step);
  if (index === -1 || index === STEPS.length - 1) return null;
  return STEPS[index + 1];
}
