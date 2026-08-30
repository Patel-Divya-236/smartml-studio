import { Shell } from './components/Shell';
import { AnalysisStep } from './steps/AnalysisStep';
import { ComparisonStep } from './steps/ComparisonStep';
import { DownloadStep } from './steps/DownloadStep';
import { ExplainabilityStep } from './steps/ExplainabilityStep';
import { FeaturesStep } from './steps/FeaturesStep';
import { ModelAdvisorStep } from './steps/ModelAdvisorStep';
import { PredictionStep } from './steps/PredictionStep';
import { PreprocessingStep } from './steps/PreprocessingStep';
import { TrainingStep } from './steps/TrainingStep';
import { UploadStep } from './steps/UploadStep';
import { VisualizationStep } from './steps/VisualizationStep';
import type { ReactElement } from 'react';
import { usePipeline } from './store/pipeline';
import type { StepId } from './api/client';

const SCREENS: Record<StepId, () => ReactElement> = {
  upload: UploadStep,
  analysis: AnalysisStep,
  visualization: VisualizationStep,
  preprocessing: PreprocessingStep,
  features: FeaturesStep,
  'model-advisor': ModelAdvisorStep,
  training: TrainingStep,
  comparison: ComparisonStep,
  prediction: PredictionStep,
  explainability: ExplainabilityStep,
  download: DownloadStep,
};

export default function App() {
  const currentStep = usePipeline((state) => state.currentStep);
  const Screen = SCREENS[currentStep];

  // Keying on the step id remounts the screen on navigation, so each module loads its
  // own data fresh rather than showing the previous step's state while fetching.
  return (
    <Shell>
      <Screen key={currentStep} />
    </Shell>
  );
}
