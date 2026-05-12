import React, { useEffect, useState } from 'react';
import { OnboardingStep } from '../types';
import WelcomeStep from './onboarding/WelcomeStep';
import ModelRecommendStep from './onboarding/ModelRecommendStep';
import SetupStep, { DownloadItemInfo } from './onboarding/SetupStep';
import IndexingGuideStep from './onboarding/IndexingGuideStep';
import IndexingProgressStep from './onboarding/IndexingProgressStep';
import logoIcon from '../assets/logo.png';

interface OnboardingProps {
  currentStep: OnboardingStep;
  setupProgress: number;
  setupItems?: DownloadItemInfo[];
  indexingProgress: number;
  indexingCompletedFiles?: number;
  indexingTotalFiles?: number;
  indexingEta?: string;
  onNext: () => void;
  onDownloadModels: () => void;
  onSkipModels: () => void;
  onSkip: () => void;
  onAddSources: () => void;
  onAddFiles?: () => void;
  onCancelIndexing?: () => void;
}

const Onboarding: React.FC<OnboardingProps> = ({
  currentStep,
  setupProgress,
  setupItems,
  indexingProgress,
  indexingCompletedFiles,
  indexingTotalFiles,
  indexingEta,
  onNext,
  onDownloadModels,
  onSkipModels,
  onSkip,
  onAddSources,
  onAddFiles,
  onCancelIndexing,
}) => {
  const [isFadingOut, setIsFadingOut] = useState(false);
  const [displayStep, setDisplayStep] = useState<OnboardingStep>(currentStep);

  // Handle completion with fade out - this should take priority
  useEffect(() => {
    if (currentStep === 'complete' && displayStep !== 'complete') {
      setIsFadingOut(true);
      const timer = setTimeout(() => {
        setDisplayStep('complete');
      }, 400);
      return () => clearTimeout(timer);
    }
  }, [currentStep, displayStep]);

  // Handle step changes with fade out effect (only for non-complete steps)
  useEffect(() => {
    if (currentStep !== displayStep && currentStep !== 'complete') {
      setIsFadingOut(true);
      const timer = setTimeout(() => {
        setDisplayStep(currentStep);
        setIsFadingOut(false);
      }, 250);
      return () => clearTimeout(timer);
    }
  }, [currentStep, displayStep]);

  if (currentStep === 'complete' && displayStep === 'complete') {
    return null;
  }

  return (
    <div className="fixed inset-0 z-[300] bg-white flex items-center justify-center">
      <div
        className="fixed top-0 left-0 right-0 h-7 z-[310]"
        style={{
          WebkitAppRegion: 'drag',
          pointerEvents: 'auto'
        } as any}
      />

      <div
        className={`w-full h-full flex items-center justify-center transition-opacity duration-[250ms] ${
          isFadingOut ? 'opacity-0' : 'opacity-100'
        }`}
      >
        {displayStep === 'welcome' && <WelcomeStep onNext={onNext} />}
        {displayStep === 'model-recommend' && <ModelRecommendStep onDownload={onDownloadModels} onSkip={onSkipModels} />}
        {displayStep === 'setup' && <SetupStep progress={setupProgress} items={setupItems} />}
        {displayStep === 'loading-models' && (
          <div className="flex flex-col items-center justify-center max-w-md text-center px-6">
            <img src={logoIcon} alt="Unfoldly" className="w-24 h-24 mb-4 opacity-80" />
            <div className="text-sm text-gray-600 font-medium">正在准备本地模型…</div>
            <p className="text-xs text-gray-400 mt-3 leading-relaxed">
              后台加载不会阻塞界面；如需退出，请使用系统红绿灯关闭窗口。
            </p>
          </div>
        )}
        {displayStep === 'indexing-guide' && <IndexingGuideStep onAddSources={onAddSources} onAddFiles={onAddFiles} onSkip={onSkip} />}
        {displayStep === 'indexing-progress' && (
          <IndexingProgressStep
            progress={indexingProgress}
            completedFiles={indexingCompletedFiles}
            totalFiles={indexingTotalFiles}
            eta={indexingEta}
            onSkip={onSkip}
            onCancel={onCancelIndexing}
          />
        )}
      </div>
    </div>
  );
};

export default Onboarding;
