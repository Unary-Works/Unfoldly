import React from 'react';
import { useTranslation } from 'react-i18next';

interface IndexingProgressStepProps {
  progress: number; // 0-100
  completedFiles?: number;
  totalFiles?: number;
  eta?: string;
  onSkip?: () => void;
  onCancel?: () => void;
}

const IndexingProgressStep: React.FC<IndexingProgressStepProps> = ({ progress, completedFiles, totalFiles, eta, onSkip, onCancel }) => {
  const { t } = useTranslation();
  const hasCounts = typeof completedFiles === 'number' && typeof totalFiles === 'number' && totalFiles > 0;
  const etaText = (eta && String(eta).trim()) || '—';

  return (
    <div className="flex flex-col items-center justify-center h-full w-full relative">
      {onCancel && (
        <button
          onClick={onCancel}
          className="absolute top-8 left-8 text-gray-400 hover:text-gray-600 transition-colors z-[110] text-xl leading-none"
          title={t('indexingWidget.cancelIndexing')}
          style={{ WebkitAppRegion: 'no-drag' } as any}
        >
          ×
        </button>
      )}
      {onSkip && (
        <button
          onClick={onSkip}
          className="absolute top-8 right-8 font-serif text-gray-400 hover:text-gray-600 transition-colors z-[320]"
          style={{ WebkitAppRegion: 'no-drag' } as any}
        >
          {t('onboarding.skip')}
        </button>
      )}
      <div className="relative flex flex-col items-center">
        <h2 className="font-serif text-2xl text-gray-900 mb-8 w-80 text-center">{t('indexingWidget.indexing')}</h2>

        <div className="relative flex items-center w-80">
          <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
            <div className="h-full bg-black transition-all duration-300 ease-out" style={{ width: `${progress}%` }} />
          </div>
          <span className="absolute left-full ml-4 font-serif text-lg text-gray-900 whitespace-nowrap">
            {Math.round(progress)}%
          </span>
        </div>

        <div className="w-80 mt-4 flex justify-between items-center text-xs text-gray-500 font-medium">
          <span>{hasCounts ? t('indexingWidget.filesProgress', { completed: completedFiles, total: totalFiles }) : t('indexingWidget.countingFiles')}</span>
          <span>{t('indexingWidget.estimatedLeft', { eta: etaText })}</span>
        </div>
      </div>
    </div>
  );
};

export default IndexingProgressStep;
