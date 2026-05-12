import React from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, X } from 'lucide-react';
import { IndexingState } from '../types';

interface IndexingWidgetProps {
  state: IndexingState;
  onClose?: () => void;
}

function formatStageDetail(state: IndexingState, t: (key: string, options?: any) => string): string {
  const stage = String(state.stage || '').trim();
  if (stage === 'analyzing_frames' || (stage === 'transcribing_audio' && (state.totalAudioSec ?? 0) <= 0)) {
    return t('indexingWidget.analyzingFrames', {
      defaultValue: `Analyzing frames...`,
    });
  }
  if (stage === 'transcribing_audio') {
    return t('indexingWidget.transcribingAudio', {
      defaultValue: `Transcribing audio...`,
    });
  }
  return '';
}

function calculateDetailedPercentage(state: IndexingState): number {
  if (!state.totalFiles || state.totalFiles <= 0) return 0;
  
  let percentage = (state.completedFiles / state.totalFiles) * 100;
  
  if (state.isIndexing && !state.isCancelling && !state.isRestoringModel) {
    const fileShare = 100 / state.totalFiles;
    
    if (state.stage === 'transcribing_audio' && state.totalAudioSec && state.totalAudioSec > 0) {
      const audioProgress = Math.min((state.currentAudioSec || 0) / state.totalAudioSec, 1);
      percentage += fileShare * 0.1 * audioProgress;
    } else if (state.stage === 'analyzing_frames' && state.totalFrames && state.totalFrames > 0) {
      const frameProgress = Math.min((state.currentFrame || 0) / state.totalFrames, 1);
      percentage += fileShare * 0.1 + (fileShare * 0.9 * frameProgress);
    }
  }
  
  return Math.min(Math.round(percentage) || 0, 100);
}

type TopIndexingWidgetProps = IndexingWidgetProps & {
  isBackendSyncing?: boolean;
};

export const TopIndexingWidget: React.FC<TopIndexingWidgetProps> = ({
  state,
  onClose,
  isBackendSyncing = false,
}) => {
  const { t } = useTranslation();
  const showIndexingCard = Boolean(state.isTopBarVisible);
  const showCard = Boolean(isBackendSyncing || showIndexingCard);
  if (!showCard) return null;

  if (isBackendSyncing) {
    return (
      <div className="w-full max-w-4xl mx-auto px-4 mb-4 mt-2">
        <div className="bg-white border border-[#E5E5E5] rounded-xl p-4 shadow-sm relative min-h-[118px]">
          <div
            className="absolute top-4 left-4 w-6 h-6 flex items-center justify-center text-gray-500"
            aria-hidden
          >
            <Loader2 size={18} className="animate-spin" />
          </div>
          <div className="pl-10">
            <p className="text-sm text-gray-700 font-medium leading-relaxed mb-2">
              {t('indexingWidget.syncingMessage')}
            </p>
            <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden mb-3">
              <div className="h-full w-[12%] rounded-full bg-gray-800 animate-pulse" />
            </div>
            <div className="flex justify-between items-center text-xs text-gray-500 font-medium">
              <span>{t('indexingWidget.preparing')}</span>
              <span />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // If totalFiles is 0, it means we are still preparing the list
  const hasKnownTotal = Boolean(state.totalFiles && state.totalFiles > 0);
  const isCancelling = Boolean(state.isCancelling);
  const isRestoringModel = Boolean(state.isRestoringModel);
  const statusMessage = String(state.statusMessage || '').trim();
  const hasStatusMessage = statusMessage.length > 0;
  const stageDetail = formatStageDetail(state, t);
  const isPreparing = state.isIndexing && state.completedFiles === 0 && !hasKnownTotal && state.stage !== 'analyzing_frames' && state.stage !== 'transcribing_audio';
  const isAnalyzingFrames = state.stage === 'analyzing_frames' || (state.stage === 'transcribing_audio' && (state.totalAudioSec ?? 0) <= 0);
  const percentage = hasKnownTotal ? calculateDetailedPercentage(state) : 0;

  const isCompleted = !state.isIndexing && !isRestoringModel && state.completedFiles > 0 && state.completedFiles === state.totalFiles;
  const isCancelled = !state.isIndexing && !isRestoringModel && !isCompleted && !isCancelling;

  return (
    <div className="w-full max-w-4xl mx-auto mb-4 mt-2">
      <div className="bg-white border border-[#E5E5E5] rounded-xl p-4 shadow-sm relative min-h-[118px]">
        {onClose ? (
          <button
            type="button"
            onClick={isCancelling || isRestoringModel ? undefined : onClose}
            disabled={isCancelling || isRestoringModel}
            className={`absolute top-4 left-4 transition-colors w-6 h-6 flex items-center justify-center rounded border ${
              (isCancelling || isRestoringModel)
                ? 'text-gray-300 border-gray-100 cursor-not-allowed bg-gray-50'
                : 'text-gray-400 hover:text-gray-600 border-gray-200 hover:bg-gray-50'
            }`}
            title={t('indexingWidget.cancelIndexing')}
          >
            <X size={14} />
          </button>
        ) : (
          <div className="absolute top-4 left-4 w-6 h-6" aria-hidden />
        )}

        <div className="pl-10">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-sm text-gray-700 font-medium min-h-[20px]">
              {hasStatusMessage ? (
                <span>{statusMessage}</span>
              ) : state.currentFile ? (
                <span>{state.currentFile}</span>
              ) : isPreparing ? (
                <span>{t('indexingWidget.indexing')}</span>
              ) : isCancelling ? (
                <span>{state.statusMessage || t('indexingWidget.cancelling')}</span>
              ) : isCancelled ? (
                <span>{t('indexingWidget.stopped')}</span>
              ) : isCompleted ? (
                <span className="text-green-600">{t('indexingWidget.allIndexed')}</span>
              ) : null}
            </div>
            {!isCancelled && !isCompleted && !isCancelling && hasKnownTotal ? (
              <span className="text-sm font-semibold text-gray-900 ml-auto">{percentage}%</span>
            ) : !isCancelled && !isCompleted && !isCancelling && !isRestoringModel && !hasKnownTotal && !isAnalyzingFrames ? (
              <span className="text-sm font-semibold text-gray-400 ml-auto">—</span>
            ) : null}
          </div>

          <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden mb-3">
            <div
              className={`h-full rounded-full transition-all duration-300 ease-out ${isCompleted ? 'bg-green-500' : isCancelled ? 'bg-gray-400' : (isCancelling || isRestoringModel) ? 'bg-gray-500 animate-pulse' : 'bg-gray-800'}`}
              style={{ width: `${(isCancelling || isRestoringModel) ? 35 : hasKnownTotal ? percentage : isPreparing ? 12 : isAnalyzingFrames ? 12 : percentage}%` }}
            />
          </div>

          <div className="flex justify-between items-center text-xs text-gray-500 font-medium">
            <span className="truncate pr-2">
              {isCancelling
                ? (state.statusMessage || t('indexingWidget.cancelling'))
                : isRestoringModel
                ? (state.statusMessage || t('indexingWidget.switchingModelWait', { defaultValue: 'Switching model...' }))
                : (
                  <>
                    <span>
                      {hasKnownTotal
                        ? t('indexingWidget.filesProgress', { completed: state.completedFiles, total: state.totalFiles })
                        : isPreparing
                          ? t('indexingWidget.preparingFileList')
                          : t('indexingWidget.countingFiles')}
                    </span>
                    {stageDetail && (
                      <span className="ml-2 text-gray-400">
                        - {stageDetail}{t('indexingWidget.mediaIndexingWarning')}
                      </span>
                    )}
                  </>
                )}
            </span>
            <span className="shrink-0">{isCompleted || isCancelled || isCancelling || isRestoringModel ? '' : t('indexingWidget.estimatedLeft', { eta: state.eta || '—' })}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export const SidebarIndexingWidget: React.FC<IndexingWidgetProps> = ({ state }) => {
  const { t } = useTranslation();
  if (!state.isIndexing) return null;
  if (!state.totalFiles || state.totalFiles <= 0) return null;

  const percentage = calculateDetailedPercentage(state);
  const stageDetail = formatStageDetail(state, t);

  return (
    <div className="border-t border-gray-200 bg-gray-50/50 p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">{t('indexingWidget.indexingFiles')}</span>
        <span className="text-xs font-semibold text-gray-700">{percentage}%</span>
      </div>

      <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden mb-2">
        <div 
          className="h-full bg-gray-800 rounded-full transition-all duration-300 ease-out"
          style={{ width: `${percentage}%` }}
        />
      </div>

      <div className="flex justify-between items-center text-[10px] text-gray-500 font-medium">
        <span className="truncate max-w-[140px]" title={state.currentFile}>{state.currentFile || t('indexingWidget.loading')}</span>
        <span className="text-gray-400 shrink-0">{stageDetail || `${state.completedFiles}/${state.totalFiles}`}</span>
      </div>
    </div>
  );
};
