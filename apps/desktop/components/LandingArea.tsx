import React from 'react';
import { useTranslation } from 'react-i18next';
import { ChevronDown, PanelRightOpen } from 'lucide-react';
import InputArea from './InputArea';
import { TopIndexingWidget } from './IndexingWidget';
import { Model, FileSource, IndexingState } from '../types';
import { formatModelName, getModelQuantBadge } from '../utils/modelDisplay';
import recommendedIcon from '../assets/recommended.png';

interface LandingAreaProps {
  selectedModel: Model | null;
  models: Model[];
  isModelStateReady?: boolean;
  onSelectModel: (model: Model) => void;
  inputValue: string;
  onInputChange: (val: string) => void;
  onSend: () => void;
  sourcesLibrary: FileSource[];
  activeSourceIds: string[];
  onToggleSource: (id: string) => void;
  onRemoveSources: (ids: string[]) => void;
  onAddSources: () => void;
  onAddFiles?: () => void;
  indexingState: IndexingState;
  onCloseIndexingTopBar: () => void;
  isBackendSyncing?: boolean;
  isRightSidebarOpen: boolean;
  onToggleRightSidebar: () => void;
  isGenerating?: boolean;
  onStopGenerating?: () => void;
  isModelSwitching?: boolean;
  onOpenManageModels?: () => void;
}

const LandingArea: React.FC<LandingAreaProps> = ({
  selectedModel,
  models,
  isModelStateReady = true,
  onSelectModel,
  inputValue,
  onInputChange,
  onSend,
  sourcesLibrary,
  activeSourceIds,
  onToggleSource,
  onRemoveSources,
  onAddSources,
  onAddFiles,
  indexingState,
  onCloseIndexingTopBar,
  isBackendSyncing = false,
  isRightSidebarOpen,
  onToggleRightSidebar,
  isGenerating = false,
  onStopGenerating,
  isModelSwitching = false,
  onOpenManageModels,
}) => {
  const { t } = useTranslation();
  const [showModelDropdown, setShowModelDropdown] = React.useState(false);
  const modelDropdownWidth = React.useMemo(() => {
    const longest = models.reduce((max, model) => {
      const badge = getModelQuantBadge(model);
      const n = formatModelName(model.name).length + (badge ? badge.length + 2 : 0);
      return Math.max(max, n);
    }, 0);
    const estimated = longest * 7 + 76;
    return Math.max(200, Math.min(300, estimated));
  }, [models]);
  const shouldShowRecommendedInChat = (m: Model): boolean => {
    const id = String(m?.id || '').toLowerCase();
    return Boolean(m?.recommended) && !id.includes('qwen3-vl');
  };
  const displayModelName = (m: Model) => {
    const quantBadge = getModelQuantBadge(m);
    return (
      <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
        <span className="whitespace-nowrap">{formatModelName(m.name)}</span>
        {quantBadge && (
          <span className="text-[11px] font-medium text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">{quantBadge}</span>
        )}
        {shouldShowRecommendedInChat(m) && (
          <img src={recommendedIcon} alt={t('modelsModal.recommended')} className="w-[24px] h-[24px] flex-shrink-0" />
        )}
      </span>
    );
  };

  return (
    <div className="flex-1 flex flex-col h-full relative bg-white">
      {/* Header controls (no visible top strip on New Chat) */}
      <div className="h-14 flex items-center justify-between px-6 z-[90]">
        <div className="relative z-[100]" style={{ WebkitAppRegion: 'no-drag' } as any}>
          <button
            onClick={() => !isGenerating && !indexingState.isIndexing && isModelStateReady && setShowModelDropdown(!showModelDropdown)}
            disabled={isGenerating || indexingState.isIndexing || !isModelStateReady}
            className={`flex items-center gap-2 text-sm font-medium transition-colors ${
              (isGenerating || indexingState.isIndexing || !isModelStateReady)
                ? 'text-gray-400 cursor-not-allowed'
                : 'text-gray-600 hover:text-gray-900'
            }`}
            title={
              isGenerating
                ? t('chat.generatingNoSwitch')
                : indexingState.isIndexing
                  ? t('chat.indexingNoSwitch')
                  : !isModelStateReady
                    ? t('chat.loadingData')
                  : t('chat.switchModel')
            }
          >
            <span>
              {selectedModel
                ? displayModelName(selectedModel)
                : (isModelStateReady ? t('landing.noModel') : t('chat.loadingData'))}
            </span>
            <ChevronDown size={14} className="text-gray-400" />
          </button>

          {showModelDropdown && (
            <div
              className="absolute top-full left-0 mt-1 bg-white border border-gray-100 shadow-lg rounded-lg py-1 z-30 animate-in fade-in zoom-in-95 duration-100"
              style={{ width: `${modelDropdownWidth}px`, maxWidth: '80vw' }}
            >
              {models.map(model => {
                const quantBadge = getModelQuantBadge(model);
                const isSelected = selectedModel?.id === model.id;
                return (
                  <button
                    key={model.id}
                    onClick={() => {
                      onSelectModel(model);
                      setShowModelDropdown(false);
                    }}
                    className={`w-full text-left px-4 py-2 text-sm hover:bg-gray-50
                      ${isSelected ? 'text-gray-900 font-medium bg-gray-50' : 'text-gray-600'}
                    `}
                  >
                    <span className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2">
                      <span className="flex items-center gap-1.5 min-w-0">
                        <span className="whitespace-nowrap truncate" title={formatModelName(model.name)}>
                          {formatModelName(model.name)}
                        </span>
                        {quantBadge && (
                          <span className="text-[11px] font-medium text-gray-500 bg-gray-100 px-1.5 py-0.5 rounded whitespace-nowrap flex-shrink-0">{quantBadge}</span>
                        )}
                      </span>
                      <span className="w-[24px] h-[24px] flex items-center justify-center">
                        {shouldShowRecommendedInChat(model) ? (
                          <img src={recommendedIcon} alt={t('modelsModal.recommended')} className="w-[24px] h-[24px] flex-shrink-0" />
                        ) : null}
                      </span>
                    </span>
                  </button>
                );
              })}
              {onOpenManageModels && (
                <div className="border-t border-gray-100 mt-1 pt-1">
                  <button
                    onClick={() => {
                      setShowModelDropdown(false);
                      onOpenManageModels();
                    }}
                    className="w-full text-left px-4 py-2 text-sm text-gray-500 hover:bg-gray-50 hover:text-gray-700"
                  >
                    + {t('landing.addModel')}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {!isRightSidebarOpen && (
          <button
            onClick={onToggleRightSidebar}
            className="text-gray-400 hover:text-gray-600 p-1.5 rounded-md hover:bg-gray-100 transition-colors relative z-[100]"
            title={t('landing.openSources')}
            style={{ WebkitAppRegion: 'no-drag' } as any}
          >
            <PanelRightOpen size={20} />
          </button>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col justify-center items-center px-8 w-full max-w-4xl mx-auto -mt-[15vh]">
        
        {/* Indexing Widget */}
        <div className="w-full mb-6">
          <TopIndexingWidget
            state={indexingState}
            onClose={onCloseIndexingTopBar}
            isBackendSyncing={isBackendSyncing}
          />
        </div>

        <h1 className="hero-heading text-4xl text-gray-800 text-center mb-6 leading-tight whitespace-pre-line">
          {t('landing.heroTitle')}
        </h1>
        
          <InputArea 
            variant="centered"
            value={inputValue}
            onChange={onInputChange}
            onSend={onSend}
            sourcesLibrary={sourcesLibrary}
            activeSourceIds={activeSourceIds}
            onToggleSource={onToggleSource}
            onRemoveSources={onRemoveSources}
            onAddSources={onAddSources}
            onAddFiles={onAddFiles}
            isIndexing={indexingState.isIndexing} 
            onOpenSidebar={onToggleRightSidebar}
            isGenerating={isGenerating}
            onStopGenerating={onStopGenerating}
            isModelSwitching={isModelSwitching}
          />
      </div>
    </div>
  );
};

export default LandingArea;
