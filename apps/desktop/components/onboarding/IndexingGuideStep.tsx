import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FileText, Plus } from 'lucide-react';

interface IndexingGuideStepProps {
  onAddSources: () => void;
  onAddFiles?: () => void;
  onSkip: () => void;
}

const IndexingGuideStep: React.FC<IndexingGuideStepProps> = ({ onAddSources, onAddFiles, onSkip }) => {
  const { t } = useTranslation();
  const [showAddMenu, setShowAddMenu] = useState(false);

  const handleAddClick = () => {
    if (!onAddFiles) {
      onAddSources();
      return;
    }
    setShowAddMenu((open) => !open);
  };

  return (
    <div className="flex flex-col items-center justify-center h-full w-full relative">
      <button
        onClick={onSkip}
        className="absolute top-8 right-8 font-sans font-light text-gray-400 hover:text-gray-600 transition-colors z-[320]"
        style={{ WebkitAppRegion: 'no-drag' } as any}
      >
        {t('onboarding.skip')}
      </button>
      <div className="flex flex-col items-center justify-center px-8 py-12">
        <div className="relative flex flex-col items-center">
          <p className="font-sans text-4xl font-light text-gray-900 mb-8 text-center leading-relaxed whitespace-nowrap">
            {t('onboarding.connectSourcesTitle')}
          </p>

          <div className="relative z-[110]" style={{ WebkitAppRegion: 'no-drag' } as any}>
            <button
              onClick={handleAddClick}
              className="px-8 py-3 rounded-lg bg-black text-white font-sans text-base font-light hover:bg-gray-800 transition-colors"
            >
              {t('onboarding.connectSourcesAdd')}
            </button>

            {showAddMenu && (
              <>
                <div className="absolute left-1/2 top-full mt-2 z-50 w-44 -translate-x-1/2">
                  <div className="bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden">
                    <button
                      type="button"
                      onClick={() => {
                        setShowAddMenu(false);
                        onAddSources();
                      }}
                      className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      <Plus size={14} />
                      <span>{t('sidebar.addFolder')}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setShowAddMenu(false);
                        onAddFiles?.();
                      }}
                      className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors border-t border-gray-100"
                    >
                      <FileText size={14} />
                      <span>{t('sidebar.addFile')}</span>
                    </button>
                  </div>
                </div>
                <div className="fixed inset-0 z-40" onClick={() => setShowAddMenu(false)} />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default IndexingGuideStep;
