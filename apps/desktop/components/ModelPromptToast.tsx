import React from 'react';
import { Box } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface ModelPromptToastProps {
  onDismiss: () => void;
  onBrowseModels: () => void;
}

const ModelPromptToast: React.FC<ModelPromptToastProps> = ({ onDismiss, onBrowseModels }) => {
  const { t } = useTranslation();

  return (
    <div className="fixed bottom-6 left-[280px] z-[200] animate-in slide-in-from-bottom-4 fade-in duration-300">
      <div className="bg-white border border-gray-200 rounded-xl shadow-lg px-5 py-4 max-w-sm">
        <div className="flex items-start gap-3">
          <Box size={18} className="text-gray-400 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-gray-900 mb-1">{t('modelPromptToast.title')}</p>
            <p className="text-xs text-gray-500 leading-relaxed mb-3">
              {t('modelPromptToast.description')}
            </p>
            <div className="flex items-center gap-2">
              <button
                onClick={onBrowseModels}
                className="px-3 py-1.5 text-xs font-medium bg-black text-white rounded-lg hover:bg-gray-800 transition-colors"
              >
                {t('modelPromptToast.browseModels')}
              </button>
              <button
                onClick={onDismiss}
                className="px-3 py-1.5 text-xs font-medium text-gray-500 hover:text-gray-700 transition-colors"
              >
                {t('modelPromptToast.later')}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelPromptToast;
