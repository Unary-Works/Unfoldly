import React, { useState } from 'react';
import { FileText, Folder, Plus, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { FileSource } from '../types';
import { countDistinctSelectedFiles } from '../utils/sourceSelection';

interface SourceSelectorProps {
  sourcesLibrary: FileSource[];
  activeSourceIds: string[];
  onToggleSource: (id: string) => void;
  onRemoveSources: (ids: string[]) => void;
  onAddSources: () => void;
  onAddFiles?: () => void;
  isIndexing?: boolean;
  isModelSwitching?: boolean;
  onOpenSidebar: () => void;
}

const SourceSelector: React.FC<SourceSelectorProps> = ({
  sourcesLibrary,
  activeSourceIds,
  onAddSources,
  onAddFiles,
  isIndexing = false,
  isModelSwitching = false,
  onOpenSidebar
}) => {
  const { t } = useTranslation();
  const [showAddMenu, setShowAddMenu] = useState(false);
  const hasSources = sourcesLibrary.length > 0;
  const selectedCount = countDistinctSelectedFiles(sourcesLibrary, activeSourceIds);

  // State 2: Indexing
  if (isIndexing) {
    return (
      <button
        type="button"
        onClick={onOpenSidebar}
        disabled={isModelSwitching}
        className={`flex items-center gap-2 text-sm font-medium px-2 py-1 rounded transition-colors ${
          isModelSwitching ? 'text-gray-400 cursor-not-allowed' : 'text-gray-500 hover:text-gray-800 hover:bg-gray-100'
        }`}
        title={t('chat.indexing')}
      >
        <Loader2 size={16} className={`animate-spin ${isModelSwitching ? 'text-gray-400' : 'text-gray-500'}`} />
        <span>{t('chat.addSources')}</span>
      </button>
    );
  }

  // State 3: Sources Loaded
  if (hasSources) {
    return (
      <button
        type="button"
        onClick={onOpenSidebar}
        disabled={isModelSwitching}
        className={`flex items-center gap-2 text-sm font-medium px-2 py-1 rounded transition-colors group ${
          isModelSwitching ? 'text-gray-400 cursor-not-allowed' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
        }`}
      >
        <Folder size={16} className={isModelSwitching ? 'text-gray-400' : 'text-gray-500 group-hover:text-gray-700'} />
        <span>{t('chat.selectedCount', { count: selectedCount })}</span>
      </button>
    );
  }

  // State 1: No Sources (Empty)
  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => {
          if (isModelSwitching) return;
          if (!onAddFiles) {
            onAddSources();
            return;
          }
          setShowAddMenu((open) => !open);
        }}
        disabled={isModelSwitching}
        className={`flex items-center gap-2 text-sm font-medium px-2 py-1 rounded transition-colors ${
          isModelSwitching ? 'text-gray-400 cursor-not-allowed' : 'text-gray-500 hover:text-gray-800 hover:bg-gray-100'
        }`}
      >
        <Plus size={16} />
        <span>{t('chat.addSources')}</span>
      </button>

      {showAddMenu && (
        <>
          <div className="absolute left-0 top-full mt-1 z-50 w-44">
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
  );
};

export default SourceSelector;
