import React, { useRef, useEffect, useCallback } from 'react';
import { ArrowUp, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import SourceSelector from './SourceSelector';
import { FileSource } from '../types';

interface InputAreaProps {
  variant: 'centered' | 'docked';
  value: string;
  onChange: (val: string) => void;
  onSend: () => void;
  sourcesLibrary: FileSource[];
  activeSourceIds: string[];
  onToggleSource: (id: string) => void;
  onRemoveSources: (ids: string[]) => void;
  onAddSources: () => void;
  onAddFiles?: () => void;
  isIndexing?: boolean;
  onOpenSidebar: () => void;
  isGenerating?: boolean;
  onStopGenerating?: () => void;
  isModelSwitching?: boolean;
}

const InputArea: React.FC<InputAreaProps> = ({
  variant,
  value,
  onChange,
  onSend,
  sourcesLibrary,
  activeSourceIds,
  onToggleSource,
  onRemoveSources,
  onAddSources,
  onAddFiles,
  isIndexing = false,
  onOpenSidebar,
  isGenerating = false,
  onStopGenerating,
  isModelSwitching = false
}) => {
  const { t } = useTranslation();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isComposingRef = useRef(false);

  const isCentered = variant === 'centered';

  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;

    el.style.height = 'auto';

    const minPx = isCentered ? 0 : 56;
    const next = Math.max(el.scrollHeight || 0, minPx);
    el.style.height = `${next}px`;
  }, [isCentered]);

  // Auto-resize textarea (value change)
  useEffect(() => {
    resizeTextarea();
  }, [value, resizeTextarea]);

  // Mount-time resize after desktop shell fonts/layout stabilize.
  useEffect(() => {
    const id = requestAnimationFrame(() => resizeTextarea());
    return () => cancelAnimationFrame(id);
  }, [resizeTextarea]);

  // Global shortcut to focus input (Cmd/Ctrl + K)
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        textareaRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    const nativeEvt = e.nativeEvent as unknown as { isComposing?: boolean; keyCode?: number };
    const isComposing = Boolean(isComposingRef.current || nativeEvt?.isComposing || nativeEvt?.keyCode === 229);
    if (isComposing) return;
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim()) onSend();
    }
  };

  return (
    <div 
      className={`
        relative bg-white border border-gray-200 transition-all duration-300
        ${isCentered 
          ? 'w-full max-w-2xl rounded-2xl shadow-sm hover:shadow-md h-40 flex flex-col' 
          : 'w-full max-w-3xl rounded-xl shadow-sm mx-auto flex flex-col min-h-[96px]'}
      `}
    >
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        onCompositionStart={() => {
          isComposingRef.current = true;
        }}
        onCompositionEnd={() => {
          isComposingRef.current = false;
        }}
        placeholder={isIndexing ? t('chat.indexing') : (isCentered ? t('chat.askQuestion') : t('chat.sendMessage'))}
        disabled={isIndexing}
        className={`
          w-full bg-transparent resize-none outline-none text-gray-800 placeholder-gray-400
          ${isCentered ? 'p-5 text-lg h-full' : 'py-3 px-4 text-base max-h-48 overflow-y-auto min-h-[56px]'}
        `}
        rows={1}
      />

      <div className={`flex items-center justify-between ${isCentered ? 'p-4 mt-auto' : 'px-2 pb-2'}`}>
        {/* Source Selector */}
        <SourceSelector 
          sourcesLibrary={sourcesLibrary}
          activeSourceIds={activeSourceIds}
          onToggleSource={onToggleSource}
          onRemoveSources={onRemoveSources}
          onAddSources={onAddSources}
          onAddFiles={onAddFiles}
          isIndexing={isIndexing}
          isModelSwitching={isModelSwitching}
          onOpenSidebar={onOpenSidebar}
        />

        {/* Send / Stop Button */}
        <button
          onClick={isGenerating ? onStopGenerating : onSend}
          disabled={isIndexing || isModelSwitching || (!isGenerating && !value.trim())}
          title={isGenerating ? t('chat.stopGenerating') : t('chat.sendMessageTitle')}
          className={`
            flex items-center justify-center rounded-lg transition-all duration-200
            ${isGenerating 
              ? 'bg-gray-900 text-white shadow-sm hover:bg-gray-700 cursor-pointer' 
              : (value.trim() && !isModelSwitching && !isIndexing)
                ? 'bg-gray-900 text-white shadow-sm hover:bg-gray-700' 
                : 'bg-transparent text-gray-300 cursor-not-allowed'}
            ${isCentered ? 'w-8 h-8' : 'w-8 h-8'}
          `}
        >
          {isGenerating ? (
            <Loader2 size={18} className="animate-spin" />
          ) : (
            <ArrowUp size={18} />
          )}
        </button>
      </div>
    </div>
  );
};

export default InputArea;
