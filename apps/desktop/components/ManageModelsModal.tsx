import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import Modal from './Modal';
import { Trash2, CheckCircle2, Globe, Star, X, Check, FolderOpen, Eye, FileText, Search, Circle, Loader2, AlertCircle, Download, CloudOff, HardDrive, Pin, Cloud, MessagesSquare } from 'lucide-react';
import type { Model } from '../types';
import { revealPath } from '../backend';
import { formatModelName, getManageModelsRowDisplay } from '../utils/modelDisplay';

import recommendedIcon from '../assets/recommended.png';

const TruncatedModelName = ({ name, tooltip }: { name: string, tooltip: React.ReactNode }) => {
  const [isTruncated, setIsTruncated] = React.useState(false);
  const textRef = React.useRef<HTMLSpanElement>(null);

  const checkTruncation = () => {
    if (textRef.current) {
      setIsTruncated(textRef.current.scrollWidth > textRef.current.clientWidth);
    }
  };

  return (
    <div 
      className="group/icon relative flex min-w-0 shrink"
      onMouseEnter={checkTruncation}
    >
      <span ref={textRef} className="text-[15px] font-medium text-gray-900 truncate cursor-default">
        {name}
      </span>
      {isTruncated && tooltip}
    </div>
  );
};

interface ManageModelsModalProps {
  isOpen: boolean;
  onClose: () => void;
  models: Model[];
  focusModelId?: string | null;
  focusSearchText?: string;
  onDownload: (id: string, source: string, quantizationFile?: string) => void;
  onSelectQuantization: (id: string, quantizationFile: string) => void;
  onSelectModel?: (id: string) => void;
  onDelete: (id: string, quantizationFile?: string) => void;
  onCancel: (id: string) => void;
}

const ManageModelsModal: React.FC<ManageModelsModalProps> = ({ 
  isOpen, 
  onClose, 
  models,
  focusModelId = null,
  focusSearchText = '',
  onDownload: _onDownload,
  onSelectQuantization: _onSelectQuantization,
  onSelectModel: _onSelectModel,
  onDelete,
  onCancel
}) => {
  const { t } = useTranslation();
  // UI filter: recommended ⭐️ vs others
  const [modelGroup, setModelGroup] = useState<'recommended' | 'others'>('recommended');
  const [pendingDeleteModelId, setPendingDeleteModelId] = useState<string | null>(null);
  const [searchText, setSearchText] = useState('');

  const normalizeForSearch = (value: string): string =>
    String(value || '')
      .toLowerCase()
      .replace(/[\s\-_/\\.]+/g, '');

  // Qwen3-VL-2B is a required core model for indexing/image workflows.
  const isProtectedCoreModel = (model: Model): boolean => {
    const id = String(model?.id || '').toLowerCase();
    const name = String(model?.name || '').toLowerCase();
    return (
      id === 'qwen3-vl-2b-instruct-gguf' ||
      id.includes('qwen3-vl-2b') ||
      (id.includes('qwen3') && id.includes('vl') && id.includes('2b')) ||
      name.includes('qwen3-vl-2b')
    );
  };

  useEffect(() => {
    if (!isOpen) {
      setPendingDeleteModelId(null);
      setSearchText('');
      return;
    }

    const normalizedFocusId = normalizeForSearch(String(focusModelId || ''));
    const normalizedFocusText = normalizeForSearch(String(focusSearchText || ''));
    const target = focusModelId
      ? models.find((m) => m.id === focusModelId) ||
        models.find((m) => {
          const key = normalizeForSearch(`${String(m.id || '')} ${String(m.name || '')}`);
          return key.includes(normalizedFocusId) || normalizedFocusId.includes(key);
        }) ||
        models.find((m) => {
          const key = normalizeForSearch(`${String(m.id || '')} ${String(m.name || '')}`);
          return Boolean(normalizedFocusText) && key.includes(normalizedFocusText);
        })
      : null;
    if (target) {
      setModelGroup(target.recommended ? 'recommended' : 'others');
    }
    if (focusSearchText.trim()) {
      setSearchText(focusSearchText);
    }
  }, [isOpen, models, focusModelId, focusSearchText]);

  const handleRevealModelLocation = async (model: Model) => {
    const targetPath = String(model.selected_model_path || model.model_dir || '').trim();
    if (!targetPath) return;
    try {
      await revealPath(targetPath);
    } catch (e) {
      console.error('Reveal model location failed', e);
    }
  };

  const canProcessImage = (model: Model): boolean => {
    const id = String(model?.id || '').toLowerCase();
    const name = String(model?.name || '').toLowerCase();
    const hasMmproj = Array.isArray((model as any)?.files)
      ? (model as any).files.some((f: string) => String(f || '').toLowerCase().includes('mmproj'))
      : Boolean((model as any)?.selected_mmproj_path);
    return id.includes('-vl-') || name.includes('-vl-') || hasMmproj;
  };

  const isIndexingPreferredModel = (model: Model): boolean => {
    const id = String(model?.id || '').toLowerCase();
    const name = String(model?.name || '').toLowerCase();
    return (
      id === 'gemma-4-e4b-it-gguf' ||
      id.includes('gemma-4-e4b') ||
      name.includes('gemma-4-e4b') ||
      name.includes('gemma 4 e4b')
    );
  };

  const isChatPreferredModel = (model: Model): boolean => {
    return !isIndexingPreferredModel(model);
  };

  const getRecommendedTooltip = (model: Model): string => {
    if (isIndexingPreferredModel(model)) return t('modelsModal.recommendedForIndexing');
    return t('modelsModal.recommendedForChatting');
  };

  const renderMacTooltip = (text: string) => (
    <span className="pointer-events-none absolute left-1/2 top-full z-50 mt-2 -translate-x-1/2 whitespace-nowrap rounded-lg border border-black/10 bg-white/90 px-2.5 py-1 text-[11px] font-medium text-gray-700 shadow-[0_12px_30px_rgba(0,0,0,0.16)] backdrop-blur-md opacity-0 translate-y-1 transition-all duration-150 group-hover/icon:opacity-100 group-hover/icon:translate-y-0">
      {text}
    </span>
  );

  const guessProvider = (model: Model): string => {
    const repoIds = [
      String(model?.sources?.modelscope?.repo_id || '').trim(),
      String(model?.sources?.hf?.repo_id || '').trim(),
    ].filter(Boolean);
    const haystack = (
      repoIds.join(' ') +
      ' ' +
      String(model?.name || '') +
      ' ' +
      String(model?.id || '')
    ).toLowerCase();

    const keywordToProvider: Array<[string, string]> = [
      ['google', 'Google'],
      ['openai', 'OpenAI'],
      ['qwen', 'Qwen'],
      ['baidu', 'Baidu'],
      ['gemma', 'Google'],
      ['ernie', 'Baidu'],
      ['gpt', 'OpenAI'],
      ['ministral', 'Mistral'],
      ['mistral', 'Mistral'],
      ['deepseek', 'DeepSeek'],
      ['llama', 'Meta'],
      ['meta', 'Meta'],
      ['falcon', 'TII'],
    ];
    for (const [k, v] of keywordToProvider) {
      if (haystack.includes(k)) return v;
    }

    const firstRepo = repoIds[0] || '';
    if (firstRepo.includes('/')) {
      const owner = firstRepo.split('/')[0].trim();
      if (owner) return owner;
    }
    return t('modelsModal.providerUnknown');
  };

  const guessModelSize = (model: Model): string => {
    const explicit = String(model?.size || '').trim();
    if (explicit) return explicit;
    const text = `${String(model?.name || '')} ${String(model?.id || '')}`;
    const m = text.match(/(\d+(?:\.\d+)?)\s*[Bb]\b/);
    if (m && m[1]) return `${m[1]}B`;
    return '-';
  };

  const asPositiveNumber = (value: unknown): number => {
    const n = Number(value);
    return Number.isFinite(n) && n > 0 ? n : 0;
  };

  const formatBytesCompact = (bytes: number): string => {
    if (!(bytes > 0)) return '-';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let idx = 0;
    while (value >= 1024 && idx < units.length - 1) {
      value /= 1024;
      idx += 1;
    }
    const precision = value >= 100 ? 0 : value >= 10 ? 1 : 2;
    return `${value.toFixed(precision)} ${units[idx]}`;
  };

  const getCaseInsensitiveMapSize = (sizes: Record<string, number> | undefined, filename: string): number => {
    if (!sizes || !filename) return 0;
    const direct = asPositiveNumber(sizes[filename]);
    if (direct > 0) return direct;
    const filenameLower = filename.toLowerCase();
    for (const [k, v] of Object.entries(sizes)) {
      if (String(k || '').toLowerCase() === filenameLower) {
        return asPositiveNumber(v);
      }
    }
    return 0;
  };

  const getDownloadSizeLabel = (model: Model, quantizationFile?: string): string => {
    const sizes = (model.file_sizes || {}) as Record<string, number>;
    const selectedQ = String(
      quantizationFile
      || model.selected_quantization
      || model.default_quantization
      || model.quantizations?.[0]?.file
      || ''
    );

    let total = 0;
    let hasAny = false;

    // Main GGUF size: quantization metadata first, then file_sizes fallback.
    if (selectedQ) {
      const q = model.quantizations?.find((it) => String(it?.file || '') === selectedQ)
        || model.quantizations?.find((it) => String(it?.file || '').toLowerCase() === selectedQ.toLowerCase());
      const qSize = asPositiveNumber(q?.size_bytes) || getCaseInsensitiveMapSize(sizes, selectedQ);
      if (qSize > 0) {
        total += qSize;
        hasAny = true;
      }
    }

    // If it is a multimodal model, add mmproj size when available.
    const files = Array.isArray((model as any)?.files) ? ((model as any).files as string[]) : [];
    const mmprojFile = files.find((f) => String(f || '').toLowerCase().includes('mmproj'));
    if (mmprojFile) {
      const mmprojSize = getCaseInsensitiveMapSize(sizes, mmprojFile);
      if (mmprojSize > 0) {
        total += mmprojSize;
        hasAny = true;
      }
    }

    // While downloading, backend may provide aggregate total_bytes directly.
    const dynamicTotal = asPositiveNumber((model as any).total_bytes);
    if (!hasAny && dynamicTotal > 0) return formatBytesCompact(dynamicTotal);

    return hasAny ? formatBytesCompact(total) : '-';
  };

  const clampPercent = (value: unknown): number => {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(100, Math.round(n)));
  };

  const formatSpeed = (bytesPerSecond: unknown): string => {
    const n = Number(bytesPerSecond);
    if (!Number.isFinite(n) || n <= 0) return '';
    const units = ['B/s', 'KB/s', 'MB/s', 'GB/s'];
    let value = n;
    let idx = 0;
    while (value >= 1024 && idx < units.length - 1) {
      value /= 1024;
      idx += 1;
    }
    const precision = value >= 10 ? 1 : 2;
    return `${value.toFixed(precision)} ${units[idx]}`;
  };

  const filteredModels = models.filter((m) => {
    const rec = Boolean((m as any).recommended);
    const inGroup = modelGroup === 'recommended' ? rec : !rec;
    if (!inGroup) return false;

    const q = searchText.trim().toLowerCase();
    if (!q) return true;
    const haystack = [
      String(m.name || ''),
      String(m.id || ''),
      guessProvider(m),
      String(m.sources?.modelscope?.repo_id || ''),
      String(m.sources?.hf?.repo_id || ''),
      String(m.description || ''),
    ].join(' ');
    const normalizedHaystack = normalizeForSearch(haystack);
    const normalizedQuery = normalizeForSearch(q);
    return normalizedHaystack.includes(normalizedQuery);
  });

  const displayRows = filteredModels.flatMap((model) => {
    const canExpandByQuantization =
      Array.isArray(model.quantizations) &&
      model.quantizations.length > 1;

    if (!canExpandByQuantization) {
      return [{ key: `${model.id}::__single`, model, quantizationFile: undefined as string | undefined }];
    }

    return (model.quantizations || [])
      .filter((q) => String(q?.file || '').trim().length > 0)
      .map((q) => ({
        key: `${model.id}::${String(q.file)}`,
        model,
        quantizationFile: String(q.file),
      }));
  });

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={t('modelsModal.title')} width="w-[820px] max-w-[96vw]">
      <div className="flex flex-col h-[500px]">
        {/* Top Controls: Search & Filter */}
        <div className="px-8 py-3 border-b border-gray-100 bg-white flex flex-col gap-3 shrink-0">
          <div className="flex items-center justify-end">
            <div className="inline-flex items-center bg-gradient-to-b from-gray-50 to-gray-100 p-1 rounded-xl border border-gray-200 shadow-sm">
              <button
                onClick={() => setModelGroup('recommended')}
                className={`flex items-center gap-1.5 px-3 py-1 text-[11px] font-semibold rounded-lg transition-all ${
                  modelGroup === 'recommended' 
                    ? 'bg-blue-600 text-white shadow-sm shadow-blue-200/70'
                    : 'text-blue-700/80 hover:text-blue-800 hover:bg-blue-50'
                }`}
              >
                <Star
                  size={13}
                  className={modelGroup === 'recommended' ? 'text-white fill-white' : 'text-blue-500 fill-blue-200'}
                />
                {t('modelsModal.recommended')}
              </button>
              <button
                onClick={() => setModelGroup('others')}
                className={`flex items-center gap-1.5 px-3 py-1 text-[11px] font-semibold rounded-lg transition-all ${
                  modelGroup === 'others' 
                    ? 'bg-amber-500 text-white shadow-sm shadow-amber-200/70'
                    : 'text-amber-700/80 hover:text-amber-800 hover:bg-amber-50'
                }`}
              >
                <Globe size={13} className={modelGroup === 'others' ? 'text-white' : 'text-amber-500'} />
                {t('modelsModal.others')}
              </button>
            </div>
          </div>

          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              placeholder={t('modelsModal.filterPlaceholder')}
              className="w-full h-8 pl-9 pr-3 text-[12px] text-gray-700 rounded-lg border border-gray-200 bg-gray-50/50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-300 transition-colors shadow-sm"
            />
          </div>
        </div>

        {/* Models List */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden p-0 bg-white">
          <table className="w-full text-left border-collapse table-fixed">
            <thead className="bg-white sticky top-0 z-10">
              <tr>
                <th className="pl-8 pr-4 py-2.5 text-[10px] font-medium text-gray-400 uppercase tracking-widest border-b border-gray-100 bg-white" style={{ width: '46%' }}>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="inline-block w-6 h-6" aria-hidden />
                    <span>{t('modelsModal.model')}</span>
                  </span>
                </th>
                <th className="px-4 py-2.5 text-[10px] font-medium text-gray-400 uppercase tracking-widest border-b border-gray-100 bg-white text-center" style={{ width: '14%' }}>{t('modelsModal.provider')}</th>
                <th className="px-4 py-2.5 text-[10px] font-medium text-gray-400 uppercase tracking-widest border-b border-gray-100 bg-white text-center" style={{ width: '9%' }}>{t('modelsModal.size')}</th>
                <th className="px-4 py-2.5 text-[10px] font-medium text-gray-400 uppercase tracking-widest border-b border-gray-100 bg-white text-center" style={{ width: '13%' }}>{t('modelsModal.downloadSize')}</th>
                <th className="px-4 py-2.5 text-[10px] font-medium text-gray-400 uppercase tracking-widest border-b border-gray-100 bg-white text-center" style={{ width: '18%' }}>{t('modelsModal.action')}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {displayRows.map((row) => {
                const model = row.model;
                const rowQuant = String(row.quantizationFile || '').trim();
                const installedQuantizations = Array.isArray(model.installed_quantizations)
                  ? model.installed_quantizations.map(q => String(q || ''))
                  : [];
                const hasQuantRows = Array.isArray(model.quantizations) && model.quantizations.length > 1;
                const activeDownloadQuant = String(model.downloading_quantization_file || '').trim();
                const isDownloading = model.status === 'downloading' && (
                  !rowQuant || !activeDownloadQuant || activeDownloadQuant === rowQuant
                );
                const isInstalled = rowQuant
                  ? installedQuantizations.includes(rowQuant)
                  : (model.status === 'installed' && !hasQuantRows);
                const isError = model.status === 'error' && (
                  !rowQuant || !activeDownloadQuant || activeDownloadQuant === rowQuant
                );
                const isProtected = isProtectedCoreModel(model);
                const downloaded = isDownloading ? Number((model as any).downloaded_bytes ?? 0) : 0;
                const total = isDownloading ? Number((model as any).total_bytes ?? 0) : 0;
                const fallbackProgress = total > 0 ? (downloaded / total) * 100 : 0;
                const progress = isDownloading ? clampPercent(model.progress ?? model.downloadProgress ?? fallbackProgress) : 0;
                const speedLabel = isDownloading ? formatSpeed((model as any).download_speed ?? (model as any).speed) : '';

                const { baseName: rowBaseName, quantBadge: rowQuantBadge } = getManageModelsRowDisplay(
                  model,
                  rowQuant || undefined,
                );
                const rowTooltipText = [rowBaseName, rowQuantBadge].filter(Boolean).join(' ');

                const installStatusTitle = isDownloading
                  ? t('modelsModal.statusDownloading')
                  : isInstalled
                    ? t('modelsModal.statusInstalled')
                    : isError
                      ? t('modelsModal.statusError')
                      : t('modelsModal.statusNotInstalled');

                return (
                  <tr key={row.key} className="group hover:bg-gray-50/50 transition-colors">
                    <td className="pl-8 pr-4 py-3">
                      <div className="flex items-center gap-2 min-w-0">
                        {isDownloading ? (
                          <span title={installStatusTitle} className="inline-flex items-center justify-center text-blue-600 flex-shrink-0 w-6 h-6">
                            <Loader2 size={15} className="animate-spin" strokeWidth={2} />
                          </span>
                        ) : isInstalled ? (
                          <span title={installStatusTitle} className="inline-flex items-center justify-center text-blue-600 flex-shrink-0 w-6 h-6">
                            <Pin size={15} className="fill-current -rotate-45" />
                          </span>
                        ) : isError ? (
                          <span title={installStatusTitle} className="inline-flex items-center justify-center text-red-500 flex-shrink-0 w-6 h-6">
                            <AlertCircle size={15} strokeWidth={2} />
                          </span>
                        ) : (
                          <span title={installStatusTitle} className="inline-flex items-center justify-center text-gray-400 flex-shrink-0 w-6 h-6">
                            <Cloud size={15} strokeWidth={2} />
                          </span>
                        )}
                        <TruncatedModelName
                          name={rowBaseName}
                          tooltip={renderMacTooltip(rowTooltipText)}
                        />
                        {rowQuantBadge && (
                          <span className="text-[11px] font-medium text-gray-500 whitespace-nowrap">
                            {rowQuantBadge}
                          </span>
                        )}
                        {model.recommended && (
                          <span className="group/icon relative inline-flex items-center justify-center flex-shrink-0">
                            <img
                              src={recommendedIcon}
                              alt={t('modelsModal.recommended')}
                              className="w-5 h-5"
                            />
                            {renderMacTooltip(getRecommendedTooltip(model))}
                          </span>
                        )}
                        {isIndexingPreferredModel(model) && (
                          <span className="group/icon relative inline-flex items-center justify-center w-[18px] h-[18px] rounded border border-emerald-300 text-emerald-700 bg-emerald-100 flex-shrink-0">
                            <FileText size={10} className="absolute left-[1px] top-[1px]" />
                            <Search size={8} className="absolute right-[1px] bottom-[1px]" />
                            {renderMacTooltip(t('modelsModal.suitableForIndexing'))}
                          </span>
                        )}
                        {isChatPreferredModel(model) && (
                          <span className="group/icon relative inline-flex items-center justify-center w-[18px] h-[18px] rounded border border-sky-300 text-sky-700 bg-sky-100 flex-shrink-0">
                            <MessagesSquare size={11} />
                            {renderMacTooltip(t('modelsModal.suitableForChatting'))}
                          </span>
                        )}
                        <span className="group/icon relative inline-flex items-center justify-center w-[18px] h-[18px] rounded border border-violet-300 text-violet-700 bg-violet-100 flex-shrink-0">
                          <FileText size={11} />
                          {renderMacTooltip(t('modelsModal.textTag'))}
                        </span>
                        {canProcessImage(model) && (
                          <span className="group/icon relative inline-flex items-center justify-center w-[18px] h-[18px] rounded border border-amber-300 text-amber-700 bg-amber-100 flex-shrink-0">
                            <Eye size={11} />
                            {renderMacTooltip(t('modelsModal.visionTag'))}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 align-middle text-center">
                      <span className="text-[12px] text-gray-700 inline-block">
                        {guessProvider(model)}
                      </span>
                    </td>
                    <td className="px-4 py-3 align-middle text-center">
                      <span className="text-[12px] text-gray-700 inline-block">
                        {guessModelSize(model)}
                      </span>
                    </td>
                    <td className="px-4 py-3 align-middle text-center">
                      <span className="text-[12px] text-gray-700 inline-block">
                        {getDownloadSizeLabel(model, row.quantizationFile)}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center align-middle">
                      {isDownloading ? (
                        <div className="flex items-center justify-center gap-2">
                           <div className="flex flex-col items-end gap-1 min-w-[84px]">
                             <div className="text-[10px] leading-none whitespace-nowrap">
                               <span className="font-medium text-blue-600">{progress}%</span>
                               {speedLabel ? <span className="ml-1 text-gray-500">{speedLabel}</span> : null}
                             </div>
                             <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                               <div
                                 className="h-full bg-blue-500 transition-all duration-300"
                                 style={{ width: `${progress}%` }}
                               />
                             </div>
                           </div>
                           <button
                             onClick={() => onCancel(model.id)}
                             className="text-gray-400 hover:text-red-600 transition-colors p-0.5 rounded-full hover:bg-red-50"
                             title={t('modelsModal.cancelDownload')}
                           >
                             <X size={14} />
                           </button>
                        </div>
                      ) : (
                        <div className="flex items-center justify-center gap-2">
                          {!isInstalled && !isError ? (
                            <button
                              onClick={() => {
                                const quant = row.quantizationFile || model.selected_quantization || model.default_quantization || model.quantizations?.[0]?.file || undefined;
                                _onDownload(model.id, 'auto', quant);
                              }}
                              className="p-1 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-md transition-all border border-transparent hover:border-blue-100 flex-shrink-0"
                              title={t('modelsModal.downloadModel')}
                            >
                              <Download size={13} />
                            </button>
                           ) : pendingDeleteModelId === row.key ? (
                            <>
                              <button
                                onClick={() => {
                                  const quantFile = rowQuant || undefined;
                                  onDelete(model.id, quantFile);
                                  setPendingDeleteModelId(null);
                                }}
                                className="inline-flex items-center gap-1 px-1.5 py-0.5 text-[11px] font-medium rounded-md border text-red-700 border-red-200 hover:bg-red-50 whitespace-nowrap"
                                title={t('modelsModal.confirmDelete')}
                              >
                                <Check size={12} />
                                {t('modelsModal.confirm')}
                              </button>
                              <button
                                onClick={() => setPendingDeleteModelId(null)}
                                className="p-1 text-gray-400 hover:text-gray-700 hover:bg-gray-100 rounded-md transition-all border border-transparent hover:border-gray-200 flex-shrink-0"
                                title={t('modelsModal.cancel')}
                              >
                                <X size={12} />
                              </button>
                            </>
                          ) : (
                            <button
                              onClick={() => setPendingDeleteModelId(row.key)}
                              className="p-1 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-md transition-all border border-transparent hover:border-red-100 flex-shrink-0"
                              title={t('modelsModal.deleteModel')}
                            >
                              <Trash2 size={12} />
                            </button>
                          )}
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          
          {displayRows.length === 0 && (
             <div className="p-12 text-center text-gray-500 text-sm">
               {models.length === 0 ? t('modelsModal.noSupportedModels') : t('modelsModal.noMatch')}
             </div>
          )}
        </div>

        {/* Recommendation hint */}
        <div className="px-8 py-3 border-t border-gray-100 bg-gray-50/50 shrink-0">
          <p className="text-xs text-gray-400 text-center">
            {t('modelsModal.hint')}
          </p>
        </div>
      </div>
    </Modal>
  );
};

export default ManageModelsModal;
