import React, { useState, useRef, useEffect, useMemo, useCallback, memo } from 'react';
import { ChevronsRight, Plus, MoreVertical, Trash2, ChevronRight, ChevronDown, Loader2, X, FileText, RefreshCw, Trash } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { FileSource, IndexingState, RelevantFile, SidebarMode, OpenedFile, Model } from '../types';
import { FileIcon } from './Icon';
import { SidebarIndexingWidget } from './IndexingWidget';
import { formatModelName } from '../utils/modelDisplay';
import { countDistinctSelectedFiles } from '../utils/sourceSelection';
import recommendedIcon from '../assets/recommended.png';

interface RightSidebarProps {
  sources: FileSource[];
  activeSourceIds: string[];
  setActiveSourceIds: React.Dispatch<React.SetStateAction<string[]>>;
  onSelectAll: (select: boolean) => void;
  onRemoveSource: (id: string) => void;
  onRemoveAllSources?: () => void;
  onSkipFile?: (filePaths: string[]) => void;
  onRefreshSource?: (folderPath: string) => void;
  refreshingFolder?: string | null;
  onAddSources: () => void;
  onAddFiles?: () => void;
  indexingState: IndexingState;
  onClose: () => void;
  mode: SidebarMode;
  onSwitchMode: (mode: SidebarMode) => void;
  openedFiles: OpenedFile[];
  activeOpenedFilePath: string | null;
  onSelectOpenedFile: (filePath: string) => void;
  onClearOpenedFiles: () => void;
  // Index model selector
  selectedIndexModel: Model | null;
  installedModels: Model[];
  isModelSwitching?: boolean;
  onSelectIndexModel: (model: Model) => void;
  onOpenManageModels: () => void;
  isRemovingSources?: boolean;
  removeProgress?: { current: number; total: number } | null;
}

const getSelectionState = (node: FileSource, activeSet: Set<string>): 'checked' | 'unchecked' | 'indeterminate' => {
  if (!node.children || node.children.length === 0) {
    return activeSet.has(node.id) ? 'checked' : 'unchecked';
  }

  const childStates = node.children.map(child => getSelectionState(child, activeSet));
  const allChecked = childStates.every(s => s === 'checked');
  const allUnchecked = childStates.every(s => s === 'unchecked');

  if (allChecked) return 'checked';
  if (allUnchecked) return 'unchecked';
  return 'indeterminate';
};

// Helper to get all child IDs recursively
const getChildIds = (node: FileSource): string[] => {
  let ids = [node.id];
  if (node.children) {
    node.children.forEach(child => {
      ids = [...ids, ...getChildIds(child)];
    });
  }
  return ids;
};

const getAggregateStatus = (node: FileSource): FileSource['status'] => {
  if (!node.children || node.children.length === 0) {
    return node.status;
  }

  const childStatuses = node.children.map(getAggregateStatus);
  if (childStatuses.length > 0 && childStatuses.every(status => status === 'indexed')) {
    return 'indexed';
  }
  if (childStatuses.some(status => status === 'indexing')) {
    return 'indexing';
  }
  if (childStatuses.some(status => status === 'pending')) {
    return 'pending';
  }
  return node.status;
};

// --- Recursive Tree Node Component ---
const FileTreeNode: React.FC<{
  node: FileSource;
  activeIdSet: Set<string>;
  onToggle: (ids: string[], isSelected: boolean) => void;
  onRemove: (id: string) => void;
  onSkipFile?: (filePaths: string[]) => void;
  onRefreshSource?: (folderPath: string) => void;
  refreshingFolder?: string | null;
  level?: number;
  currentFile?: string;
  currentPath?: string;
  isGloballyIndexing?: boolean;
}> = memo(({ node, activeIdSet, onToggle, onRemove, onSkipFile, onRefreshSource, refreshingFolder, level = 0, currentFile, currentPath, isGloballyIndexing }) => {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [skipConfirmOpen, setSkipConfirmOpen] = useState(false);
  const [removeConfirm, setRemoveConfirm] = useState(false);
  const isRefreshingThis = refreshingFolder === node.path;
  
  const selectionState = getSelectionState(node, activeIdSet);
  const isChecked = selectionState === 'checked';
  const isIndeterminate = selectionState === 'indeterminate';
  
  const checkboxRef = useRef<HTMLInputElement>(null);
  
  useEffect(() => {
    if (checkboxRef.current) {
      checkboxRef.current.indeterminate = isIndeterminate;
    }
  }, [isIndeterminate]);

  const handleToggle = () => {
    const idsToToggle = getChildIds(node);
    // If currently checked or indeterminate, we uncheck all. If unchecked, we check all.
    const shouldSelect = selectionState === 'unchecked';
    onToggle(idsToToggle, shouldSelect);
  };

  const hasChildren = node.children && node.children.length > 0;
  
  // Dynamically compute display status based on backend status and active indexing job
  let displayStatus = getAggregateStatus(node); // 'indexed', 'indexing', 'pending'
  
  // If the backend marked this node as indexing or pending, we refine it with real-time info
  if (isGloballyIndexing && (displayStatus === 'indexing' || displayStatus === 'pending')) {
    if (node.type === 'file') {
      // It's a file
      if ((currentPath && node.path === currentPath) || (currentFile && node.name === currentFile)) {
        displayStatus = 'indexing'; // Spinner
      } else {
        // If it's not the currently indexing file, keep its original status (which was calculated by backend)
        // or force it to 'pending' if it was generically marked 'indexing'
        if (displayStatus === 'indexing') {
          displayStatus = 'pending'; 
        }
      }
    } else {
      // It's a folder, check if currentPath is inside this folder
      if (displayStatus !== 'indexed' && currentPath && (currentPath === node.path || currentPath.startsWith(node.path + '/') || currentPath.startsWith(node.path + '\\'))) {
        displayStatus = 'indexing';
      } else {
        if (displayStatus === 'indexing') {
          displayStatus = 'pending';
        }
      }
    }
  }

  const isIndexing = displayStatus === 'indexing';

  return (
    <div>
    <div 
      className={`group flex items-center py-1 pr-2 rounded-md hover:bg-gray-100/50 transition-colors ${node.type === 'folder' ? 'cursor-pointer' : ''}`}
      style={{ paddingLeft: `${level * 16 + 8}px` }}
      onClick={(e) => {
        if (node.type === 'folder' && e.target === e.currentTarget) {
          setExpanded(!expanded);
        }
      }}
    >
      {/* Expand/Collapse Chevron */}
      <div 
        className="w-5 flex-shrink-0 flex items-center justify-center"
        onClick={(e) => {
          if (node.type === 'folder') {
            e.stopPropagation();
            setExpanded(!expanded);
          }
        }}
      >
        {node.type === 'folder' && (
          <button className="text-gray-400 hover:text-gray-600">
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
        )}
      </div>

        {/* Icon / Menu Trigger */}
        <div className="relative w-6 h-6 flex-shrink-0 flex items-center justify-center mr-1">
          {/* Only show menu if NOT indexing */}
          {!isIndexing && (
             <div 
                className={`absolute inset-0 z-10 flex items-center justify-center cursor-pointer text-gray-500 hover:text-gray-800 transition-opacity ${menuOpen ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}
                onClick={(e) => {
                  e.stopPropagation();
                  setMenuOpen(!menuOpen);
                }}
             >
               <MoreVertical size={14} />
             </div>
          )}

          <div className={`flex items-center justify-center transition-opacity ${menuOpen ? 'opacity-0' : 'opacity-100 group-hover:opacity-0'}`}>
            <FileIcon type={node.iconType} className="w-4 h-4" />
          </div>

          {menuOpen && (
            <>
              <div 
                className="fixed inset-0 z-40" 
                onClick={(e) => {
                  e.stopPropagation();
                  setMenuOpen(false);
                  setRemoveConfirm(false);
                }}
              />
              <div className="absolute top-5 left-0 w-36 bg-white border border-gray-200 shadow-lg rounded-md p-1 z-50">
                {/* Refresh — for folders and files */}
                {onRefreshSource && node.path && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setMenuOpen(false);
                      onRefreshSource(node.path!);
                    }}
                    className="flex items-center gap-2 w-full text-left px-2 py-1 text-xs rounded text-blue-600 hover:bg-blue-50 transition-colors"
                  >
                    {isRefreshingThis
                      ? <Loader2 size={12} className="animate-spin" />
                      : <RefreshCw size={12} />}
                    <span>{isRefreshingThis ? t('sidebar.updating', 'Updating...') : t('sidebar.update', 'Update')}</span>
                  </button>
                )}
                <button 
                  onClick={(e) => {
                    e.stopPropagation();
                    if (removeConfirm) {
                      onRemove(node.id);
                      setMenuOpen(false);
                      setRemoveConfirm(false);
                    } else {
                      setRemoveConfirm(true);
                      setTimeout(() => setRemoveConfirm(false), 3000);
                    }
                  }}
                  className={`flex items-center gap-2 w-full text-left px-2 py-1 text-xs rounded transition-colors ${
                    removeConfirm 
                      ? 'text-white bg-red-500 hover:bg-red-600' 
                      : 'text-red-600 hover:bg-red-50'
                  }`}
                >
                  <Trash2 size={12} />
                  <span>{removeConfirm ? t('sidebar.confirmRemove', 'Confirm?') : t('sidebar.remove')}</span>
                </button>
              </div>
            </>
          )}
        </div>

      {/* Name */}
      <div 
        className="flex-1 min-w-0 mr-2"
        onClick={(e) => {
          if (node.type === 'folder') {
            e.stopPropagation();
            setExpanded(!expanded);
          }
        }}
      >
          <p className="text-sm text-gray-800 truncate select-none" title={node.name}>
            {node.name}
          </p>
        </div>

        {/* Checkbox or Spinner */}
        <div className="flex-shrink-0 flex items-center justify-center w-5 h-5">
          {isIndexing || isRefreshingThis ? (
            <Loader2 size={14} className="animate-spin text-gray-400" />
          ) : displayStatus === 'pending' ? (
            <div className="relative">
              <button
                className="w-4 h-4 rounded-full border-2 border-gray-300 border-dashed flex items-center justify-center transition-colors hover:border-red-400 hover:bg-red-50"
                title={t('common.cancel')}
                onClick={(e) => {
                  e.stopPropagation();
                  e.preventDefault();
                  setSkipConfirmOpen(!skipConfirmOpen);
                }}
              >
                <X size={8} className="text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
              </button>
              {skipConfirmOpen && (
                <>
                  <div
                    className="fixed inset-0 z-40"
                    onClick={(e) => { e.stopPropagation(); setSkipConfirmOpen(false); }}
                  />
                  <div className="absolute top-5 right-0 w-28 bg-white border border-gray-200 shadow-lg rounded-md p-1 z-50">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSkipConfirmOpen(false);
                        if (onSkipFile) {
                          const collectPaths = (n: FileSource): string[] => {
                            const paths: string[] = [];
                            if (n.path) paths.push(n.path);
                            if (n.children) n.children.forEach(c => paths.push(...collectPaths(c)));
                            return paths;
                          };
                          const paths = collectPaths(node);
                          if (paths.length > 0) onSkipFile(paths);
                        }
                      }}
                      className="flex items-center gap-2 w-full text-left px-2 py-1 text-xs text-red-600 hover:bg-red-50 rounded"
                    >
                      <X size={12} />
                      <span>{t('common.cancel')}</span>
                    </button>
                  </div>
                </>
              )}
            </div>
          ) : (
            <input 
              ref={checkboxRef}
              type="checkbox"
              checked={isChecked}
              onChange={handleToggle}
              className="w-3.5 h-3.5 text-gray-900 border-gray-300 rounded focus:ring-gray-900 cursor-pointer"
              title="已索引 (选择以检索)"
            />
          )}
        </div>
      </div>

      {/* Children */}
      {expanded && hasChildren && (
        <div className="mt-0.5">
          {node.children!.map(child => (
            <FileTreeNode 
              key={child.id} 
              node={child} 
              activeIdSet={activeIdSet}
              onToggle={onToggle}
              onRemove={onRemove}
              onSkipFile={onSkipFile}
              onRefreshSource={onRefreshSource}
              refreshingFolder={refreshingFolder}
              level={level + 1}
              currentFile={currentFile}
              currentPath={currentPath}
              isGloballyIndexing={isGloballyIndexing}
            />
          ))}
        </div>
      )}
    </div>
  );
});


const RightSidebar: React.FC<RightSidebarProps> = ({
  sources,
  activeSourceIds,
  setActiveSourceIds,
  onSelectAll,
  onRemoveSource,
  onRemoveAllSources,
  onSkipFile,
  onRefreshSource,
  refreshingFolder,
  onAddSources,
  onAddFiles,
  indexingState,
  onClose,
  mode,
  onSwitchMode,
  openedFiles,
  activeOpenedFilePath,
  onSelectOpenedFile,
  onClearOpenedFiles,
  selectedIndexModel,
  installedModels,
  isModelSwitching = false,
  onSelectIndexModel,
  onOpenManageModels,
  isRemovingSources = false,
  removeProgress = null,
}) => {
  const { t } = useTranslation();
  const [imagePreview, setImagePreview] = useState<{ src: string; title: string } | null>(null);
  const [showAddMenu, setShowAddMenu] = useState(false);
  const [showIndexModelDropdown, setShowIndexModelDropdown] = useState(false);
  const [removeAllConfirm, setRemoveAllConfirm] = useState(false);
  const removeAllTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const selectAllRef = useRef<HTMLInputElement>(null);
  const activeIdSet = useMemo(() => new Set(activeSourceIds), [activeSourceIds]);
  const allIds = useMemo(
    () =>
      getChildIds({ id: 'root', name: '', type: 'folder', iconType: 'folder', children: sources }).filter(
        id => id !== 'root'
      ),
    [sources]
  );
  const allSelected = allIds.length > 0 && allIds.every(id => activeIdSet.has(id));
  const isIndeterminate = !allSelected && activeSourceIds.length > 0;
  const selectedFileCount = useMemo(() => countDistinctSelectedFiles(sources, activeSourceIds), [sources, activeSourceIds]);
  const totalFileCount = useMemo(() => countDistinctSelectedFiles(sources, allIds), [sources, allIds]);
  const indexModelOptions = useMemo(() => {
    if (!selectedIndexModel) return installedModels;
    const exists = installedModels.some(m => m.id === selectedIndexModel.id);
    return exists ? installedModels : [selectedIndexModel, ...installedModels];
  }, [installedModels, selectedIndexModel]);

  const indexModelDropdownWidth = useMemo(() => {
    const longest = indexModelOptions.reduce((max, model) => {
      const n = formatModelName(model.name).length;
      return Math.max(max, n);
    }, 0);
    const estimated = longest * 8 + 84;
    return Math.max(220, Math.min(280, estimated));
  }, [indexModelOptions]);
  const shouldShowRecommendedInSources = (m: Model): boolean => {
    const id = String(m?.id || '').toLowerCase();
    return Boolean(m?.recommended) && id !== 'qwen3-4b-gguf';
  };

  useEffect(() => {
    if (selectAllRef.current) {
      selectAllRef.current.indeterminate = isIndeterminate;
    }
  }, [isIndeterminate, allSelected]);

  const handleTreeToggle = useCallback((ids: string[], isSelected: boolean) => {
    setActiveSourceIds(prev => {
      const newSet = new Set(prev);
      ids.forEach(id => {
        if (isSelected) newSet.add(id);
        else newSet.delete(id);
      });
      return Array.from(newSet);
    });
  }, [setActiveSourceIds]);

  return (
    <div className="w-[300px] flex-shrink-0 h-full bg-[#FAFAFA] border-l border-gray-200 flex flex-col shadow-xl lg:shadow-none absolute right-0 lg:relative h-full z-[150]">
      {/* Header */}
      <div 
        className="h-14 flex items-center justify-between px-4 border-b border-gray-200/50 flex-shrink-0 bg-[#FAFAFA] relative z-[160]"
      >
        <div className="flex items-center gap-2 min-w-0 relative z-[170]" style={{ WebkitAppRegion: 'no-drag', pointerEvents: 'auto' } as any}>
            <div className="relative">
              <button
                onClick={() => setShowIndexModelDropdown(!showIndexModelDropdown)}
                disabled={indexingState.isIndexing || isModelSwitching}
                className={`flex items-center gap-2 text-sm font-medium transition-colors ${
                  (indexingState.isIndexing || isModelSwitching)
                    ? 'text-gray-400 cursor-not-allowed'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
                title="Index Model"
              >
                <span className="inline-flex items-center gap-1.5 min-w-0">
                  <span className="truncate max-w-[170px]" title={selectedIndexModel ? formatModelName(selectedIndexModel.name) : 'No Model'}>
                    {selectedIndexModel ? formatModelName(selectedIndexModel.name) : 'No Model'}
                  </span>
                  {selectedIndexModel && shouldShowRecommendedInSources(selectedIndexModel) && (
                    <img src={recommendedIcon} alt="Recommended" className="w-[24px] h-[24px] flex-shrink-0" />
                  )}
                </span>
                <ChevronDown size={14} className="text-gray-400" />
              </button>

              {showIndexModelDropdown && (
                <div
                  className="absolute top-full right-0 mt-1 bg-white border border-gray-100 shadow-lg rounded-lg py-1 z-30 animate-in fade-in zoom-in-95 duration-100"
                  style={{ width: `${indexModelDropdownWidth}px` }}
                >
                  {indexModelOptions.map(model => (
                    <button
                      key={model.id}
                      onClick={() => {
                        if (isModelSwitching || indexingState.isIndexing) return;
                        onSelectIndexModel(model);
                        setShowIndexModelDropdown(false);
                      }}
                      className={`w-full text-left px-4 py-2 text-sm hover:bg-gray-50
                        ${selectedIndexModel?.id === model.id ? 'text-gray-900 font-medium bg-gray-50' : 'text-gray-600'}
                      `}
                    >
                      <span className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2 w-full">
                        <span className="truncate whitespace-nowrap" title={formatModelName(model.name)}>
                          {formatModelName(model.name)}
                        </span>
                        <span className="w-[24px] h-[24px] flex items-center justify-center">
                          {shouldShowRecommendedInSources(model) ? (
                            <img src={recommendedIcon} alt="Recommended" className="w-[24px] h-[24px] flex-shrink-0" />
                          ) : null}
                        </span>
                      </span>
                    </button>
                  ))}
                  <div className="border-t border-gray-100 mt-1 pt-1">
                    <button
                      onClick={() => {
                        setShowIndexModelDropdown(false);
                        onOpenManageModels();
                      }}
                      className="w-full text-left px-4 py-2 text-sm text-gray-500 hover:bg-gray-50 hover:text-gray-700"
                    >
                      + Add Model
                    </button>
                  </div>
                </div>
              )}
            </div>
        </div>
        <div className="relative z-[170]" style={{ WebkitAppRegion: 'no-drag', pointerEvents: 'auto' } as any}>
          <div className="flex items-center gap-1">
            <button 
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 p-1 rounded hover:bg-gray-100 transition-colors"
              title="Collapse Sidebar"
            >
              <ChevronsRight size={18} />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-0 relative">
        {/* Removing overlay — covers the whole content area */}
        {isRemovingSources && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-4 bg-white/85 backdrop-blur-sm">
            <Loader2 size={24} className="animate-spin text-gray-400" />
            <div className="flex flex-col items-center gap-2 w-40">
              <span className="text-sm font-medium text-gray-600">
                {t('sidebar.removingSource', 'Removing...')}
              </span>
              {removeProgress && removeProgress.total > 1 && (
                <>
                  <div className="w-full bg-gray-200 rounded-full h-1">
                    <div
                      className="bg-gray-500 h-1 rounded-full transition-all duration-200"
                      style={{ width: `${(removeProgress.current / removeProgress.total) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs tabular-nums text-gray-400">
                    {removeProgress.current} / {removeProgress.total}
                  </span>
                </>
              )}
            </div>
          </div>
        )}
        
        {/* Tree View Mode */}
        {mode === 'tree' && (
          <>
            {/* ── Fixed top: Add Sources button + toolbar ── */}
            <div className="flex-shrink-0 px-4 pt-4 pb-0 bg-[#FAFAFA]">
              <div className="relative mb-3">
                <button
                  onClick={() => {
                    if (isModelSwitching) return;
                    if (installedModels.length === 0) {
                      onAddSources();
                      return;
                    }
                    setShowAddMenu(!showAddMenu);
                  }}
                  disabled={isModelSwitching}
                  className={`w-full flex items-center justify-center gap-2 px-4 py-2 border border-gray-200 rounded-lg bg-white text-sm font-medium transition-all shadow-sm ${
                    isModelSwitching
                      ? 'text-gray-400 cursor-not-allowed'
                      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 hover:border-gray-300'
                  }`}
                >
                  <Plus size={16} />
                  <span>{t('sidebar.addSources')}</span>
                </button>

                {showAddMenu && (
                  <div className="absolute left-0 right-0 top-full mt-1 z-50">
                    <div className="bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden">
                      <button
                        onClick={() => {
                          if (isModelSwitching) return;
                          setShowAddMenu(false);
                          onAddSources();
                        }}
                        className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                      >
                        <Plus size={14} />
                        <span>{t('sidebar.addFolder')}</span>
                      </button>
                      {onAddFiles && (
                        <button
                          onClick={() => {
                            if (isModelSwitching) return;
                            setShowAddMenu(false);
                            onAddFiles();
                          }}
                          className={`w-full flex items-center gap-2 px-4 py-2 text-sm transition-colors border-t border-gray-100 ${
                            isModelSwitching ? 'text-gray-400 cursor-not-allowed' : 'text-gray-700 hover:bg-gray-50'
                          }`}
                        >
                          <FileText size={14} />
                          <span>{t('sidebar.addFile')}</span>
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {showAddMenu && (
                <div className="fixed inset-0 z-40" onClick={() => setShowAddMenu(false)} />
              )}

              {sources.length > 0 && (
                <div className="flex items-center justify-between py-1.5 border-b border-gray-100">
                  <span className="text-xs text-gray-400 whitespace-nowrap tabular-nums">
                    {selectedFileCount === totalFileCount
                      ? `${totalFileCount} files`
                      : `${selectedFileCount} / ${totalFileCount} files`}
                  </span>
                  <div className="flex items-center gap-1">
                    {onRemoveAllSources && (
                      removeAllConfirm ? (
                        <button
                          onClick={() => {
                            if (removeAllTimerRef.current) clearTimeout(removeAllTimerRef.current);
                            setRemoveAllConfirm(false);
                            onRemoveAllSources();
                          }}
                          className="flex items-center gap-1 text-xs px-1.5 py-0.5 rounded bg-red-500 text-white hover:bg-red-600 transition-colors whitespace-nowrap"
                        >
                          <Trash size={11} />
                          <span>{t('sidebar.confirmRemove', 'Sure?')}</span>
                        </button>
                      ) : (
                        <button
                          onClick={() => {
                            setRemoveAllConfirm(true);
                            removeAllTimerRef.current = setTimeout(() => setRemoveAllConfirm(false), 3000);
                          }}
                          disabled={indexingState.isIndexing || isRemovingSources}
                          className="w-6 h-6 flex items-center justify-center rounded text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                          title={t('sidebar.removeSelected', 'Remove Selected Sources')}
                        >
                          <Trash size={13} />
                        </button>
                      )
                    )}
                    {onRefreshSource && (
                      <button
                        onClick={() => onRefreshSource('__ALL__')}
                        disabled={refreshingFolder === '__ALL__'}
                        className={`w-6 h-6 flex items-center justify-center rounded text-gray-400 hover:text-blue-500 hover:bg-blue-50 transition-colors ${refreshingFolder === '__ALL__' ? 'cursor-not-allowed opacity-50' : ''}`}
                        title={refreshingFolder === '__ALL__' ? t('sidebar.updating', 'Updating...') : t('sidebar.updateAll', 'Update All')}
                      >
                        {refreshingFolder === '__ALL__' ? (
                          <Loader2 size={13} className="animate-spin" />
                        ) : (
                          <RefreshCw size={13} />
                        )}
                      </button>
                    )}
                    <div className="w-px h-3.5 bg-gray-200 mx-1" />
                    <input
                      type="checkbox"
                      ref={selectAllRef}
                      checked={allSelected}
                      onChange={(e) => onSelectAll(e.target.checked)}
                      className="w-3.5 h-3.5 text-gray-900 border-gray-300 rounded focus:ring-gray-900 cursor-pointer"
                      title={t('sidebar.selectAllSources')}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* ── Scrollable file list ── */}
            <div className="flex-1 overflow-y-auto px-4 pt-2 pb-4">
              <div className="space-y-1">
                {sources.map(source => (
                  <FileTreeNode
                    key={source.id}
                    node={source}
                    activeIdSet={activeIdSet}
                    onToggle={handleTreeToggle}
                    onRemove={onRemoveSource}
                    onSkipFile={onSkipFile}
                    onRefreshSource={onRefreshSource}
                    refreshingFolder={refreshingFolder}
                    currentFile={indexingState?.currentFile}
                    currentPath={indexingState?.currentPath}
                    isGloballyIndexing={indexingState?.isIndexing}
                  />
                ))}
              </div>
            </div>
          </>
        )}


        {/* Opened File Mode */}
        {mode === 'openedFile' && (
          <div className="flex-1 overflow-y-auto px-4 pb-4 pt-4">
            {openedFiles.length === 0 ? (
              <div className="text-center py-10 text-sm text-gray-400">
                {t('sidebar.noOpenedFiles')}
              </div>
            ) : (
              <>
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
                  {t('sidebar.openedCount', { count: openedFiles.length })}
                </p>
                <div className="space-y-2">
                  {openedFiles.map((f) => {
                    const isActive = f.file_path === activeOpenedFilePath || (!activeOpenedFilePath && openedFiles[0]?.file_path === f.file_path);
                    return (
                      <button
                        key={f.file_path}
                        type="button"
                        onClick={() => onSelectOpenedFile(f.file_path)}
                        className={`w-full text-left flex items-center gap-3 p-2 rounded transition-colors border ${
                          isActive ? 'bg-gray-900 text-white border-gray-900' : 'bg-white hover:bg-gray-50 border-gray-200'
                        }`}
                        title={f.file_path}
                      >
                        <div className={`p-1 rounded border ${isActive ? 'border-white/20 bg-white/10' : 'border-gray-100 bg-gray-50'}`}>
                          <FileIcon type={f.iconType} className="w-4 h-4" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="text-sm font-medium truncate">{f.file_name}</div>
                          <div className={`text-xs truncate ${isActive ? 'text-white/70' : 'text-gray-500'}`}>{f.file_path}</div>
                        </div>
                      </button>
                    );
                  })}
                </div>

                {(() => {
                  const active = openedFiles.find(x => x.file_path === activeOpenedFilePath) || openedFiles[0];
                  if (!active) return null;
                  return (
                    <div className="mt-4 bg-white border border-gray-200 rounded-lg p-3">
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <FileText size={14} className="text-gray-500 flex-shrink-0" />
                          <div className="text-sm font-semibold text-gray-800 truncate" title={active.file_path}>
                            {active.file_name}
                          </div>
                        </div>
                        {active.truncated && (
                          <span className="text-[11px] px-2 py-0.5 rounded bg-yellow-50 text-yellow-700 border border-yellow-200">
                            {t('sidebar.truncated')}
                          </span>
                        )}
                      </div>
                      {active.iconType === 'image' && typeof active.content === 'string' && active.content.startsWith('data:image/') ? (
                        <div className="space-y-2">
                          <div className="text-[11px] text-gray-500">
                            {t('sidebar.clickToEnlarge')}
                          </div>
                          <button
                            type="button"
                            onClick={() => setImagePreview({ src: active.content, title: active.file_name })}
                            className="w-full border border-gray-200 rounded-md overflow-hidden bg-gray-50 hover:bg-gray-100 transition-colors"
                            title={t('sidebar.clickToEnlarge')}
                          >
                            <img
                              src={active.content}
                              alt={active.file_name}
                              className="w-full max-h-[45vh] object-contain"
                            />
                          </button>
                        </div>
                      ) : (
                        <pre className="text-xs text-gray-700 whitespace-pre-wrap break-words max-h-[45vh] overflow-y-auto">
                          {active.content || t('sidebar.empty')}
                        </pre>
                      )}
                    </div>
                  );
                })()}
              </>
            )}
          </div>
        )}
      </div>

      {/* Image preview overlay */}
      {imagePreview && (
        <div
          className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-6"
          onClick={() => setImagePreview(null)}
        >
          <div
            className="bg-white rounded-lg shadow-xl max-w-[92vw] max-h-[92vh] w-auto overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-gray-200">
              <div className="text-sm font-medium text-gray-800 truncate" title={imagePreview.title}>
                {imagePreview.title}
              </div>
              <button
                type="button"
                onClick={() => setImagePreview(null)}
                className="p-1 rounded hover:bg-gray-100 text-gray-600"
                title={t('common.close')}
              >
                <X size={16} />
              </button>
            </div>
            <div className="p-3 bg-black">
              <img
                src={imagePreview.src}
                alt={imagePreview.title}
                className="max-w-[88vw] max-h-[80vh] object-contain mx-auto"
              />
            </div>
          </div>
        </div>
      )}

      {/* Indexing Widget Footer (Always visible if indexing) */}
      <SidebarIndexingWidget state={indexingState} />
    </div>
  );
};

export default RightSidebar;
