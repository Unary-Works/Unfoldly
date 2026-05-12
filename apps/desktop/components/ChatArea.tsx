import React, { useRef, useEffect, useCallback } from 'react';
import { ThumbsUp, ThumbsDown, Copy, Check, ChevronDown, PanelRightOpen, FolderOpen } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import InputArea from './InputArea';
import { TopIndexingWidget } from './IndexingWidget';
import { Message, Model, FileSource, IndexingState } from '../types';
import { FileIcon } from './Icon';
import { formatModelName, getModelQuantBadge } from '../utils/modelDisplay';
import recommendedIcon from '../assets/recommended.png';

interface TreeNode {
  id: string;
  name: string;
  type: string;
  path: string;
  children: TreeNode[];
  file?: any;
}

const TREE_COMPRESS_FOLDER_NAMES = new Set(
  [
    'users',
    'library',
    'volumes',
    'mnt',
    'home',
    'documents',
    'downloads',
    'desktop',
    '桌面',
    '下载',
    '文档',
  ].map((s) => s.toLowerCase()),
);

function shouldCompressFolderSegment(name: string): boolean {
  return TREE_COMPRESS_FOLDER_NAMES.has(String(name || '').trim().toLowerCase());
}

function buildFileTree(files: any[]): TreeNode[] {
  const root: TreeNode = { id: 'root', name: 'root', type: 'folder', path: '', children: [] };
  
  files.forEach(f => {
    const rawPath = String((f.tree_path != null && f.tree_path !== '' ? f.tree_path : f.path) || f.id || '').trim();
    if (!rawPath) {
      root.children.push({ id: f.id, name: f.name, type: f.type, path: rawPath, children: [], file: f });
      return;
    }
    
    // Split by / or \
    const parts = rawPath.split(/[/\\]/).filter(Boolean);
    if (!parts.length) {
      root.children.push({ id: f.id, name: f.name, type: f.type, path: rawPath, children: [], file: f });
      return;
    }
    
    let current = root;
    let currentPath = '';
    for (let i = 0; i < parts.length - 1; i++) {
      currentPath += (currentPath ? '/' : '') + parts[i];
      if (rawPath.startsWith('/') && i === 0 && !currentPath.startsWith('/')) {
         currentPath = '/' + currentPath;
      } else if (/^[A-Za-z]:/.test(rawPath) && i === 0 && !currentPath.includes(':')) {
         currentPath = parts[0]; // e.g. C:
      }
      let child = current.children.find(c => c.name === parts[i] && c.type === 'folder');
      if (!child) {
        child = { id: currentPath, name: parts[i], type: 'folder', path: currentPath, children: [] };
        current.children.push(child);
      }
      current = child;
    }
    // add file
    current.children.push({ id: f.id, name: f.name, type: f.type, path: rawPath, children: [], file: f });
  });

  function compress(node: TreeNode): TreeNode {
    if (node.type !== 'folder') return node;
    node.children = node.children.map(compress);

    if (node.children.length === 1 && node.id !== 'root') {
      const child = node.children[0];
      if (child.type === 'folder' && shouldCompressFolderSegment(node.name)) {
        return compress(child);
      }
    }
    return node;
  }
  
  root.children = root.children.map(compress);
  return root.children;
}

const SHOW_TRACE_DEBUG = false;
const RELATED_FILES_PAGE_SIZE = 20;

interface ChatAreaProps {
  messages: Message[];
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
  isRightSidebarOpen: boolean;
  onToggleRightSidebar: () => void;
  indexingState?: IndexingState;
  onCloseIndexingTopBar?: () => void;
  isBackendSyncing?: boolean;
  isGenerating?: boolean;
  onStopGenerating?: () => void;
  isModelSwitching?: boolean;
  onOpenManageModels?: () => void;
}

const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
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
  isRightSidebarOpen,
  onToggleRightSidebar,
  indexingState,
  onCloseIndexingTopBar,
  isBackendSyncing = false,
  isGenerating = false,
  onStopGenerating,
  isModelSwitching = false,
  onOpenManageModels,
}) => {
  const { t } = useTranslation();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [copiedId, setCopiedId] = React.useState<string | null>(null);
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
  const [isUserScrolling, setIsUserScrolling] = React.useState(false);
  const [showScrollToBottom, setShowScrollToBottom] = React.useState(false);
  const [relatedFilesExpandedLimit, setRelatedFilesExpandedLimit] = React.useState<Record<string, number>>({});
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
  const isIndexing = Boolean(indexingState?.isIndexing);
  const effectiveIndexingState: IndexingState = indexingState || {
    isIndexing: false,
    totalFiles: 0,
    completedFiles: 0,
    eta: '—',
    isTopBarVisible: false,
  };

  const isNearBottom = useCallback(() => {
    if (!scrollRef.current) return true;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    return distanceFromBottom < 100;
  }, []);

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      setShowScrollToBottom(false);
      setIsUserScrolling(false);
    }
  }, []);

  const handleScroll = useCallback(() => {
    if (!scrollRef.current) return;
    const nearBottom = isNearBottom();
    
    if (nearBottom) {
      setIsUserScrolling(false);
      setShowScrollToBottom(false);
    } else {
      setIsUserScrolling(true);
      setShowScrollToBottom(true);
    }
  }, [isNearBottom]);

  useEffect(() => {
    if (!isUserScrolling && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isUserScrolling]);

  const handleMarkdownLinkClick = useCallback(async (href?: string) => {
    const rawHref = String(href || '').trim();
    if (!rawHref.startsWith('unfoldly://open?path=')) return false;
    try {
      const url = new URL(rawHref);
      const path = url.searchParams.get('path') || '';
      if (!path) return true;
      const { openPath } = await import('../backend');
      await openPath(path);
    } catch {}
    return true;
  }, []);

  return (
    <div className="flex-1 flex flex-col h-full bg-white relative">
      {/* Header */}
      <div 
        className="h-14 border-b border-gray-100 flex items-center justify-between px-6 sticky top-0 bg-white/80 backdrop-blur-sm z-[90]"
      >
          <div className="relative z-[100]" style={{ WebkitAppRegion: 'no-drag' } as any}>
            <button
              onClick={() => !isGenerating && !isIndexing && isModelStateReady && setShowModelDropdown(!showModelDropdown)}
              disabled={isGenerating || isIndexing || !isModelStateReady}
              className={`flex items-center gap-2 text-sm font-medium transition-colors ${
                (isGenerating || isIndexing || !isModelStateReady)
                  ? 'text-gray-400 cursor-not-allowed' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              title={
                isGenerating
                  ? 'Chat Model — generating, cannot switch'
                  : isIndexing
                    ? 'Chat Model — indexing, cannot switch'
                    : !isModelStateReady
                      ? t('chat.loadingData')
                    : 'Chat Model'
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
              className="absolute top-full left-0 mt-1 bg-white border border-gray-100 shadow-lg rounded-lg py-1 z-30"
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
                    + Add Model
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Sidebar Toggle */}
        {!isRightSidebarOpen && (
          <button 
            onClick={onToggleRightSidebar}
            className="text-gray-400 hover:text-gray-600 p-1.5 rounded-md hover:bg-gray-100 transition-colors relative z-[100]"
            title="Open Sources"
            style={{ WebkitAppRegion: 'no-drag' } as any}
          >
            <PanelRightOpen size={20} />
          </button>
        )}
      </div>

      <div className="flex-1 flex flex-col items-center overflow-hidden relative w-full pt-4">
        {/* Indexing Progress (Chat view) */}
        {(isBackendSyncing || indexingState?.isTopBarVisible) && (
          <div className="px-6 w-full max-w-4xl mx-auto pt-2 flex-shrink-0">
            <TopIndexingWidget
              state={effectiveIndexingState}
              onClose={indexingState?.isTopBarVisible ? onCloseIndexingTopBar : undefined}
              isBackendSyncing={isBackendSyncing}
            />
          </div>
        )}

        {/* Messages */}
        <div 
          ref={scrollRef} 
          className="flex-1 overflow-y-auto overflow-x-hidden px-6 py-8 space-y-8 scroll-smooth w-full max-w-4xl mx-auto min-w-0"
          onScroll={handleScroll}
        >
        {messages.map((msg) => (
          <div 
            key={msg.id} 
            className={`flex w-full min-w-0 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.role === 'user' ? (
              <div className="max-w-2xl w-fit min-w-0 overflow-hidden bg-gray-100 text-gray-800 rounded-2xl rounded-tr-sm px-5 py-3 text-base leading-relaxed shadow-sm">
                <div className="chat-break-anywhere whitespace-pre-wrap select-text">{msg.content}</div>
              </div>
            ) : (
              <div className="max-w-3xl w-full min-w-0 overflow-hidden">

                <div className="pl-0 min-w-0 overflow-hidden">
                  {/* Thinking process (Cursor-style collapsible with streaming tokens) */}
                  {(msg.thinkingContent || (msg as any).statusText) && (() => {
                    const isThinking = !msg.content;
                    const thinkLines = (msg.thinkingContent || '').split('\n').filter(Boolean);
                    const lastLine = thinkLines[thinkLines.length - 1] || '';
                    const THINKING_LABEL = 'Thinking...';
                    const THINKING_PROCESS_LABEL = 'Thinking process';
                    const summaryText = msg.thinkingContent
                      ? (isThinking ? lastLine || THINKING_LABEL : THINKING_PROCESS_LABEL)
                      : ((msg as any).statusText || THINKING_LABEL);
                    return (
                    <div className="mb-4 min-w-0 overflow-hidden">
                      <details className="group" open={isThinking || undefined}>
                        <summary className="cursor-pointer list-none flex items-center gap-1.5 select-none w-fit max-w-full">
                          <svg className="w-3 h-3 text-gray-400 group-open:rotate-90 transition-transform flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                          <span className={`text-sm font-medium truncate ${isThinking ? 'text-gray-500' : 'text-gray-400'}`}>
                            {summaryText}
                            {isThinking && <span className="inline-block w-[5px] h-[14px] bg-gray-400 ml-0.5 align-middle animate-pulse rounded-sm" />}
                          </span>
                        </summary>
                        {msg.thinkingContent ? (
                          <div
                            className="mt-2 ml-[18px] pl-3 border-l-2 border-gray-200 max-h-40 overflow-y-auto overflow-x-hidden min-w-0"
                            ref={el => { if (el && isThinking) el.scrollTop = el.scrollHeight; }}
                          >
                            <pre className="chat-break-anywhere text-xs text-gray-400 whitespace-pre-wrap font-mono leading-relaxed select-text">
                              {msg.thinkingContent}
                              {isThinking && <span className="inline-block w-[4px] h-[12px] bg-gray-300 ml-px align-middle animate-pulse rounded-sm" />}
                            </pre>
                          </div>
                        ) : (
                          <div className="mt-2 ml-[18px] pl-3 border-l-2 border-gray-200">
                            <span className="text-xs text-gray-400">{(msg as any).statusText}</span>
                          </div>
                        )}
                      </details>
                    </div>
                    );
                  })()}

                  {SHOW_TRACE_DEBUG && (msg as any).trace?.length > 0 && (
                    <details className="mb-4 group">
                      <summary className="cursor-pointer text-xs font-medium text-gray-400 hover:text-gray-600 transition-colors list-none flex items-center gap-1 select-none w-fit">
                        <div className="flex items-center gap-1">
                          <svg className="w-3 h-3 group-open:rotate-90 transition-transform text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                          <span>{t('chat.viewThinking', { count: (msg as any).trace.length })}</span>
                        </div>
                      </summary>
                      
                      <div className="mt-3 space-y-2 pl-2 border-l-2 border-gray-100 ml-1.5">
                        {(msg as any).trace.map((t: any, idx: number) => {
                          const toolMap: Record<string, string> = {
                            'list_directory': '读取目录',
                            'read_file': '读取文件',
                            'count_documents_files': '统计数据库',
                            'search_documents': '检索数据库',
                          };
                          
                          let title = '';
                          if (t.type === 'tool') {
                            const rawName = t.tool || 'unknown';
                            title = `调用工具: ${toolMap[rawName] || rawName}`;
                          } else {
                            title = t.title || 'Agent 规划';
                          }

                          const preview = t.preview || t.result_preview || t.args_preview || '';
                          
                          return (
                            <details key={idx} className="group/item rounded-lg border border-gray-200 bg-white px-3 py-2">
                              <summary className="cursor-pointer list-none flex items-center justify-between gap-3">
                                <div className="min-w-0 flex-1">
                                  <div className="text-sm font-medium text-gray-700 truncate">{title}</div>
                                </div>
                                <span className="text-xs text-gray-400 group-open/item:hidden">{t('chat.expand')}</span>
                                <span className="text-xs text-gray-400 hidden group-open/item:inline">{t('chat.collapse')}</span>
                              </summary>
                              
                              <div className="mt-2 text-xs text-gray-600 font-mono whitespace-pre-wrap break-all bg-gray-50 p-2 rounded select-text">
                                {t.tool_calls && (
                                  <div className="mb-1">
                                    <span className="text-gray-400 mr-2">Params:</span>
                                    {JSON.stringify(t.tool_calls, null, 2)}
                                  </div>
                                )}
                                {t.args && (
                                  <div className="mb-1">
                                    <span className="text-gray-400 mr-2">Args:</span>
                                    {JSON.stringify(t.args, null, 2)}
                                  </div>
                                )}
                                {preview && (
                                  <div>
                                    <span className="text-gray-400 mr-2">Result:</span>
                                    {String(preview)}
                                  </div>
                                )}
                              </div>
                            </details>
                          );
                        })}
                      </div>
                    </details>
                  )}

                  {(msg as any).relevantFiles?.length ? (
                    <div className="mb-6 min-w-0 overflow-hidden">
                      <div className="border border-gray-100 bg-gray-50/50 rounded-xl p-4 min-w-0 overflow-hidden">
                        {(() => {
                          const allFiles = (Array.isArray((msg as any).relevantFilesAll) && (msg as any).relevantFilesAll.length > 0)
                            ? (msg as any).relevantFilesAll
                            : (msg.relevantFiles || []);
                          const rawTotal = Number((msg as any).relevantFilesTotal ?? allFiles.length);
                          const rawShown = Number((msg as any).relevantFilesShown ?? allFiles.length);
                          const total = Number.isFinite(rawTotal) ? Math.max(rawTotal, allFiles.length) : allFiles.length;
                          const shown = Number.isFinite(rawShown)
                            ? Math.max(Math.min(rawShown, total), allFiles.length)
                            : allFiles.length;
                          const _folderCards = allFiles.filter((f: any) => f.is_matched_folder);
                          const _folderChainFiles = allFiles.filter((f: any) => !f.is_matched_folder && f.from_folder_chain);
                          const _semanticFiles = allFiles.filter((f: any) => !f.is_matched_folder && !f.from_folder_chain);
                          const expandedLimit = relatedFilesExpandedLimit[msg.id] ?? Math.min(RELATED_FILES_PAGE_SIZE, _semanticFiles.length);
                          const expandedSemanticFiles = _semanticFiles.slice(0, expandedLimit);
                          const canLoadMore = expandedLimit < _semanticFiles.length;
                          const _chainByRoot = new Map<string, any[]>();
                          for (const f of _folderChainFiles) {
                            const root = String(f.folder_chain_root || f.path || f.id || '');
                            if (!_chainByRoot.has(root)) _chainByRoot.set(root, []);
                            _chainByRoot.get(root)!.push(f);
                          }
                          const renderFolderCard = (fc: any) => {
                            const childFiles: any[] = _chainByRoot.get(String(fc.path || '')) || [];
                            const displayCnt = childFiles.length || (typeof fc.child_file_count === 'number' ? fc.child_file_count : 0);
                            const p = String(fc.path || '').trim();
                            return (
                              <div key={fc.id} className="space-y-0.5">
                                <div className="flex items-center gap-2 p-1.5 rounded hover:bg-blue-50/60 transition-colors group min-w-0">
                                  <div className="p-1 bg-blue-50 rounded border border-blue-100 shadow-sm flex-shrink-0">
                                    <FolderOpen size={14} className="text-blue-500" />
                                  </div>
                                  <span className="text-sm font-semibold text-gray-700 truncate flex-1 min-w-0">{fc.name}</span>
                                  {displayCnt > 0 && (
                                    <span className="text-[11px] text-gray-400 whitespace-nowrap flex-shrink-0">
                                      {displayCnt} {t('chat.files', 'files')}
                                    </span>
                                  )}
                                  {p && (
                                    <button type="button"
                                      onClick={async () => { try { const { revealPath } = await import('../backend'); await revealPath(p); } catch {} }}
                                      title={t('chat.showInFinder')}
                                      className="flex-shrink-0 text-gray-300 hover:text-gray-600 p-1 rounded hover:bg-gray-200/60 transition-colors opacity-0 group-hover:opacity-100">
                                      <FolderOpen size={14} />
                                    </button>
                                  )}
                                </div>
                                {childFiles.length > 0 && (
                                  <div className="pl-5 border-l-2 border-blue-100 ml-3 space-y-0.5 min-w-0">
                                    {childFiles.map(f => renderRow(f))}
                                  </div>
                                )}
                              </div>
                            );
                          };

                          const renderRow = (file: any) => {
                            const guessPath = () => {
                              const p = String(file?.path || '').trim();
                              if (p) return p;
                              const id = String(file?.id || '').trim();
                              if (!id) return '';
                              const isPosixAbs = id.startsWith('/');
                              const isWinAbs = /^[A-Za-z]:\\/.test(id);
                              return (isPosixAbs || isWinAbs) ? id : '';
                            };
                            const path = guessPath();
                            const canOpen = Boolean(path);
                            const openFile = async () => {
                              if (!canOpen) return;
                              try {
                                const { openPath } = await import('../backend');
                                await openPath(path);
                              } catch {}
                            };
                            const revealFile = async () => {
                              if (!path) return;
                              try {
                                const { revealPath } = await import('../backend');
                                await revealPath(path);
                              } catch {}
                            };
                            return (
                              <div
                                key={file.id}
                                className="flex items-center gap-3 p-1.5 rounded hover:bg-gray-100/80 transition-colors group min-w-0"
                              >
                                <div className="p-1 bg-white rounded border border-gray-200 shadow-sm flex-shrink-0">
                                  <FileIcon type={file.type} className="w-4 h-4" />
                                </div>
                                <button
                                  type="button"
                                  onClick={openFile}
                                  disabled={!canOpen}
                                  title={canOpen ? t('chat.openLocal') : t('chat.pathUnavailable')}
                                  className={`text-sm font-medium text-left underline-offset-2 decoration-blue-600/30 min-w-0 flex-1 ${
                                    canOpen
                                      ? 'text-gray-600 hover:text-blue-600 hover:underline cursor-pointer'
                                      : 'text-gray-400 cursor-not-allowed'
                                  }`}
                                >
                                  <span className="block truncate">{file.name}</span>
                                </button>
                                <button
                                  type="button"
                                  onClick={(e) => { e.stopPropagation(); void revealFile(); }}
                                  disabled={!path}
                                  title={t('chat.showInFinder')}
                                  className="ml-auto flex-shrink-0 text-gray-400 hover:text-gray-700 disabled:text-gray-200 disabled:cursor-not-allowed p-1 rounded hover:bg-gray-200/60 transition-colors"
                                >
                                  <FolderOpen size={16} />
                                </button>
                              </div>
                            );
                          };

                          const renderTreeNode = (node: TreeNode, depth: number = 0): React.ReactNode => {
                            if (node.type === 'folder') {
                              return (
                                <details key={node.id} className="group/folder ml-2 mt-1" open={depth < 2}>
                                  <summary className="flex items-center gap-2 p-1.5 cursor-pointer hover:bg-gray-100/80 rounded transition-colors list-none select-none min-w-0">
                                    <FolderOpen size={16} className="text-gray-400 group-open/folder:text-blue-500 flex-shrink-0" />
                                    <span className="text-sm font-medium text-gray-700 truncate">{node.name}</span>
                                  </summary>
                                  <div className="pl-3 border-l border-gray-200 ml-2 space-y-1 mt-1 min-w-0">
                                    {node.children.map(c => renderTreeNode(c, depth + 1))}
                                  </div>
                                </details>
                              );
                            } else {
                              return (
                                <div key={node.id} className={`${depth > 0 ? 'ml-2' : ''}`}>
                                  {renderRow(node.file)}
                                </div>
                              );
                            }
                          };

                          return (
                            <details
                              className="rf-details"
                              onToggle={(e) => {
                                const isOpen = Boolean((e.currentTarget as HTMLDetailsElement).open);
                                if (!isOpen) return;
                                setRelatedFilesExpandedLimit((prev) => {
                                  if (prev[msg.id]) return prev;
                                  return { ...prev, [msg.id]: Math.min(RELATED_FILES_PAGE_SIZE, allFiles.length) };
                                });
                              }}
                            >
                              <summary className="cursor-pointer list-none flex items-center justify-between gap-3 mb-0 w-full select-none">
                                <div className="min-w-0 flex-1 pr-2 text-left">
                                  <div className="text-sm font-medium text-gray-700 truncate">
                                    {(() => {
                                      const nFolders = _folderCards.length;
                                      const nFiles = _semanticFiles.length;
                                      const fileLabel = (count: number) => count === 1
                                        ? t('chat.relevantFile', 'relevant file')
                                        : t('chat.relevantFiles', 'relevant files');
                                      if (nFolders > 0 && nFiles > 0) {
                                        return `${nFolders} ${t('chat.relevantFolders', nFolders === 1 ? 'folder' : 'folders')} · ${nFiles} ${fileLabel(nFiles)}`;
                                      }
                                      if (nFolders > 0) {
                                        return `${nFolders} ${t('chat.relevantFolders', nFolders === 1 ? 'relevant folder' : 'relevant folders')}`;
                                      }
                                      return `${total} ${fileLabel(total)}${shown < total ? ` (${t('chat.showingCount', { count: shown, defaultValue: `showing ${shown}` })})` : ''}`;
                                    })()}
                                  </div>
                                </div>
                                <span className="rf-expand-label text-xs text-blue-600/80 hover:text-blue-600 font-medium whitespace-nowrap flex-shrink-0">
                                  {t('chat.expand')}
                                </span>
                                <span className="rf-collapse-label text-xs text-blue-600/80 hover:text-blue-600 font-medium whitespace-nowrap flex-shrink-0">
                                  {t('chat.collapse')}
                                </span>
                              </summary>

                              {/* ── Collapsed preview panel ── */}
                              <div className="rf-collapsed-panel space-y-2 border-t border-gray-100/80 pt-3 mt-3">
                                {_folderCards.length > 0 && (
                                  <div className="space-y-1">
                                    {_folderCards.length > 0 && _semanticFiles.length > 0 && (
                                      <div className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide px-1 mb-1">{t('chat.matchedFolders', 'Matched folders')}</div>
                                    )}
                                    {_folderCards.map(renderFolderCard)}
                                  </div>
                                )}
                                {_semanticFiles.length > 0 && (
                                  <div>
                                    {_folderCards.length > 0 && (
                                      <div className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide px-1 mb-1 mt-2">{t('chat.matchedFiles', 'Matched files')}</div>
                                    )}
                                    {buildFileTree(_semanticFiles.slice(0, 5)).map(n => renderTreeNode(n, 0))}
                                  </div>
                                )}
                              </div>

                              {/* ── Expanded panel ── */}
                              <div className="rf-expanded-panel space-y-2 border-t border-gray-100/80 pt-3 mt-3">
                                {_folderCards.length > 0 && (
                                  <div className="space-y-1">
                                    {_folderCards.length > 0 && _semanticFiles.length > 0 && (
                                      <div className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide px-1 mb-1">{t('chat.matchedFolders', 'Matched folders')}</div>
                                    )}
                                    {_folderCards.map(renderFolderCard)}
                                  </div>
                                )}
                                {expandedSemanticFiles.length > 0 && (
                                  <div>
                                    {_folderCards.length > 0 && (
                                      <div className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide px-1 mb-1 mt-2">{t('chat.matchedFiles', 'Matched files')}</div>
                                    )}
                                    {buildFileTree(expandedSemanticFiles).map(n => renderTreeNode(n, 0))}
                                  </div>
                                )}
                                {canLoadMore && (
                                  <button type="button"
                                    onClick={() => setRelatedFilesExpandedLimit(prev => ({ ...prev, [msg.id]: Math.min((prev[msg.id] || RELATED_FILES_PAGE_SIZE) + RELATED_FILES_PAGE_SIZE, _semanticFiles.length) }))}
                                    className="mt-1 text-xs text-blue-600 hover:text-blue-700 underline-offset-2 hover:underline">
                                    {t('chat.loadMoreFiles', { count: Math.min(RELATED_FILES_PAGE_SIZE, _semanticFiles.length - expandedLimit), defaultValue: 'Load more' })}
                                  </button>
                                )}
                              </div>
                            </details>
                          );

                        })()}

                        {/* Progress Bar (Only when scanning) */}
                        {msg.isSearch && msg.scanState === 'thinking' && (
                          <div className="mt-4 h-1 w-full bg-gray-200 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gray-500 rounded-full transition-all duration-300 ease-out"
                              style={{ width: `${msg.scanProgress}%` }}
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  ) : null}
                  
                  <div className="chat-markdown text-gray-800 text-base leading-7 min-w-0 overflow-hidden break-words select-text">
                    <ReactMarkdown
                      urlTransform={(value: string) => value}
                      remarkPlugins={[remarkGfm, remarkBreaks, [remarkMath, { singleDollarTextMath: true }]]}
                      rehypePlugins={[rehypeKatex]}
                      components={{
                        strong: ({node, ...props}) => <span className="font-bold text-gray-900" {...props} />,
                        em: ({node, ...props}) => <span className="italic text-gray-800" {...props} />,
                        h1: ({node, ...props}) => <h1 className="text-2xl font-bold mt-6 mb-4 text-gray-900" {...props} />,
                        h2: ({node, ...props}) => <h2 className="text-xl font-bold mt-5 mb-3 text-gray-900" {...props} />,
                        h3: ({node, ...props}) => <h3 className="text-lg font-bold mt-4 mb-2 text-gray-900" {...props} />,
                        ul: ({node, ...props}) => <ul className="list-disc pl-5 space-y-1 my-2" {...props} />,
                        ol: ({node, ...props}) => <ol className="list-decimal pl-5 space-y-1 my-2" {...props} />,
                        li: ({node, ...props}) => <li className="pl-1" {...props} />,
                        p: ({node, ...props}) => <p className="mb-3 last:mb-0 min-w-0" {...props} />,
                        blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-gray-200 pl-4 py-1 my-3 text-gray-600 italic bg-gray-50 rounded-r" {...props} />,
                        code: ({node, className, children, ...props}: any) => {
                          const match = /language-(\w+)/.exec(className || '');
                          const isInline = !match && !String(children).includes('\n');
                          return isInline 
                            ? <code className="chat-break-anywhere bg-gray-100 text-red-600 px-1.5 py-0.5 rounded text-sm font-mono" {...props}>{children}</code>
                            : <code className="block max-w-full bg-gray-50 border border-gray-200 rounded p-3 text-sm font-mono overflow-x-auto my-2" {...props}>{children}</code>;
                        },
                        a: ({node, href, children, ...props}: any) => {
                          const rawHref = String(href || props?.href || '').trim();
                          if (rawHref.startsWith('unfoldly://open?path=')) {
                            return (
                              <button
                                type="button"
                                title={t('chat.openLocal')}
                                className="inline p-0 m-0 border-0 bg-transparent text-blue-600 hover:underline align-baseline"
                                onClick={(e) => {
                                  e.preventDefault();
                                  void handleMarkdownLinkClick(rawHref);
                                }}
                              >
                                {children}
                              </button>
                            );
                          }
                          return <a className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer" href={rawHref} {...props}>{children}</a>;
                        },
                      }}
                    >
                      {(() => {
                        // Preprocess content to fix "abrupt" LLM artifacts like "=======" headers
                        let content = msg.content || '';

                        // 🔥 Fix "Markdown in a box" syndrome: if the LLM wraps the ENTIRE response in a code block, unwrap it.
                        // This allows headers and lists inside to actually be rendered as rich text by ReactMarkdown.
                        const trimmed = content.trim();
                        if (trimmed.startsWith('```') && trimmed.endsWith('```')) {
                          // Matches ```markdown [content] ``` or ``` [content] ```
                          const match = trimmed.match(/^```(?:markdown)?\n?([\s\S]*?)\n?```$/i);
                          if (match && match[1]) {
                            content = match[1];
                          }
                        }
                        
                        content = content.replace(/\[Generation interrupted\]/gi, `[${t('chat.generationInterrupted')}]`);
                        content = content.replace(/\[生成已中断\]/g, `[${t('chat.generationInterrupted')}]`);
                        
                        const normalized = content
                          .trim()
                          .replace(/^⚠️\s*/u, '')
                          .replace(/^error:\s*/i, '')
                          .replace(/^发生错误[:：]\s*/u, '')
                          .trim()
                          .toLowerCase();
                          
                        if (normalized === 'generation interrupted' || normalized === '生成已中断' || normalized === `[${t('chat.generationInterrupted').toLowerCase()}]`) {
                             content = t('chat.generationInterrupted');
                        }
                        
                        // 1. Handle "Sandwich" headers:
                        // ==========
                        // Title
                        // ==========
                        // -> ### Title
                        content = content.replace(/(?:^|\n)={3,}\s*\n(.+?)\n={3,}(?:\n|$)/gm, '\n### $1\n');

                        // 2. Handle standalone divider lines (===) -> Horizontal Rule
                        content = content.replace(/(?:^|\n)={3,}\s*(?:\n|$)/gm, '\n\n---\n\n');

                        return content;
                      })()}
                    </ReactMarkdown>
                  </div>

                  <div className="flex items-center gap-3 mt-6">
                    <button
                      className="text-gray-300 hover:text-gray-500 transition-colors"
                      onClick={() => {
                        if (msg.content) {
                          navigator.clipboard.writeText(msg.content);
                          setCopiedId(msg.id);
                          setTimeout(() => setCopiedId(null), 2000);
                        }
                      }}
                      title="Copy to clipboard"
                    >
                      {copiedId === msg.id ? <Check size={16} className="text-green-500" /> : <Copy size={16} />}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {showScrollToBottom && (
        <button
          onClick={scrollToBottom}
          className="fixed bottom-32 right-8 bg-white border border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-gray-900 rounded-full p-3 shadow-lg transition-all duration-200 z-30 flex items-center gap-2"
          title={t('chat.backToBottom')}
        >
          <ChevronDown size={20} />
          <span className="text-sm font-medium">{t('chat.backToBottom')}</span>
        </button>
      )}

      {/* Docked Input */}
      <div className="p-6 pt-2 pb-6 max-w-4xl w-full mx-auto">
        <InputArea 
          variant="docked"
          value={inputValue}
          onChange={onInputChange}
          onSend={onSend}
          sourcesLibrary={sourcesLibrary}
          activeSourceIds={activeSourceIds}
          onToggleSource={onToggleSource}
          onRemoveSources={onRemoveSources}
          onAddSources={onAddSources}
          onAddFiles={onAddFiles}
          isIndexing={isIndexing} 
          onOpenSidebar={onToggleRightSidebar}
          isGenerating={isGenerating}
          onStopGenerating={onStopGenerating}
          isModelSwitching={isModelSwitching}
        />
        <div className="text-center mt-2">
           <span className="text-[10px] text-gray-400">AI can make mistakes. Please verify important information.</span>
        </div>
      </div>
      </div>
    </div>
  );
};

export default ChatArea;
