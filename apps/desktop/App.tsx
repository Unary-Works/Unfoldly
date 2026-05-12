import React, { useState, useCallback, useRef, useEffect, useLayoutEffect } from 'react';
import { getCurrentWebview } from '@tauri-apps/api/webview';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { stat } from '@tauri-apps/plugin-fs';
import { useTranslation } from 'react-i18next';
import { v4 as uuidv4 } from 'uuid';
import LeftSidebar from './components/LeftSidebar';
import LandingArea from './components/LandingArea';
import ChatArea from './components/ChatArea';
import RightSidebar from './components/RightSidebar';
import SettingsModal from './components/SettingsModal';
import ManageModelsModal from './components/ManageModelsModal';
import Onboarding from './components/Onboarding';
import ModelPromptToast from './components/ModelPromptToast';
import ModelSwitchIndicator from './components/ModelSwitchIndicator';
import { TopIndexingWidget } from './components/IndexingWidget';
import type { DownloadItemInfo } from './components/onboarding/SetupStep';
import { INITIAL_MODELS, INITIAL_HISTORY, INITIAL_SOURCES } from './constants';
import { Conversation, Message, Model, FileSource, IndexingState, RelevantFile, SidebarMode, OpenedFile } from './types';
import { fetchSources, queryBackendStream, removeSource, removeSourcesBatch, refreshSource, selectFolder, selectFiles, checkMediaFiles, startIndex, indexFiles, getIndexStatus, cancelIndex, skipFiles, fetchHistory, syncHistory, deleteHistory, fetchModels, downloadModel, deleteModel, cancelDownloadModel, selectModel, selectModelQuantization, fetchCoreModelsStatus, startCoreModelsDownload, cancelCoreModelsDownload, fetchAsrModelStatus, startAsrModelDownload, cancelAsrModelDownload, getActiveIndexJob, notifyUIReady, getSettings, updateSettings } from './backend';
import { checkOnboardingStatus, markOnboardingComplete, getOnboardingStep, saveOnboardingStep } from './utils/onboarding';
import { dedupeEffectiveSourceIdsByPath, isIndexedKbFile } from './utils/sourceSelection';
import { formatModelName } from './utils/modelDisplay';
import i18n from './i18n';

async function yieldToMain(): Promise<void> {
  await new Promise<void>((resolve) => {
    requestAnimationFrame(() => {
      setTimeout(resolve, 0);
    });
  });
}

  // Helper to collect all IDs from the tree
  const getAllIds = (nodes: FileSource[]): string[] => {
    let ids: string[] = [];
    for (const node of nodes) {
      if (node.type === 'file') {
        ids.push(node.id);
      } else {
        ids.push(node.id);
      }
      if (node.children) {
        ids = ids.concat(getAllIds(node.children));
      }
    }
    return ids;
  };



function hasIndexingSource(nodes: FileSource[]): boolean {
  for (const node of nodes) {
    if (node.status === 'indexing') return true;
    if (node.children && hasIndexingSource(node.children)) return true;
  }
  return false;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function finalizeSourcesAfterIndex(
  setSourcesLibrary: React.Dispatch<React.SetStateAction<FileSource[]>>,
  setActiveSourceIds: React.Dispatch<React.SetStateAction<string[]>>,
) {
  const delays = [0, 250, 500, 900, 1400];
  for (let attempt = 0; attempt < delays.length; attempt += 1) {
    if (delays[attempt] > 0) await sleep(delays[attempt]);
    try {
      const sources = await fetchSources();
      setSourcesLibrary(sources);
      setActiveSourceIds(getAllIds(sources));
      if (!hasIndexingSource(sources)) return;
      const active = await getActiveIndexJob();
      if ((active as any)?.active || (active as any)?.job?.is_indexing) continue;
    } catch {
      /* retry while the backend finishes clearing persisted index state */
    }
  }
}

const INDEX_STATUS_POLL_MS = 450;
const TEXT_ONLY_INDEX_WARNING_DISABLED_KEY = 'index_text_only_warning_disabled';
const INDEX_MODEL_PREFERENCE_KEY = 'selected_index_model_id';
const INDEX_MODEL_MANUAL_LOCK_KEY = 'index_model_manual_locked';
const CHAT_MODEL_MANUAL_LOCK_KEY = 'chat_model_manual_locked';

function parseBooleanSetting(value: unknown): boolean {
  if (value === true) return true;
  if (value === false || value == null) return false;
  if (typeof value === 'number') return value === 1;
  if (typeof value === 'string') {
    const v = value.trim().toLowerCase();
    if (v === 'true' || v === '1' || v === 'yes' || v === 'on') return true;
    if (v === 'false' || v === '0' || v === 'no' || v === 'off' || v === '') return false;
  }
  return false;
}

function formatIndexEta(job: any): string {
  const etaSeconds = Number(job?.eta_seconds) || 0;
  const etaFromServer = typeof job?.eta === 'string' ? job.eta.trim() : '';
  if (etaFromServer && etaFromServer !== '—') return etaFromServer;
  if (etaSeconds <= 0) return '—';
  if (etaSeconds < 60) return `${etaSeconds}s`;
  if (etaSeconds < 3600) return `${Math.floor(etaSeconds / 60)}m`;
  return `${Math.floor(etaSeconds / 3600)}h`;
}

function buildLiveIndexStatusMessage(job: any): string {
  const fileName = String(job?.current_file || '').trim();
  if (!fileName) return '';
  return `Indexing ${fileName}`;
}

function App() {
  const { t } = useTranslation();
  // Unified model: Gemma 4B serves both chat and vision roles
  const UNIFIED_MODEL_ID = 'gemma-4-e4b-it-gguf';
  const UNIFIED_MODEL_QF = 'gemma-4-E4B-it-Q5_K_S.gguf';
  // Legacy aliases for minimal diff — both point to the same model
  const VL_MODEL_ID = UNIFIED_MODEL_ID;
  const VL_QF = UNIFIED_MODEL_QF;
  const CHAT_MODEL_ID = UNIFIED_MODEL_ID;
  const CHAT_QF = UNIFIED_MODEL_QF;

  // --- Onboarding State ---
  const [onboardingHydrated, setOnboardingHydrated] = useState(false);
  const [isOnboardingComplete, setIsOnboardingComplete] = useState<boolean>(false);
  const [onboardingStep, setOnboardingStep] = useState<'welcome' | 'model-recommend' | 'setup' | 'indexing-guide' | 'indexing-progress' | 'complete'>('welcome');
  const [setupProgress, setSetupProgress] = useState<number>(0);
  const [setupItems, setSetupItems] = useState<DownloadItemInfo[]>([]);
  const [setupDownloadErrors, setSetupDownloadErrors] = useState<Record<string, string>>({});
  const setupDownloadErrorsRef = useRef<Record<string, string>>({});
  const setupIntervalRef = useRef<number | null>(null);
  const setupTickRunningRef = useRef(false);
  const setupTickNowRef = useRef<(() => void) | null>(null);
  const setupAutoRetryAtRef = useRef<{ core: number; asr: number; vl: number; chat: number }>({ core: 0, asr: 0, vl: 0, chat: 0 });
  const setupWasOfflineRef = useRef(false);
  const setupOfflineAbortIssuedRef = useRef(false);
  const skippedModelsRef = useRef(false); // true when user chose "Skip for now" on model-recommend
  const onboardingFastTrackReadyRef = useRef(false);

  const normalizeStartupStep = useCallback((step: string, complete: boolean) => {
    if (complete) return 'complete';
    if (step === 'setup' || step === 'loading-models') return 'welcome';
    return step;
  }, []);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const complete = await checkOnboardingStatus();
        const rawStep = complete ? 'complete' : await getOnboardingStep();
        const startupStep = normalizeStartupStep(String(rawStep || 'welcome'), complete);

        const [coreStatus, asrStatus] = await Promise.all([
          Promise.race<any>([
            fetchCoreModelsStatus().catch(() => null),
            new Promise<null>((resolve) => setTimeout(() => resolve(null), 2500)),
          ]),
          Promise.race<any>([
            fetchAsrModelStatus().catch(() => null),
            new Promise<null>((resolve) => setTimeout(() => resolve(null), 2500)),
          ]),
        ]);
        
        let shouldForceWelcome = false;
        if (coreStatus || asrStatus) {
          // Backend responded: check actual installation state
          const coreReady = coreStatus
            ? Boolean((coreStatus as any)?.embedding?.installed) && Boolean((coreStatus as any)?.reranker?.installed)
            : complete;
          const asrReady = asrStatus
            ? Boolean(((asrStatus as any)?.asr ?? asrStatus)?.installed)
            : complete;
          shouldForceWelcome = !coreReady || !asrReady;
        } else if (!complete) {
          // Backend timed out/failed, and onboarding wasn't complete anyway
          shouldForceWelcome = true;
        }
        // If status calls time out AND complete is true, assume backend is just slow
        // and preserve the user's completed state.

        const nextComplete = shouldForceWelcome ? false : complete;
        const nextStep = shouldForceWelcome ? 'welcome' : startupStep;

        if (!nextComplete && nextStep === 'welcome' && rawStep !== 'welcome') {
          try {
            await saveOnboardingStep('welcome');
          } catch {
            /* ignore */
          }
        } else if (shouldForceWelcome) {
          try {
            await Promise.all([
              updateSettings('onboarding_complete', false),
              updateSettings('onboarding_step', 'welcome'),
            ]);
          } catch {
            /* ignore */
          }
        }
        if (!mounted) return;
        setIsOnboardingComplete(nextComplete);
        setOnboardingStep(nextStep as any);
      } catch {
        if (!mounted) return;
        setIsOnboardingComplete(false);
        setOnboardingStep('welcome');
      } finally {
        if (mounted) setOnboardingHydrated(true);
      }
    })();
    return () => {
      mounted = false;
    };
  }, [normalizeStartupStep]);

  // --- State ---
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  
  // Drag Drop Queue State
  const dragQueueRef = useRef<{ folders: string[], files: string[], isProcessing: boolean }>({ folders: [], files: [], isProcessing: false });
  const [isDragOver, setIsDragOver] = useState(false);
  const dragCounter = useRef(0);

  // Model State
  const [models, setModels] = useState<Model[]>([]);
  const [modelsHydrated, setModelsHydrated] = useState(false);
  const installedModels = models.filter(m => m.status === 'installed');
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);       // Chat model
  const [selectedIndexModel, setSelectedIndexModel] = useState<Model | null>(null); // Index model
  const userPickedChatModelRef = useRef(false);
  const userPickedIndexModelRef = useRef(false);
  const prevInstalledModelIdsRef = useRef<string[] | null>(null);
  const indexPrevModelIdRef = useRef<string | null>(null);
  const indexPrevQuantRef = useRef<string | null>(null);

  const isMultimodal = (m: Model) => {
    const id = String(m?.id || '').toLowerCase();
    const name = String(m?.name || '').toLowerCase();
    const hasMmproj = Array.isArray((m as any)?.files)
      ? (m as any).files.some((f: string) => String(f || '').toLowerCase().includes('mmproj'))
      : Boolean((m as any)?.selected_mmproj_path);
    return id.includes('-vl-') || name.includes('-vl-') || hasMmproj;
  };

  const modelSizeInB = (m: Model): number => {
    const raw = String(m.name || m.id || '');
    const hit = raw.match(/(\d+(?:\.\d+)?)\s*[Bb]/);
    return hit ? Number(hit[1]) : 999;
  };

  const modelTierForRole = useCallback((role: 'chat' | 'index', m: Model): number => {
    const vl = isMultimodal(m);
    const rec = Boolean((m as any)?.recommended);
    if (role === 'chat') {
      // chat: vl < text < recommended(text)
      if (!vl && rec) return 2;
      if (!vl) return 1;
      return 0;
    }
    // index: text < vl < recommended(vl)
    if (vl && rec) return 2;
    if (vl) return 1;
    return 0;
  }, []);

  const isBetterModelForRole = useCallback(
    (role: 'chat' | 'index', candidate: Model | null, current: Model | null): boolean => {
      if (!candidate) return false;
      if (!current) return true;
      return modelTierForRole(role, candidate) > modelTierForRole(role, current);
    },
    [modelTierForRole]
  );

  const pickCandidateByPriority = (candidates: Model[], role: 'chat' | 'index'): Model | null => {
    if (!candidates.length) return null;
    const sorted = [...candidates].sort((a, b) => {
      const tierDiff = modelTierForRole(role, b) - modelTierForRole(role, a);
      if (tierDiff !== 0) return tierDiff;
      if (role === 'index') {
        const sizeDiff = modelSizeInB(a) - modelSizeInB(b);
        if (sizeDiff !== 0) return sizeDiff;
      }
      return String(a.name || a.id).localeCompare(String(b.name || b.id));
    });
    return sorted[0] ?? null;
  };

  const pickAutoModelForRole = useCallback((role: 'chat' | 'index', current: Model | null): Model | null => {
    if (installedModels.length === 0) return null;

    const preferMultimodal = role === 'index';
    const primary = installedModels.filter((m) => preferMultimodal ? isMultimodal(m) : !isMultimodal(m));
    const fallback = installedModels.filter((m) => preferMultimodal ? !isMultimodal(m) : isMultimodal(m));
    const currentInstalled = current ? (installedModels.find((m) => m.id === current.id) || null) : null;
    const currentInPrimary = !!currentInstalled && primary.some((m) => m.id === currentInstalled.id);
    const currentRecommended = Boolean(currentInstalled?.recommended);

    if (currentInstalled && currentRecommended && (currentInPrimary || primary.length === 0)) {
      return currentInstalled;
    }

    return pickCandidateByPriority(primary, role) || pickCandidateByPriority(fallback, role);
  }, [installedModels, modelTierForRole]);

  const installedModelsSignature = installedModels
    .map((m) => `${m.id}:${m.recommended ? '1' : '0'}:${isMultimodal(m) ? '1' : '0'}`)
    .join('|');

  // Auto-assign index and chat models when installed models change
  useEffect(() => {
    if (!modelsHydrated) return;

    const installedIds = installedModels.map((m) => m.id);
    const prevInstalledIds = prevInstalledModelIdsRef.current;
    const addedInstalledIds = prevInstalledIds
      ? installedIds.filter((id) => !prevInstalledIds.includes(id))
      : [];
    prevInstalledModelIdsRef.current = installedIds;
    const newlyInstalledModels = installedModels.filter((m) => addedInstalledIds.includes(m.id));

    if (installedModels.length === 0) {
      if (selectedIndexModel) setSelectedIndexModel(null);
      if (selectedModel) setSelectedModel(null);
      userPickedChatModelRef.current = false;
      userPickedIndexModelRef.current = false;
      return;
    }

    // --- Index model ---
    const currentIndexStillInstalled = selectedIndexModel ? (installedModels.find((m) => m.id === selectedIndexModel.id) || null) : null;
    if (!currentIndexStillInstalled) {
      if (userPickedIndexModelRef.current && selectedIndexModel) {
        void updateSettings(INDEX_MODEL_MANUAL_LOCK_KEY, false);
        void updateSettings(INDEX_MODEL_PREFERENCE_KEY, '');
      }
      userPickedIndexModelRef.current = false;
      const nextIndex = pickAutoModelForRole('index', null);
      if (nextIndex) {
        setSelectedIndexModel(nextIndex);
      }
    } else if (newlyInstalledModels.length > 0) {
      const bestNewIndex = pickCandidateByPriority(newlyInstalledModels, 'index');
      if (
        bestNewIndex &&
        bestNewIndex.id !== currentIndexStillInstalled.id &&
        isBetterModelForRole('index', bestNewIndex, currentIndexStillInstalled)
      ) {
        userPickedIndexModelRef.current = false;
        setSelectedIndexModel(bestNewIndex);
        void updateSettings(INDEX_MODEL_MANUAL_LOCK_KEY, false);
        void updateSettings(INDEX_MODEL_PREFERENCE_KEY, '');
      }
    }

    // --- Chat model ---
    const currentChatStillInstalled = selectedModel ? (installedModels.find((m) => m.id === selectedModel.id) || null) : null;
    if (!currentChatStillInstalled) {
      if (userPickedChatModelRef.current && selectedModel) {
        void updateSettings(CHAT_MODEL_MANUAL_LOCK_KEY, false);
      }
      userPickedChatModelRef.current = false;
      const nextChat = pickAutoModelForRole('chat', null);
      if (nextChat) {
        setSelectedModel(nextChat);
        void selectModel(nextChat.id).catch((e) => {
          console.warn('[Auto Chat Switch] failed to select fallback model:', e);
        });
      }
    } else if (newlyInstalledModels.length > 0) {
      const bestNewChat = pickCandidateByPriority(newlyInstalledModels, 'chat');
      if (
        bestNewChat &&
        bestNewChat.id !== currentChatStillInstalled.id &&
        isBetterModelForRole('chat', bestNewChat, currentChatStillInstalled)
      ) {
        userPickedChatModelRef.current = false;
        setSelectedModel(bestNewChat);
        void selectModel(bestNewChat.id).catch((e) => {
          console.warn('[Auto Chat Switch] failed to select better new model:', e);
        });
        void updateSettings(CHAT_MODEL_MANUAL_LOCK_KEY, false);
      }
    }
  }, [modelsHydrated, installedModelsSignature, pickAutoModelForRole, isBetterModelForRole, selectedIndexModel, selectedModel]);

  // Polling ref for model downloads
  const modelPollIntervalRef = useRef<number | null>(null);

  // Modal State
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isManageModelsOpen, setIsManageModelsOpen] = useState(false);
  const [manageModelsFocus, setManageModelsFocus] = useState<{ modelId?: string; searchText?: string } | null>(null);
  
  // UI State
  const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(false); // Collapsed by default on landing
  const [sidebarMode, setSidebarMode] = useState<SidebarMode>('tree');
  
  // Input State
  const [inputValue, setInputValue] = useState('');
  
  // Source State (Global Library + Active Selection)
  const [sourcesLibrary, setSourcesLibrary] = useState<FileSource[]>(INITIAL_SOURCES);
  // Default select all from initial
  const [activeSourceIds, setActiveSourceIds] = useState<string[]>([]);
  const [searchRelevantFiles, setSearchRelevantFiles] = useState<RelevantFile[]>([]);

  // Opened file viewer (RightSidebar)
  const [openedFiles, setOpenedFiles] = useState<OpenedFile[]>([]);
  const [activeOpenedFilePath, setActiveOpenedFilePath] = useState<string | null>(null);

  const [indexingState, setIndexingState] = useState<IndexingState>({
    isIndexing: false,
    totalFiles: 0,
    completedFiles: 0,
    eta: '—',
    isTopBarVisible: false,
    isCancelling: false,
    isRestoringModel: false,
    statusMessage: '',
    currentFrame: 0,
    totalFrames: 0,
    currentAudioSec: 0,
    totalAudioSec: 0,
    stageRate: 0,
    stage: '',
  });

  const indexingIntervalRef = useRef<number | null>(null);
  const fileIndexPollIntervalRef = useRef<number | null>(null);
  const fileIndexSourcesRefreshRef = useRef<{ at: number; completed: number; total: number; currentPath: string }>({
    at: 0,
    completed: -1,
    total: -1,
    currentPath: '',
  });
  const [indexJobId, setIndexJobId] = useState<string | null>(null);
  const [indexCancelConfirmOpen, setIndexCancelConfirmOpen] = useState(false);
  const [pendingDeleteChatId, setPendingDeleteChatId] = useState<string | null>(null);
  const [cancelSummaryDialog, setCancelSummaryDialog] = useState<{
    completed: number;
    cancelled: number;
    failed: number;
  } | null>(null);
  const indexingCompleteDialogOpenRef = useRef(false);
  const indexingCompleteDialogLastRef = useRef<{ key: string; at: number }>({ key: '', at: 0 });
  const [connectionError, setConnectionError] = useState<boolean>(false);

  // --- Derived State ---
  const activeConversation = conversations.find(c => c.id === activeConversationId);
  const messages = activeConversation ? activeConversation.messages : [];
  const isLanding = !activeConversationId && messages.length === 0;

  const indexingProgress = indexingState.totalFiles > 0 ? Math.round((indexingState.completedFiles / indexingState.totalFiles) * 100) : 0;

  const [isBackendSyncing, setIsBackendSyncing] = useState(false);
  const syncPanelInMain = isBackendSyncing && !connectionError && isOnboardingComplete;

  // --- Model prompt toast ---
  const [showModelPrompt, setShowModelPrompt] = useState(false);
  const hasInstalledModels = installedModels.length > 0;
  const [isModelSwitching, setIsModelSwitching] = useState(false);
  const [switchingToModelId, setSwitchingToModelId] = useState<string | null>(null);
  const [modelSwitchError, setModelSwitchError] = useState<boolean>(false);
  const modelSwitchingCountRef = useRef(0);
  const [textOnlyIndexWarningOpen, setTextOnlyIndexWarningOpen] = useState(false);
  const [textOnlyIndexWarningSkipNext, setTextOnlyIndexWarningSkipNext] = useState(false);
  const [textOnlyIndexWarningDisabled, setTextOnlyIndexWarningDisabled] = useState(false);
  const [textOnlyIndexWarningModelName, setTextOnlyIndexWarningModelName] = useState('');
  const textOnlyIndexWarningResolverRef = useRef<((proceed: boolean) => void) | null>(null);

  const openManageModels = useCallback((focus?: { modelId?: string; searchText?: string }) => {
    setManageModelsFocus(focus ?? null);
    setIsManageModelsOpen(true);
  }, []);

  const closeManageModels = useCallback(() => {
    setIsManageModelsOpen(false);
    setManageModelsFocus(null);
  }, []);

  const ensureIndexModelForIndexing = useCallback((): Model | null => {
    if (installedModels.length === 0) {
      setShowModelPrompt(true);
      return null;
    }

    if (userPickedIndexModelRef.current && selectedIndexModel) {
      const selectedInstalled = installedModels.find((m) => m.id === selectedIndexModel.id) || null;
      if (selectedInstalled) return selectedInstalled;
      window.alert('Selected index model is no longer installed. Please reselect an installed model.');
      userPickedIndexModelRef.current = false;
      setSelectedIndexModel(null);
      void updateSettings(INDEX_MODEL_MANUAL_LOCK_KEY, false);
      void updateSettings(INDEX_MODEL_PREFERENCE_KEY, '');
      return null;
    }

    const currentInstalled = selectedIndexModel
      ? (installedModels.find((m) => m.id === selectedIndexModel.id) || null)
      : null;
    return currentInstalled || pickAutoModelForRole('index', null);
  }, [installedModels, selectedIndexModel, pickAutoModelForRole]);

  const persistIndexModelForIndexing = useCallback(async (model: Model): Promise<boolean> => {
    if (!model?.id) return false;
    setSelectedIndexModel(model);
    try {
      await updateSettings(INDEX_MODEL_PREFERENCE_KEY, model.id);
      return true;
    } catch (e) {
      console.warn('[Sources] Failed to persist index model before indexing:', e);
      return false;
    }
  }, []);

  const persistTextOnlyIndexWarningDisabled = useCallback(async (disabled: boolean) => {
    setTextOnlyIndexWarningDisabled(disabled);
    try {
      await updateSettings(TEXT_ONLY_INDEX_WARNING_DISABLED_KEY, disabled);
    } catch (e) {
      console.warn('Failed to persist text-only index warning setting:', e);
    }
  }, []);

  const resolveTextOnlyIndexWarning = useCallback(async (proceed: boolean) => {
    const resolver = textOnlyIndexWarningResolverRef.current;
    textOnlyIndexWarningResolverRef.current = null;
    const disableFuture = proceed && textOnlyIndexWarningSkipNext;
    setTextOnlyIndexWarningOpen(false);
    setTextOnlyIndexWarningSkipNext(false);
    setTextOnlyIndexWarningModelName('');
    if (disableFuture) {
      await persistTextOnlyIndexWarningDisabled(true);
    }
    resolver?.(proceed);
  }, [textOnlyIndexWarningSkipNext, persistTextOnlyIndexWarningDisabled]);

  const confirmTextOnlyIndexWarningIfNeeded = useCallback(async (model: Model): Promise<boolean> => {
    if (isMultimodal(model) || textOnlyIndexWarningDisabled) return true;
    return await new Promise<boolean>((resolve) => {
      textOnlyIndexWarningResolverRef.current = resolve;
      setTextOnlyIndexWarningModelName(formatModelName(model.name || model.id));
      setTextOnlyIndexWarningSkipNext(false);
      setTextOnlyIndexWarningOpen(true);
    });
  }, [textOnlyIndexWarningDisabled]);

  // Show model prompt toast when entering main UI without models
  useEffect(() => {
    if (!onboardingHydrated || !modelsHydrated) return;
    if (isOnboardingComplete && !hasInstalledModels) {
      const timer = setTimeout(() => {
        setShowModelPrompt(true);
      }, 2000);
      return () => clearTimeout(timer);
    } else {
      setShowModelPrompt(false);
    }
  }, [onboardingHydrated, modelsHydrated, isOnboardingComplete, hasInstalledModels]);

  // --- Generating state (for AI response) ---
  const [isGenerating, setIsGenerating] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const sendInFlightRef = useRef(false);

  // --- Initial Load: sources & history & models from backend ---
  const loadSourcesFromBackend = useCallback(async () => {
    try {
      const sources = await fetchSources();
      setSourcesLibrary(sources);
      setActiveSourceIds(getAllIds(sources));
      setConnectionError(false);
      return true;
    } catch (e) {
      console.warn('Failed to fetch sources:', e);
      setConnectionError(true);
      return false;
    }
  }, []);

  // Fetch models from backend
  const refreshModels = useCallback(async () => {
    try {
      const data = await fetchModels();
      setModels(prev => {
        if (prev.length > 0 && data.length === 0) return prev;
        const prevInstalled = prev.filter(m => m.status === 'installed').length;
        const newInstalled = data.filter((m: any) => m.status === 'installed').length;
        if (prevInstalled > 0 && newInstalled === 0 && data.some((m: any) => m.status === 'downloading')) {
          return prev;
        }
        return data;
      });
      setModelsHydrated(true);
      if (indexPrevModelIdRef.current) {
        setSelectedModel(prev => {
          if (!prev) return prev;
          const refreshed = data.find((m: any) => m.id === prev.id && m.status === 'installed');
          return refreshed || prev;
        });
      } else {
        setSelectedModel(prev => {
          const backendSelected = data.find((m: any) => m.selected && m.status === 'installed');
          if (backendSelected) return backendSelected;
          if (prev && data.find((m: any) => m.id === prev.id && m.status === 'installed')) {
            return data.find((m: any) => m.id === prev.id) || prev;
          }
          const installed = data.filter((m: any) => m.status === 'installed');
          if (installed.length > 0) return installed[0];
          return null;
        });
      }
    } catch (e) {
      console.warn('Failed to fetch models:', e);
    }
  }, []);

  const snapshotModelBeforeIndexing = useCallback(() => {
    const selectedByBackend = models.find((m: any) => (m as any).selected);
    const current = selectedModel || selectedByBackend || null;
    indexPrevModelIdRef.current = current?.id || null;
    indexPrevQuantRef.current = current?.selected_quantization || null;
  }, [models, selectedModel]);

  const switchModelAndSyncUI = useCallback(async (modelId: string, quantFile?: string | null, opts?: { updateChatSelection?: boolean }) => {
    const updateChat = opts?.updateChatSelection !== false;
    modelSwitchingCountRef.current += 1;
    setIsModelSwitching(true);
    setSwitchingToModelId(modelId);
    setModelSwitchError(false);
    try {
      if (quantFile) {
        const qRes = await selectModelQuantization(modelId, quantFile);
        if ((qRes as any)?.ok === false) {
          throw new Error(String((qRes as any)?.error || 'select quantization failed'));
        }
      }
      const mRes = await selectModel(modelId);
      if ((mRes as any)?.ok === false) {
        throw new Error(String((mRes as any)?.error || 'select model failed'));
      }
      if (updateChat) {
        const localHit = models.find(m => m.id === modelId);
        if (localHit) setSelectedModel(localHit);
      }
      await refreshModels();
    } catch (e: any) {
      setModelSwitchError(true);
      throw e;
    } finally {
      modelSwitchingCountRef.current = Math.max(0, modelSwitchingCountRef.current - 1);
      if (modelSwitchingCountRef.current === 0) {
        setIsModelSwitching(false);
      }
    }
  }, [models, refreshModels]);

  const restoreModelAfterIndexing = useCallback(async () => {
    const prevId = indexPrevModelIdRef.current;
    const prevQ = indexPrevQuantRef.current;
    if (!prevId) return;
    if (selectedIndexModel && prevId === selectedIndexModel.id) {
      indexPrevModelIdRef.current = null;
      indexPrevQuantRef.current = null;
      return;
    }
    for (let w = 0; w < 120; w++) {
      try {
        const jobId = indexJobId;
        if (jobId) {
          const st = await getIndexStatus(jobId);
          if (st?.job?.is_indexing) {
            await new Promise(r => setTimeout(r, 500));
            continue;
          }
        }
        break;
      } catch {
        await new Promise(r => setTimeout(r, 500));
      }
    }
    await new Promise(r => setTimeout(r, 1000));
    await switchModelAndSyncUI(prevId, prevQ);
    indexPrevModelIdRef.current = null;
    indexPrevQuantRef.current = null;
  }, [selectedIndexModel, indexJobId, switchModelAndSyncUI]);

  const loadFinalIndexCounts = useCallback(async (
    jobId: string | null,
    fallbackTotal: number,
    fallbackCompleted: number,
  ) => {
    if (!jobId) {
      return { total: fallbackTotal, completed: fallbackCompleted, failed: 0, isIndexing: false };
    }
    try {
      const st = await getIndexStatus(jobId);
      const job = st?.job;
      if (!job) {
        return { total: fallbackTotal, completed: fallbackCompleted, failed: 0, isIndexing: false };
      }
      return {
        total: Number(job.total_files || fallbackTotal || 0),
        completed: Number(job.completed_files || fallbackCompleted || 0),
        failed: Number(job.failed_files || 0),
        isIndexing: Boolean(job.is_indexing),
      };
    } catch {
      return { total: fallbackTotal, completed: fallbackCompleted, failed: 0, isIndexing: false };
    }
  }, []);

  const showCancelSummaryDialog = useCallback(async (total: number, completed: number, failed: number = 0) => {
    const t = Math.max(0, Number(total || 0));
    const c = Math.max(0, Number(completed || 0));
    const f = Math.max(0, Number(failed || 0));
    const cancelled = Math.max(0, t - c - f);
    setCancelSummaryDialog({
      completed: c,
      cancelled,
      failed: f,
    });
  }, []);

  const showIndexingCompleteDialog = useCallback(async (
    total: number,
    completed: number,
    failed: number,
    jobKey?: string | null,
    errorMessage?: string | null,
  ) => {
    const t = Math.max(0, Number(total || 0));
    const c = Math.max(0, Number(completed || 0));
    const rawF = Math.max(0, Number(failed || 0));
    const normalizedError = String(errorMessage || '').trim();
    const hasError = normalizedError !== '' && normalizedError.toLowerCase() !== 'cancelled';
    const f = rawF > 0 ? rawF : (hasError ? Math.max(0, t - c) : rawF);
    const succeeded = Math.max(0, Math.min(t || c, c) - f);
    const stableJobKey = String(jobKey || '').trim() || 'nojob';
    const dedupeKey = `${stableJobKey}:${succeeded}:${f}:${hasError ? normalizedError : ''}`;
    const now = Date.now();
    if (indexingCompleteDialogOpenRef.current) return;
    if (
      indexingCompleteDialogLastRef.current.key === dedupeKey
      && now - indexingCompleteDialogLastRef.current.at < 15_000
    ) {
      return;
    }
    try {
      indexingCompleteDialogOpenRef.current = true;
      const { message } = await import('@tauri-apps/plugin-dialog');
      const summary = i18n.t('indexingWidget.indexingCompleteBody', { completed: succeeded, failed: f });
      const body = hasError
        ? `${summary}\n\n${i18n.t('indexingWidget.indexingErrorLine', { error: normalizedError })}`
        : summary;
      const titleKey = hasError
        ? (c > 0 ? 'indexingWidget.indexingFinishedWithErrorsTitle' : 'indexingWidget.indexingFailedTitle')
        : 'indexingWidget.indexingCompleteTitle';
      await message(body, {
        title: i18n.t(titleKey),
        kind: hasError ? 'error' : 'info',
      });
      indexingCompleteDialogLastRef.current = { key: dedupeKey, at: Date.now() };
    } catch (e) {
      console.warn('show indexing summary failed:', e);
    } finally {
      indexingCompleteDialogOpenRef.current = false;
    }
  }, []);

  useLayoutEffect(() => {
    document.body.classList.add('app-shell-ready');
    return () => {
      document.body.classList.remove('app-shell-ready');
    };
  }, []);

  useEffect(() => {
    if (!onboardingHydrated) return;

    let mounted = true;
    let retryCount = 0;
    const maxRetries = import.meta.env.DEV ? 30 : 180;
    const baseRetryDelay = 200;

    const attachToIndexJob = async (jobId: string, topBarVisible: boolean) => {
      let currentJobId = jobId;
      setIndexJobId(currentJobId);
      setIndexingState(prev => ({
        ...prev,
        isIndexing: true,
        isTopBarVisible: topBarVisible,
        isCancelling: false,
        isRestoringModel: false,
        statusMessage: '',
      }));

      const clearIndexPoll = () => {
        if (indexingIntervalRef.current != null) {
          window.clearTimeout(indexingIntervalRef.current);
          indexingIntervalRef.current = null;
        }
      };
      clearIndexPoll();

      const scheduleNext = () => {
        indexingIntervalRef.current = window.setTimeout(runTick, INDEX_STATUS_POLL_MS);
      };

      const runTick = async () => {
        indexingIntervalRef.current = null;
        try {
          const st = await getIndexStatus(currentJobId);
          if (st?.error === 'job_not_found') {
            clearIndexPoll();
            setIndexJobId(null);
            setIndexingState(prev => ({
              ...prev,
              isIndexing: false,
              isRestoringModel: false,
              isTopBarVisible: prev.isCancelling ? true : false,
            }));
            await finalizeSourcesAfterIndex(setSourcesLibrary, setActiveSourceIds);
            return;
          }
          const job = st?.job;
          if (!job) {
            scheduleNext();
            return;
          }

          let isIndexing = Boolean(job.is_indexing);

          if (!isIndexing && job.error !== 'cancelled') {
             await new Promise(r => setTimeout(r, 500));
             try {
               const activeRes = await getActiveIndexJob();
               const nextJob = (activeRes as any)?.job;
               if (nextJob && nextJob.is_indexing && nextJob.job_id !== currentJobId) {
                   console.log('[Sources] 恢复追踪队列后续任务:', nextJob.job_id);
                   currentJobId = nextJob.job_id;
                   setIndexJobId(currentJobId);
                   scheduleNext();
                   return;
               }
             } catch(e) {}
          }

          setIndexingState(prev => ({
            ...prev,
            isIndexing,
            isRestoringModel: false,
            totalFiles: Number(job.total_files || 0),
            completedFiles: Math.min(Number(job.completed_files || 0), Number(job.total_files || 0)),
            eta: formatIndexEta(job),
            isTopBarVisible: job.is_indexing ? prev.isTopBarVisible : false,
            statusMessage: buildLiveIndexStatusMessage(job),
            currentFile: job.current_file,
            currentPath: job.current_path,
            currentFrame: Number(job.current_frame || 0),
            totalFrames: Number(job.total_frames || 0),
            currentAudioSec: Number(job.current_audio_sec || 0),
            totalAudioSec: Number(job.total_audio_sec || 0),
            stageRate: Number(job.stage_rate || 0),
            stage: job.stage || '',
          }));

          if (job.is_indexing) {
            try {
              const sources = await fetchSources();
              setSourcesLibrary(sources);
              // Auto-check newly indexed files immediately
              setActiveSourceIds(getAllIds(sources));
            } catch {
              /* ignore */
            }
            scheduleNext();
            return;
          }

          clearIndexPoll();
          setIndexJobId(null);
          setIndexingState(prev => ({
            ...prev,
            isIndexing: false,
            isRestoringModel: false,
            isTopBarVisible: prev.isCancelling ? true : false,
          }));
          
          await finalizeSourcesAfterIndex(setSourcesLibrary, setActiveSourceIds);

          if (!isOnboardingComplete && onboardingStep === 'indexing-progress') {
            await refreshModels();
            await markOnboardingComplete();
            setIsOnboardingComplete(true);
            setOnboardingStep('complete');
            await saveOnboardingStep('complete');
          }

          if (job && job.error !== 'cancelled') {
            await showIndexingCompleteDialog(
              Number(job.total_files || 0),
              Number(job.completed_files || 0),
              Number(job.failed_files || 0),
              String((job as any)?.job_id || currentJobId || ''),
              String(job.error || ''),
            );
          }
        } catch {
          scheduleNext();
        }
      };

      indexingIntervalRef.current = window.setTimeout(runTick, 0);
    };

    const loadData = async () => {
      try {
        setIsBackendSyncing(true);
        const [history, modelsData, coreStatus, settingsData] = await Promise.all([
          fetchHistory(),
          fetchModels(),
          fetchCoreModelsStatus().catch(() => null),
          getSettings().catch(() => null),
        ]);
        if (!mounted) return;
        await yieldToMain();

        setModels(modelsData);
        const textOnlyWarningDisabled = parseBooleanSetting((settingsData as any)?.[TEXT_ONLY_INDEX_WARNING_DISABLED_KEY]);
        setTextOnlyIndexWarningDisabled(textOnlyWarningDisabled);

        const persistedIndexModelId = String((settingsData as any)?.[INDEX_MODEL_PREFERENCE_KEY] || '').trim();
        const persistedIndexModelLocked = parseBooleanSetting((settingsData as any)?.[INDEX_MODEL_MANUAL_LOCK_KEY]);
        const persistedChatModelLocked = parseBooleanSetting((settingsData as any)?.[CHAT_MODEL_MANUAL_LOCK_KEY]);
        if (persistedIndexModelLocked && persistedIndexModelId) {
          const persistedIndexModel = (modelsData as any[]).find(
            (m: any) => m?.id === persistedIndexModelId && m?.status === 'installed'
          ) || null;
          if (persistedIndexModel) {
            userPickedIndexModelRef.current = true;
            setSelectedIndexModel(persistedIndexModel);
          } else {
            userPickedIndexModelRef.current = false;
            void updateSettings(INDEX_MODEL_MANUAL_LOCK_KEY, false);
            void updateSettings(INDEX_MODEL_PREFERENCE_KEY, '');
          }
        } else {
          userPickedIndexModelRef.current = false;
        }

        const previouslySelected = modelsData.find((m: any) => m.selected && m.status === 'installed');
        const installed = modelsData.filter((m: any) => m.status === 'installed');

        if (previouslySelected) {
          setSelectedModel(previouslySelected);
        } else if (installed.length > 0) {
          setSelectedModel(installed[0]);
        } else {
          setSelectedModel(null);
        }
        if (persistedChatModelLocked && previouslySelected) {
          userPickedChatModelRef.current = true;
        } else if (persistedChatModelLocked && !previouslySelected) {
          userPickedChatModelRef.current = false;
          void updateSettings(CHAT_MODEL_MANUAL_LOCK_KEY, false);
        } else {
          userPickedChatModelRef.current = false;
        }
        setModelsHydrated(true);

        setConnectionError(false);

        if (Array.isArray(history)) {
          setConversations(history);
        }

        const sources = await fetchSources();
        if (!mounted) return;
        await yieldToMain();
        setSourcesLibrary(sources);
        const outIds = getAllIds(sources);
        setActiveSourceIds(outIds);

        try {
          const already = await checkOnboardingStatus();
          const rawCurrentStep = await getOnboardingStep();
          const currentStep = normalizeStartupStep(String(rawCurrentStep || 'welcome'), already) as any;
          if (!already && currentStep === 'welcome' && rawCurrentStep !== 'welcome') {
            void saveOnboardingStep('welcome');
          }

          const coreReady = Boolean((coreStatus as any)?.embedding?.installed) && Boolean((coreStatus as any)?.reranker?.installed);
          const startupReady = coreReady;
          const hasSources = Array.isArray(sources) && sources.length > 0;
          const allIndexed = hasSources && sources.every((s: any) => (s?.status || 'indexed') !== 'indexing');

          if (!already && currentStep === 'welcome') {
            onboardingFastTrackReadyRef.current = Boolean(allIndexed && startupReady);
          } else {
            onboardingFastTrackReadyRef.current = false;
          }

          if (!already) {
            if (allIndexed && startupReady && currentStep !== 'welcome') {
              await markOnboardingComplete();
              await saveOnboardingStep('complete');
              setIsOnboardingComplete(true);
              setOnboardingStep('complete');
            } else if (startupReady && !hasSources && currentStep !== 'welcome') {
              setOnboardingStep('indexing-guide');
              await saveOnboardingStep('indexing-guide');
            } else if (!startupReady) {
              setIsOnboardingComplete(false);
              // Don't override welcome or model-recommend — let user go through them first
              setOnboardingStep((prev) => {
                if (prev !== 'welcome' && prev !== 'model-recommend') {
                  void saveOnboardingStep('welcome');
                  return 'welcome';
                }
                return prev;
              });
            }
          } else if (!startupReady) {
            setIsOnboardingComplete(false);
            setOnboardingStep((prev) => {
              if (prev !== 'welcome' && prev !== 'model-recommend') {
                return 'welcome';
              }
              return prev;
            });
            await Promise.all([
              updateSettings('onboarding_complete', false),
              updateSettings('onboarding_step', 'welcome'),
            ]);
          }
        } catch {
          // ignore
        }

        try {
          await yieldToMain();
          const active = await getActiveIndexJob();
          const job = (active as any)?.job;
          const persisted = (active as any)?.persisted;
          const jobId = job?.job_id || persisted?.job_id;

          if (job || persisted) {
            const shouldIndex =
              jobId &&
              ((job && job.is_indexing && job.error !== 'cancelled') ||
                (persisted && persisted.is_indexing && persisted.error !== 'cancelled'));

            if (shouldIndex) {
              console.log('[App] Resuming active index job:', jobId);
              await attachToIndexJob(String(jobId), true);
            }
          }
        } catch (e) {
          console.warn('[App] Failed to resume active index job:', e);
        }

        try {
          await notifyUIReady();
          const updatedSources = await fetchSources();
          setSourcesLibrary(updatedSources);
          setActiveSourceIds(getAllIds(updatedSources));
        } catch (e) {
          console.warn('[App] Failed to notify UI ready or refresh sources:', e);
        }

        if (mounted) setIsBackendSyncing(false);
      } catch (e) {
        console.warn('Failed to fetch initial data:', e);
        if (!mounted) return;
        
        // Retry with exponential backoff (capped at 2 seconds)
        if (retryCount < maxRetries) {
          retryCount++;
          const delay = Math.min(baseRetryDelay * Math.pow(1.5, retryCount - 1), 2000);
          console.log(`Retrying in ${Math.round(delay)}ms... (${retryCount}/${maxRetries})`);
          setTimeout(loadData, delay);
        } else {
          setConnectionError(true);
          setIsBackendSyncing(false);
        }
      }
    };

    let raf1 = 0;
    let raf2 = 0;
    raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => {
        if (!mounted) return;
        void loadData();
      });
    });

    return () => {
      mounted = false;
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, [onboardingHydrated, normalizeStartupStep]);

  const startIndexingWithFolder = useCallback(async (folder: string) => {
    const indexModel = ensureIndexModelForIndexing();
    if (!indexModel) return;
    const indexModelPersisted = await persistIndexModelForIndexing(indexModel);
    if (!indexModelPersisted) return;

    setIsRightSidebarOpen(true);

    const startRes: any = await startIndex(folder);

    if (startRes?.appended) {
      console.log('[Sources] 文件夹已追加到索引队列:', folder);
      try {
        const sources = await fetchSources();
        setSourcesLibrary(sources);
        setActiveSourceIds(getAllIds(sources));
      } catch {}
      return;
    }

    const job_id = String(startRes?.job_id || '').trim();
    if (!job_id) {
      const err = String(startRes?.error || '');
      const msg = String(startRes?.message || startRes?.error || '启动索引失败，请稍后重试。');
      if (err === 'core_models_not_ready') {
        setIsOnboardingComplete(false);
        setOnboardingStep('welcome');
          void Promise.all([
            updateSettings('onboarding_complete', false),
            updateSettings('onboarding_step', 'welcome'),
          ]);
      }
      console.warn('[Sources] 启动索引失败:', msg);
      return;
    }
    let currentJobId = job_id;
    setIndexJobId(currentJobId);
    setIndexingState({
      isIndexing: true,
      totalFiles: 0,
      completedFiles: 0,
      eta: '—',
      isTopBarVisible: true,
      isCancelling: false,
      isRestoringModel: false,
      statusMessage: '',
      currentFrame: 0,
      totalFrames: 0,
      currentAudioSec: 0,
      totalAudioSec: 0,
      stageRate: 0,
      stage: '',
    });

    try {
      const sources = await fetchSources();
      setSourcesLibrary(sources);
      setActiveSourceIds(getAllIds(sources));
    } catch {}

    if (indexingIntervalRef.current != null) {
      window.clearTimeout(indexingIntervalRef.current);
      indexingIntervalRef.current = null;
    }

    const clearFolderPoll = () => {
      if (indexingIntervalRef.current != null) {
        window.clearTimeout(indexingIntervalRef.current);
        indexingIntervalRef.current = null;
      }
    };

    const scheduleFolderNext = () => {
      indexingIntervalRef.current = window.setTimeout(runFolderTick, INDEX_STATUS_POLL_MS);
    };

    const runFolderTick = async () => {
      indexingIntervalRef.current = null;
      try {
        const st = await getIndexStatus(currentJobId);
        const job = st?.job;
        if (!job) {
          try {
            const activeRes = await getActiveIndexJob();
            const nextJob = (activeRes as any)?.job;
            const persisted = (activeRes as any)?.persisted;
            if ((nextJob && nextJob.is_indexing) || (persisted && persisted.is_indexing)) {
              scheduleFolderNext();
              return;
            }
          } catch {
            scheduleFolderNext();
            return;
          }
          clearFolderPoll();
          setIndexJobId(null);
          setIndexingState(prev => ({
            ...prev,
            isIndexing: false,
            isRestoringModel: false,
            isTopBarVisible: prev.isCancelling ? true : false,
          }));
          await finalizeSourcesAfterIndex(setSourcesLibrary, setActiveSourceIds);
          return;
        }

        let isIndexing = Boolean(job.is_indexing);

        if (!isIndexing && job.error !== 'cancelled') {
           await new Promise(r => setTimeout(r, 500));
           try {
             const activeRes = await getActiveIndexJob();
             const nextJob = (activeRes as any)?.job;
             if (nextJob && nextJob.is_indexing && nextJob.job_id !== currentJobId) {
                 console.log('[Sources] 接力追踪队列后续任务:', nextJob.job_id);
                 currentJobId = nextJob.job_id;
                 setIndexJobId(currentJobId);
                 scheduleFolderNext();
                 return;
             }
           } catch(e) {}
        }

        setIndexingState(prev => ({
          ...prev,
          isIndexing,
          isRestoringModel: false,
          totalFiles: Number(job.total_files || 0),
          completedFiles: Math.min(Number(job.completed_files || 0), Number(job.total_files || 0)),
          eta: formatIndexEta(job),
          isTopBarVisible: job.is_indexing ? prev.isTopBarVisible : (prev.isCancelling ? true : false),
          statusMessage: buildLiveIndexStatusMessage(job),
          currentFile: job.current_file,
          currentPath: job.current_path,
          currentFrame: Number(job.current_frame || 0),
          totalFrames: Number(job.total_frames || 0),
          currentAudioSec: Number(job.current_audio_sec || 0),
          totalAudioSec: Number(job.total_audio_sec || 0),
          stageRate: Number(job.stage_rate || 0),
          stage: job.stage || '',
        }));

        if (job.is_indexing) {
          const indexedPaths: string[] = job.indexed_paths || [];
          if (indexedPaths.length > 0) {
            const indexedSet = new Set(indexedPaths);
            setSourcesLibrary(prev => {
              const markIndexed = (nodes: FileSource[]): FileSource[] =>
                nodes.map(n => {
                  const updated = { ...n };
                  if (n.path && indexedSet.has(n.path) && n.status !== 'indexed') {
                    updated.status = 'indexed';
                  }
                  if (n.children) {
                    updated.children = markIndexed(n.children);
                    const childStatuses = updated.children.map(c => c.status);
                    if (childStatuses.length > 0 && childStatuses.every(s => s === 'indexed')) {
                      updated.status = 'indexed';
                    } else if (childStatuses.some(s => s === 'indexing')) {
                      updated.status = 'indexing';
                    } else if (childStatuses.some(s => s === 'pending')) {
                      updated.status = 'pending';
                    }
                  }
                  return updated;
                });
              return markIndexed(prev);
            });
          }
          scheduleFolderNext();
          return;
        }

        clearFolderPoll();
        setIndexJobId(null);
        const totalFiles = Number(job.total_files || 0);
        const completedFiles = Number(job.completed_files || 0);
        const failedFiles = Number(job.failed_files || 0);

        if (job.error !== 'cancelled') {
          setIndexingState(prev => ({
            ...prev,
            isIndexing: false,
            isTopBarVisible: true,
            isCancelling: false,
            isRestoringModel: true,
            statusMessage: i18n.t('indexingWidget.switchingModelWait'),
            totalFiles,
            completedFiles,
          }));
        } else {
          setIndexingState(prev => ({
            ...prev,
            isIndexing: false,
            isRestoringModel: false,
            isTopBarVisible: prev.isCancelling ? true : false,
          }));
        }

        let modelSwitchCostMs = 0;
        const switchStart = Date.now();
        if (job.error !== 'cancelled') {
          try {
            console.log('[Sources] 文件夹索引完成，恢复原模型...');
            await restoreModelAfterIndexing();
          } catch (e) {
            console.warn('[Sources] 恢复原模型失败:', e);
            setIndexingState(prev => ({
              ...prev,
              isRestoringModel: false,
              isTopBarVisible: true,
              statusMessage: i18n.t('indexingWidget.switchModelFailedManual'),
            }));
            await finalizeSourcesAfterIndex(setSourcesLibrary, setActiveSourceIds);
            return;
          }
          modelSwitchCostMs = Math.max(0, Date.now() - switchStart);
          setIndexingState(prev => ({ ...prev, isRestoringModel: false }));
        }

        if (!isOnboardingComplete && onboardingStep === 'indexing-progress') {
          try {
            const onboardingChatModel = pickAutoModelForRole('chat', selectedModel);
            if (onboardingChatModel) {
              await switchModelAndSyncUI(onboardingChatModel.id);
              indexPrevModelIdRef.current = null;
              indexPrevQuantRef.current = null;
            }
          } catch (e) {
            console.warn('[Onboarding] 切换到聊天模型失败:', e);
          }
          await markOnboardingComplete();
          setIsOnboardingComplete(true);
          setOnboardingStep('complete');
          await saveOnboardingStep('complete');
        }
        if (job.error !== 'cancelled') {
          modelSwitchCostMs = Math.max(0, Date.now() - switchStart);
        }

        await finalizeSourcesAfterIndex(setSourcesLibrary, setActiveSourceIds);
        if (job.error !== 'cancelled') {
          setIndexingState(prev => ({
            ...prev,
            isIndexing: false,
            isTopBarVisible: false,
            isCancelling: false,
            isRestoringModel: false,
            statusMessage: '',
          }));
          await showIndexingCompleteDialog(
            totalFiles,
            completedFiles,
            failedFiles,
            String((job as any)?.job_id || currentJobId || ''),
            String(job.error || ''),
          );
        }
      } catch (e) {
        console.warn('index status poll failed', e);
        scheduleFolderNext();
      }
    };

    indexingIntervalRef.current = window.setTimeout(runFolderTick, 0);
  }, [indexingState.isIndexing, restoreModelAfterIndexing, isOnboardingComplete, onboardingStep, switchModelAndSyncUI, showIndexingCompleteDialog, pickAutoModelForRole, selectedModel, ensureIndexModelForIndexing, persistIndexModelForIndexing]);

  const handleStartIndexing = useCallback(async () => {
    if (isModelSwitching) return;
    const isAlreadyIndexing = indexingState.isIndexing;
    const indexModel = ensureIndexModelForIndexing();
    if (!indexModel) return;
    const folder = await selectFolder();
    if (!folder) return;
    const confirmed = await confirmTextOnlyIndexWarningIfNeeded(indexModel);
    if (!confirmed) return;
    const indexModelPersisted = await persistIndexModelForIndexing(indexModel);
    if (!indexModelPersisted) return;

    try {
      const { confirm } = await import('@tauri-apps/plugin-dialog');
      const hasMedia = await checkMediaFiles(folder);
      const messageKey = hasMedia ? 'confirmIndexDialog.messageMediaFolder' : 'confirmIndexDialog.messageFolder';
      const shouldIndex = await confirm(i18n.t(messageKey), { title: i18n.t('confirmIndexDialog.title'), kind: 'info' });
      if (!shouldIndex) return;
    } catch (e) {
      console.warn('Failed to show confirm dialog', e);
    }

    if (isAlreadyIndexing) {
      console.log('[Sources] 索引进行中，追加文件夹到队列:', folder);
      try {
        await startIndexingWithFolder(folder);
      } catch (e) {
        console.warn('[Sources] 追加文件夹失败:', e);
      }
      return;
    }

    try {
      await startIndexingWithFolder(folder);
    } catch (e) {
      console.warn('[Sources] 启动索引失败:', e);
    }
  }, [isModelSwitching, ensureIndexModelForIndexing, indexingState.isIndexing, startIndexingWithFolder, confirmTextOnlyIndexWarningIfNeeded, persistIndexModelForIndexing]);

  const dismissIndexCancelConfirm = useCallback(() => {
    setIndexCancelConfirmOpen(false);
  }, []);

  const confirmCancelIndexing = useCallback(async () => {
    setIndexCancelConfirmOpen(false);
    const jobIdAtCancel = indexJobId;
    const fallbackTotal = Number(indexingState.totalFiles || 0);
    const fallbackCompleted = Number(indexingState.completedFiles || 0);
    setIndexingState(prev => ({
      ...prev,
      isCancelling: true,
      isRestoringModel: false,
      isTopBarVisible: true,
      statusMessage: i18n.t('indexingWidget.cancelling'),
    }));
    try {
      let cancelErr: string | null = null;
      try {
        const res = await cancelIndex(indexJobId);
        if ((res as any)?.ok === false) {
          const err = String((res as any)?.error || 'cancel_failed');
          if (err === 'job_id_mismatch') {
            const retry = await cancelIndex(null);
            if ((retry as any)?.ok === false) {
              const retryErr = String((retry as any)?.error || 'cancel_failed');
              if (retryErr !== 'no_active_job') cancelErr = retryErr;
            }
          } else if (err !== 'no_active_job') {
            cancelErr = err;
          }
        }
      } catch (e: any) {
        cancelErr = String(e?.message || e || 'cancel_failed');
      }

      const finalCounts = await loadFinalIndexCounts(jobIdAtCancel, fallbackTotal, fallbackCompleted);
      if (cancelErr && finalCounts.isIndexing) {
        throw new Error(cancelErr);
      }
      try {
        await restoreModelAfterIndexing();
      } catch (e) {
        console.warn('restore model after cancel failed:', e);
      }
      await showCancelSummaryDialog(finalCounts.total, finalCounts.completed, finalCounts.failed);
      setIndexingState(prev => ({
        ...prev,
        isIndexing: false,
        isTopBarVisible: false,
        isCancelling: false,
        isRestoringModel: false,
        statusMessage: '',
      }));
    } catch (e) {
      console.warn('cancel indexing failed:', e);
      const finalCounts = await loadFinalIndexCounts(jobIdAtCancel, fallbackTotal, fallbackCompleted);
      if (!finalCounts.isIndexing) {
        try {
          await restoreModelAfterIndexing();
        } catch (restoreErr) {
          console.warn('restore model after cancel failed (catch fallback):', restoreErr);
        }
        await showCancelSummaryDialog(finalCounts.total, finalCounts.completed, finalCounts.failed);
        setIndexingState(prev => ({
          ...prev,
          isIndexing: false,
          isTopBarVisible: false,
          isCancelling: false,
          isRestoringModel: false,
          statusMessage: '',
        }));
      } else {
        setIndexingState(prev => ({
          ...prev,
          isCancelling: true,
          isRestoringModel: false,
          isTopBarVisible: true,
          statusMessage: i18n.t('indexingWidget.cancelling'),
        }));
      }
    }
    fetchSources().then(sources => {
      setSourcesLibrary(sources);
    }).catch(() => {/* ignore */});
  }, [
    indexJobId,
    indexingState.totalFiles,
    indexingState.completedFiles,
    loadFinalIndexCounts,
    restoreModelAfterIndexing,
    showCancelSummaryDialog,
  ]);

  const handleCloseIndexingTopBar = useCallback(() => {
    if (indexingState.isRestoringModel) {
      return;
    }
    if (indexingState.isIndexing) {
      setIndexCancelConfirmOpen(true);
      return;
    }
    setIndexingState(prev => ({ ...prev, isTopBarVisible: false }));
  }, [indexingState.isIndexing, indexingState.isRestoringModel]);

  useEffect(() => {
    if (!indexCancelConfirmOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        setIndexCancelConfirmOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [indexCancelConfirmOpen]);

  useEffect(() => {
    if (!pendingDeleteChatId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        setPendingDeleteChatId(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [pendingDeleteChatId]);


  // --- Model Management ---
  
  // Poll models status when modal is open or downloads in progress
  const hasActiveDownloads = models.some(m => m.status === 'downloading');
  const shouldPollModels = isManageModelsOpen || hasActiveDownloads;

  useEffect(() => {
    if (!shouldPollModels) {
      if (modelPollIntervalRef.current) {
        window.clearInterval(modelPollIntervalRef.current);
        modelPollIntervalRef.current = null;
      }
      refreshModels();
      const delayedRefresh = window.setTimeout(refreshModels, 1500);
      return () => window.clearTimeout(delayedRefresh);
    }

    refreshModels();
    const id = window.setInterval(refreshModels, 500);
    modelPollIntervalRef.current = id;

    return () => {
      window.clearInterval(id);
      modelPollIntervalRef.current = null;
    };
  }, [shouldPollModels, refreshModels]);

  const handleDownloadModel = useCallback(async (id: string, source: string, quantizationFile?: string) => {
    // Optimistic update
    setModels(prev => prev.map(m => m.id === id ? { ...m, status: 'downloading', progress: 0, downloadProgress: 0 } : m));
    try {
      await downloadModel(id, source, quantizationFile);
      refreshModels(); // Refresh to get real job status
    } catch (e) {
      console.error("Download failed", e);
      refreshModels();
    }
  }, [refreshModels]);

  const handleCancelDownloadModel = useCallback(async (id: string) => {
    try {
      await cancelDownloadModel(id);
      refreshModels();
    } catch (e) {
      console.error("Cancel failed", e);
    }
  }, [refreshModels]);

  const handleDeleteModel = useCallback(async (id: string, quantizationFile?: string) => {
    try {
      const res = await deleteModel(id, quantizationFile);
      if (res?.ok === false) {
        const detail = String(res?.error || 'Unknown error');
        window.alert(`删除模型失败：${detail}`);
      }
      refreshModels();
    } catch (e) {
      console.error("Delete failed", e);
      window.alert(`删除模型失败：${e instanceof Error ? e.message : String(e)}`);
    }
  }, [refreshModels]);

  const handleSelectModelQuantization = useCallback(async (id: string, quantizationFile: string) => {
    try {
      await selectModelQuantization(id, quantizationFile);
      refreshModels();
    } catch (e) {
      console.error("Select quantization failed", e);
    }
  }, [refreshModels]);


  // --- Actions ---

  const handleNewChat = useCallback(() => {
    setActiveConversationId(null);
    setInputValue('');
  }, []);

  // Keyboard Shortcuts
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'n') {
        e.preventDefault();
        handleNewChat();
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [handleNewChat]);

  const handleDeleteChat = useCallback((id: string) => {
    setPendingDeleteChatId(id);
  }, []);

  const dismissDeleteChatConfirm = useCallback(() => {
    setPendingDeleteChatId(null);
  }, []);

  const confirmDeleteChat = useCallback(async () => {
    const id = pendingDeleteChatId;
    if (!id) return;
    setPendingDeleteChatId(null);
    try {
      await deleteHistory(id);
      setConversations(prev => prev.filter(c => c.id !== id));
      if (activeConversationId === id) {
        setActiveConversationId(null);
        setInputValue('');
      }
    } catch (e) {
      console.warn('delete chat failed', e);
    }
  }, [activeConversationId, pendingDeleteChatId]);

  const handleSelectChat = useCallback((id: string) => {
    setActiveConversationId(id);
  }, []);

  const handleSelectModel = useCallback(async (model: Model) => {
    if (indexingState.isIndexing || isModelSwitching) return;
    userPickedChatModelRef.current = true;
    void updateSettings(CHAT_MODEL_MANUAL_LOCK_KEY, true);
    try {
      await switchModelAndSyncUI(model.id);
    } catch (e) {
      console.warn('Failed to select model:', e);
    }
  }, [indexingState.isIndexing, isModelSwitching, switchModelAndSyncUI]);

  const handleSendMessage = useCallback(async () => {
    if (!inputValue.trim() || isGenerating || sendInFlightRef.current || isModelSwitching) return;
    if (!modelsHydrated) return;
    if (!hasInstalledModels || !selectedModel) {
      setShowModelPrompt(true);
      return;
    }

    // Create abort controller for this generation
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    sendInFlightRef.current = true;
    setIsGenerating(true);

    const userText = inputValue;
    const isSearchQuery = /find|search|summary|risk|documents/i.test(userText);

    const newUserMsg: Message = {
      id: uuidv4(),
      role: 'user',
      content: userText,
      timestamp: Date.now()
    };

    let currentConvId = activeConversationId;
    let newConversations = [...conversations];

    if (!currentConvId) {
      const newConv: Conversation = {
        id: uuidv4(),
        title: userText.slice(0, 30) + (userText.length > 30 ? '...' : ''),
        messages: [],
        lastActive: Date.now()
      };
      newConversations = [newConv, ...newConversations];
      currentConvId = newConv.id;
      setActiveConversationId(newConv.id);
    }

    newConversations = newConversations.map(c => {
      if (c.id === currentConvId) {
        return { ...c, messages: [...c.messages, newUserMsg], lastActive: Date.now() };
      }
      return c;
    });
    setConversations(newConversations);
    setInputValue('');
    
    // Sync user message immediately
    const convToSync = newConversations.find(c => c.id === currentConvId);
    if (convToSync) {
      syncHistory(convToSync).catch(console.warn);
    }

    const responseId = uuidv4();

    const initialAiMsg: Message = {
      id: responseId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      statusText: 'Thinking...',
      model: selectedModel || undefined,
      ...(isSearchQuery ? { isSearch: true, scanState: 'thinking', scanProgress: 0, relevantFiles: [] } : { relevantFiles: [] })
    };
    setConversations(prev => prev.map(c => 
      c.id === currentConvId ? { ...c, messages: [...c.messages, initialAiMsg] } : c
    ));

    if (isSearchQuery) {
      setIsRightSidebarOpen(true);
    }

    // --- Compute Effective Source IDs for Backend ---
    const getEffectiveSourceIds = (nodes: FileSource[], activeIds: string[]): string[] => {
      let effective: string[] = [];
      
      const getSelectionState = (node: FileSource): 'checked' | 'unchecked' | 'indeterminate' => {
        if (!node.children || node.children.length === 0) {
          return activeIds.includes(node.id) ? 'checked' : 'unchecked';
        }
        const childStates = node.children.map(getSelectionState);
        if (childStates.every(s => s === 'checked')) return 'checked';
        if (childStates.every(s => s === 'unchecked')) return 'unchecked';
        return 'indeterminate';
      };

      const traverse = (node: FileSource) => {
        const state = getSelectionState(node);
        // CRITICAL FIX: To prevent backend's `startswith` matching from including
        // unselected files in the same folder, we MUST extract exactly the leaf file IDs,
        // even if the entire folder is checked. 
        if (state === 'checked') {
          const collectLeaves = (n: FileSource) => {
            if (!n.children || n.children.length === 0) {
              if (isIndexedKbFile(n)) {
                effective.push(n.id);
              }
            } else {
              n.children.forEach(collectLeaves);
            }
          };
          collectLeaves(node);
        } else if (state === 'indeterminate' && node.children) {
          node.children.forEach(traverse);
        }
      };

      nodes.forEach(traverse);
      return Array.from(new Set(effective));
    };

    const getDirectSelectedIndexedLeafIds = (nodes: FileSource[], activeIds: string[]): string[] => {
      const selected = new Set(activeIds);
      const out: string[] = [];
      const walk = (node: FileSource) => {
        if (!node.children || node.children.length === 0) {
          if (isIndexedKbFile(node) && selected.has(node.id)) {
            out.push(node.id);
          }
          return;
        }
        node.children.forEach(walk);
      };
      nodes.forEach(walk);
      return Array.from(new Set(out));
    };

    let effectiveSourceIds = dedupeEffectiveSourceIdsByPath(
      sourcesLibrary,
      getEffectiveSourceIds(sourcesLibrary, activeSourceIds),
    );
    if (effectiveSourceIds.length === 0 && activeSourceIds.length > 0) {
      const fallbackIds = dedupeEffectiveSourceIdsByPath(
        sourcesLibrary,
        getDirectSelectedIndexedLeafIds(sourcesLibrary, activeSourceIds),
      );
      if (fallbackIds.length > 0) {
        effectiveSourceIds = fallbackIds;
      }
    }
    const langRaw = String(i18n.resolvedLanguage || i18n.language || 'en').toLowerCase();
    const promptLanguage = langRaw.startsWith('zh') ? 'zh' : 'en';

    try {
      
      let accumulatedContent = '';
      let rafId: number | null = null;
      let pendingUpdate = false;
      let finalizedContentFromDone: string | null = null;
      let hasRenderedFirstTextChunk = false;

      const applyContentImmediately = (content: string) => {
        setConversations(prev => prev.map(c => {
          if (c.id !== currentConvId) return c;
          return {
            ...c,
            messages: c.messages.map(m => m.id === responseId ? { ...m, content } : m)
          };
        }));
      };
      
      const scheduleUpdate = () => {
        if (pendingUpdate) return;
        pendingUpdate = true;
        
        rafId = requestAnimationFrame(() => {
          pendingUpdate = false;
          const content = accumulatedContent;
          
          setConversations(prev => prev.map(c => {
            if (c.id !== currentConvId) return c;
            return {
              ...c,
              messages: c.messages.map(m => m.id === responseId ? { ...m, content } : m)
            };
          }));
        });
      };

      await queryBackendStream(
        {
          message: userText,
          active_source_ids: effectiveSourceIds,
          model_id: selectedModel.id,
          session_id: currentConvId || undefined,
          language: promptLanguage,
          opened_file_path: sidebarMode === 'openedFile' && openedFiles.length > 0 ? openedFiles[openedFiles.length - 1].file_path : undefined,
        },
        (event, data) => {
          console.log('[queryBackendStream]', event, data);
          if (event === 'status') {
            const txt = (data as any)?.message || 'Processing...';
            setConversations(prev => prev.map(c => {
              if (c.id !== currentConvId) return c;
              return {
                ...c,
                messages: c.messages.map(m => m.id === responseId ? { ...m, statusText: txt } : m)
              };
            }));
            return;
          }
          if (event === 'thinking') {
            const delta = (data as any)?.delta || '';
            if (!delta) return;
            setConversations(prev => prev.map(c => {
              if (c.id !== currentConvId) return c;
              return {
                ...c,
                messages: c.messages.map(m => {
                  if (m.id !== responseId) return m;
                  return { ...m, thinkingContent: (m.thinkingContent || '') + delta };
                })
              };
            }));
            return;
          }
          if (event === 'trace_append') {
            const item = (data as any)?.item;
            if (!item) return;
            setConversations(prev => prev.map(c => {
              if (c.id !== currentConvId) return c;
              return {
                ...c,
                messages: c.messages.map(m => {
                  if (m.id !== responseId) return m;
                  const next = [...((m as any).trace || []), item];
                  return { ...m, trace: next };
                })
              };
            }));
            return;
          }
          if (event === 'files') {
            const preview = ((data as any)?.preview || []) as any[];
            const all = ((data as any)?.all || []) as any[];
            const parsedTotal = Number((data as any)?.total_matches ?? (data as any)?.total ?? all.length ?? 0);
            const parsedShown = Number((data as any)?.shown_count ?? all.length ?? 0);
            const totalMatches = Number.isFinite(parsedTotal) ? Math.max(0, parsedTotal) : all.length;
            const shownCount = Number.isFinite(parsedShown) ? Math.max(0, parsedShown) : all.length;
            setSearchRelevantFiles(
              preview.map((f: any) => ({
                id: f.file_path || f.file_name || f.id,
                name: f.file_name || f.name,
                type: (f.iconType || f.type || 'doc') as any,
                path: f.file_path || f.path,
                tree_path: f.tree_path != null && f.tree_path !== '' ? String(f.tree_path) : undefined,
                from_folder_chain: Boolean(f.from_folder_chain),
                is_matched_folder: Boolean(f.is_matched_folder),
                child_file_count: typeof f.child_file_count === 'number' ? f.child_file_count : undefined,
                folder_chain_root: f.folder_chain_root ? String(f.folder_chain_root) : undefined,
                doc_summary: f.doc_summary != null ? String(f.doc_summary) : '',
              })),
            );
            setConversations(prev => prev.map(c => {
              if (c.id !== currentConvId) return c;
              return {
                ...c,
                messages: c.messages.map(m => {
                  if (m.id !== responseId) return m;
                  const toRF = (x: any) => ({
                    id: x.file_path || x.path || x.file_name || x.id,
                    name: x.file_name || x.name,
                    type: (x.iconType || x.type || 'doc'),
                    path: x.file_path || x.path,
                    tree_path: x.tree_path != null && x.tree_path !== '' ? String(x.tree_path) : undefined,
                    from_folder_chain: Boolean(x.from_folder_chain),
                    is_matched_folder: Boolean(x.is_matched_folder),
                    child_file_count: typeof x.child_file_count === 'number' ? x.child_file_count : undefined,
                    folder_chain_root: x.folder_chain_root ? String(x.folder_chain_root) : undefined,
                    doc_summary: x.doc_summary != null ? String(x.doc_summary) : '',
                  });
                  return {
                    ...m,
                    relevantFiles: preview.map(toRF),
                    relevantFilesAll: all.map(toRF),
                    relevantFilesTotal: Math.max(totalMatches, all.length),
                    relevantFilesShown: Math.max(Math.min(shownCount, Math.max(totalMatches, all.length)), all.length),
                  };
                })
              };
            }));
            return;
          }
          if (event === 'opened_file') {
            const file = (data as any)?.file;
            const content = (data as any)?.content ?? '';
            const truncated = Boolean((data as any)?.truncated);
            if (file?.file_path) {
              const of: OpenedFile = {
                file_path: String(file.file_path),
                file_name: String(file.file_name || file.name || file.file_path),
                iconType: (file.iconType || file.type || 'doc'),
                content: String(content),
                truncated,
                openedAt: Date.now()
              };
              setOpenedFiles(prev => {
                const next = prev.filter(x => x.file_path !== of.file_path);
                return [of, ...next].slice(0, 20);
              });
              setActiveOpenedFilePath(of.file_path);
              setSidebarMode('openedFile');
              setIsRightSidebarOpen(true);
            }
            return;
          }
          if (event === 'text') {
            const delta = (data as any)?.delta ?? (data as any)?.content ?? '';
            if (!delta) return;
            accumulatedContent += delta;
            if (!hasRenderedFirstTextChunk) {
              hasRenderedFirstTextChunk = true;
              if (rafId) {
                cancelAnimationFrame(rafId);
                rafId = null;
              }
              pendingUpdate = false;
              applyContentImmediately(accumulatedContent);
              return;
            }
            scheduleUpdate();
            return;
          }
          if (event === 'done') {
            const isError = (data as any)?.ok === false;
            const queryType = String((data as any)?.query_type || '').toLowerCase();
            const rawErrorMsg = String((data as any)?.error || (data as any)?.message || '');
            const normalizedRaw = rawErrorMsg.trim().toLowerCase();
            const isInterrupted =
              queryType === 'interrupted' ||
              normalizedRaw === '生成已中断' ||
              normalizedRaw === 'generation interrupted';
            const localizedError = isInterrupted
              ? t('chat.generationInterrupted')
              : (rawErrorMsg || t('chat.unknownError'));
            
            let finalContent = accumulatedContent;
            if (isError) {
                if (isInterrupted) {
                    finalContent = accumulatedContent ? `${accumulatedContent}\n\n[${localizedError}]` : localizedError;
                } else {
                    finalContent = accumulatedContent ? `${accumulatedContent}\n\n⚠️ ${t('chat.errorOccurred', { message: localizedError })}` : `⚠️ ${t('chat.errorOccurred', { message: localizedError })}`;
                }
            }
            finalizedContentFromDone = finalContent;
            
            if (rafId) {
              cancelAnimationFrame(rafId);
              rafId = null;
            }
            
            setConversations(prev => {
              const nextState = prev.map(c => {
                if (c.id !== currentConvId) return c;
                return {
                  ...c,
                  messages: c.messages.map(m => {
                    if (m.id !== responseId) return m;
                    const base: any = { ...m, content: finalContent || (isError ? t('chat.unknownError') : ''), statusText: '' };
                    if (isSearchQuery) base.scanState = 'completed', base.scanProgress = 100;
                    return base;
                  })
                };
              });
              
              // Sync final state (AI response complete)
              const finalConv = nextState.find(c => c.id === currentConvId);
              if (finalConv) {
                syncHistory(finalConv).catch(console.warn);
              }
              
              return nextState;
            });
          }
        },
        abortController.signal
      );
      
      if (rafId) {
        cancelAnimationFrame(rafId);
      }
      if (finalizedContentFromDone === null) {
        setConversations(prev => prev.map(c => {
          if (c.id !== currentConvId) return c;
          return {
            ...c,
            messages: c.messages.map(m => m.id === responseId ? { ...m, content: accumulatedContent } : m)
          };
        }));
      }
    } catch (e: any) {
      const detail = e?.message || String(e);
      const isInterrupted = e?.name === 'AbortError' || detail.includes('interrupted') || detail.includes('中断');
      const msg = isInterrupted
        ? t('chat.generationInterrupted')
        : t('chat.backendRequestFailed', { message: detail });
        
      setConversations(prev => {
        const nextState = prev.map(c => {
          if (c.id !== currentConvId) return c;
          return {
            ...c,
            messages: c.messages.map(m => {
              if (m.id !== responseId) return m;
              const currentContent = String(m.content || '');
              let appendedContent = currentContent;
              if (isInterrupted) {
                 appendedContent = currentContent ? `${currentContent}\n\n[${msg}]` : msg;
              } else {
                 appendedContent = currentContent ? `${currentContent}\n\n⚠️ ${msg}` : msg;
              }
              return { 
                ...m, 
                content: appendedContent, 
                ...(isSearchQuery ? { scanState: 'completed' as const, scanProgress: 100 } : {}) 
              };
            })
          };
        });
        
        // Sync error state
        const errConv = nextState.find(c => c.id === currentConvId);
        if (errConv) syncHistory(errConv).catch(console.warn);
        
        return nextState;
      });
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
      sendInFlightRef.current = false;
    }
  }, [inputValue, activeConversationId, conversations, activeSourceIds, sourcesLibrary, selectedModel?.id, isGenerating, isModelSwitching, modelsHydrated, hasInstalledModels, selectedModel]);

  const handleStopGenerating = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    sendInFlightRef.current = false;
    setIsGenerating(false);
  }, []);

  const handleAddSources = () => {
    if (!modelsHydrated || isModelSwitching) return;
    if (!hasInstalledModels) {
      setShowModelPrompt(true);
      return;
    }
    handleStartIndexing();
  };

  useEffect(() => {
    setupDownloadErrorsRef.current = setupDownloadErrors;
  }, [setupDownloadErrors]);

  const triggerSetupModelDownload = useCallback(async (modelId: string, quantizationFile: string) => {
    try {
      const res = await downloadModel(modelId, 'auto', quantizationFile);
      const ok = res?.ok !== false;
      const err = String(res?.error || '');
      const alreadyRunning = err.toLowerCase().includes('already in progress');
      if (ok || alreadyRunning) {
        setSetupDownloadErrors(prev => {
          if (!prev[modelId]) return prev;
          const next = { ...prev };
          delete next[modelId];
          return next;
        });
        return;
      }
      setSetupDownloadErrors(prev => ({ ...prev, [modelId]: err || '下载任务启动失败' }));
    } catch (e: any) {
      const msg = e?.message || String(e) || '下载任务启动失败';
      setSetupDownloadErrors(prev => ({ ...prev, [modelId]: msg }));
    }
  }, []);

  const triggerSetupAsrDownload = useCallback(async () => {
    const key = 'asr';
    try {
      const res = await startAsrModelDownload();
      const ok = res?.ok !== false;
      const err = String(res?.error || '');
      const alreadyRunning = err.toLowerCase().includes('already') || Boolean(res?.already_running);
      if (ok || alreadyRunning) {
        setSetupDownloadErrors(prev => {
          if (!prev[key]) return prev;
          const next = { ...prev };
          delete next[key];
          return next;
        });
        return;
      }
      setSetupDownloadErrors(prev => ({ ...prev, [key]: err || 'ASR 下载任务启动失败' }));
    } catch (e: any) {
      const msg = e?.message || String(e) || 'ASR 下载任务启动失败';
      setSetupDownloadErrors(prev => ({ ...prev, [key]: msg }));
    }
  }, []);

  const buildSetupItems = useCallback((core: any, mm: any[], asrStatus?: any): DownloadItemInfo[] => {
    const errors = setupDownloadErrorsRef.current;
    const asPositiveNumber = (value: unknown): number => {
      const n = Number(value);
      return Number.isFinite(n) && n > 0 ? n : 0;
    };
    const clampPercent = (value: unknown): number => {
      const n = Number(value);
      if (!Number.isFinite(n)) return 0;
      return Math.max(0, Math.min(100, n));
    };
    const resolveQuantSize = (model: any, quantFile: string): number => {
      if (!quantFile) return 0;
      const q = Array.isArray(model?.quantizations)
        ? model.quantizations.find((it: any) => String(it?.file || '') === quantFile)
        : null;
      return asPositiveNumber(q?.size_bytes) || asPositiveNumber(model?.file_sizes?.[quantFile]);
    };
    const mergeVlCompositeProgress = (model: any, quantFile: string) => {
      const mmprojFile = Array.isArray(model?.files)
        ? model.files.find((f: any) => String(f || '').toLowerCase().includes('mmproj'))
        : '';
      const mmprojSize = asPositiveNumber(model?.file_sizes?.[mmprojFile || '']);
      const quantSize = resolveQuantSize(model, quantFile);
      const currentTotal = asPositiveNumber(model?.total_bytes);
      const currentDownloaded = Math.min(
        asPositiveNumber(model?.downloaded_bytes),
        currentTotal > 0 ? currentTotal : Number.MAX_SAFE_INTEGER
      );
      const compositeTotal = mmprojSize + quantSize;

      if (compositeTotal <= 0) {
        const fallbackTotal = currentTotal;
        const fallbackDownloaded = Math.min(currentDownloaded, fallbackTotal || currentDownloaded);
        const fallbackPercent = fallbackTotal > 0
          ? clampPercent((fallbackDownloaded / fallbackTotal) * 100)
          : clampPercent(model?.progress ?? 0);
        return {
          percent: fallbackPercent,
          downloaded_bytes: fallbackDownloaded,
          total_bytes: fallbackTotal,
        };
      }

      if (currentTotal > 0) {
        const nearComposite = Math.abs(currentTotal - compositeTotal) <= Math.max(8 * 1024 * 1024, compositeTotal * 0.06);
        if (nearComposite) {
          const normalizedDownloaded = Math.min(Math.max(0, currentDownloaded), currentTotal);
          return {
            percent: clampPercent((normalizedDownloaded / currentTotal) * 100),
            downloaded_bytes: normalizedDownloaded,
            total_bytes: currentTotal,
          };
        }
      }

      let compositeDownloaded = currentDownloaded;
      if (currentTotal > 0 && quantSize > 0 && mmprojSize > 0) {
        const tolMain = Math.max(5 * 1024 * 1024, quantSize * 0.05);
        const tolMmproj = Math.max(3 * 1024 * 1024, mmprojSize * 0.08);
        const isMainStage = Math.abs(currentTotal - quantSize) <= tolMain || currentTotal > mmprojSize * 1.2;
        const isMmprojStage = Math.abs(currentTotal - mmprojSize) <= tolMmproj;
        if (isMainStage) {
          compositeDownloaded = mmprojSize + currentDownloaded;
        } else if (isMmprojStage) {
          compositeDownloaded = currentDownloaded;
        }
      }

      const normalizedDownloaded = Math.min(Math.max(0, compositeDownloaded), compositeTotal);
      return {
        percent: clampPercent((normalizedDownloaded / compositeTotal) * 100),
        downloaded_bytes: normalizedDownloaded,
        total_bytes: compositeTotal,
      };
    };

    const coreItem = (label: string, c: any, errorKey?: string): DownloadItemInfo => {
      const startErr = errorKey ? errors[errorKey] : '';
      const rawStatus = String(c?.status ?? 'idle') as DownloadItemInfo['status'];
      const normalizedStatus: DownloadItemInfo['status'] = (rawStatus === 'installed' && !Boolean(c?.installed))
        ? 'idle'
        : rawStatus;
      const status: DownloadItemInfo['status'] = (startErr && normalizedStatus !== 'installed' && normalizedStatus !== 'downloading')
        ? 'error'
        : normalizedStatus;
      return {
        label,
        status,
        percent: c?.percent,
        speed: c?.speed,
        eta: c?.eta,
        downloaded_bytes: c?.downloaded_bytes,
        total_bytes: c?.total_bytes,
        error: c?.error || startErr,
      };
    };
    const modelItem = (label: string, id: string, quantizationFile?: string): DownloadItemInfo => {
      const m = (mm as any[]).find((x: any) => x.id === id);
      const startErr = errors[id];
      if (!m) {
        return startErr ? { label, status: 'error', error: startErr } : { label, status: 'idle' };
      }
      if (m.status === 'installed') {
        return { label, status: 'installed', percent: 100 };
      }
      if (m.status === 'downloading') {
        if (id === VL_MODEL_ID) {
          const qf = String(
            quantizationFile
              || m.selected_quantization
              || m.default_quantization
              || m.quantizations?.[0]?.file
              || ''
          );
          const merged = mergeVlCompositeProgress(m, qf);
          return {
            label,
            status: 'downloading',
            percent: merged.percent,
            speed: m.download_speed ?? m.speed,
            eta: m.eta_seconds ?? m.eta,
            downloaded_bytes: merged.downloaded_bytes,
            total_bytes: merged.total_bytes,
          };
        }
        return {
          label,
          status: 'downloading',
          percent: Number(m.progress ?? 0) || 0,
          speed: m.download_speed ?? m.speed,
          eta: m.eta_seconds ?? m.eta,
          downloaded_bytes: m.downloaded_bytes,
          total_bytes: m.total_bytes,
        };
      }
      if (m.status === 'error') {
        return {
          label,
          status: 'error',
          percent: Number(m.progress ?? 0) || 0,
          error: m.error || startErr || '下载失败',
          downloaded_bytes: m.downloaded_bytes,
          total_bytes: m.total_bytes,
        };
      }
      if (startErr) {
        return { label, status: 'error', error: startErr };
      }
      if (m.status === 'available') {
        return { label, status: 'available', percent: Number(m.progress ?? 0) || 0 };
      }
      return { label, status: 'idle', percent: Number(m.progress ?? 0) || 0 };
    };

    return [
      coreItem('Embedding', core?.embedding),
      coreItem('Reranker', core?.reranker),
      coreItem('Whisper ASR', asrStatus?.asr ?? asrStatus, 'asr'),
      modelItem('AI Model (Gemma 4B)', UNIFIED_MODEL_ID, UNIFIED_MODEL_QF),
    ];
  }, []);

  const computeSetupProgress = useCallback((items: DownloadItemInfo[]) => {
    if (!items.length) return 0;
    const total = items.reduce((sum, item) => {
      if (item.status === 'installed') return sum + 100;
      if (item.status === 'downloading') return sum + (item.percent ?? 0);
      return sum;
    }, 0);
    return Math.round(total / items.length);
  }, []);

  const startSetupPolling = useCallback((coreOnly: boolean) => {
    if (setupIntervalRef.current) window.clearInterval(setupIntervalRef.current);
    const runTick = async () => {
      if (setupTickRunningRef.current) return;
      setupTickRunningRef.current = true;
      try {
        const [core, mm, asr] = await Promise.all([fetchCoreModelsStatus(), fetchModels(), fetchAsrModelStatus()]);
        const modelRows = (mm as any[]);

        const isOnlineHint = typeof navigator === 'undefined' ? true : navigator.onLine !== false;
        const now = Date.now();
        const cooldownMs = 6000;

        const corePending = !Boolean((core as any)?.embedding?.installed) || !Boolean((core as any)?.reranker?.installed);
        const asrItem = (asr as any)?.asr ?? asr;
        const asrPending = !Boolean(asrItem?.installed);
        const modelStatus = (modelId: string) => String(modelRows.find((x: any) => x?.id === modelId)?.status || '');
        const modelPending = (modelId: string) => modelStatus(modelId) !== 'installed';

        const canKick = (key: 'core' | 'asr' | 'vl' | 'chat') => (now - setupAutoRetryAtRef.current[key]) >= cooldownMs;
        const markKick = (key: 'core' | 'asr' | 'vl' | 'chat') => {
          setupAutoRetryAtRef.current[key] = now;
        };

        if (!isOnlineHint) {
          setupWasOfflineRef.current = true;
          if (!setupOfflineAbortIssuedRef.current) {
            setupOfflineAbortIssuedRef.current = true;
            cancelCoreModelsDownload().catch(() => {});
            cancelAsrModelDownload().catch(() => {});
            if (!coreOnly) {
              cancelDownloadModel(UNIFIED_MODEL_ID).catch(() => {});
            }
          }
        } else if (setupWasOfflineRef.current) {
          setupWasOfflineRef.current = false;
          setupOfflineAbortIssuedRef.current = false;
          setupAutoRetryAtRef.current = { core: 0, asr: 0, vl: 0, chat: 0 };
          if (corePending) {
            markKick('core');
            startCoreModelsDownload().catch(() => {});
          }
          if (asrPending) {
            markKick('asr');
            void triggerSetupAsrDownload();
          }
          if (!coreOnly) {
            if (modelPending(UNIFIED_MODEL_ID)) {
              markKick('vl');
              void triggerSetupModelDownload(UNIFIED_MODEL_ID, UNIFIED_MODEL_QF);
            }
          }
        }

        if (isOnlineHint) {
          if (corePending && canKick('core')) {
            markKick('core');
            startCoreModelsDownload().catch(() => {});
          }
          const asrWorkerActive = String(asrItem?.status || '') === 'downloading' || Boolean((asr as any)?.is_downloading);
          if (asrPending && !asrWorkerActive && canKick('asr')) {
            markKick('asr');
            void triggerSetupAsrDownload();
          }
          if (!coreOnly) {
            // Guard against re-triggering download when Gemma just finished:
            // - 'downloading' means a worker is already active, no need to kick again
            // - progress >= 100 means the download worker completed; the job is still
            //   in its grace window before filesystem detection stabilises.
            //   Re-triggering here causes the "99.9% → restart" loop.
            const vlRow = modelRows.find((x: any) => x.id === UNIFIED_MODEL_ID);
            const vlProgress = Number(vlRow?.progress ?? 0);
            const vlWorkerActive = modelStatus(UNIFIED_MODEL_ID) === 'downloading' || vlProgress >= 100;
            if (modelPending(UNIFIED_MODEL_ID) && !vlWorkerActive && canKick('vl')) {
              markKick('vl');
              void triggerSetupModelDownload(UNIFIED_MODEL_ID, UNIFIED_MODEL_QF);
            }
          }
        }

        setModels(modelRows);
        const allItems: DownloadItemInfo[] = buildSetupItems(core, modelRows, asr);
        const visibleItems = coreOnly ? allItems.slice(0, 3) : allItems;
        setSetupItems(visibleItems);

        const total = computeSetupProgress(visibleItems);
        setSetupProgress(total);

        const allInstalled = visibleItems.length > 0 && visibleItems.every(item => item.status === 'installed');

        if (allInstalled) {
          if (setupIntervalRef.current) window.clearInterval(setupIntervalRef.current);
          setupIntervalRef.current = null;
          setupTickNowRef.current = null;
          setTimeout(async () => {
            if (coreOnly) {
              await refreshModels();
              await markOnboardingComplete();
              await saveOnboardingStep('complete');
              setIsOnboardingComplete(true);
              setOnboardingStep('complete');
              return;
            }
            try {
              const srcs = await fetchSources();
              const hasSources = Array.isArray(srcs) && srcs.length > 0;
              if (hasSources) {
                await refreshModels();
                await markOnboardingComplete();
                await saveOnboardingStep('complete');
                setIsOnboardingComplete(true);
                setOnboardingStep('complete');
                return;
              }
            } catch {}
            setOnboardingStep('indexing-guide');
            await saveOnboardingStep('indexing-guide');
          }, 500);
        }
      } catch {
        // ignore, next tick will retry
      } finally {
        setupTickRunningRef.current = false;
      }
    };
    setupTickNowRef.current = () => {
      void runTick();
    };
    void runTick();
    setupIntervalRef.current = window.setInterval(() => {
      void runTick();
    }, 500);
  }, [buildSetupItems, computeSetupProgress, startCoreModelsDownload, triggerSetupAsrDownload, triggerSetupModelDownload]);

  useEffect(() => {
    const triggerSetupSync = () => {
      setupAutoRetryAtRef.current = { core: 0, asr: 0, vl: 0, chat: 0 };
      const fn = setupTickNowRef.current;
      if (fn) fn();
    };
    const onVisibility = () => {
      if (document.visibilityState === 'visible') triggerSetupSync();
    };
    window.addEventListener('online', triggerSetupSync);
    window.addEventListener('setup-network-recovered', triggerSetupSync);
    window.addEventListener('focus', triggerSetupSync);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      window.removeEventListener('online', triggerSetupSync);
      window.removeEventListener('setup-network-recovered', triggerSetupSync);
      window.removeEventListener('focus', triggerSetupSync);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, []);

  const handleOnboardingNext = useCallback(() => {
    if (onboardingStep !== 'welcome') return;
    if (onboardingFastTrackReadyRef.current) {
      void (async () => {
        await markOnboardingComplete();
        setIsOnboardingComplete(true);
        setOnboardingStep('complete');
        await saveOnboardingStep('complete');
      })();
      return;
    }
    setOnboardingStep('model-recommend');
    void saveOnboardingStep('model-recommend');
  }, [onboardingStep]);

  // User chose "Download" on model-recommend page → start downloading and go to setup
  const startModelDownloads = useCallback(() => {
    skippedModelsRef.current = false;
    setOnboardingStep('setup');
    void saveOnboardingStep('setup');
    setSetupDownloadErrors({});
    setSetupProgress(0);
    setSetupItems([]);

    startCoreModelsDownload().catch(() => {});
    void triggerSetupAsrDownload();

    // Single unified model download (Gemma 4B serves both chat & vision)
    void triggerSetupModelDownload(UNIFIED_MODEL_ID, UNIFIED_MODEL_QF);
    startSetupPolling(false);
  }, [startSetupPolling, triggerSetupAsrDownload, triggerSetupModelDownload]);

  // User chose "Skip for now" on model-recommend page.
  // Still show setup, but only download core models (Embedding/Reranker).
  const handleSkipModels = useCallback(async () => {
    skippedModelsRef.current = true;
    setIsOnboardingComplete(false);
    setOnboardingStep('setup');
    await saveOnboardingStep('setup');
    setSetupDownloadErrors({});
    setSetupProgress(0);
    setSetupItems([]);

    startCoreModelsDownload().catch(() => {});
    void triggerSetupAsrDownload();
    startSetupPolling(true);
  }, [startSetupPolling, triggerSetupAsrDownload]);

  useEffect(() => {
    if (onboardingStep !== 'setup' && onboardingStep as any !== 'loading-models') return;
    if (setupIntervalRef.current) return;

    if (onboardingStep === 'setup') {
      startCoreModelsDownload().catch(() => {});
      void triggerSetupAsrDownload();
      if (!skippedModelsRef.current) {
        void triggerSetupModelDownload(UNIFIED_MODEL_ID, UNIFIED_MODEL_QF);
      }
    }

    startSetupPolling(skippedModelsRef.current);
    return () => {
      if (setupIntervalRef.current) {
        window.clearInterval(setupIntervalRef.current);
        setupIntervalRef.current = null;
      }
    };
  }, [onboardingStep, startSetupPolling, triggerSetupAsrDownload, triggerSetupModelDownload]);

  const handleOnboardingSkip = useCallback(async () => {
    await markOnboardingComplete();
    setIsOnboardingComplete(true);
    setOnboardingStep('complete');
    await saveOnboardingStep('complete');
  }, []);

  const handleOnboardingCancelIndexing = useCallback(async () => {
    if (!window.confirm("确定要取消索引吗？\n\n已成功建立索引的文件将会保留，未完成的文件将被取消。")) {
      return;
    }
    
    const jobIdAtCancel = indexJobId;
    const fallbackTotal = Number(indexingState.totalFiles || 0);
    const fallbackCompleted = Number(indexingState.completedFiles || 0);
    try {
      if (indexJobId) {
        await cancelIndex(indexJobId);
      } else {
        await cancelIndex(null);
      }
      await restoreModelAfterIndexing();
      const finalCounts = await loadFinalIndexCounts(jobIdAtCancel, fallbackTotal, fallbackCompleted);
      await showCancelSummaryDialog(finalCounts.total, finalCounts.completed, finalCounts.failed);
    } catch (e) {
      console.warn('cancel indexing failed:', e);
    } finally {
      await markOnboardingComplete();
      setIsOnboardingComplete(true);
      setOnboardingStep('complete');
      await saveOnboardingStep('complete');

      try {
        const sources2 = await fetchSources();
        setSourcesLibrary(sources2);
        setActiveSourceIds(getAllIds(sources2));
      } catch {
        // ignore
      }
      setIndexingState(prev => ({ ...prev, isTopBarVisible: true, isRestoringModel: false }));
    }
  }, [
    indexJobId,
    indexingState.totalFiles,
    indexingState.completedFiles,
    loadFinalIndexCounts,
    restoreModelAfterIndexing,
    showCancelSummaryDialog,
  ]);

  const handleOnboardingAddSources = useCallback(async () => {
    if (isModelSwitching) return;
    const indexModel = ensureIndexModelForIndexing();
    if (!indexModel) return;
    const folder = await selectFolder();
    if (!folder) return;
    const confirmed = await confirmTextOnlyIndexWarningIfNeeded(indexModel);
    if (!confirmed) return;
    const indexModelPersisted = await persistIndexModelForIndexing(indexModel);
    if (!indexModelPersisted) return;

    setOnboardingStep('indexing-progress');
    await saveOnboardingStep('indexing-progress');

    try {
      await startIndexingWithFolder(folder);
    } catch (e) {
      console.warn('[Onboarding] 启动索引失败:', e);
    }
  }, [isModelSwitching, ensureIndexModelForIndexing, startIndexingWithFolder, confirmTextOnlyIndexWarningIfNeeded, persistIndexModelForIndexing]);

  useEffect(() => {
    if (onboardingStep !== 'indexing-progress') return;
    if (indexingState.isIndexing) return;
    if (indexingState.isRestoringModel) return;
    if (indexingProgress < 100) return;

    if (dragQueueRef.current.isProcessing && (dragQueueRef.current.folders.length > 0 || dragQueueRef.current.files.length > 0)) {
      return;
    }

    setOnboardingStep('complete');

    const doSwitch = async () => {
      try {
        if (indexPrevModelIdRef.current) {
          await restoreModelAfterIndexing();
        } else {
          const fallbackChat = pickAutoModelForRole('chat', selectedModel);
          if (fallbackChat) {
            await switchModelAndSyncUI(fallbackChat.id);
          }
        }
      } catch (e) {
        console.warn('Failed to switch to chat model after indexing:', e);
      } finally {
        setTimeout(async () => {
          await markOnboardingComplete();
          setIsOnboardingComplete(true);
          await saveOnboardingStep('complete');
        }, 1000);
      }
    };
    doSwitch();
  }, [onboardingStep, indexingState.isIndexing, indexingState.isRestoringModel, indexingProgress, restoreModelAfterIndexing, switchModelAndSyncUI, pickAutoModelForRole, selectedModel]);

  useEffect(() => {
    return () => {
      if (setupIntervalRef.current) window.clearInterval(setupIntervalRef.current);
      setupTickNowRef.current = null;
      setupTickRunningRef.current = false;
      if (indexingIntervalRef.current != null) window.clearTimeout(indexingIntervalRef.current);
      if (modelPollIntervalRef.current) window.clearInterval(modelPollIntervalRef.current);
      if (fileIndexPollIntervalRef.current != null) window.clearTimeout(fileIndexPollIntervalRef.current);
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      if (textOnlyIndexWarningResolverRef.current) {
        textOnlyIndexWarningResolverRef.current(false);
        textOnlyIndexWarningResolverRef.current = null;
      }
    };
  }, []);

  // Refresh sources when opening sidebar (in case initial load failed)
  const handleToggleRightSidebar = useCallback(async () => {
    const willOpen = !isRightSidebarOpen;
    setIsRightSidebarOpen(willOpen);
    
    // If opening and no sources loaded, try to fetch
    if (willOpen && sourcesLibrary.length === 0) {
      await loadSourcesFromBackend();
    }
  }, [isRightSidebarOpen, sourcesLibrary.length, loadSourcesFromBackend]);

  const handleToggleSource = (id: string) => {
    setActiveSourceIds(prev => 
      prev.includes(id) ? prev.filter(sid => sid !== id) : [...prev, id]
    );
  };

  const handleSelectAllSources = (shouldSelect: boolean) => {
    if (shouldSelect) {
      setActiveSourceIds(getAllIds(sourcesLibrary));
    } else {
      setActiveSourceIds([]);
    }
  };

  const findNodeById = (nodes: FileSource[], id: string): FileSource | null => {
    for (const n of nodes) {
      if (n.id === id) return n;
      if (n.children?.length) {
        const found = findNodeById(n.children, id);
        if (found) return found;
      }
    }
    return null;
  };

  const handleAddFilesRef = useRef<any>(null);

  // --- Drag and Drop Handling ---
  const processDropQueue = useCallback(async () => {
    const q = dragQueueRef.current;
    if (q.folders.length === 0 && q.files.length === 0) {
      q.isProcessing = false;
      return;
    }
    if (q.folders.length > 0) {
      const folder = q.folders.shift()!;
      try {
        await startIndexingWithFolder(folder);
      } catch (e) {
        console.warn('Failed to start folder indexing from queue', e);
        // Continue queue despite error
        setTimeout(processDropQueue, 500); 
      }
      return;
    }
    if (q.files.length > 0) {
      const files = [...q.files];
      q.files = [];
      try {
        if (handleAddFilesRef.current) {
          await handleAddFilesRef.current(files);
        }
      } catch (e) {
        console.warn('Failed to start file indexing from queue', e);
      }
      // handleAddFiles will clear queue processing or finish naturally... wait, handleAddFiles doesn't reset isProcessing!
      // But it's fine, files are processed, processDropQueue will run again if indexingState flip happens.
      // Actually, let's just mark it done.
      if (dragQueueRef.current.folders.length === 0 && dragQueueRef.current.files.length === 0) {
        dragQueueRef.current.isProcessing = false;
      }
    }
  }, [startIndexingWithFolder]);

  const processDropQueueRef = useRef(processDropQueue);
  useLayoutEffect(() => {
    processDropQueueRef.current = processDropQueue;
  }, [processDropQueue]);

  useEffect(() => {
    if (!dragQueueRef.current.isProcessing) return;
    if (indexingState.isIndexing) return;
    if (isModelSwitching) return;
    
    // Ensure react state cycle settled
    const timer = setTimeout(() => {
      processDropQueue();
    }, 500);
    return () => clearTimeout(timer);
  }, [indexingState.isIndexing, isModelSwitching, processDropQueue]);

  const handleDropPaths = useCallback(async (paths: string[]) => {
    if (isModelSwitching || !hasInstalledModels) {
      if (!hasInstalledModels) setShowModelPrompt(true);
      return;
    }
    const indexModel = ensureIndexModelForIndexing();
    if (!indexModel) return;

    let localFolders: string[] = [];
    let localFiles: string[] = [];
    for (const p of paths) {
      try {
        const s = await stat(p);
        if (s.isDirectory) localFolders.push(p);
        else if (s.isFile) localFiles.push(p);
      } catch(e) {
        console.warn('Failed to stat path', p, e);
      }
    }

    if (localFolders.length === 0 && localFiles.length === 0) return;

    dragQueueRef.current.folders.push(...localFolders);
    dragQueueRef.current.files.push(...localFiles);

    if (!dragQueueRef.current.isProcessing) {
      dragQueueRef.current.isProcessing = true;
      processDropQueueRef.current();
    }
  }, [hasInstalledModels, isModelSwitching, ensureIndexModelForIndexing]);

  const isRightSidebarOpenRef = useRef(isRightSidebarOpen);
  const handleDropPathsRef = useRef(handleDropPaths);
  useLayoutEffect(() => {
    isRightSidebarOpenRef.current = isRightSidebarOpen;
    handleDropPathsRef.current = handleDropPaths;
  }, [isRightSidebarOpen, handleDropPaths]);

  useEffect(() => {
    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
      if (!isRightSidebarOpenRef.current) {
        if (e.dataTransfer) e.dataTransfer.dropEffect = 'none';
        return;
      }
      
      // RightSidebar is open, check if we're over the panel (320px width)
      const isTarget = e.clientX >= window.innerWidth - 320;
      if (e.dataTransfer) {
        e.dataTransfer.dropEffect = isTarget ? 'copy' : 'none';
      }
    };
    
    const handleDrop = (e: DragEvent) => {
      e.preventDefault(); // Prevent browser default (like opening the file)
    };

    window.addEventListener('dragover', handleDragOver);
    window.addEventListener('drop', handleDrop);

    let unlistenDragDrop: (() => void) | undefined;
    (async () => {
      try {
        unlistenDragDrop = await getCurrentWindow().onDragDropEvent(async (event) => {
          if (event.payload.type === 'enter') {
            dragCounter.current += 1;
            try {
              await getCurrentWindow().setFocus();
              await getCurrentWindow().unminimize();
            } catch (e) {
              console.warn('Failed to focus window on drag-enter', e);
            }
          }
          
          if (event.payload.type === 'leave') {
            dragCounter.current = Math.max(0, dragCounter.current - 1);
            if (dragCounter.current === 0) setIsDragOver(false);
          }

          if (event.payload.type === 'over') {
            if (!isRightSidebarOpenRef.current) return;
            const pos = event.payload.position;
            if (pos && typeof pos.x === 'number') {
              // Tauri V2 window onDragDropEvent passes logical pixels in pos.x! 
              // Do NOT divide by devicePixelRatio.
              const logicalX = pos.x;
              
              // Only target if mouse is over the right-side Add Sources panel
              // (usually 300px wide, 320px for margin)
              const isTarget = logicalX >= window.innerWidth - 320;
              setIsDragOver(prev => {
                if (prev !== isTarget) return isTarget;
                return prev;
              });
            }
          }

          if (event.payload.type === 'drop') {
            dragCounter.current = 0;
            setIsDragOver(false);

            if (!isRightSidebarOpenRef.current) return;
            // Verify drop position
            const pos = event.payload.position;
            if (!pos || typeof pos.x !== 'number') return;
            
            // Tauri V2 Window API passes logical pixels here.
            const logicalX = pos.x;
            
            // only accept if dropped within RightSidebar
            if (logicalX < window.innerWidth - 320) return;

            const paths = event.payload.paths;
            if (!paths || paths.length === 0) return;
            handleDropPathsRef.current(paths);
          }
        });
      } catch (e) {
        console.error('Failed to setup drag listener:', e);
      }
    })();
    return () => {
      window.removeEventListener('dragover', handleDragOver);
      window.removeEventListener('drop', handleDrop);
      if (unlistenDragDrop) unlistenDragDrop();
    };
  }, []);

  useLayoutEffect(() => {
    handleAddFilesRef.current = handleAddFiles;
  });


  const handleAddFiles = useCallback(async (
    droppedFiles?: string[],
    options?: { onReadyToIndex?: () => void | Promise<void> },
  ) => {
    if (!modelsHydrated) {
      console.log('[Sources] handleAddFiles aborted: modelsHydrated == false');
      return;
    }
    if (isModelSwitching) {
      console.log('[Sources] handleAddFiles aborted: isModelSwitching == true');
      return;
    }
    if (!hasInstalledModels) {
      setShowModelPrompt(true);
      return;
    }
    const indexModel = ensureIndexModelForIndexing();
    if (!indexModel) {
      console.log('[Sources] handleAddFiles aborted: ensureIndexModelForIndexing returned false/null/undefined');
      return;
    }
    const files = droppedFiles && droppedFiles.length > 0 ? droppedFiles : await selectFiles();
    if (!files || files.length === 0) {
      console.log('[Sources] handleAddFiles aborted: no files selected');
      return;
    }
    const confirmed = await confirmTextOnlyIndexWarningIfNeeded(indexModel);
    if (!confirmed) return;
    const indexModelPersisted = await persistIndexModelForIndexing(indexModel);
    if (!indexModelPersisted) return;

    try {
      const { confirm } = await import('@tauri-apps/plugin-dialog');
      
      const mediaExts = ['.mp3', '.mp4', '.wav', '.m4a', '.webm', '.ogg', '.mov', '.avi', '.mkv', '.flac', '.aac'];
      const hasMedia = files.some(f => {
        const lower = f.toLowerCase();
        return mediaExts.some(ext => lower.endsWith(ext));
      });
      const messageKey = hasMedia ? 'confirmIndexDialog.messageMediaFiles' : 'confirmIndexDialog.messageFiles';
      
      const shouldIndex = await confirm(i18n.t(messageKey), { title: i18n.t('confirmIndexDialog.title'), kind: 'info' });
      if (!shouldIndex) return;
    } catch (e) {
      console.warn('Failed to show confirm dialog', e);
    }

    if (options?.onReadyToIndex) {
      await options.onReadyToIndex();
    }

    if (indexingState.isIndexing) {
      console.log('[Sources] 索引进行中，追加文件到队列:', files.length);
      try {
        const result = await indexFiles(files);
        if (result.ok || (result as any)?.appended) {
          const sources = await fetchSources();
          setSourcesLibrary(sources);
          setActiveSourceIds(getAllIds(sources));
        }
      } catch (e) {
        console.warn('[Sources] 追加文件失败:', e);
      }
      return;
    }

    try {
      if (fileIndexPollIntervalRef.current != null) {
        window.clearTimeout(fileIndexPollIntervalRef.current);
        fileIndexPollIntervalRef.current = null;
      }

      const result = await indexFiles(files);
      if (!result.ok) {
        console.error('[Sources] 添加文件失败:', result.error);
        const err = String(result.error || '');
        const msg = String((result as any)?.message || err || '添加文件失败，请稍后重试。');
        if (err === 'core_models_not_ready') {
          setIsOnboardingComplete(false);
          setOnboardingStep('welcome');
          void Promise.all([
            updateSettings('onboarding_complete', false),
            updateSettings('onboarding_step', 'welcome'),
          ]);
        } else {
          console.warn('[Sources] 添加文件失败:', msg);
        }
        try {
          await restoreModelAfterIndexing();
        } catch (e) {
          console.warn('[Sources] 恢复原模型失败:', e);
        }
        return;
      }

      const jobId = result.job_id;
      try {
        const src = await fetchSources();
        setSourcesLibrary(src);
        setActiveSourceIds(getAllIds(src));
        fileIndexSourcesRefreshRef.current = {
          at: Date.now(),
          completed: 0,
          total: files.length,
          currentPath: '',
        };
      } catch {
        /* ignore */
      }
      setIndexingState({ 
        isIndexing: true, 
        totalFiles: files.length, 
        completedFiles: 0, 
        eta: '…',
        isTopBarVisible: true,
        isCancelling: false,
        isRestoringModel: false,
        statusMessage: '',
        currentFrame: 0,
        totalFrames: 0,
        currentAudioSec: 0,
        totalAudioSec: 0,
        stageRate: 0,
        stage: '',
      });

      const FILE_INDEX_POLL_MS = 500;

      const clearFilePoll = () => {
        if (fileIndexPollIntervalRef.current != null) {
          window.clearTimeout(fileIndexPollIntervalRef.current);
          fileIndexPollIntervalRef.current = null;
        }
      };

      const scheduleFileNext = () => {
        fileIndexPollIntervalRef.current = window.setTimeout(runFileTick, FILE_INDEX_POLL_MS);
      };

      const runFileTick = async () => {
        fileIndexPollIntervalRef.current = null;
        try {
          const status = await getIndexStatus(jobId);
          if (!status || !status.job) {
            try {
              const activeRes = await getActiveIndexJob();
              const nextJob = (activeRes as any)?.job;
              const persisted = (activeRes as any)?.persisted;
              if ((nextJob && nextJob.is_indexing) || (persisted && persisted.is_indexing)) {
                scheduleFileNext();
                return;
              }
            } catch {
              scheduleFileNext();
              return;
            }
            clearFilePoll();
            setIndexJobId(null);
            setIndexingState(prev => ({
              ...prev,
              isIndexing: false,
              isRestoringModel: false,
              isTopBarVisible: prev.isCancelling ? true : false,
            }));
            await finalizeSourcesAfterIndex(setSourcesLibrary, setActiveSourceIds);
            return;
          }

          const job = status.job;

          setIndexingState(prev => ({
            ...prev,
            isIndexing: Boolean(job.is_indexing),
            isRestoringModel: false,
            totalFiles: job.total_files || files.length,
            completedFiles: job.completed_files || 0,
            eta: formatIndexEta(job),
            isTopBarVisible: job.is_indexing ? true : (prev.isCancelling ? true : false),
            statusMessage: buildLiveIndexStatusMessage(job),
            currentFile: job.current_file,
            currentPath: job.current_path,
            currentFrame: Number(job.current_frame || 0),
            totalFrames: Number(job.total_frames || 0),
            currentAudioSec: Number(job.current_audio_sec || 0),
            totalAudioSec: Number(job.total_audio_sec || 0),
            stageRate: Number(job.stage_rate || 0),
            stage: job.stage || '',
          }));

          if (job.is_indexing) {
            const now = Date.now();
            const nextCompleted = Number(job.completed_files || 0);
            const nextTotal = Number(job.total_files || files.length || 0);
            const nextPath = String(job.current_path || '');
            const lastRefresh = fileIndexSourcesRefreshRef.current;
            const shouldRefreshSources =
              lastRefresh.at === 0 ||
              nextCompleted !== lastRefresh.completed ||
              nextTotal !== lastRefresh.total ||
              nextPath !== lastRefresh.currentPath ||
              (now - lastRefresh.at) >= 10000;

            if (shouldRefreshSources) {
              try {
                const src = await fetchSources();
                setSourcesLibrary(src);
                // Auto-check newly indexed files immediately
                setActiveSourceIds(getAllIds(src));
                fileIndexSourcesRefreshRef.current = {
                  at: now,
                  completed: nextCompleted,
                  total: nextTotal,
                  currentPath: nextPath,
                };
              } catch {
                /* ignore */
              }
            }
            scheduleFileNext();
            return;
          }

          clearFilePoll();
          setIndexJobId(null);
          const totalFiles = Number(job.total_files || files.length || 0);
          const completedFiles = Number(job.completed_files || 0);
          const failedFiles = Number(job.failed_files || 0);

          if (job.error !== 'cancelled') {
            setIndexingState(prev => ({
              ...prev,
              isIndexing: false,
              isTopBarVisible: true,
              isCancelling: false,
              isRestoringModel: true,
              statusMessage: i18n.t('indexingWidget.switchingModelWait'),
              totalFiles,
              completedFiles,
            }));
          }

          let modelSwitchCostMs = 0;
          if (job.error !== 'cancelled') {
            const switchStart = Date.now();
            try {
              console.log('[Sources] 文件索引完成，恢复原模型...');
              await restoreModelAfterIndexing();
            } catch (e) {
              console.warn('[Sources] 恢复原模型失败:', e);
              setIndexingState(prev => ({
                ...prev,
                isRestoringModel: false,
                isTopBarVisible: true,
                statusMessage: i18n.t('indexingWidget.switchModelFailedManual'),
              }));
              await finalizeSourcesAfterIndex(setSourcesLibrary, setActiveSourceIds);
              return;
            }
            modelSwitchCostMs = Math.max(0, Date.now() - switchStart);
            setIndexingState(prev => ({ ...prev, isRestoringModel: false }));
          }

          await finalizeSourcesAfterIndex(setSourcesLibrary, setActiveSourceIds);
          setConnectionError(false);
          if (job.error !== 'cancelled') {
            setIndexingState(prev => ({
              ...prev,
              isIndexing: false,
              isTopBarVisible: false,
              isCancelling: false,
              isRestoringModel: false,
              statusMessage: '',
            }));
            await showIndexingCompleteDialog(
              totalFiles,
              completedFiles,
              failedFiles,
              String((job as any)?.job_id || jobId || ''),
              String(job.error || ''),
            );
          }
          console.log('[Sources] ✓ 文件索引完成');
        } catch {
          scheduleFileNext();
        }
      };

      fileIndexPollIntervalRef.current = window.setTimeout(runFileTick, 0);
    } catch (e) {
      console.error('[Sources] 添加文件失败:', e);
      
      try {
        await restoreModelAfterIndexing();
      } catch (e) {
        console.warn('[Sources] 恢复原模型失败:', e);
      }
      
      if (fileIndexPollIntervalRef.current != null) {
        window.clearTimeout(fileIndexPollIntervalRef.current);
        fileIndexPollIntervalRef.current = null;
      }
    }
  }, [isModelSwitching, ensureIndexModelForIndexing, hasInstalledModels, modelsHydrated, restoreModelAfterIndexing, showIndexingCompleteDialog, confirmTextOnlyIndexWarningIfNeeded, persistIndexModelForIndexing, indexingState.isIndexing]);

  const handleOnboardingAddFiles = useCallback(async () => {
    await handleAddFiles(undefined, {
      onReadyToIndex: async () => {
        setOnboardingStep('indexing-progress');
        await saveOnboardingStep('indexing-progress');
      },
    });
  }, [handleAddFiles, saveOnboardingStep]);

  const [isRemovingSources, setIsRemovingSources] = useState(false);
  const [removeProgress, setRemoveProgress] = useState<{ current: number; total: number } | null>(null);
  const [refreshingFolder, setRefreshingFolder] = useState<string | null>(null);
  const [refreshToast, setRefreshToast] = useState<{ added: number; updated: number; deleted: number; skipped: number } | null>(null);
  const refreshPollRef = useRef<number | null>(null);
  const handleRemoveSources = async (ids: string[]) => {
    const toRemovePaths: string[] = [];
    ids.forEach(id => {
      const node = findNodeById(sourcesLibrary, id);
      if (node?.path) toRemovePaths.push(node.path);
    });

    if (toRemovePaths.length === 0) {
      setActiveSourceIds(prev => prev.filter(sid => !ids.includes(sid)));
      return;
    }

    setIsRemovingSources(true);
    setRemoveProgress({ current: 0, total: toRemovePaths.length });
    try {
      let totalDeleted = 0;
      if (toRemovePaths.length > 1) {
        const result = await removeSourcesBatch(toRemovePaths);
        if (result.ok && result.deleted_count) totalDeleted = result.deleted_count;
        if (!result.ok) console.error('[Sources] 批量移除失败:', result.error);
        setRemoveProgress({ current: toRemovePaths.length, total: toRemovePaths.length });
      } else {
        for (let i = 0; i < toRemovePaths.length; i++) {
          const result = await removeSource(toRemovePaths[i]);
          if (result.ok && result.deleted_count) totalDeleted += result.deleted_count;
          if (!result.ok) console.error(`[Sources] 移除失败 ${toRemovePaths[i]}:`, result.error);
          setRemoveProgress({ current: i + 1, total: toRemovePaths.length });
        }
      }
      const sources = await fetchSources();
      setSourcesLibrary(sources);
      setActiveSourceIds(getAllIds(sources));
      if (totalDeleted > 0) {
        console.log(`[Sources] ✓ 已移除 ${toRemovePaths.length} 个项目，删除 ${totalDeleted} 条索引文档`);
      }
    } catch (e) {
      console.error('[Sources] 移除失败:', e);
    } finally {
      setIsRemovingSources(false);
      setRemoveProgress(null);
    }
  };

  const handleRefreshSource = useCallback(async (folderPath: string) => {
    if (refreshingFolder || indexingState.isIndexing) return;
    setRefreshingFolder(folderPath);
    try {
      const res = await refreshSource(folderPath);
      const jobId = res.job_id;
      if (!jobId) {
        setRefreshingFolder(null);
        return;
      }

      const POLL_MS = 1000;
      const clearPoll = () => {
        if (refreshPollRef.current != null) {
          window.clearTimeout(refreshPollRef.current);
          refreshPollRef.current = null;
        }
      };

      const tick = async () => {
        refreshPollRef.current = null;
        try {
          const status = await getIndexStatus(jobId);
          const job = status?.job;
          if (!job) { clearPoll(); setRefreshingFolder(null); return; }

          if (job.is_indexing) {
            // Only show progress (and lock features) if it enters Phase 2 (total_files > 0)
            if (job.total_files && job.total_files > 0) {
              setIndexingState(prev => ({
                ...prev,
                isIndexing: true,
                totalFiles: job.total_files,
                completedFiles: job.completed_files || 0,
                eta: formatIndexEta(job),
                isTopBarVisible: true,
                statusMessage: buildLiveIndexStatusMessage(job),
                currentFile: job.current_file,
                currentPath: job.current_path,
                currentFrame: Number(job.current_frame || 0),
                totalFrames: Number(job.total_frames || 0),
                currentAudioSec: Number(job.current_audio_sec || 0),
                totalAudioSec: Number(job.total_audio_sec || 0),
                stageRate: Number(job.stage_rate || 0),
                stage: job.stage || '',
              }));
            }
            // Still running — keep polling
            refreshPollRef.current = window.setTimeout(tick, POLL_MS);
            return;
          }

          // Done — parse result from job.message
          clearPoll();
          setRefreshingFolder(null);

          const totalFiles = Number(job.total_files || 0);

          if (totalFiles > 0 && job.error !== 'cancelled') {
            setIndexingState(prev => ({
              ...prev,
              isIndexing: false,
              isTopBarVisible: true,
              isRestoringModel: true,
              statusMessage: i18n.t('indexingWidget.switchingModelWait'),
            }));
            try {
              console.log('[Refresh] 增量索引完成，恢复原模型...');
              await restoreModelAfterIndexing();
            } catch (e) {
              console.warn('[Refresh] 恢复原模型失败:', e);
            }
          }

          setIndexingState(prev => ({
            ...prev,
            isIndexing: false,
            isTopBarVisible: false,
            isRestoringModel: false,
            statusMessage: ''
          }));

          const msg: string = job.message || '';
          if (msg.startsWith('refresh_done|')) {
            const parse = (key: string) => {
              const m = msg.match(new RegExp(`${key}=(\\d+)`));
              return m ? parseInt(m[1], 10) : 0;
            };
            const toastData = {
              added: parse('added'),
              updated: parse('updated'),
              deleted: parse('deleted'),
              skipped: parse('skipped'),
            };
            setRefreshToast(toastData);
            setTimeout(() => setRefreshToast(null), 5000);
          }

          // Refresh sources list
          const sources = await fetchSources();
          setSourcesLibrary(sources);
          setActiveSourceIds(getAllIds(sources));
        } catch {
          refreshPollRef.current = window.setTimeout(tick, POLL_MS);
        }
      };

      refreshPollRef.current = window.setTimeout(tick, 500);
    } catch (e) {
      console.error('[Refresh] 刷新失败:', e);
      setRefreshingFolder(null);
    }
  }, [refreshingFolder, indexingState.isIndexing]);

  const showOnboarding = onboardingHydrated && !isOnboardingComplete;
  const showMainUI = onboardingHydrated && isOnboardingComplete;

  return (
    <div className="flex flex-col h-full w-full bg-white overflow-hidden font-sans text-gray-900 relative">
      {isDragOver && isRightSidebarOpen && (
        <div className="absolute top-0 right-0 bottom-0 w-[300px] z-[9999] bg-blue-500/10 backdrop-blur-sm flex items-center justify-center pointer-events-none transition-all duration-200">
          <div className="bg-white px-6 py-5 rounded-2xl shadow-xl flex flex-col items-center gap-3 animate-in zoom-in duration-200 shadow-black/10 text-center mx-4">
            <div className="w-12 h-12 bg-black/5 text-gray-700 rounded-full flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            </div>
            <h3 className="text-lg font-bold text-gray-900 tracking-tight">Drop to Index</h3>
            <p className="text-xs text-gray-500 leading-tight">Supports folders and various document formats</p>
          </div>
        </div>
      )}

      {isOnboardingComplete && (
        <div 
          className="absolute top-0 left-0 w-24 h-7 z-[120]"
          style={{ 
            WebkitAppRegion: 'drag',
            pointerEvents: 'auto'
          } as any}
        />
      )}
      
      <div className="flex-none flex flex-col w-full z-[120]">
        {connectionError && (
          <div className="bg-red-50 border-b border-red-200 px-4 py-2 text-sm text-red-700 flex justify-between items-center h-10">
            <span>
              <strong>Backend Disconnected:</strong> Could not connect to the local AI agent. Please check if the backend is running.
            </span>
            <button 
              onClick={() => window.location.reload()} 
              className="text-xs bg-white border border-red-300 px-2 py-1 rounded hover:bg-red-50"
            >
              Retry
            </button>
          </div>
        )}
      </div>

      {showOnboarding && (
        <Onboarding
          currentStep={onboardingStep}
          setupProgress={setupProgress}
          setupItems={setupItems}
          indexingProgress={indexingProgress}
          indexingCompletedFiles={indexingState.completedFiles}
          indexingTotalFiles={indexingState.totalFiles}
          indexingEta={indexingState.eta}
          onNext={handleOnboardingNext}
          onDownloadModels={startModelDownloads}
          onSkipModels={handleSkipModels}
          onSkip={handleOnboardingSkip}
          onAddSources={handleOnboardingAddSources}
          onAddFiles={handleOnboardingAddFiles}
          onCancelIndexing={handleOnboardingCancelIndexing}
        />
      )}

      <div className="flex-1 flex flex-row min-h-0 relative w-full overflow-hidden">
        {/* Left Sidebar */}
        {showMainUI && (
          <LeftSidebar
            history={conversations}
            activeConversationId={activeConversationId}
            onNewChat={handleNewChat}
            onSelectChat={handleSelectChat}
            onDeleteChat={handleDeleteChat}
            onOpenSettings={() => setIsSettingsOpen(true)}
            onOpenManageModels={() => openManageModels()}
            isGenerating={isGenerating}
            hasModels={modelsHydrated ? hasInstalledModels : true}
            onShowModelPrompt={() => setShowModelPrompt(true)}
          />
        )}

        {/* Main Center Panel */}
        <main className="flex-1 flex flex-col relative min-w-0 h-full bg-white">
          <ModelSwitchIndicator 
            isSwitching={isModelSwitching}
            targetModelId={switchingToModelId}
            installedModels={installedModels}
            switchError={modelSwitchError}
          />
          {showMainUI && (isLanding ? (
            <LandingArea 
              selectedModel={selectedModel}
              models={installedModels}
              isModelStateReady={modelsHydrated}
              onSelectModel={handleSelectModel}
              inputValue={inputValue}
              onInputChange={setInputValue}
              onSend={handleSendMessage}
              sourcesLibrary={sourcesLibrary}
              activeSourceIds={activeSourceIds}
              onToggleSource={handleToggleSource}
              onRemoveSources={handleRemoveSources}
              onAddSources={handleAddSources}
              onAddFiles={handleAddFiles}
              indexingState={indexingState}
              onCloseIndexingTopBar={handleCloseIndexingTopBar}
              isBackendSyncing={syncPanelInMain}
              isRightSidebarOpen={isRightSidebarOpen}
              onToggleRightSidebar={handleToggleRightSidebar}
              isGenerating={isGenerating}
              onStopGenerating={handleStopGenerating}
              onOpenManageModels={() => openManageModels()}
            />
          ) : (
            <ChatArea
              messages={messages}
              selectedModel={selectedModel}
              models={installedModels}
              isModelStateReady={modelsHydrated}
              onSelectModel={handleSelectModel}
              inputValue={inputValue}
              onInputChange={setInputValue}
              onSend={handleSendMessage}
              isGenerating={isGenerating}
              onStopGenerating={handleStopGenerating}
              sourcesLibrary={sourcesLibrary}
              activeSourceIds={activeSourceIds}
              onToggleSource={handleToggleSource}
              onRemoveSources={handleRemoveSources}
              onAddSources={handleAddSources}
              onAddFiles={handleAddFiles}
              isRightSidebarOpen={isRightSidebarOpen}
              onToggleRightSidebar={handleToggleRightSidebar}
              indexingState={indexingState}
              onCloseIndexingTopBar={handleCloseIndexingTopBar}
              isBackendSyncing={syncPanelInMain}
              onOpenManageModels={() => openManageModels()}
            />
          ))}
        </main>

        {/* Right Sidebar */}
        {showMainUI && isRightSidebarOpen && (
          <RightSidebar
            sources={sourcesLibrary}
            activeSourceIds={activeSourceIds}
            setActiveSourceIds={setActiveSourceIds}
            onSelectAll={handleSelectAllSources}
            onRemoveSource={(id) => handleRemoveSources([id])}
            onRemoveAllSources={() => handleRemoveSources(
              activeSourceIds.length > 0 ? activeSourceIds : sourcesLibrary.map(s => s.id)
            )}
            onSkipFile={async (filePaths) => {
              try {
                const res = await skipFiles(filePaths);
                const skippedCount = (res as any)?.skipped_count || filePaths.length;
                console.log('[Sources] 已跳过文件:', skippedCount);
                const skippedSet = new Set(filePaths);
                const removeSkipped = (nodes: FileSource[]): FileSource[] => {
                  return nodes
                    .filter(n => !n.path || !skippedSet.has(n.path))
                    .map(n => n.children ? { ...n, children: removeSkipped(n.children) } : n);
                };
                setSourcesLibrary(prev => removeSkipped(prev));
                const skippedIds = new Set<string>();
                const collectIds = (nodes: FileSource[]) => {
                  for (const n of nodes) {
                    if (n.path && skippedSet.has(n.path)) skippedIds.add(n.id);
                    if (n.children) collectIds(n.children);
                  }
                };
                collectIds(sourcesLibrary);
                if (skippedIds.size > 0) {
                  setActiveSourceIds(prev => prev.filter(id => !skippedIds.has(id)));
                }
                setIndexingState(prev => ({
                  ...prev,
                  totalFiles: Math.max(0, prev.totalFiles - skippedCount),
                }));
              } catch (e) {
                console.warn('[Sources] 跳过文件失败:', e);
              }
            }}
            onAddSources={handleAddSources}
            onAddFiles={handleAddFiles}
            indexingState={indexingState}
            onClose={() => setIsRightSidebarOpen(false)}
            mode={sidebarMode}
            onSwitchMode={setSidebarMode}
            openedFiles={openedFiles}
            activeOpenedFilePath={activeOpenedFilePath}
            onSelectOpenedFile={setActiveOpenedFilePath}
            onClearOpenedFiles={() => { setOpenedFiles([]); setActiveOpenedFilePath(null); }}
            selectedIndexModel={selectedIndexModel}
            installedModels={installedModels}
            isModelSwitching={isModelSwitching}
            onSelectIndexModel={(model: Model) => {
              if (isModelSwitching || indexingState.isIndexing) return;
              userPickedIndexModelRef.current = true;
              setSelectedIndexModel(model);
              void updateSettings(INDEX_MODEL_PREFERENCE_KEY, model.id);
              void updateSettings(INDEX_MODEL_MANUAL_LOCK_KEY, true);
            }}
            onOpenManageModels={() => openManageModels()}
            isRemovingSources={isRemovingSources}
            removeProgress={removeProgress}
            onRefreshSource={handleRefreshSource}
            refreshingFolder={refreshingFolder}
          />
        )}
      </div>

      {/* Refresh Result Toast */}
      {refreshToast && (
        <div className="fixed bottom-6 right-6 z-[500] animate-in fade-in slide-in-from-bottom-2 duration-300">
          <div className="bg-gray-900 text-white text-sm rounded-xl px-4 py-3 shadow-2xl flex items-start gap-3 max-w-xs">
            <div className="mt-0.5 w-4 h-4 rounded-full bg-green-400 flex-shrink-0" />
            <div>
              <div className="font-semibold mb-1">{t('sidebar.updateDone', 'Update Complete')}</div>
              <div className="text-gray-300 text-xs space-y-0.5">
                {refreshToast.updated > 0 && <div>↺ {refreshToast.updated} {t('sidebar.updateUpdated', 'updated')}</div>}
                {refreshToast.added > 0 && <div>+ {refreshToast.added} {t('sidebar.updateAdded', 'added')}</div>}
                {refreshToast.deleted > 0 && <div>− {refreshToast.deleted} {t('sidebar.updateDeleted', 'removed')}</div>}
                {refreshToast.updated === 0 && refreshToast.added === 0 && refreshToast.deleted === 0 && (
                  <div>{t('sidebar.updateNoChange', 'Already up to date')}</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {isOnboardingComplete && (
        <>
          <SettingsModal 
            isOpen={isSettingsOpen} 
            onClose={() => setIsSettingsOpen(false)} 
          />
          <ManageModelsModal 
            isOpen={isManageModelsOpen}
            onClose={closeManageModels}
            models={models}
            focusModelId={manageModelsFocus?.modelId ?? null}
            focusSearchText={manageModelsFocus?.searchText ?? ''}
            onDownload={handleDownloadModel}
            onSelectQuantization={handleSelectModelQuantization}
            onSelectModel={async (id) => {
              if (isModelSwitching || indexingState.isIndexing) return;
              try {
                userPickedChatModelRef.current = true;
                void updateSettings(CHAT_MODEL_MANUAL_LOCK_KEY, true);
                await switchModelAndSyncUI(id);
              } catch (e) {
                console.error("Select model failed", e);
              }
            }}
            onDelete={handleDeleteModel}
            onCancel={handleCancelDownloadModel}
          />
        </>
      )}

      {showModelPrompt && modelsHydrated && !hasInstalledModels && isOnboardingComplete && (
        <ModelPromptToast
          onDismiss={() => setShowModelPrompt(false)}
          onBrowseModels={() => {
            setShowModelPrompt(false);
            openManageModels();
          }}
        />
      )}

      {indexCancelConfirmOpen && (
        <div
          className="fixed inset-0 z-[400] flex items-center justify-center p-4 bg-black/25"
          style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
          role="presentation"
          onClick={dismissIndexCancelConfirm}
        >
          <div
            className="w-full max-w-md rounded-xl border border-gray-200 bg-white p-5 shadow-xl"
            role="dialog"
            aria-modal="true"
            aria-labelledby="index-cancel-title"
            onClick={e => e.stopPropagation()}
          >
            <h2 id="index-cancel-title" className="text-base font-semibold text-gray-900 mb-3">
              {t('indexingWidget.stopIndexingConfirmTitle')}
            </h2>
            <p className="text-sm text-gray-600 leading-relaxed">
              {t('indexingWidget.stopIndexingConfirmBody')}
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                className="px-4 py-2 text-sm font-medium text-gray-700 rounded-lg border border-gray-200 hover:bg-gray-50"
                onClick={dismissIndexCancelConfirm}
              >
                {t('common.cancel')}
              </button>
              <button
                type="button"
                className="px-4 py-2 text-sm font-medium text-white rounded-lg bg-gray-900 hover:bg-gray-800"
                onClick={() => void confirmCancelIndexing()}
              >
                {t('indexingWidget.stop')}
              </button>
            </div>
          </div>
        </div>
      )}

      {pendingDeleteChatId && (
        <div
          className="fixed inset-0 z-[405] flex items-center justify-center p-4 bg-black/25"
          style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
          role="presentation"
          onClick={dismissDeleteChatConfirm}
        >
          <div
            className="w-full max-w-md rounded-xl border border-gray-200 bg-white p-5 shadow-xl"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-chat-title"
            onClick={e => e.stopPropagation()}
          >
            <h2 id="delete-chat-title" className="text-base font-semibold text-gray-900 mb-3">
              {t('sidebar.deleteChatConfirmTitle')}
            </h2>
            <p className="text-sm text-gray-600 leading-relaxed">
              {t('sidebar.deleteChatConfirmBody')}
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                className="px-4 py-2 text-sm font-medium text-gray-700 rounded-lg border border-gray-200 hover:bg-gray-50"
                onClick={dismissDeleteChatConfirm}
              >
                {t('common.cancel')}
              </button>
              <button
                type="button"
                className="px-4 py-2 text-sm font-medium text-white rounded-lg bg-gray-900 hover:bg-gray-800"
                onClick={() => void confirmDeleteChat()}
              >
                {t('common.confirm')}
              </button>
            </div>
          </div>
        </div>
      )}

      {cancelSummaryDialog && (
        <div
          className="fixed inset-0 z-[410] flex items-center justify-center p-4 bg-black/25"
          style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
          role="presentation"
          onClick={() => setCancelSummaryDialog(null)}
        >
          <div
            className="w-full max-w-md rounded-xl border border-gray-200 bg-white p-5 shadow-xl"
            role="dialog"
            aria-modal="true"
            aria-labelledby="cancel-summary-title"
            onClick={e => e.stopPropagation()}
          >
            <h2 id="cancel-summary-title" className="text-lg font-semibold text-gray-900 text-center">
              {t('indexingWidget.indexingCancelled')}
            </h2>

            <div className="mt-5 grid grid-cols-3 gap-3">
              <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-center">
                <div className="text-xs font-medium text-gray-500">{t('indexingWidget.completedFiles')}</div>
                <div className="mt-1 text-2xl font-semibold text-gray-900">{cancelSummaryDialog.completed}</div>
                <div className="text-xs text-gray-500">{t('indexingWidget.files')}</div>
              </div>
              <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-center">
                <div className="text-xs font-medium text-gray-500">{t('indexingWidget.cancelledFiles')}</div>
                <div className="mt-1 text-2xl font-semibold text-gray-900">{cancelSummaryDialog.cancelled}</div>
                <div className="text-xs text-gray-500">{t('indexingWidget.files')}</div>
              </div>
              <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-center">
                <div className="text-xs font-medium text-gray-500">{t('indexingWidget.failedFiles')}</div>
                <div className="mt-1 text-2xl font-semibold text-gray-900">{cancelSummaryDialog.failed}</div>
                <div className="text-xs text-gray-500">{t('indexingWidget.files')}</div>
              </div>
            </div>

            <div className="mt-5 flex justify-center">
              <button
                type="button"
                className="min-w-[140px] px-5 py-2 text-sm font-medium text-white rounded-lg bg-blue-600 hover:bg-blue-700"
                onClick={() => setCancelSummaryDialog(null)}
              >
                {t('common.confirm')}
              </button>
            </div>
          </div>
        </div>
      )}

      {textOnlyIndexWarningOpen && (
        <div
          className="fixed inset-0 z-[420] flex items-center justify-center p-4 bg-black/30"
          style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
          role="presentation"
          onClick={() => { void resolveTextOnlyIndexWarning(false); }}
        >
          <div
            className="w-full max-w-md rounded-2xl border border-gray-200 bg-white p-5 shadow-xl"
            role="dialog"
            aria-modal="true"
            aria-labelledby="text-only-index-warning-title"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 id="text-only-index-warning-title" className="text-base font-semibold text-gray-900">
              {t('textOnlyIndexWarningDialog.title')}
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-gray-600">
              {t('textOnlyIndexWarningDialog.message', { model: textOnlyIndexWarningModelName || '-' })}
            </p>
            <label className="mt-4 flex items-center gap-2 text-sm text-gray-700 select-none">
              <input
                type="checkbox"
                className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                checked={textOnlyIndexWarningSkipNext}
                onChange={(e) => setTextOnlyIndexWarningSkipNext(e.target.checked)}
              />
              <span>{t('textOnlyIndexWarningDialog.dontShowAgain')}</span>
            </label>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                className="px-4 py-2 text-sm font-medium text-gray-700 rounded-lg border border-gray-200 hover:bg-gray-50"
                onClick={() => { void resolveTextOnlyIndexWarning(false); }}
              >
                {t('textOnlyIndexWarningDialog.cancel')}
              </button>
              <button
                type="button"
                className="px-4 py-2 text-sm font-medium text-white rounded-lg bg-blue-600 hover:bg-blue-700"
                onClick={() => { void resolveTextOnlyIndexWarning(true); }}
              >
                {t('textOnlyIndexWarningDialog.confirm')}
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;
