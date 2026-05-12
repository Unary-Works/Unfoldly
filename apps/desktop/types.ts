export interface Model {
  id: string;
  name: string;
  model_dir?: string;
  selected_model_path?: string;
  selected_mmproj_path?: string;
  size?: string;
  status?: 'installed' | 'downloading' | 'available' | 'error' | 'cancelled';
  progress?: number; // 0-100
  downloaded_bytes?: number;
  total_bytes?: number;
  download_speed?: number;
  eta_seconds?: number | null;
  downloadingSource?: 'modelscope' | 'hf';
  downloading_quantization_file?: string;
  error?: string;
  sources?: {
    modelscope?: { repo_id?: string };
    hf?: { repo_id?: string };
  };
  default_quantization?: string;
  selected_quantization?: string;
  quantizations?: Array<{ file: string; level?: string; note?: string; size_bytes?: number; recommended?: boolean }>;
  recommended?: boolean;
  recommended_quantization?: string;
  type?: string;
  format?: 'gguf' | 'hf';
  file_sizes?: Record<string, number>;
  installed_quantizations?: string[];
  selected?: boolean;
  downloadProgress?: number; // 0-100
  description?: string;
}

export interface FileSource {
  id: string;
  name: string;
  type: 'file' | 'folder';
  path?: string;
  size?: string;
  fileCount?: number; // For folders
  addedAt?: string;
  iconType: 'pdf' | 'doc' | 'image' | 'sheet' | 'folder';
  children?: FileSource[];
  status?: 'indexing' | 'indexed' | 'pending';
}

export interface RelevantFile {
  id: string;
  name: string;
  type: 'pdf' | 'doc' | 'image' | 'sheet' | 'folder';
  path?: string;
  tree_path?: string;
  from_folder_chain?: boolean;
  doc_summary?: string;
  is_matched_folder?: boolean;
  child_file_count?: number;
  folder_chain_root?: string;
}

export interface OpenedFile {
  file_path: string;
  file_name: string;
  iconType: 'pdf' | 'doc' | 'image' | 'sheet' | 'folder';
  content: string;
  truncated?: boolean;
  openedAt: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string; // supports markdown-like structure visually
  timestamp: number;
  // Search specific fields
  isSearch?: boolean;
  scanState?: 'thinking' | 'completed';
  scanProgress?: number; // 0 to 100
  relevantFiles?: RelevantFile[];
  trace?: any[];
  statusText?: string;
  thinkingContent?: string;
  relevantFilesAll?: RelevantFile[];
  relevantFilesTotal?: number;
  relevantFilesShown?: number;
  model?: Model;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  lastActive: number;
}

export interface IndexingState {
  isIndexing: boolean;
  totalFiles: number;
  completedFiles: number;
  eta: string;
  isTopBarVisible: boolean;
  isCancelling?: boolean;
  isRestoringModel?: boolean;
  statusMessage?: string;
  currentFile?: string;
  currentPath?: string;
  currentFrame?: number;
  totalFrames?: number;
  stage?: string;
}

export type ViewState = 'landing' | 'conversation';
export type SidebarMode = 'tree' | 'openedFile';

// --- Onboarding ---
export type OnboardingStep = 'welcome' | 'model-recommend' | 'setup' | 'loading-models' | 'indexing-guide' | 'indexing-progress' | 'complete';

export interface OnboardingState {
  step: OnboardingStep;
  isComplete: boolean;
  setupProgress: number; // 0-100 for model download
}