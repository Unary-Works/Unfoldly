import { Model, FileSource, Conversation } from './types';

export const INITIAL_MODELS: Model[] = [
  { id: 'qwen3-8b', name: 'Qwen3-8B', size: '4.8 GB', status: 'installed' },
  { id: 'llama-3-70b', name: 'Llama-3-70B', size: '42 GB', status: 'installed' },
  { id: 'mistral-large', name: 'Mistral Large', size: '24 GB', status: 'installed' },
  { id: 'gemma-7b', name: 'Gemma-7B-it', size: '5.2 GB', status: 'installed' },
  { id: 'deepseek-coder', name: 'Deepseek Coder 6.7B', size: '6.7 GB', status: 'available' },
  { id: 'phi-3-mini', name: 'Phi-3 Mini', size: '2.4 GB', status: 'available' },
  { id: 'falcon-180b', name: 'Falcon 180B', size: '100+ GB', status: 'available' },
];

// Fallback for simple usage if needed, but App.tsx will use INITIAL_MODELS
export const MOCK_MODELS = INITIAL_MODELS; 

export const INITIAL_SOURCES: FileSource[] = [];

export const INITIAL_HISTORY: Conversation[] = [];