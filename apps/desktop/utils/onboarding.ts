import { invoke } from '@tauri-apps/api/core';
import type { OnboardingStep } from '../types';

export const ONBOARDING_STORAGE_KEY = 'unfoldly-onboarding-complete';
const LEGACY_STEP_KEY = `${ONBOARDING_STORAGE_KEY}-step`;

const SETTINGS_COMPLETE_KEY = 'onboarding_complete';
const SETTINGS_STEP_KEY = 'onboarding_step';
const SETTINGS_MIGRATED_KEY = 'onboarding_legacy_migrated';

const VALID_STEPS: OnboardingStep[] = [
  'welcome',
  'model-recommend',
  'setup',
  'loading-models',
  'indexing-guide',
  'indexing-progress',
  'complete',
];

function normalizeStep(step: unknown): OnboardingStep {
  const s = String(step || '').trim();
  if (VALID_STEPS.includes(s as OnboardingStep)) {
    return s as OnboardingStep;
  }
  return 'welcome';
}

type OnboardingPersistedState = {
  isComplete: boolean;
  step: OnboardingStep;
};

function readLegacyLocalStorage(): OnboardingPersistedState {
  try {
    const complete = localStorage.getItem(ONBOARDING_STORAGE_KEY) === 'true';
    const rawStep = localStorage.getItem(LEGACY_STEP_KEY);
    const step = rawStep ? normalizeStep(rawStep) : (complete ? 'complete' : 'welcome');
    return { isComplete: complete, step };
  } catch {
    return { isComplete: false, step: 'welcome' };
  }
}

async function loadFromBackendSettings(): Promise<OnboardingPersistedState | null> {
  try {
    const settings = (await invoke('get_settings')) as Record<string, any>;
    const hasComplete = Object.prototype.hasOwnProperty.call(settings || {}, SETTINGS_COMPLETE_KEY);
    const hasStep = Object.prototype.hasOwnProperty.call(settings || {}, SETTINGS_STEP_KEY);
    if (!hasComplete && !hasStep) return null;

    const isComplete = Boolean(settings?.[SETTINGS_COMPLETE_KEY]);
    const step = normalizeStep(settings?.[SETTINGS_STEP_KEY]);
    return { isComplete, step: isComplete ? 'complete' : step };
  } catch {
    return null;
  }
}

async function saveToBackend(state: OnboardingPersistedState): Promise<void> {
  await Promise.all([
    invoke('update_settings', { key: SETTINGS_COMPLETE_KEY, value: state.isComplete }),
    invoke('update_settings', { key: SETTINGS_STEP_KEY, value: state.step }),
    // Guard key: once written, future startups won't re-import stale legacy localStorage.
    invoke('update_settings', { key: SETTINGS_MIGRATED_KEY, value: true }),
  ]);
}

async function migrateLegacyIfNeeded(): Promise<OnboardingPersistedState> {
  try {
    const settings = (await invoke('get_settings')) as Record<string, any>;
    const hasComplete = Object.prototype.hasOwnProperty.call(settings || {}, SETTINGS_COMPLETE_KEY);
    const hasStep = Object.prototype.hasOwnProperty.call(settings || {}, SETTINGS_STEP_KEY);
    const migrated = Boolean(settings?.[SETTINGS_MIGRATED_KEY]);
    if (hasComplete || hasStep) {
      const isComplete = Boolean(settings?.[SETTINGS_COMPLETE_KEY]);
      const step = normalizeStep(settings?.[SETTINGS_STEP_KEY]);
      return { isComplete, step: isComplete ? 'complete' : step };
    }
    if (migrated) {
      // Backend has already completed a migration before; do not resurrect from legacy localStorage again.
      const fresh: OnboardingPersistedState = { isComplete: false, step: 'welcome' };
      await saveToBackend(fresh);
      try {
        localStorage.removeItem(ONBOARDING_STORAGE_KEY);
        localStorage.removeItem(LEGACY_STEP_KEY);
      } catch {
        // ignore
      }
      return fresh;
    }
  } catch {
    // ignore and try legacy fallback below
  }

  const legacy = readLegacyLocalStorage();
  try {
    await saveToBackend(legacy);
    try {
      localStorage.removeItem(ONBOARDING_STORAGE_KEY);
      localStorage.removeItem(LEGACY_STEP_KEY);
    } catch {
      // ignore
    }
  } catch {
    // ignore backend save error and still return legacy snapshot
  }
  return legacy;
}

export async function getOnboardingState(): Promise<OnboardingPersistedState> {
  return await migrateLegacyIfNeeded();
}

export async function checkOnboardingStatus(): Promise<boolean> {
  const st = await getOnboardingState();
  return st.isComplete;
}

export async function markOnboardingComplete(): Promise<void> {
  await saveToBackend({ isComplete: true, step: 'complete' });
}

export async function getOnboardingStep(): Promise<OnboardingStep> {
  const st = await getOnboardingState();
  return st.isComplete ? 'complete' : st.step;
}

export async function saveOnboardingStep(step: OnboardingStep): Promise<void> {
  const current = await getOnboardingState();
  const nextStep = normalizeStep(step);
  await saveToBackend({ isComplete: current.isComplete, step: nextStep });
}

export async function resetOnboarding(): Promise<void> {
  try {
    await saveToBackend({ isComplete: false, step: 'welcome' });
    try {
      localStorage.removeItem(ONBOARDING_STORAGE_KEY);
      localStorage.removeItem(LEGACY_STEP_KEY);
    } catch {
      // ignore
    }
    window.location.reload();
  } catch {
    // ignore
  }
}

if (typeof window !== 'undefined' && import.meta.env.DEV) {
  (window as any).resetOnboarding = () => {
    void resetOnboarding();
  };
}
