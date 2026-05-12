import React, { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import Modal from './Modal';
import { RefreshCw, Globe, Keyboard, Shield, Info, ExternalLink, X } from 'lucide-react';
import { getVersion } from '@tauri-apps/api/app';
import { Channel, invoke } from '@tauri-apps/api/core';
import { relaunch } from '@tauri-apps/plugin-process';
import { check, type DownloadEvent } from '@tauri-apps/plugin-updater';
import { openExternalUrl, updateSettings } from '../backend';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type SettingsTab = 'updates' | 'privacy' | 'general' | 'about';

const HOMEPAGE_URL = 'https://www.unfoldly.io/';
const PRIVACY_POLICY_URL = 'https://www.unfoldly.io/privacy';
const GITHUB_REPOSITORY_URL = 'https://github.com/Unary-Works/Unfoldly';

type PendingUpdate = NonNullable<Awaited<ReturnType<typeof check>>>;
type UpdateStatus = 'idle' | 'checking' | 'up-to-date' | 'available' | 'downloading' | 'ready' | 'error';
type AvailableUpdateInfo = {
  version: string;
  notes?: string;
};
type CancelableDownloadEvent = DownloadEvent | { event: 'Cancelled' };

const UPDATE_DOWNLOAD_CANCELLED = 'update_download_cancelled';

function shortReleaseNotes(notes?: string): string {
  return String(notes || '')
    .trim()
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean)
    .slice(0, 2)
    .join('\n\n');
}

const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<SettingsTab>('general');
  const [appVersion, setAppVersion] = useState<string>('1.0.0');
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus>('idle');
  const [availableUpdate, setAvailableUpdate] = useState<AvailableUpdateInfo | null>(null);
  const [pendingUpdate, setPendingUpdate] = useState<PendingUpdate | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<number>(0);
  const downloadRunRef = useRef<number>(0);
  const { t, i18n } = useTranslation();

  useEffect(() => {
    let cancelled = false;
    getVersion()
      .then((version) => {
        if (!cancelled && version) setAppVersion(version);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const handleLanguageChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newLang = String(e.target.value || '').toLowerCase().startsWith('zh') ? 'zh' : 'en';
    i18n.changeLanguage(newLang);
    try {
      await updateSettings('language', newLang);
    } catch (err) {
      console.warn('Failed to persist language setting:', err);
    }
  };

  const handleOpenExternalUrl = async (url: string) => {
    try {
      await openExternalUrl(url);
    } catch (err) {
      console.warn('Failed to open external URL:', err);
    }
  };

  const handleCheckUpdate = async () => {
    setUpdateStatus('checking');
    setAvailableUpdate(null);
    setPendingUpdate(null);
    setDownloadProgress(0);
    try {
      const update = await check();
      if (!update) {
        setUpdateStatus('up-to-date');
        return;
      }
      setPendingUpdate(update);
      setAvailableUpdate({
        version: update.version,
        notes: update.body || '',
      });
      setUpdateStatus('available');
    } catch (err) {
      console.warn('Failed to check for updates:', err);
      setUpdateStatus('error');
    }
  };

  const handleDownloadAndInstall = async () => {
    if (!pendingUpdate) return;
    const downloadRunId = downloadRunRef.current + 1;
    downloadRunRef.current = downloadRunId;
    setUpdateStatus('downloading');
    setDownloadProgress(0);
    try {
      let downloaded = 0;
      let contentLength = 0;
      const onEvent = new Channel<CancelableDownloadEvent>();
      onEvent.onmessage = (event) => {
        if (downloadRunRef.current !== downloadRunId) return;
        switch (event.event) {
          case 'Started':
            downloaded = 0;
            contentLength = Number(event.data?.contentLength || 0);
            setDownloadProgress(0);
            break;
          case 'Progress':
            downloaded += Number(event.data?.chunkLength || 0);
            if (contentLength > 0) {
              setDownloadProgress(Math.min(99, Math.round((downloaded / contentLength) * 100)));
            }
            break;
          case 'Finished':
            setDownloadProgress(100);
            break;
        }
      };
      await invoke('download_and_install_update_cancelable', {
        rid: pendingUpdate.rid,
        onEvent,
      });
      if (downloadRunRef.current !== downloadRunId) return;
      setUpdateStatus('ready');
    } catch (err) {
      if (downloadRunRef.current !== downloadRunId || String(err).includes(UPDATE_DOWNLOAD_CANCELLED)) {
        return;
      }
      console.warn('Failed to download and install update:', err);
      setUpdateStatus('error');
    }
  };

  const handleCancelUpdateDownload = () => {
    if (updateStatus !== 'downloading') return;
    downloadRunRef.current += 1;
    setDownloadProgress(0);
    setUpdateStatus(pendingUpdate ? 'available' : 'idle');
    void invoke('cancel_update_download').catch((err) => {
      console.warn('Failed to cancel update download:', err);
    });
  };

  const releaseNotes = shortReleaseNotes(availableUpdate?.notes);
  const statusText =
    updateStatus === 'checking'
      ? t('settings.checkingUpdate')
      : updateStatus === 'available'
        ? t('settings.updateAvailable', { version: availableUpdate?.version || '' })
        : updateStatus === 'downloading'
          ? t('settings.downloadingUpdate', { progress: downloadProgress })
          : updateStatus === 'ready'
            ? t('settings.updateReady')
            : updateStatus === 'error'
              ? t('settings.updateCheckFailed')
              : t('settings.upToDate');

  const updateButtonText =
    updateStatus === 'checking'
      ? t('settings.checkingShort')
      : updateStatus === 'available'
        ? t('settings.updateNow')
        : updateStatus === 'downloading'
          ? t('settings.downloadingShort', { progress: downloadProgress })
          : updateStatus === 'ready'
            ? t('settings.restartToUpdate')
            : updateStatus === 'error'
              ? t('settings.tryAgain')
              : t('settings.checkUpdate');

  const handleUpdateAction = () => {
    if (updateStatus === 'available') {
      void handleDownloadAndInstall();
      return;
    }
    if (updateStatus === 'ready') {
      void relaunch();
      return;
    }
    if (updateStatus === 'checking' || updateStatus === 'downloading') {
      return;
    }
    void handleCheckUpdate();
  };

  const updateButtonDisabled = updateStatus === 'checking' || updateStatus === 'downloading';
  const updateButtonPrimary = updateStatus === 'available' || updateStatus === 'ready';

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={t('settings.title')} width="w-[700px]">
      <div className="flex h-[400px]">
        {/* Sidebar Nav */}
        <div className="w-48 bg-gray-50/50 border-r border-gray-100 p-3 flex flex-col gap-1">
          <button
            onClick={() => setActiveTab('updates')}
            className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2
              ${activeTab === 'updates' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'}
            `}
          >
            <RefreshCw size={16} />
            {t('settings.updates')}
          </button>
          <button
            onClick={() => setActiveTab('privacy')}
            className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2
              ${activeTab === 'privacy' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'}
            `}
          >
            <Shield size={16} />
            {t('settings.privacy')}
          </button>
          <button
            onClick={() => setActiveTab('general')}
            className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2
              ${activeTab === 'general' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'}
            `}
          >
            <Globe size={16} />
            {t('settings.general')}
          </button>
          <button
            onClick={() => setActiveTab('about')}
            className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2
              ${activeTab === 'about' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'}
            `}
          >
            <Info size={16} />
            {t('settings.about', 'About Us')}
          </button>
        </div>

        {/* Content Area */}
        <div className="flex-1 p-6 overflow-y-auto bg-white">
          
          {activeTab === 'updates' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-base font-semibold text-gray-900 mb-1">{t('settings.appUpdates')}</h3>
                <p className="text-sm text-gray-500">{t('settings.currentVersion', { version: appVersion })}</p>
                <p className="text-xs text-gray-400 mt-0.5">
                  {t('settings.buildDate', 'Build Date')}: {typeof __BUILD_DATE__ !== 'undefined' ? __BUILD_DATE__ : 'dev'}
                </p>
              </div>
              
              <div className="flex max-w-sm flex-col items-start gap-4">
                <div className="space-y-2">
                  <div className={`text-sm font-medium ${
                    updateStatus === 'error'
                      ? 'text-red-600'
                      : updateStatus === 'available' || updateStatus === 'downloading' || updateStatus === 'ready'
                        ? 'text-blue-700'
                        : 'text-gray-700'
                  }`}>
                    {statusText}
                  </div>
                  {releaseNotes && updateStatus === 'available' && (
                    <p className="max-w-sm whitespace-pre-wrap text-sm leading-6 text-gray-500">{releaseNotes}</p>
                  )}
                </div>
                {updateStatus === 'downloading' && (
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-72 overflow-hidden rounded-full bg-gray-100">
                      <div
                        className="h-full rounded-full bg-gray-900 transition-all"
                        style={{ width: `${downloadProgress}%` }}
                      />
                    </div>
                    <button
                      type="button"
                      onClick={handleCancelUpdateDownload}
                      aria-label={t('settings.cancelUpdate')}
                      title={t('settings.cancelUpdate')}
                      className="flex h-6 w-6 items-center justify-center rounded-full text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-700 active:scale-95"
                    >
                      <X size={14} strokeWidth={2} />
                    </button>
                  </div>
                )}
                <button
                  type="button"
                  onClick={handleUpdateAction}
                  disabled={updateButtonDisabled}
                  className={`min-w-40 px-4 py-2 rounded-lg border text-sm font-medium transition-all shadow-sm active:scale-95 disabled:cursor-not-allowed disabled:opacity-60 ${
                    updateButtonPrimary
                      ? 'border-gray-900 bg-gray-900 text-white hover:bg-gray-800'
                      : 'border-gray-200 bg-white text-gray-700 hover:bg-gray-50 hover:border-gray-300'
                  }`}
                >
                  {updateButtonText}
                </button>
              </div>
            </div>
          )}

          {activeTab === 'privacy' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-base font-semibold text-gray-900 mb-1">{t('settings.privacyTitle')}</h3>
                <p className="text-sm text-gray-500 max-w-sm">
                  {t('settings.privacyDesc')}
                </p>
              </div>
              
              <button
                type="button"
                onClick={() => void handleOpenExternalUrl(PRIVACY_POLICY_URL)}
                className="inline-flex items-center text-sm font-medium text-blue-600 hover:text-blue-800 hover:underline"
              >
                {t('settings.readPrivacy')}
              </button>
            </div>
          )}

          {activeTab === 'general' && (
            <div className="space-y-8">
              {/* Language */}
              <div className="space-y-3">
                <label className="block text-sm font-medium text-gray-700">{t('settings.language')}</label>
                <div className="relative inline-block w-64">
                   <select 
                     value={i18n.language}
                     onChange={handleLanguageChange}
                     className="block w-full pl-3 pr-10 py-2 text-sm border-gray-300 focus:outline-none focus:ring-gray-900 focus:border-gray-900 sm:text-sm rounded-md border bg-white shadow-sm cursor-pointer"
                   >
                    <option value="en">English (United States)</option>
                    <option value="zh">简体中文</option>
                   </select>
                </div>
              </div>

              {/* Shortcuts */}
              <div className="space-y-3">
                 <h4 className="text-sm font-medium text-gray-700 flex items-center gap-2">
                   <Keyboard size={16} className="text-gray-400" />
                   {t('settings.keyboardShortcuts')}
                 </h4>
                 <div className="border border-gray-100 rounded-lg overflow-hidden">
                   <div className="flex items-center justify-between px-4 py-2.5 bg-gray-50 border-b border-gray-100">
                     <span className="text-sm text-gray-600">{t('settings.newChat')}</span>
                     <kbd className="px-2 py-0.5 bg-white border border-gray-200 rounded text-xs font-sans text-gray-500 shadow-sm">⌘ N</kbd>
                   </div>
                   <div className="flex items-center justify-between px-4 py-2.5 bg-white border-b border-gray-100">
                     <span className="text-sm text-gray-600">{t('settings.quickSearch')}</span>
                     <kbd className="px-2 py-0.5 bg-white border border-gray-200 rounded text-xs font-sans text-gray-500 shadow-sm">⌘ K</kbd>
                   </div>
                   <div className="flex items-center justify-between px-4 py-2.5 bg-gray-50">
                     <span className="text-sm text-gray-600">{t('settings.closeWindow')}</span>
                     <kbd className="px-2 py-0.5 bg-white border border-gray-200 rounded text-xs font-sans text-gray-500 shadow-sm">⌘ W</kbd>
                   </div>
                 </div>
              </div>
            </div>
          )}

          {activeTab === 'about' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-1">Unfoldly</h3>
                <p className="text-sm text-gray-500 max-w-sm">
                  {t('settings.aboutDesc', 'Search your files by memory.')}
                </p>
              </div>
              
              <div className="space-y-4">
                <button
                  type="button"
                  onClick={() => void handleOpenExternalUrl(HOMEPAGE_URL)}
                  className="flex w-full items-center gap-3 text-left text-sm font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-50 p-4 rounded-xl border border-gray-100 transition-all shadow-sm"
                >
                  <div className="w-8 h-8 rounded-full bg-blue-50 flex items-center justify-center">
                     <Globe size={16} className="text-blue-600" />
                  </div>
                  <div className="flex-1">
                    <div className="text-gray-900">{t('settings.officialHomepage', 'Official Homepage')}</div>
                    <div className="text-xs text-gray-500 font-normal text-left">{HOMEPAGE_URL}</div>
                  </div>
                  <ExternalLink size={16} className="text-gray-300" />
                </button>

                <button
                  type="button"
                  onClick={() => void handleOpenExternalUrl(GITHUB_REPOSITORY_URL)}
                  className="flex w-full items-center gap-3 text-left text-sm font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-50 p-4 rounded-xl border border-gray-100 transition-all shadow-sm"
                >
                  <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">
                     <svg height="16" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" className="fill-current text-gray-700"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path></svg>
                  </div>
                  <div className="flex-1">
                    <div className="text-gray-900">{t('settings.githubRepo', 'GitHub Repository')}</div>
                    <div className="text-xs text-gray-500 font-normal text-left">{GITHUB_REPOSITORY_URL}</div>
                  </div>
                  <ExternalLink size={16} className="text-gray-300" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </Modal>
  );
};

export default SettingsModal;
