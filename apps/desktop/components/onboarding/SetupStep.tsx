import React from 'react';
import { useTranslation } from 'react-i18next';

export interface DownloadItemInfo {
  label: string;
  status: 'idle' | 'available' | 'downloading' | 'installed' | 'error';
  percent?: number;
  speed?: number;       // bytes/sec
  eta?: number;         // seconds
  downloaded_bytes?: number;
  total_bytes?: number;
  error?: string;
}

interface SetupStepProps {
  progress: number;
  items?: DownloadItemInfo[];
}

function isRetryableTransientError(message: string): boolean {
  const s = String(message || '').trim().toLowerCase();
  if (!s) return false;
  const keys = [
    'stalled',
    'stall',
    'network',
    'offline',
    'timeout',
    'timed out',
    'connection',
    'unreachable',
    'refused',
    'reset by peer',
    'failed to fetch',
    'dns',
    'host',
    'ssl',
    'certificate',
    'econn',
    'enet',
    'ehostunreach',
    'eai_again',
    'nodename nor servname',
    'proxy',
    '断网',
    '网络',
    '超时',
    '无法连接',
    '连接失败',
    '无法解析',
    '证书',
    '离线',
    '代理',
  ];
  return keys.some((k) => s.includes(k));
}

function isOfflineLikeError(message: string): boolean {
  const s = String(message || '').trim().toLowerCase();
  if (!s) return false;
  const keys = [
    'offline',
    'disconnected',
    'stalled',
    'timeout',
    'timed out',
    'network is unreachable',
    'ehostunreach',
    'enetunreach',
    'eai_again',
    'dns',
    'nodename nor servname',
    'unable to resolve',
    '网络连接已断开',
    '网络不可达',
    '无法解析',
    '离线',
    '断网',
  ];
  return keys.some((k) => s.includes(k));
}

function formatBytes(bytes: number): string {
  if (bytes <= 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatSpeed(bytesPerSec: number): string {
  if (bytesPerSec <= 0) return '--';
  return `${formatBytes(bytesPerSec)}/s`;
}

function formatEta(seconds: number): string {
  if (seconds <= 0 || !isFinite(seconds)) return '--';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function statusIcon(status: string, retryableError: boolean, isOffline: boolean) {
  if (status === 'installed') return <span className="text-green-600">&#10003;</span>;
  if (status === 'error') {
    if (retryableError) {
      if (!isOffline) return <span className="text-gray-400">&bull;</span>;
      return <span className="text-amber-500 animate-pulse">&#8635;</span>;
    }
    return <span className="text-amber-500">!</span>;
  }
  if (status === 'downloading') return <span className="text-gray-500 animate-pulse">&darr;</span>;
  return <span className="text-gray-400">&bull;</span>;
}

const SPEED_ZERO_STALL_SEC = 5;

const SetupStep: React.FC<SetupStepProps> = ({ progress, items }) => {
  const { t } = useTranslation();

  const [isOffline, setIsOffline] = React.useState(false);
  React.useEffect(() => {
    let mounted = true;
    let prevIsOffline = false;
    const probe = async () => {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 1500);
        await fetch('https://huggingface.co/favicon.ico', {
          mode: 'no-cors',
          cache: 'no-store',
          signal: controller.signal,
        });
        clearTimeout(timer);
        if (mounted) {
          setIsOffline(false);
          if (prevIsOffline) {
            window.dispatchEvent(new Event('setup-network-recovered'));
          }
          prevIsOffline = false;
        }
      } catch {
        if (mounted) {
          setIsOffline(true);
          prevIsOffline = true;
        }
      }
    };
    probe();
    const pollId = window.setInterval(probe, 2000);
    const handleOnline = () => { if (mounted) setIsOffline(false); probe(); };
    const handleOffline = () => { if (mounted) setIsOffline(true); };
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      mounted = false;
      window.clearInterval(pollId);
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);
  
  const zeroSpeedTimerRef = React.useRef<number>(0);
  const prevHadSpeedRef = React.useRef<boolean>(false);
  const [zeroSpeedStalled, setZeroSpeedStalled] = React.useState(false);

  React.useEffect(() => {
    const downloadingItems = (items || []).filter((it) => it.status === 'downloading');
    if (downloadingItems.length === 0) {
      zeroSpeedTimerRef.current = 0;
      prevHadSpeedRef.current = false;
      setZeroSpeedStalled(false);
      return;
    }
    const hasSpeed = downloadingItems.some((it) => (it.speed ?? 0) > 0);
    if (hasSpeed) {
      zeroSpeedTimerRef.current = 0;
      prevHadSpeedRef.current = true;
      setZeroSpeedStalled(false);
      return;
    }
    const now = Date.now() / 1000;
    if (prevHadSpeedRef.current || zeroSpeedTimerRef.current === 0) {
      zeroSpeedTimerRef.current = now;
      prevHadSpeedRef.current = false;
    }
    if (now - zeroSpeedTimerRef.current >= SPEED_ZERO_STALL_SEC) {
      setZeroSpeedStalled(true);
    }
  }, [items]);

  const hasOfflineLikeError = Boolean((items || []).some((item) => {
    if (item.status !== 'error') return false;
    return isOfflineLikeError(String(item.error || ''));
  }));
  const downloadingItemsArr = (items || []).filter((it) => it.status === 'downloading');
  const hasLiveDownloading = downloadingItemsArr.length > 0;
  const hasSpeed = downloadingItemsArr.some((it) => (it.speed ?? 0) > 0);
  
  const showNetworkNotice = !hasSpeed && (isOffline || zeroSpeedStalled || (!hasLiveDownloading && hasOfflineLikeError));

  return (
    <div className="flex flex-col items-center justify-center h-full w-full">
      <div className="flex flex-col items-center w-96">
        <h2 className="font-serif text-2xl text-gray-900 mb-6 text-center">
          {t('onboarding.setupProgressTitle')}
        </h2>

        <div className="flex items-center w-full gap-3 mb-4">
          <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-black transition-all duration-300 ease-out"
              style={{ width: `${Math.min(progress, 100)}%` }}
            />
          </div>
          <span className="font-serif text-base text-gray-900 w-12 text-right">
            {Math.round(progress)}%
          </span>
        </div>

        {showNetworkNotice && (
          <div className="w-full mb-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-amber-800">
            <div className="text-[13px] font-semibold">{t('onboarding.setupNetworkNoticeTitle')}</div>
            <div className="text-xs mt-1 leading-relaxed">{t('onboarding.setupNetworkNoticeDesc')}</div>
          </div>
        )}

        {items && items.length > 0 && (
          <div className="w-full space-y-2">
            {items.map((item, i) => {
              const retryableError = item.status === 'error' && isRetryableTransientError(String(item.error || ''));
              return (
                <div key={i} className="flex flex-col bg-gray-50 rounded-lg px-3 py-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-1.5 text-gray-700 font-medium">
                      {statusIcon(item.status, retryableError, isOffline)} {item.label}
                    </span>
                    <span className={`text-xs ${item.status === 'error' && (isOffline || !retryableError) ? 'text-amber-600' : 'text-gray-500'}`}>
                      {item.status === 'installed' && t('onboarding.setupStatusInstalled')}
                      {item.status === 'error' && (retryableError ? (isOffline ? t('onboarding.setupStatusRetrying') : t('onboarding.setupStatusWaiting')) : (item.error || t('onboarding.setupStatusError')))}
                      {item.status === 'downloading' && item.percent != null && `${item.percent.toFixed(1)}%`}
                      {item.status === 'available' && t('onboarding.setupStatusPending')}
                      {item.status === 'idle' && t('onboarding.setupStatusWaiting')}
                    </span>
                  </div>
                  {item.status === 'downloading' && (
                    <>
                      <div className="w-full h-1 bg-gray-200 rounded-full mt-1.5 overflow-hidden">
                        <div
                          className="h-full bg-blue-500 transition-all duration-300"
                          style={{ width: `${item.percent ?? 0}%` }}
                        />
                      </div>
                      <div className="flex justify-between text-xs text-gray-400 mt-1">
                        <span>
                          {item.downloaded_bytes != null && item.total_bytes != null && item.total_bytes > 0
                            ? `${formatBytes(Math.min(item.downloaded_bytes, item.total_bytes))} / ${formatBytes(item.total_bytes)}`
                            : item.downloaded_bytes != null
                              ? formatBytes(item.downloaded_bytes)
                              : ''}
                        </span>
                        <span>
                          {item.speed != null && item.speed > 0 ? formatSpeed(item.speed) : '--'}
                          {' · '}
                          {item.eta != null && item.eta > 0 ? formatEta(item.eta) : '--'}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default SetupStep;
