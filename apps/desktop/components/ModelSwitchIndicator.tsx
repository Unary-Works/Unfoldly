import React, { useEffect, useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { CheckCircle2, XCircle } from 'lucide-react';
import { Model } from '../types';
import { formatModelName } from '../utils/modelDisplay';

interface ModelSwitchIndicatorProps {
  isSwitching: boolean;
  targetModelId: string | null;
  installedModels: Model[];
  switchError?: boolean;
}

const ModelSwitchIndicator: React.FC<ModelSwitchIndicatorProps> = ({
  isSwitching,
  targetModelId,
  installedModels,
  switchError = false,
}) => {
  const { t } = useTranslation();
  const [showSuccess, setShowSuccess] = useState(false);
  const [showError, setShowError] = useState(false);
  const prevSwitchingRef = useRef(false);

  useEffect(() => {
    const wasSwitching = prevSwitchingRef.current;
    
    // When finishing a switch
    if (wasSwitching && !isSwitching) {
      let timer: ReturnType<typeof setTimeout>;
      if (switchError) {
        setShowError(true);
        timer = setTimeout(() => setShowError(false), 3000);
      } else {
        setShowSuccess(true);
        timer = setTimeout(() => setShowSuccess(false), 2000);
      }
      
      // Update ref to false
      prevSwitchingRef.current = isSwitching;
      return () => clearTimeout(timer);
    }
    
    // When starting a switch
    if (!wasSwitching && isSwitching) {
      setShowSuccess(false);
      setShowError(false);
    }
    
    prevSwitchingRef.current = isSwitching;
  }, [isSwitching, switchError]);

  if (!isSwitching && !showSuccess && !showError) return null;

  const targetModel = installedModels.find(m => m.id === targetModelId);
  const modelName = formatModelName(targetModel?.name || targetModelId || 'Model');

  return (
    <div className="absolute top-6 left-1/2 -translate-x-1/2 z-[200] pointer-events-none transition-all duration-300">
      <div className="flex items-center gap-3 px-4 py-2.5 bg-white/95 backdrop-blur-sm rounded-full shadow-lg border border-gray-100/50 min-w-[280px]">
        {showSuccess ? (
          <>
            <CheckCircle2 className="w-5 h-5 text-green-500 shrink-0" />
            <span className="text-sm font-medium text-gray-700">
              {t('chat.switchModelSuccess', { defaultValue: 'Switched to' })} {modelName}
            </span>
          </>
        ) : showError ? (
          <>
            <XCircle className="w-5 h-5 text-red-500 shrink-0" />
            <span className="text-sm font-medium text-gray-700">
              {t('chat.switchModelFailed', { defaultValue: 'Failed to switch to' })} {modelName}
            </span>
          </>
        ) : (
          <>
            <div className="flex-1 flex flex-col gap-1.5">
              <span className="text-sm font-medium text-gray-700 whitespace-nowrap">
                {t('chat.switchingModel', { defaultValue: 'Switching to' })} {modelName}...
              </span>
              <div className="h-1 w-full bg-gray-100 rounded-full overflow-hidden">
                <div className="h-full bg-black rounded-full w-1/3 animate-[slide_1.5s_ease-in-out_infinite]" />
              </div>
            </div>
          </>
        )}
      </div>
      <style>{`
        @keyframes slide {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(300%); }
        }
      `}</style>
    </div>
  );
};

export default ModelSwitchIndicator;
