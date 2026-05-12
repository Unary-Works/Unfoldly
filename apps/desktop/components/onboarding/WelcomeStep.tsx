import React from 'react';
import { useTranslation } from 'react-i18next';

import logoIcon from '../../assets/logo.png';

interface WelcomeStepProps {
  onNext: () => void;
}

const WelcomeStep: React.FC<WelcomeStepProps> = ({ onNext }) => {
  const { t, i18n } = useTranslation();
  const isZh = String(i18n.language || '').toLowerCase().startsWith('zh');
  const manropeTitleStyle: React.CSSProperties = {
    fontFamily: "'Manrope', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    fontWeight: 300,
    letterSpacing: 0,
  };
  const loraAccentStyle: React.CSSProperties = {
    fontFamily: "'Lora', Georgia, serif",
    fontStyle: 'italic',
    fontWeight: 400,
    letterSpacing: 0,
  };

  return (
    <div className="flex flex-col items-center justify-center h-full w-full">
      <div className="mb-2">
        <img
          src={logoIcon}
          alt="Logo"
          className="object-contain transform scale-125"
          style={{ width: '201.6px', height: '201.6px' }}
          onError={(e) => {
            console.error('[WelcomeStep] Logo failed to load:', logoIcon);
          }}
        />
      </div>

      {isZh ? (
        <h1
          className="text-4xl text-gray-900 mb-12 text-center leading-tight"
          style={manropeTitleStyle}
        >
          {t('onboarding.welcomeTitle')}
        </h1>
      ) : (
        <h1
          className="text-[42px] text-gray-900 mb-12 text-center leading-[1.08]"
          style={manropeTitleStyle}
        >
          <span className="block">Search everything with AI</span>
          <span className="block text-[48px]" style={loraAccentStyle}>locally</span>
        </h1>
      )}

      <button
        onClick={onNext}
        className="px-8 py-3 rounded-lg bg-black text-white font-sans text-base font-light hover:bg-gray-800 transition-colors relative z-[110]"
        style={{ WebkitAppRegion: 'no-drag' } as any}
      >
        {t('onboarding.getStarted')}
      </button>
    </div>
  );
};

export default WelcomeStep;
