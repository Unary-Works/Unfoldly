import React from 'react';
import { useTranslation } from 'react-i18next';
import { Shield, EyeOff, Download } from 'lucide-react';

interface ModelRecommendStepProps {
  onDownload: () => void;
  onSkip: () => void;
}

const ModelRecommendStep: React.FC<ModelRecommendStepProps> = ({ onDownload, onSkip }) => {
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
    <div className="flex items-center justify-center h-full w-full">
      <div className="flex w-full max-w-4xl mx-auto px-8">
        {/* Left side */}
        <div className="flex flex-col justify-center flex-1 pr-16">
          {isZh ? (
            <h1
              className="text-5xl text-gray-900 leading-tight mb-12 whitespace-pre-line"
              style={manropeTitleStyle}
            >
              {t('onboarding.modelRecommendTitle')}
            </h1>
          ) : (
            <h1
              className="text-[54px] text-gray-900 leading-[1.04] mb-12"
              style={manropeTitleStyle}
            >
              <span className="block whitespace-nowrap">
                <span>Get </span>
                <span style={loraAccentStyle}>SOTA local</span>
              </span>
              <span className="block">AI models</span>
            </h1>
          )}

          <div className="flex flex-col gap-3 w-80">
            <button
              onClick={onDownload}
              className="w-full px-8 py-3.5 rounded-lg bg-black text-white font-sans text-base font-light hover:bg-gray-800 transition-colors"
              style={{ WebkitAppRegion: 'no-drag' } as any}
            >
              {t('onboarding.modelRecommendDownload')}
            </button>
            <button
              onClick={onSkip}
              className="w-full px-8 py-3.5 rounded-lg bg-white text-gray-600 font-sans text-base font-light border border-gray-200 hover:bg-gray-50 transition-colors"
              style={{ WebkitAppRegion: 'no-drag' } as any}
            >
              {t('onboarding.modelRecommendSkip')}
            </button>
          </div>
        </div>

        {/* Divider */}
        <div className="w-px bg-gray-200 self-stretch my-8" />

        {/* Right side */}
        <div className="flex flex-col justify-center flex-1 pl-16">
          <ul className="space-y-10">
            <li className="flex items-start gap-4">
              <div className="mt-0.5 flex-shrink-0">
                <Shield className="w-5 h-5 text-gray-400" strokeWidth={1.5} />
              </div>
              <span className="font-sans text-lg font-light text-gray-500 leading-relaxed">
                {t('onboarding.modelRecommendPoint1')}
              </span>
            </li>
            <li className="flex items-start gap-4">
              <div className="mt-0.5 flex-shrink-0">
                <EyeOff className="w-5 h-5 text-gray-400" strokeWidth={1.5} />
              </div>
              <span className="font-sans text-lg font-light text-gray-500 leading-relaxed">
                {t('onboarding.modelRecommendPoint2')}
              </span>
            </li>
            <li className="flex items-start gap-4">
              <div className="mt-0.5 flex-shrink-0">
                <Download className="w-5 h-5 text-gray-400" strokeWidth={1.5} />
              </div>
              <span className="font-sans text-lg font-light text-gray-500 leading-relaxed">
                {t('onboarding.modelRecommendPoint3')}
              </span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ModelRecommendStep;
