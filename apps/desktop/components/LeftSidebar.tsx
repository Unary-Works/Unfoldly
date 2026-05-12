import React from 'react';
import { PlusCircle, Settings, Box, MessageSquare, Trash2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Conversation } from '../types';

interface LeftSidebarProps {
  history: Conversation[];
  activeConversationId: string | null;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  onDeleteChat?: (id: string) => void;
  onOpenSettings: () => void;
  onOpenManageModels: () => void;
  isGenerating?: boolean;
  hasModels?: boolean;
  onShowModelPrompt?: () => void;
}

const LeftSidebar: React.FC<LeftSidebarProps> = ({ 
  history, 
  activeConversationId, 
  onNewChat, 
  onSelectChat,
  onDeleteChat,
  onOpenSettings,
  onOpenManageModels,
  isGenerating = false,
  hasModels = true,
  onShowModelPrompt,
}) => {
  const { t } = useTranslation();
  return (
    <div className="w-[260px] flex-shrink-0 h-full bg-[#F7F7F5] border-r border-gray-200 flex flex-col text-gray-700 select-none">
      {/* Top: New Chat */}
      <div className="h-14 flex items-center px-4 relative z-[110]">
        <button
          onClick={() => {
            if (!hasModels && onShowModelPrompt) {
              onShowModelPrompt();
              return;
            }
            onNewChat();
          }}
          className="group flex items-center gap-2 text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors w-full px-2 py-1.5 rounded-md hover:bg-gray-200/50 relative z-[120]"
          style={{ WebkitAppRegion: 'no-drag' } as any}
        >
          <PlusCircle size={18} className="text-gray-400 group-hover:text-gray-600" />
          <span>{t('sidebar.newChat')}</span>
        </button>
      </div>

      {/* Middle: History */}
      <div className="flex-1 overflow-y-auto px-4 py-2">
        <div className="text-xs font-semibold text-gray-400 mb-2 px-2 uppercase tracking-wide">
          {t('sidebar.chatHistory')}
        </div>
        <div className="space-y-0.5">
          {history.map((chat) => (
            <div
              key={chat.id}
              className={`group w-full px-2 py-1.5 rounded-md flex items-center justify-between gap-2 cursor-pointer transition-colors
                ${activeConversationId === chat.id 
                  ? 'bg-gray-200 text-gray-900 font-medium' 
                  : 'text-gray-600 hover:bg-gray-200/50 hover:text-gray-900'
                }`}
              onClick={() => onSelectChat(chat.id)}
            >
              <div className="flex-1 truncate text-sm">{chat.title}</div>
              {onDeleteChat && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteChat(chat.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-600 transition-opacity rounded hover:bg-gray-300/50"
                  title={t('sidebar.deleteChat')}
                >
                  <Trash2 size={14} />
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Bottom: Settings */}
      <div className="p-4 border-t border-gray-200/50 space-y-1">
        <button 
          onClick={onOpenSettings}
          className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-sm text-gray-600 hover:bg-gray-200/50 hover:text-gray-900 transition-colors"
        >
          <Settings size={16} />
          <span>{t('sidebar.settings')}</span>
        </button>
        <button 
          onClick={onOpenManageModels}
          className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-sm text-gray-600 hover:bg-gray-200/50 hover:text-gray-900 transition-colors"
        >
          <Box size={16} />
          <span>{t('sidebar.manageModels')}</span>
        </button>
      </div>
    </div>
  );
};

export default LeftSidebar;