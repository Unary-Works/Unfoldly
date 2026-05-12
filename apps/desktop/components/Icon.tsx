import React from 'react';
import { FileText, FileSpreadsheet, Image, Folder, File, FileVideo, FileAudio } from 'lucide-react';

interface FileIconProps {
  type: 'pdf' | 'doc' | 'image' | 'sheet' | 'folder' | 'video' | 'audio';
  className?: string;
}

export const FileIcon: React.FC<FileIconProps> = ({ type, className = "w-4 h-4" }) => {
  switch (type) {
    case 'pdf':
      return <FileText className={`text-red-500 ${className}`} />;
    case 'doc':
      return <FileText className={`text-blue-500 ${className}`} />;
    case 'sheet':
      return <FileSpreadsheet className={`text-green-500 ${className}`} />;
    case 'image':
      return <Image className={`text-purple-500 ${className}`} />;
    case 'folder':
      return <Folder className={`text-gray-500 ${className}`} />;
    case 'video':
      return <FileVideo className={`text-pink-500 ${className}`} />;
    case 'audio':
      return <FileAudio className={`text-yellow-500 ${className}`} />;
    default:
      return <File className={`text-gray-400 ${className}`} />;
  }
};