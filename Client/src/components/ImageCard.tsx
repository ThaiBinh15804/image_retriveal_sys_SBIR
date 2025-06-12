import { imageQueryType } from "../App";
import { useState } from "react";

export function ImageCard({ image, onClick }: { image: imageQueryType; onClick: () => void }) {
  const [loadError, setLoadError] = useState<string>("");

  // Tách file ID từ URL (đồng bộ với App)
  const extractFileId = (url: string) => {
    let match = url.match(/[?&]id=([^&]+)/);
    if (match) return match[1];
    match = url.match(/\/d\/([a-zA-Z0-9_-]+)/);
    if (match) return match[1];
    match = url.match(/file\/d\/([a-zA-Z0-9_-]+)/);
    if (match) return match[1];
    match = url.match(/open\?id=([^&]+)/);
    if (match) return match[1];
    return null;
  };

  // Chuyển fileId thành link thumbnail Google Drive (đồng bộ với App)
  const getImageSrc = (url: string) => {
    const fileId = extractFileId(url);
    return fileId ? `https://drive.google.com/thumbnail?id=${fileId}&sz=w300` : '';
  };

  const getNameImage = (name: string) => {
    return name.split("#")[1];
  };

  // Xử lý lỗi tải ảnh: chỉ hiện thông báo đơn giản
  const handleImageError = () => {
    setLoadError("Không thể tải ảnh. File có thể không công khai hoặc đã bị xóa.");
  };

  return (
    <div
      className="group relative bg-white rounded-xl overflow-hidden shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105 cursor-pointer"
      onClick={onClick}
    >
      <div className="aspect-square relative">
        <img
          src={getImageSrc(image.url.value)}
          alt={image.image.value}
          className="w-full h-full object-cover"
          loading="lazy"
          onError={handleImageError}
        />
        {loadError && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 text-white text-sm p-2">
            {loadError}
          </div>
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-purple-900/70 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      </div>
      <div className="absolute bottom-0 left-0 right-0 p-4 text-white transform translate-y-full group-hover:translate-y-0 transition-transform duration-300">
        <p className="text-sm font-medium truncate">{getNameImage(image.image.value)}</p>
      </div>
    </div>
  );
}