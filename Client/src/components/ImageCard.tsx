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

  // Chuyển fileId thành direct link Googleusercontent (đồng bộ với App)
  const getImageSrc = (url: string) => {
    const fileId = extractFileId(url);
    return fileId ? `https://lh3.googleusercontent.com/d/${fileId}=w1000` : '';
  };

  const getNameImage = (name: string) => {
    return name.split("#")[1];
  };

  // Xử lý lỗi tải ảnh: hiện thông báo cụ thể (có kiểm tra lỗi 429)
  const handleImageError = async () => {
    const url = image.url.value || "";
    const fileId = extractFileId(url);
    const directLink = fileId ? `https://lh3.googleusercontent.com/d/${fileId}=w1000` : '';
    if (!fileId) {
      setLoadError("Không lấy được fileId từ link ảnh. Link không đúng định dạng hoặc thiếu thông tin.");
      return;
    }
    try {
      const response = await fetch(directLink, { method: "GET" });
      if (response.status === 429) {
        setLoadError("Bạn đã tải quá nhiều ảnh từ Google Drive trong thời gian ngắn. Vui lòng chờ một lúc rồi thử lại (429 Too Many Requests).");
      } else if (!response.ok) {
        setLoadError(`Lỗi tải ảnh từ Google Drive: ${response.status} ${response.statusText}`);
      } else {
        setLoadError("Ảnh không thể tải được. Lỗi không xác định.");
      }
    } catch (e: any) {
      setLoadError("Không thể lấy chi tiết lỗi từ Google Drive (có thể do CORS). Ảnh không thể tải được.");
    }
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