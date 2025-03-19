import { useLocation } from "react-router-dom";
import { useState, useEffect } from "react";

const convertGoogleDriveThumbnail = (url: string | null, width = 1000, height = 1000) => {
  if (!url) return null;
  const match = url.match(/id=([^&]+)/);
  return match ? `https://drive.google.com/thumbnail?id=${match[1]}&sz=w${width}-h${height}` : url;
};

export function ImageView() {
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  const imageURL = convertGoogleDriveThumbnail(params.get("imageURL"));
  const imageName = params.get("imageName");

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (imageURL) {
      const img = new Image();
      img.src = imageURL;
      img.onload = () => setLoading(false);
    }
  }, [imageURL]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-2xl font-bold text-gray-800 mb-4">{imageName}</h1>
      {imageURL ? (
        <div className="relative w-[400px] h-[400px] flex items-center justify-center bg-gray-300 rounded-lg shadow-lg">
          {loading ? (
            <div className="w-full h-full animate-pulse bg-gray-400" />
          ) : (
            <img
              src={imageURL}
              alt={imageName || "Image"}
              className="rounded-lg shadow-lg object-contain max-w-full max-h-full"
              loading="lazy"
              referrerPolicy="no-referrer"
            />
          )}
        </div>
      ) : (
        <p className="text-gray-600">Không tìm thấy ảnh.</p>
      )}
    </div>
  );
}
