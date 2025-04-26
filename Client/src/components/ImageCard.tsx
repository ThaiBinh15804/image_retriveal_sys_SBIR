import { imageQueryType } from "../App";
import { useState } from "react";
import { gapi } from "gapi-script";

export function ImageCard({ image, onClick }: { image: imageQueryType; onClick: () => void }) {
  const [loadError, setLoadError] = useState<string>("");

  const convertURLImage = (url: string) => {
    const urlNew = url.split("id=")[1];
    return `https://drive.google.com/thumbnail?id=${urlNew}&sz=w1000`;
  };

  const getNameImage = (name: string) => {
    return name.split("#")[1];
  };

  const checkFileStatus = async (fileId: string) => {
    try {
      const response = await gapi.client.drive.files.get({
        fileId: fileId,
        fields: "id, name, mimeType, webContentLink, error",
      });
      return { success: true, data: response.result };
    } catch (err: any) {
      const errorDetails = err.result?.error || { message: "Unknown error", code: "N/A" };
      return {
        success: false,
        error: `Google Drive API Error: ${errorDetails.message} (Code: ${errorDetails.code})`,
      };
    }
  };

  const handleImageError = async (fileId: string) => {
    console.log(`Image load failed in ImageCard for fileId: ${fileId}`);
    
    const fileStatus = await checkFileStatus(fileId);
    if (!fileStatus.success) {
      setLoadError(fileStatus.error || "An unknown error occurred.");
      console.error(fileStatus.error || "An unknown error occurred.");
      return;
    }

    const { data } = fileStatus;
    console.log("File details:", data);
    if (!data.webContentLink) {
      setLoadError("Application Error: File exists but no webContentLink available.");
    } else {
      setLoadError("Network Error: File exists but failed to load. Possible CORS or network issue.");
    }
  };

  return (
    <div
      className="group relative bg-white rounded-xl overflow-hidden shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105 cursor-pointer"
      onClick={onClick}
    >
      <div className="aspect-square relative">
        <img
          src={convertURLImage(image.url.value)}
          alt={image.image.value}
          className="w-full h-full object-cover"
          loading="lazy"
          onError={() => handleImageError(image.url.value.split("id=")[1])}
          onLoad={() => console.log(`ImageCard loaded successfully: ${image.url.value}`)}
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