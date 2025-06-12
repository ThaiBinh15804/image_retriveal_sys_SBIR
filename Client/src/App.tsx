import { useState } from "react";
import { SearchSection } from "./components/SearchSection";
import { ResultsSection } from "./components/ResultsSection";
import axios from "axios";

export type imageQueryType = {
  image: { type: string; value: string };
  url: { type: string; value: string };
};

export function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [searchResults, setSearchResults] = useState<imageQueryType[]>([]);
  const [sparqlQuery, setSparqlQuery] = useState("");
  const [description, setDescription] = useState("");
  const [selectedImage, setSelectedImage] = useState<imageQueryType | null>(null);
  const [imageLoadError, setImageLoadError] = useState<string>("");

  // Tách file ID từ URL (hỗ trợ nhiều dạng URL Google Drive)
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

  // Xử lý lỗi tải ảnh: in ra nguyên nhân cụ thể từ Google Drive nếu có thể
  const handleImageError = async () => {
    const url = selectedImage?.url.value || "";
    const fileId = extractFileId(url);
    const directLink = fileId ? `https://lh3.googleusercontent.com/d/${fileId}=w1000` : "";
    if (!fileId) {
      setImageLoadError("Không lấy được fileId từ link ảnh. Link không đúng định dạng hoặc thiếu thông tin.");
      return;
    }
    try {
      const response = await fetch(directLink, { method: "GET" });
      if (response.status === 429) {
        setImageLoadError("Bạn đã tải quá nhiều ảnh từ Google Drive trong thời gian ngắn. Vui lòng chờ một lúc rồi thử lại (429 Too Many Requests).");
      } else if (!response.ok) {
        setImageLoadError(`Lỗi tải ảnh từ Google Drive: ${response.status} ${response.statusText}`);
      } else {
        setImageLoadError("Ảnh không thể tải được. Lỗi không xác định.");
      }
    } catch (e: any) {
      setImageLoadError("Không thể lấy chi tiết lỗi từ Google Drive (có thể do CORS). Ảnh không thể tải được.");
    }
  };

  const handleSearch = async (query: string | File, type: string) => {
    setIsLoading(true);
    setError("");
    setImageLoadError("");

    try {
      let sparql = type === "text" && typeof query === "string" ? query : "";
      let fromImage = false;
      let main_entity = "";

      if (type === "image" && query instanceof File) {
        const formData = new FormData();
        formData.append("file", query);

        const { data } = await axios.post("http://localhost:8080/analyze/", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });

        if (!data?.description) throw new Error("⚠️ Image analysis failed.");
        sparql = data.description;
        setDescription(data.description);
        fromImage = true;
        main_entity = data.main_entity;
      }

      const { data: results } = await axios.post("http://localhost:2020/query", {
        body: sparql,
        from_image: fromImage,
        main_entity: main_entity,
      });

      setSparqlQuery(results.sparql_query);
      setIsLoading(false);
      if (Array.isArray(results.bindings)) setSearchResults(results.bindings as imageQueryType[]);
      else throw new Error("⚠️ Unexpected response format.");
    } catch (err) {
      setError(String(err));
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen w-full bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      <header className="bg-gradient-to-r from-purple-600 to-pink-600 py-6 px-6 shadow-lg">
        <div className="container mx-auto">
          <h1 className="text-3xl font-bold text-white tracking-tight">
            Image Search<span className="text-yellow-300">.</span>
          </h1>
          <p className="text-purple-100 mt-2">Find the perfect image in seconds ✨</p>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8 flex-grow">
        <SearchSection onSearch={handleSearch} sparqlQuery={sparqlQuery} description={description} />
        <ResultsSection results={searchResults} isLoading={isLoading} error={error} onSelectImage={setSelectedImage} />
      </div>

      {selectedImage && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="relative bg-white rounded-lg shadow-lg p-4 max-w-4xl w-full">
            <button
              className="absolute top-3 right-3 text-gray-700 hover:text-red-500 text-2xl font-bold"
              onClick={() => setSelectedImage(null)}
            >
              ×
            </button>
            <div className="flex flex-col justify-center items-center w-full">
              {selectedImage ? (
                <>
                  <img
                    src={(() => {
                      const fileId = extractFileId(selectedImage.url.value);
                      return fileId ? `https://lh3.googleusercontent.com/d/${fileId}=w1000` : "";
                    })()}
                    alt={selectedImage.image.value}
                    className="max-w-full max-h-[80vh] object-contain rounded-md"
                    onError={handleImageError}
                  />
                  {imageLoadError && (
                    <p className="text-red-500 mt-2 text-center">{imageLoadError}</p>
                  )}
                </>
              ) : (
                <p>Đang tải ảnh...</p>
              )}
            </div>
            <p className="text-center text-gray-700 mt-2">{selectedImage.image.value.split("#")[1]}</p>
          </div>
        </div>
      )}

      <footer className="bg-white/80 backdrop-blur-sm py-4 px-6 text-center text-sm text-gray-600">
        <p>Thuận - Bình - Khánh</p>
      </footer>
    </main>
  );
}

export default App;