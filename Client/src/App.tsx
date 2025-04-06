import { useState, useEffect } from "react";
import { SearchSection } from "./components/SearchSection";
import { ResultsSection } from "./components/ResultsSection";
import axios from "axios";
import { gapi } from "gapi-script";

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
  const [imageUrl, setImageUrl] = useState<string>("");

  const API_KEY = "YOUR_API_KEY_HERE"; // Thay bằng API key của bạn

  // Khởi tạo Google API Client
  useEffect(() => {
    const initClient = () => {
      gapi.client
        .init({
          apiKey: API_KEY,
          discoveryDocs: ["https://www.googleapis.com/discovery/v1/apis/drive/v3/rest"],
        })
        .then(() => {
          console.log("Google API Client đã sẵn sàng");
        })
        .catch((err: unknown) => {
          // Xử lý lỗi linh hoạt
          const errorMessage = err instanceof Error ? err.message : JSON.stringify(err);
          setError("Không thể khởi tạo Google API Client: " + errorMessage);
        });
    };
    gapi.load("client", initClient);
  }, []);

  // // Hàm lấy URL ảnh từ Google Drive API
  // const fetchImageUrl = async (fileId: string) => {
  //   try {
  //     const response = await gapi.client.drive.files.get({
  //       fileId: fileId,
  //       fields: "webContentLink",
  //     });
  //     setImageUrl(response.result.webContentLink);
  //   } catch (err) {
  //     // Xử lý lỗi linh hoạt
  //     const errorMessage = err instanceof Error ? err.message : JSON.stringify(err);
  //     setError("Không thể lấy URL ảnh: " + errorMessage);
  //   }
  // };

  // // Khi chọn ảnh, gọi API để lấy URL
  // useEffect(() => {
  //   if (selectedImage) {
  //     const fileId = selectedImage.url.value.split("id=")[1];
  //     fetchImageUrl(fileId);
  //   }
  // }, [selectedImage]);

  const handleSearch = async (query: string | File, type: string) => {
    setIsLoading(true);
    setError("");

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

            <div className="flex justify-center items-center w-full">
              {selectedImage ? (
                <img
                  // src={imageUrl}
                  src={`https://drive.google.com/thumbnail?id=${selectedImage.url.value.split("id=")[1]}&sz=w1000`}
                  alt={selectedImage.image.value}
                  className="max-w-full max-h-[80vh] object-contain rounded-md"
                />
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