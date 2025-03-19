import { useState } from "react"
import { SearchSection } from "./components/SearchSection"
import { ResultsSection } from "./components/ResultsSection"
import axios from "axios"

export type imageQueryType = {
  image: {
    type: string
    value: string
  }
  url: {
    type: string
    value: string
  }
}

export function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [searchResults, setSearchResults] = useState<imageQueryType[]>([])
  const [sparqlQuery, setSparqlQuery] = useState(""); // Thêm state để lưu SPARQL query
  const [description, setDescription] = useState(""); // Thêm state để lưu mô tả ảnh

  // State quản lý modal
  const [selectedImage, setSelectedImage] = useState<imageQueryType | null>(null)

  const handleSearch = async (query: string | File, type: string) => {
    setIsLoading(true);
    setError("");
  
    try {
      let sparql = type === "text" && typeof query === "string" ? query : "";
  
      if (type === "image" && query instanceof File) {
        const formData = new FormData();
        formData.append("file", query);
  
        const { data } = await axios.post("http://localhost:8080/analyze/", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
  
        if (!data?.description) throw new Error("⚠️ Image analysis failed.");
        sparql = data.description;
        setDescription(data.description); // Lưu mô tả ảnh vào state
        console.log(sparql)
      }
  
      const { data: results } = await axios.post("http://localhost:2020/query", { body: sparql });
      console.log(results.bindings)
      console.log(results.sparql_query)
      setSparqlQuery(results.sparql_query); // Lưu SPARQL query vào state
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
            Image Search
            <span className="text-yellow-300">.</span>
          </h1>
          <p className="text-purple-100 mt-2">Find the perfect image in seconds ✨</p>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8 flex-grow">
        <SearchSection onSearch={handleSearch} sparqlQuery={sparqlQuery} description={description} /> {/* Truyền sparqlQuery */}
        {/* Truyền hàm setSelectedImage xuống */}
        <ResultsSection results={searchResults} isLoading={isLoading} error={error} onSelectImage={setSelectedImage} />
      </div>

      {/* Modal hiển thị ảnh */}
      {selectedImage && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="relative bg-white rounded-lg shadow-lg p-4 max-w-4xl w-full">
            {/* Nút đóng */}
            <button
              className="absolute top-3 right-3 text-gray-700 hover:text-red-500 text-2xl font-bold"
              onClick={() => setSelectedImage(null)}
            >
              ×
            </button>

            {/* Hiển thị ảnh */}
            <div className="flex justify-center items-center w-full">
              <img
                src={`https://drive.google.com/thumbnail?id=${selectedImage.url.value.split("id=")[1]}&sz=w1000`}
                alt={selectedImage.image.value}
                className="max-w-full max-h-[80vh] object-contain rounded-md"
              />
            </div>

            {/* Tên ảnh */}
            <p className="text-center text-gray-700 mt-2">{selectedImage.image.value.split("#")[1]}</p>
          </div>
        </div>
      )}

      <footer className="bg-white/80 backdrop-blur-sm py-4 px-6 text-center text-sm text-gray-600">
        <p>Thuận - Bình - Khánh</p>
      </footer>
    </main>
  )
}

export default App
