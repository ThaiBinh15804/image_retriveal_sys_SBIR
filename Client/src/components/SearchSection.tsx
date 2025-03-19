import { useState } from "react"
import { SearchIcon, UploadIcon } from "lucide-react"

export function SearchSection({ onSearch, sparqlQuery, description }: { onSearch: (query: string | File, type: string) => void, sparqlQuery: string, description : string }) {
  const [activeTab, setActiveTab] = useState("text")
  const [textQuery, setTextQuery] = useState("")
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string>("")

  const handleTextSearch = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (textQuery.trim()) {
      onSearch(textQuery.trim(), "text");
    }
  };

  const handleImageSearch = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (imageFile) {
      onSearch(imageFile, "image");
    }
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setImageFile(file)
      const reader = new FileReader()
      reader.onload = () => {
        setPreviewUrl(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const clearImage = () => {
    setImageFile(null)
    setPreviewUrl("")
  }

  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 mb-8 border border-purple-100">
      <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-purple-600 to-pink-600 text-transparent bg-clip-text">
        Find Your Perfect Image
      </h2>

      <div className="flex space-x-1 border-b border-purple-100 mb-6">
        <button
          className={`flex items-center px-6 py-3 -mb-px border-b-2 font-medium text-sm transition-all duration-300 hover:scale-105 ${
            activeTab === "text"
              ? "border-purple-500 text-purple-600 bg-purple-50/50"
              : "border-transparent text-gray-500 hover:text-purple-600"
          } rounded-t-lg`}
          onClick={() => setActiveTab("text")}
        >
          <SearchIcon className="w-4 h-4 mr-2" />
          Text Search
        </button>
        <button
          className={`flex items-center px-6 py-3 -mb-px border-b-2 font-medium text-sm transition-all duration-300 hover:scale-105 ${
            activeTab === "image"
              ? "border-purple-500 text-purple-600 bg-purple-50/50"
              : "border-transparent text-gray-500 hover:text-purple-600"
          } rounded-t-lg`}
          onClick={() => setActiveTab("image")}
        >
          <UploadIcon className="w-4 h-4 mr-2" />
          Image Search
        </button>
      </div>

      <div className="bg-purple-50/50 p-6 rounded-2xl backdrop-blur-sm">
        {activeTab === "text" ? (
          <form onSubmit={handleTextSearch} className="flex flex-col gap-4">
            <div>
              <label htmlFor="textQuery" className="block text-sm font-medium text-purple-700 mb-2">
                What kind of image are you looking for? ✨
              </label>
              <input
                type="text"
                id="textQuery"
                className="w-full px-4 py-3 border border-purple-200 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-purple-500 bg-white/80 backdrop-blur-sm transition-all duration-300"
                placeholder="e.g., sunset over mountains"
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
              />
            </div>
            <button
              type="submit"
              className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-6 py-3 rounded-xl hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:hover:scale-100 w-full md:w-auto md:self-end"
              disabled={!textQuery.trim()}
            >
              Search with Text ✨
            </button>
          </form>
        ) : (
          <form onSubmit={handleImageSearch} className="flex flex-col gap-4">
            <div>
              <label htmlFor="imageUpload" className="block text-sm font-medium text-purple-700 mb-2">
                Upload an image to find similar ones ✨
              </label>
              <input
                type="file"
                id="imageUpload"
                accept="image/*"
                className="w-full text-sm text-gray-500 file:mr-4 file:py-3 file:px-6 file:rounded-xl file:border-0 
                file:text-sm file:font-semibold file:bg-purple-500 file:text-white hover:file:bg-purple-600 
                file:transition-colors file:cursor-pointer"
                onChange={handleImageChange}
              />
            </div>

            {previewUrl && (
              <div className="mt-2 relative">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="h-48 object-contain rounded-xl border border-purple-200 bg-white/50 backdrop-blur-sm p-2"
                />
                <button
                  type="button"
                  className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full text-xs hover:bg-red-600 transition-colors"
                  onClick={clearImage}
                >
                  ✕
                </button>
              </div>
            )}
            
            <button
              type="submit"
              className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-6 py-3 rounded-xl hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:hover:scale-100 w-full md:w-auto md:self-end"
              disabled={!imageFile}
            >
              Search with Image ✨
            </button>

            {/* Display description from parent */}
            {description && (
              <div className="mt-6 bg-gradient-to-r from-gray-100 to-gray-200 p-6 rounded-xl shadow-lg border border-gray-300">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">Description:</h3>
                <p className="text-gray-700 leading-relaxed">{description}</p>
              </div>
            )}
          </form>
          
        )}
      </div>

      

      {/* Display SPARQL query from parent */}
      {sparqlQuery && (
        <div className="mt-6 bg-gray-100 p-4 rounded-lg shadow-inner border border-gray-300">
          <h3 className="text-lg font-semibold text-gray-700 mb-2">Generated SPARQL Query:</h3>
          <pre className="bg-gray-50 p-4 rounded-lg overflow-auto text-sm text-gray-800 border border-gray-200">
            <code>
              {sparqlQuery.split('\n').map((line, index) => (
                <div key={index}>
                  <span className="text-gray-500">{index + 1}    </span>
                  {line}
                </div>
              ))}
            </code>
          </pre>
        </div>
      )}
    </div>
  )
}
