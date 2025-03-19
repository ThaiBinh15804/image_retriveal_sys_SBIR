import { imageQueryType } from "../App"
import { ImageCard } from "./ImageCard"

export function ResultsSection({
  results,
  isLoading,
  error,
  onSelectImage,
}: {
  results: imageQueryType[]
  isLoading: boolean
  error: string
  onSelectImage: (image: imageQueryType) => void
}) {
  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-purple-100">
      <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-purple-600 to-pink-600 text-transparent bg-clip-text">
        Results
      </h2>

      {/* Hiển thị khi đang tải */}
      {isLoading && (
        <div className="flex flex-col items-center justify-center py-12">
          <div className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
          <p className="mt-4 text-purple-600 animate-pulse">Finding amazing images for you... ✨</p>
        </div>
      )}

      {/* Hiển thị khi có lỗi */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-6 py-4 rounded-xl">
          <p>{error}</p>
        </div>
      )}

      {/* Hiển thị khi không có kết quả */}
      {!isLoading && !error && results.length === 0 && (
        <div className="text-center py-12 text-purple-600">
          <p>Start your search to discover amazing images! ✨</p>
        </div>
      )}

      {/* Hiển thị danh sách ảnh */}
      {!isLoading && !error && results.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {results.map((image) => (
            <ImageCard key={image.image.value} image={image} onClick={() => onSelectImage(image)} />
          ))}
        </div>
      )}
    </div>
  )
}
