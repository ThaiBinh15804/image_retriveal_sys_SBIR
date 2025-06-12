import { useState } from "react"
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
  const [page, setPage] = useState(1)
  const pageSize = 10
  const totalPages = Math.ceil(results.length / pageSize)
  const pagedResults = results.slice((page - 1) * pageSize, page * pageSize)

  // Hiệu ứng chuyển trang mượt mà
  // Sử dụng key cho grid để React remount khi đổi trang (kích hoạt animation)
  // Nút phân trang đẹp, hiển thị số trang, nhảy nhanh, disable hợp lý
  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) setPage(newPage)
  }

  // Tạo dãy số trang hiển thị (ví dụ: 1 ... 4 5 6 ... 10)
  const getPageNumbers = () => {
    const pages = []
    if (totalPages <= 5) {
      for (let i = 1; i <= totalPages; i++) pages.push(i)
    } else {
      if (page <= 3) {
        pages.push(1, 2, 3, 4, '...', totalPages)
      } else if (page >= totalPages - 2) {
        pages.push(1, '...', totalPages - 3, totalPages - 2, totalPages - 1, totalPages)
      } else {
        pages.push(1, '...', page - 1, page, page + 1, '...', totalPages)
      }
    }
    return pages
  }

  // Xử lý nhập số trang trực tiếp
  const [inputPage, setInputPage] = useState("");
  const handleInputPageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value.replace(/[^0-9]/g, "");
    setInputPage(val);
  };
  const handleInputPageGo = () => {
    const num = Number(inputPage);
    if (num >= 1 && num <= totalPages) setPage(num);
    setInputPage("");
  };

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
        <>
          <div
            key={page}
            className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6 transition-all duration-500 animate-fadein"
          >
            {pagedResults.map((image, idx) => (
              <ImageCard key={idx} image={image} onClick={() => onSelectImage(image)} />
            ))}
          </div>
          {/* Pagination controls */}
          {totalPages > 1 && (
            <>
              <div className="flex flex-col md:flex-row justify-center items-center gap-2 mt-8 select-none">
                {/* Dòng 1: Nút đầu, trước, số trang, sau, cuối */}
                <div className="flex flex-row flex-wrap gap-1 items-center justify-center">
                  <button
                    className="px-3 py-1 rounded-lg font-semibold bg-gradient-to-r from-purple-200 to-pink-200 text-purple-700 hover:from-purple-300 hover:to-pink-300 shadow disabled:opacity-50 disabled:cursor-not-allowed"
                    onClick={() => handlePageChange(1)}
                    disabled={page === 1}
                    aria-label="Trang đầu"
                  >
                    <span className="hidden md:inline">« Đầu</span>
                    <span className="md:hidden">«</span>
                  </button>
                  <button
                    className="px-3 py-1 rounded-lg font-semibold bg-gradient-to-r from-purple-200 to-pink-200 text-purple-700 hover:from-purple-300 hover:to-pink-300 shadow disabled:opacity-50 disabled:cursor-not-allowed"
                    onClick={() => handlePageChange(page - 1)}
                    disabled={page === 1}
                    aria-label="Trang trước"
                  >
                    <span className="hidden md:inline">← Trước</span>
                    <span className="md:hidden">←</span>
                  </button>
                  {getPageNumbers().map((p, i) =>
                    p === '...'
                      ? <span key={i} className="px-2 text-gray-400">...</span>
                      : <button
                          key={i}
                          className={`px-3 py-1 rounded-lg font-semibold transition-all duration-200 ${p === page ? 'bg-purple-600 text-white shadow-lg scale-110' : 'bg-gray-100 text-purple-700 hover:bg-purple-200'}`}
                          onClick={() => handlePageChange(Number(p))}
                          disabled={p === page}
                        >
                          {p}
                        </button>
                  )}
                  <button
                    className="px-3 py-1 rounded-lg font-semibold bg-gradient-to-r from-pink-200 to-purple-200 text-purple-700 hover:from-pink-300 hover:to-purple-300 shadow disabled:opacity-50 disabled:cursor-not-allowed"
                    onClick={() => handlePageChange(page + 1)}
                    disabled={page === totalPages}
                    aria-label="Trang sau"
                  >
                    <span className="hidden md:inline">Sau →</span>
                    <span className="md:hidden">→</span>
                  </button>
                  <button
                    className="px-3 py-1 rounded-lg font-semibold bg-gradient-to-r from-pink-200 to-purple-200 text-purple-700 hover:from-pink-300 hover:to-purple-300 shadow disabled:opacity-50 disabled:cursor-not-allowed"
                    onClick={() => handlePageChange(totalPages)}
                    disabled={page === totalPages}
                    aria-label="Trang cuối"
                  >
                    <span className="hidden md:inline">Cuối »</span>
                    <span className="md:hidden">»</span>
                  </button>
                </div>
              </div>
              {/* Dòng 2: Ô nhập số trang nằm dưới cùng */}
              <div className="flex justify-center items-center gap-1 mt-4">
                <input
                  type="text"
                  value={inputPage}
                  onChange={handleInputPageChange}
                  className="w-16 px-2 py-1 rounded border border-purple-200 focus:outline-none focus:ring-2 focus:ring-purple-400 text-center"
                  placeholder="Trang..."
                  onKeyDown={e => { if (e.key === 'Enter') handleInputPageGo(); }}
                />
                <button
                  className="px-3 py-1 rounded bg-purple-500 text-white hover:bg-purple-600"
                  onClick={handleInputPageGo}
                  disabled={!inputPage || Number(inputPage) < 1 || Number(inputPage) > totalPages}
                >
                  Đến
                </button>
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}
