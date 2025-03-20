import nltk
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util

# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

# Load SBERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_expanded_synonyms(word, sentence, max_synonyms=5):
    """
    Lấy từ đồng nghĩa, từ có nghĩa rộng hơn (hypernyms), từ có nghĩa hẹp hơn (hyponyms),
    và lọc theo ngữ cảnh bằng SBERT.
    """
    if not isinstance(max_synonyms, int):
        raise TypeError(f"Expected max_synonyms to be an int, but got {type(max_synonyms).__name__}")

    synonyms = set()
    hypernyms = set()
    hyponyms = set()

    # Lấy từ loại (POS tag) của từ cần tìm
    pos_tag = nltk.pos_tag([word])[0][1]

    # Lấy từ đồng nghĩa từ WordNet
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            lemma_name = lemma.name().replace("_", " ")
            if lemma_name.lower() != word.lower():
                synonyms.add(lemma_name)

        # Lấy từ có nghĩa rộng hơn (hypernyms)
        for hyper in synset.hypernyms():
            for lemma in hyper.lemmas():
                hypernyms.add(lemma.name().replace("_", " "))

        # Lấy từ có nghĩa hẹp hơn (hyponyms)
        for hypo in synset.hyponyms():
            for lemma in hypo.lemmas():
                hyponyms.add(lemma.name().replace("_", " "))

    # Tổng hợp danh sách từ ứng viên
    candidate_words = list(synonyms | hypernyms | hyponyms)

    if not candidate_words:
        return []

    # Loại bỏ từ có độ dài quá lớn (tránh các từ như "geological formation")
    candidate_words = [word for word in candidate_words if len(word.split()) <= 2]

    # Tạo embedding SBERT cho từ gốc và các từ ứng viên
    word_embedding = model.encode(word, convert_to_tensor=True)
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    candidate_embeddings = model.encode(candidate_words, convert_to_tensor=True)

    # Tính độ tương đồng giữa từ gốc và từ ứng viên
    similarities = util.pytorch_cos_sim(word_embedding, candidate_embeddings).squeeze(0).tolist()

    # Tính độ tương đồng giữa câu và từ ứng viên (để kiểm tra ngữ cảnh)
    sentence_similarities = util.pytorch_cos_sim(sentence_embedding, candidate_embeddings).squeeze(0).tolist()

    # Tổng hợp điểm số và sắp xếp
    sorted_candidates = sorted(
        zip(candidate_words, similarities, sentence_similarities),
        key=lambda x: (x[1] + x[2]) / 2,  # Trung bình độ tương đồng giữa từ và câu
        reverse=True
    )

    # Chọn max_synonyms từ phù hợp nhất
    best_synonyms = [word for word, _, _ in sorted_candidates[:max_synonyms]]

    return best_synonyms

# Ví dụ sử dụng
sentence = "A man is running on the beach near the ocean."
word = "black"
print(get_expanded_synonyms(word, sentence, max_synonyms=10))
