import os
import time
import nltk
from nltk.corpus import wordnet
from text_preprocessor import TextPreprocessor
from classification import EntityClassifier
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_synonyms(word: str, sentence: str = None, max_synonyms: int = 5):
    """Lấy danh sách từ đồng nghĩa, hypernyms (từ có nghĩa rộng hơn) và hyponyms (từ có nghĩa hẹp hơn) 
       và lọc theo ngữ cảnh bằng SBERT nếu có câu."""
    if not isinstance(max_synonyms, int):
        raise TypeError(f"Expected max_synonyms to be an int, but got {type(max_synonyms).__name__}")

    synonyms = set()
    hypernyms = set()
    hyponyms = set()
    lemma_frequencies = []

    # Lấy từ loại (POS tag) của từ đầu vào
    pos_tag = nltk.pos_tag([word])[0][1]

    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            lemma_name = lemma.name().replace("_", " ")
            if lemma_name.lower() != word.lower():
                lemma_frequencies.append((lemma_name, lemma.count()))

        # Thêm hypernyms (từ rộng hơn)
        for hyper in synset.hypernyms():
            for lemma in hyper.lemmas():
                hypernyms.add(lemma.name().replace("_", " "))

        # Thêm hyponyms (từ hẹp hơn)
        for hypo in synset.hyponyms():
            for lemma in hypo.lemmas():
                hyponyms.add(lemma.name().replace("_", " "))

    # Gộp từ đồng nghĩa, hypernyms và hyponyms
    candidate_words = list(set([word for word, _ in lemma_frequencies]) | hypernyms | hyponyms)

    # Nếu không có danh sách từ, trả về rỗng
    if not candidate_words:
        return []

    # Lọc từ có độ dài quá lớn để tránh những cụm từ không tự nhiên
    candidate_words = [w for w in candidate_words if len(w.split()) <= 2]

    # Nếu không có câu để lọc theo ngữ cảnh, trả về danh sách thô
    if sentence is None:
        return candidate_words[:max_synonyms]

    # Tạo embedding SBERT để lọc từ phù hợp với ngữ cảnh
    word_embedding = model.encode(word, convert_to_tensor=True)
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    candidate_embeddings = model.encode(candidate_words, convert_to_tensor=True)

    # Tính độ tương đồng giữa từ gốc và từ ứng viên
    similarities = util.pytorch_cos_sim(word_embedding, candidate_embeddings).squeeze(0).tolist()

    # Tính độ tương đồng giữa câu và từ ứng viên
    sentence_similarities = util.pytorch_cos_sim(sentence_embedding, candidate_embeddings).squeeze(0).tolist()

    # Sắp xếp từ theo mức độ phù hợp
    sorted_candidates = sorted(
        zip(candidate_words, similarities, sentence_similarities),
        key=lambda x: (x[1] + x[2]) / 2,  # Trung bình độ tương đồng giữa từ và câu
        reverse=True
    )

    # Trả về danh sách từ tốt nhất
    best_synonyms = [word for word, _, _ in sorted_candidates[:max_synonyms]]
    return best_synonyms




def generate_sparql_query(data):
    """Sinh truy vấn SPARQL từ dữ liệu phân loại."""
    select_clause = "SELECT DISTINCT ?image ?url WHERE {\n"
    where_clauses = []
    entity_vars = {}

    # Xử lý thực thể
    for entity, entity_class in data["classified_entities"].items():
        var_name = f"{entity_class}Name_{entity.replace(' ', '_')}"
        entity_var = f'?{var_name}'
        entity_vars[entity] = entity_var
        where_clauses.append(f"    {entity_var} a :{entity_class}.")
        relation = ":hasContext" if entity_class == "Context" else ":contains"
        where_clauses.append(f"    ?image {relation} {entity_var}.")

        # Lấy từ đồng nghĩa có ngữ cảnh
        synonyms = get_synonyms(entity, sentence=data, max_synonyms=10)
        synonym_var = f"?synonym_{var_name}"

        if synonyms:
            if entity not in synonyms:
                synonyms.append(entity)
            where_clauses.append(f"    VALUES {synonym_var} {{ {' '.join(f'\"{syn}\"' for syn in synonyms)} }}")
            where_clauses.append(f"    OPTIONAL {{ {entity_var} :Wordnet {synonym_var}. }}")
        else:
            where_clauses.append(f"    {entity_var} :ObjectName \"{entity}\".")

        if entity_class in ["PhysicalObject", "Animal", "Person"]:
            where_clauses.append(f"    {entity_var} :ObjectName {synonym_var}.")
        elif entity_class == "Context":
            where_clauses.append(f"    {entity_var} :ContextName {synonym_var}.")

    for entity, attributes in data["classified_attributes"].items():
        if entity in entity_vars:
            entity_var = entity_vars[entity]
            for attr, values in attributes.items():
                for value in values:
                    # Lấy từ đồng nghĩa của giá trị thuộc tính
                    attr_synonyms = get_synonyms(value, sentence=data["original_sentence"], max_synonyms=5)
                    attr_synonym_var = f"?synonym_{attr}_{value.replace(' ', '_')}"

                    if attr_synonyms:
                        where_clauses.append(f"    VALUES {attr_synonym_var} {{ {' '.join(f'\"{syn}\"' for syn in attr_synonyms)} }}")
                        where_clauses.append(f"    OPTIONAL {{ {entity_var} :{attr} {attr_synonym_var}. }}")
                    else:
                        where_clauses.append(f"    {entity_var} :{attr} \"{value}\".")

    for relation in data["classified_relations"]:
        subj_var = entity_vars.get(relation["subject"])
        obj_var = entity_vars.get(relation["object"]) if relation["object"] != "-" else None
        if not subj_var:
            continue

        action_name = relation["predicate"].replace(" ", "_")
        action_var = f"?ActionName_{action_name}"
        where_clauses.append(f"    {action_var} a :Action.")

        # Lấy từ đồng nghĩa của quan hệ
        synonyms = get_synonyms(relation['predicate'], sentence=data["original_sentence"], max_synonyms=5)
        if synonyms:
            synonym_var = f"?synonym_ActionName_{action_name}"
            where_clauses.append(f"    VALUES {synonym_var} {{ {' '.join(f'\"{syn}\"' for syn in synonyms)} }}")
            where_clauses.append(f"    OPTIONAL {{ {action_var} :Wordnet {synonym_var}. }}")
        else:
            where_clauses.append(f"    {action_var} :ActionName \"{relation['predicate']}\".")

        where_clauses.append(f"    {action_var} :hasAgent {subj_var}.")
        if obj_var:
            where_clauses.append(f"    {action_var} :hasObject {obj_var}.")

    where_clauses.append("    ?image a :Image;")
    where_clauses.append("           :ImageURL ?url.")

    return select_clause + "\n".join(where_clauses) + "\n}"

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Đo thời gian bắt đầu
    total_start_time = time.time()

    # Khởi tạo bộ phân loại thực thể
    start_time = time.time()
    classifier = EntityClassifier(
        os.path.join(current_dir, "classicalNoun.txt"),
        os.path.join(current_dir, "attributes.txt")
    )
    print(f"Initialized EntityClassifier in {time.time() - start_time:.2f} seconds")

    # Khởi tạo xử lý NLP
    start_time = time.time()
    processor = TextPreprocessor()
    print(f"Initialized NLP processor in {time.time() - start_time:.2f} seconds")

    # Câu đầu vào
    comment = "A man is walking on the grass"

    # Xử lý NLP
    start_time = time.time()
    parsed_data = processor.preprocess_text(comment)
    print(f"Processed NLP for comment in {time.time() - start_time:.2f} seconds")
    # Chuẩn bị dữ liệu phân loại
    classification_input = {
        "entities": list(parsed_data["entities"]),
        "attributes": parsed_data["attributes"],
        "relations": parsed_data["relations"],
    }

    # Phân loại thực thể, thuộc tính, quan hệ
    start_time = time.time()
    classified_data = classifier.classify_input(classification_input)
    print(f"Classified entities in {time.time() - start_time:.2f} seconds")
    print(classified_data)
    # Sinh truy vấn SPARQL
    sparql_query = generate_sparql_query(classified_data)
    print(sparql_query)

    # Tổng thời gian thực thi
    print(f"Total execution time: {time.time() - total_start_time:.2f} seconds")


if __name__ == "__main__":
    main()
