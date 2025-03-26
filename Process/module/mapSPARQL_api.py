import os
import nltk
import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import wordnet
from text_preprocessor import TextPreprocessor
from classification import EntityClassifier
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

app = FastAPI()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chỉnh sửa để giới hạn domain nếu cần
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load NLP Processor và Entity Classifier
current_dir = os.path.dirname(os.path.abspath(__file__))
processor = TextPreprocessor()
classifier = EntityClassifier(
    os.path.join(current_dir, "classicalNoun.txt"),
    os.path.join(current_dir, "attributes.txt")
)


# Định nghĩa cấu trúc request
class QueryRequest(BaseModel):
    text: str


def get_synonyms(word: str, sentence: str = None, max_synonyms: int = 5, similarity_threshold: float = 0.6):
    """Lấy từ đồng nghĩa chính xác với ngưỡng tương đồng tối thiểu 60% dựa trên ngữ cảnh."""
    if not isinstance(max_synonyms, int):
        raise TypeError(f"Expected max_synonyms to be an int, got {type(max_synonyms).__name__}")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("Similarity threshold must be between 0 and 1")

    synonyms = set()

    # Xác định POS tag từ ngữ cảnh câu hoặc từ đơn
    if sentence:
        doc = nlp(sentence)
        target_token = next((token for token in doc if token.text.lower() == word.lower()), None)
        wn_pos = {
            'NOUN': wordnet.NOUN, 'VERB': wordnet.VERB, 'ADJ': wordnet.ADJ, 'ADV': wordnet.ADV
        }.get(target_token.pos_ if target_token else 'NOUN', wordnet.NOUN)
    else:
        pos_tag = nltk.pos_tag([word])[0][1]
        wn_pos = {
            'NN': wordnet.NOUN, 'VB': wordnet.VERB, 'JJ': wordnet.ADJ, 'RB': wordnet.ADV
        }.get(pos_tag[:2], wordnet.NOUN)

    # Lấy từ đồng nghĩa từ WordNet với POS phù hợp
    for synset in wordnet.synsets(word, pos=wn_pos):
        definition = synset.definition().lower()
        if any(kw in definition for kw in ["slang", "drug", "person", "food", "tool", "machine"]):
            continue  # Loại bỏ các nghĩa không phù hợp
        for lemma in synset.lemmas():
            lemma_name = lemma.name().replace("_", " ")
            if (lemma_name.lower() != word.lower() and 
                not lemma_name[0].isupper() and 
                len(lemma_name.split()) <= 2):
                synonyms.add(lemma_name)

    if not synonyms:
        return [word]  # Trả về từ gốc nếu không tìm thấy từ đồng nghĩa

    candidate_words = list(synonyms)

    # Nếu không có câu, trả về danh sách thô từ WordNet
    if not sentence:
        return candidate_words[:max_synonyms]

    # Tạo ngữ cảnh từ các từ liên quan trong câu
    doc = nlp(sentence)
    context_words = set()
    for token in doc:
        if token.text.lower() == word.lower():
            context_words.add(token.head.text.lower())
            for child in token.children:
                context_words.add(child.text.lower())
    context_sentence = " ".join(context_words) if context_words else sentence

    # Tính embedding cho từ gốc, ngữ cảnh, và các ứng viên
    embeddings = model.encode([word, context_sentence] + candidate_words, convert_to_tensor=True)
    word_embedding = embeddings[0]
    context_embedding = embeddings[1]
    candidate_embeddings = embeddings[2:]

    # Tính độ tương đồng giữa từ gốc và các ứng viên
    word_similarities = util.pytorch_cos_sim(word_embedding, candidate_embeddings).squeeze(0).tolist()

    # Tính độ tương đồng giữa ngữ cảnh và các ứng viên
    context_similarities = util.pytorch_cos_sim(context_embedding, candidate_embeddings).squeeze(0).tolist()

    # Kết hợp độ tương đồng (trung bình giữa từ gốc và ngữ cảnh)
    combined_similarities = [(w, (word_sim + context_sim) / 2) 
                             for w, word_sim, context_sim in zip(candidate_words, word_similarities, context_similarities)]

    # Lọc các từ có độ tương đồng >= 60%
    filtered_candidates = [(w, sim) for w, sim in combined_similarities if sim >= similarity_threshold]
    sorted_candidates = sorted(filtered_candidates, key=lambda x: x[1], reverse=True)

    # Chọn các từ đồng nghĩa tốt nhất
    best_synonyms = [word for word, _ in sorted_candidates[:max_synonyms]]
    
    # Đảm bảo từ gốc luôn có mặt nếu không có từ nào vượt ngưỡng
    if not best_synonyms:
        return [word]
    if word.lower() not in [w.lower() for w in best_synonyms]:
        best_synonyms.append(word)

    return best_synonyms[:max_synonyms]



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
        synonyms = get_synonyms(entity, max_synonyms=10)
        synonym_var = f"?synonym_{var_name}"

        if synonyms:
            if entity not in synonyms:
                synonyms.append(entity)
            where_clauses.append(f"""    VALUES {synonym_var} {{ {' '.join(f'"{syn}"' for syn in synonyms)} }}""")
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
                    attr_synonyms = get_synonyms(value,  max_synonyms=5)
                    attr_synonym_var = f"?synonym_{attr}_{value.replace(' ', '_')}"

                    if attr_synonyms:
                        where_clauses.append(f"""    VALUES {attr_synonym_var} {{ {' '.join(f'"{syn}"' for syn in attr_synonyms)} }}""")
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
        synonyms = get_synonyms(relation['predicate'],  max_synonyms=5)
        if synonyms:
            if entity not in synonyms:
                synonyms.append(action_name)
            synonym_var = f"?synonym_ActionName_{action_name}"
            where_clauses.append(f"""    VALUES {synonym_var} {{ {' '.join(f'"{syn}"' for syn in synonyms)} }}""")
            where_clauses.append(f"    OPTIONAL {{ {action_var} :Wordnet {synonym_var}. }}")
        else:
            where_clauses.append(f"    {action_var} :ActionName \"{relation['predicate']}\".")

        where_clauses.append(f"    {action_var} :hasAgent {subj_var}.")
        if obj_var:
            where_clauses.append(f"    {action_var} :hasObject {obj_var}.")

    where_clauses.append("    ?image a :Image;")
    where_clauses.append("           :ImageURL ?url.")

    return select_clause + "\n".join(where_clauses) + "\n}"


@app.post("/generate_sparql")
def generate_sparql(request: QueryRequest):
    """API nhận câu truy vấn, xử lý NLP và sinh SPARQL"""
    # start_time = time.time()

    parsed_data = processor.preprocess_text(request.text)
    classification_input = {
        "entities": list(parsed_data["entities"]),
        "attributes": parsed_data["attributes"],
        "relations": parsed_data["relations"],
    }
    
    classified_data = classifier.classify_input(classification_input)
    sparql_query = "PREFIX : <http://www.semanticweb.org/asus/ontologies/2025/1/untitled-ontology-26#> " + generate_sparql_query(classified_data)

    return {
        "sparql_query": sparql_query,
        # "execution_time": round(time.time() - start_time, 2)
    }
