import os
# import time
import nltk
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import wordnet
from text_preprocessor import TextPreprocessor
from classification import EntityClassifier
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

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


def get_synonyms(word, max_synonyms=5):
    """Lấy danh sách từ đồng nghĩa phổ biến và phù hợp ngữ cảnh của một từ"""
    synonyms = set()
    lemma_frequencies = []

    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            lemma_name = lemma.name().replace("_", " ")
            if lemma_name.lower() != word.lower():  # Tránh trùng lặp
                lemma_frequencies.append((lemma_name, lemma.count()))

    # Sắp xếp các từ đồng nghĩa theo tần suất sử dụng (giảm dần)
    sorted_lemmas = sorted(lemma_frequencies, key=lambda x: x[1], reverse=True)

    for lemma_name, _ in sorted_lemmas:
        synonyms.add(lemma_name)
        if len(synonyms) >= max_synonyms:
            break

    return list(synonyms)


def generate_sparql_query(data):
    """Sinh truy vấn SPARQL từ dữ liệu phân loại"""
    select_clause = "SELECT DISTINCT ?image ?url WHERE {\n"
    where_clauses = []
    entity_vars = {}

    for entity, entity_class in data["classified_entities"].items():
        var_name = f"{entity_class}Name_{entity.replace(' ', '_')}"
        entity_var = f'?{var_name}'
        entity_vars[entity] = entity_var
        where_clauses.append(f"    {entity_var} a :{entity_class}.")
        relation = ":hasContext" if entity_class == "Context" else ":contains"
        where_clauses.append(f"    ?image {relation} {entity_var}.")
        
        synonyms = get_synonyms(entity)
        synonyms.append(entity)
        synonym_var = f"?synonym_{var_name}"
        where_clauses.append(f"    VALUES {synonym_var} {{ {' '.join(f'\"{syn}\"' for syn in synonyms)} }}")
        
        if entity_class in ["PhysicalObject", "Animal", "Person"]:
            where_clauses.append(f"    {entity_var} :ObjectName {synonym_var}.")
        elif entity_class == "Context":
            where_clauses.append(f"    {entity_var} :ContextName {synonym_var}.")
    
    for entity, attributes in data["classified_attributes"].items():
        if entity in entity_vars:
            entity_var = entity_vars[entity]
            for attr, values in attributes.items():
                for value in values:
                    where_clauses.append(f"    {entity_var} :{attr} \"{value}\".")
    
    for relation in data["classified_relations"]:
        subj_var = entity_vars.get(relation["subject"])
        obj_var = entity_vars.get(relation["object"]) if relation["object"] != "-" else None
        if not subj_var:
            continue

        action_name = relation["predicate"].replace(" ", "_")
        action_var = f"?ActionName_{action_name}"
        where_clauses.append(f"    {action_var} a :Action.")

        synonyms = get_synonyms(relation['predicate'])
        synonyms.append(relation['predicate'])
        synonym_var = f"?synonym_ActionName_{action_name}"
        where_clauses.append(f"    VALUES {synonym_var} {{ {' '.join(f'\"{syn}\"' for syn in synonyms)} }}")
        where_clauses.append(f"    {action_var} :ActionName {synonym_var}.")
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
