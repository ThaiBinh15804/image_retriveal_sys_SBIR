import os
import time
import nltk
from nltk.corpus import wordnet
from text_preprocessor import TextPreprocessor
from classification import EntityClassifier

def get_synonyms(word, max_synonyms=5):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            lemma_name = lemma.name().replace("_", " ")
            if lemma_name.lower() != word.lower():  # Tránh lặp lại từ gốc
                synonyms.add(lemma_name)
            if len(synonyms) >= max_synonyms:
                return list(synonyms)
    return list(synonyms)


def generate_sparql_query(data):
    select_clause = "SELECT DISTINCT ?image ?url WHERE {\n"
    where_clauses = []
    entity_vars = {}

    # Xử lý thực thể
    for entity, entity_class in data["classified_entities"].items():
        var_name = f"{entity_class}Name_{entity.replace(' ', '_')}"
        entity_var = f'?{var_name}'
        entity_vars[entity] = entity_var
        where_clauses.append(f"    {entity_var} a :{entity_class}.")
        
        # Liên kết với ảnh
        relation = ":hasContext" if entity_class == "Context" else ":contains"
        where_clauses.append(f"    ?image {relation} {entity_var}.")

        # Mở rộng từ đồng nghĩa
        synonyms = get_synonyms(entity)
        # if synonyms:  # Chỉ thêm VALUES nếu có từ đồng nghĩa
        synonyms.append(entity)
        synonym_var = f"?synonym_{var_name}"
        where_clauses.append(f"    VALUES {synonym_var} {{ {' '.join(f'\"{syn}\"' for syn in synonyms)} }}")

        if entity_class in ["PhysicalObject", "Animal", "Person"]:
            where_clauses.append(f"    {entity_var} :ObjectName {synonym_var}.")
        elif entity_class == "Context":
            where_clauses.append(f"    {entity_var} :ContextName {synonym_var}.")
    
    # Xử lý thuộc tính thực thể
    for entity, attributes in data["classified_attributes"].items():
        if entity in entity_vars:
            entity_var = entity_vars[entity]
            for attr, values in attributes.items():
                for value in values:
                    where_clauses.append(f"    {entity_var} :{attr} \"{value}\".")

    # Xử lý quan hệ hành động
    for relation in data["classified_relations"]:
        subj_var = entity_vars.get(relation["subject"])
        obj_var = entity_vars.get(relation["object"]) if relation["object"] != "-" else None

        if not subj_var:
            continue  # Bỏ qua nếu không tìm thấy thực thể thực hiện hành động

        action_name = relation["predicate"].replace(" ", "_")
        action_var = f"?ActionName_{action_name}"
        where_clauses.append(f"    {action_var} a :Action.")

        # Mở rộng từ đồng nghĩa cho ActionName
        synonyms = get_synonyms(relation['predicate'])
        if not synonyms:
            synonyms = [relation['predicate']]
        else:
            synonyms.append(relation['predicate'])

        synonym_var = f"?synonym_ActionName_{action_name}"
        where_clauses.append(f"    VALUES {synonym_var} {{ {' '.join(f'\"{syn}\"' for syn in synonyms)} }}")
        where_clauses.append(f"    {action_var} :ActionName {synonym_var}.")

        where_clauses.append(f"    {action_var} :hasAgent {subj_var}.")
        
        if obj_var:  # Chỉ thêm :hasObject nếu có đối tượng
            where_clauses.append(f"    {action_var} :hasObject {obj_var}.")
    
    # Thêm thông tin ảnh
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
    comment = "two dog are playing together on the grass"

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
