import os
import nltk
import spacy
import random  # NEW: Thêm để chọn ngẫu nhiên 50% điều kiện
from fastapi import FastAPI, Query
from pydantic import BaseModel
from nltk.corpus import wordnet
from text_preprocessor import TextPreprocessor
from classification import EntityClassifier
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util

# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger_eng')

app = FastAPI()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
processor = TextPreprocessor()
classifier = EntityClassifier(
    os.path.join(current_dir, "classicalNoun.txt"),
    os.path.join(current_dir, "attributes.txt")
)

class QueryRequest(BaseModel):
    text: str

def get_synonyms(word: str, sentence: str = None, max_synonyms: int = 5, similarity_threshold: float = 0.6):
    """Lấy từ đồng nghĩa chính xác với ngưỡng tương đồng tối thiểu 60% dựa trên ngữ cảnh."""
    if not isinstance(max_synonyms, int):
        raise TypeError(f"Expected max_synonyms to be an int, got {type(max_synonyms).__name__}")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("Similarity threshold must be between 0 and 1")

    synonyms = set()

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

    for synset in wordnet.synsets(word, pos=wn_pos):
        definition = synset.definition().lower()
        if any(kw in definition for kw in ["slang", "drug", "person", "food", "tool", "machine"]):
            continue
        for lemma in synset.lemmas():
            lemma_name = lemma.name().replace("_", " ")
            if (lemma_name.lower() != word.lower() and 
                not lemma_name[0].isupper() and 
                len(lemma_name.split()) <= 2):
                synonyms.add(lemma_name)

    if not synonyms:
        return [word]

    candidate_words = list(synonyms)

    if not sentence:
        return candidate_words[:max_synonyms]

    doc = nlp(sentence)
    context_words = set()
    for token in doc:
        if token.text.lower() == word.lower():
            context_words.add(token.head.text.lower())
            for child in token.children:
                context_words.add(child.text.lower())
    context_sentence = " ".join(context_words) if context_words else sentence

    embeddings = model.encode([word, context_sentence] + candidate_words, convert_to_tensor=True)
    word_embedding = embeddings[0]
    context_embedding = embeddings[1]
    candidate_embeddings = embeddings[2:]

    word_similarities = util.pytorch_cos_sim(word_embedding, candidate_embeddings).squeeze(0).tolist()
    context_similarities = util.pytorch_cos_sim(context_embedding, candidate_embeddings).squeeze(0).tolist()

    combined_similarities = [(w, (word_sim + context_sim) / 2) 
                             for w, word_sim, context_sim in zip(candidate_words, word_similarities, context_similarities)]

    filtered_candidates = [(w, sim) for w, sim in combined_similarities if sim >= similarity_threshold]
    sorted_candidates = sorted(filtered_candidates, key=lambda x: x[1], reverse=True)

    best_synonyms = [word for word, _ in sorted_candidates[:max_synonyms]]
    
    if not best_synonyms:
        return [word]
    if word.lower() not in [w.lower() for w in best_synonyms]:
        best_synonyms.append(word)

    return best_synonyms[:max_synonyms]

def generate_sparql_query(data, from_image=False, main_entity=None):
    """Sinh truy vấn SPARQL từ dữ liệu phân loại, ưu tiên thực thể khi input từ hình ảnh."""
    select_clause = "SELECT DISTINCT ?image ?url WHERE {\n"
    where_clauses = []
    entity_vars = {}

    priority_order = ["Person", "Animal", "PhysicalObject", "Context", "Action"]

    if from_image:
        desc_entities = data["classified_entities"]
        main_entities = {}
        pair_similarities = {}
        
        if main_entity:
            parsed_data_main = processor.preprocess_text(main_entity)
            classification_input_main = {
                "entities": list(parsed_data_main["entities"]),
                "attributes": parsed_data_main["attributes"],
                "relations": parsed_data_main["relations"],
            }
            classified_data_main = classifier.classify_input(classification_input_main)
            main_entities = classified_data_main["classified_entities"]
            
            for main_ent, main_class in main_entities.items():
                best_similarity = -1
                best_desc_ent = None
                main_embedding = model.encode(main_ent, convert_to_tensor=True)
                for desc_ent, desc_class in desc_entities.items():
                    if main_class == desc_class:
                        desc_embedding = model.encode(desc_ent, convert_to_tensor=True)
                        similarity = util.pytorch_cos_sim(main_embedding, desc_embedding).item()
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_desc_ent = desc_ent
                if best_desc_ent:
                    pair_similarities[main_ent] = (best_desc_ent, best_similarity)

            entity_order = [e.strip() for e in main_entity.split(",")]
            if entity_order and entity_order[0] in pair_similarities:
                primary_entity = entity_order[0]
                primary_class = main_entities[primary_entity]
            else:
                ordered_entities = []
                for cls in priority_order:
                    class_entities = [(entity, entity_class) for entity, entity_class in main_entities.items() if entity_class == cls]
                    class_entities = sorted(class_entities, key=lambda x: entity_order.index(x[0]) if x[0] in entity_order else len(entity_order))
                    for entity, entity_class in class_entities:
                        if (entity, entity_class) not in ordered_entities:
                            ordered_entities.append((entity, entity_class))
                
                if not ordered_entities:
                    ordered_desc_entities = []
                    for cls in priority_order:
                        class_entities = [(entity, entity_class) for entity, entity_class in desc_entities.items() if entity_class == cls]
                    for entity, entity_class in class_entities:
                        if (entity, entity_class) not in ordered_desc_entities:
                            ordered_desc_entities.append((entity, entity_class))
                    if ordered_desc_entities:
                        primary_entity, primary_class = ordered_desc_entities[0]
                    else:
                        raise ValueError("No entities available from main_entity or desc_entities")
                else:
                    primary_entity, primary_class = ordered_entities[0]

        else:
            ordered_desc_entities = []
            for cls in priority_order:
                class_entities = [(entity, entity_class) for entity, entity_class in desc_entities.items() if entity_class == cls]
                for entity, entity_class in class_entities:
                    if (entity, entity_class) not in ordered_desc_entities:
                        ordered_desc_entities.append((entity, entity_class))
            if not ordered_desc_entities:
                raise ValueError("No entities available from desc_entities when main_entity is empty")
            primary_entity, primary_class = ordered_desc_entities[0]

        primary_var_name = f"{primary_class}Name_{primary_entity.replace(' ', '_')}"
        primary_entity_var = f'?{primary_var_name}'
        entity_vars[primary_entity] = primary_entity_var

        where_clauses.append(f"    {primary_entity_var} a :{primary_class} .")
        relation = ":hasContext" if primary_class == "Context" else ":contains"
        where_clauses.append(f"    ?image {relation} {primary_entity_var} .")

        synonyms = []
        if not (primary_class == "Person" and primary_entity in pair_similarities):
            synonyms = get_synonyms(primary_entity, max_synonyms=10)
            if primary_entity not in synonyms:
                synonyms.append(primary_entity)
        
        if primary_entity in pair_similarities:
            desc_ent = pair_similarities[primary_entity][0]
            if desc_ent:
                synonyms.extend(get_synonyms(desc_ent, max_synonyms=10))
                if desc_ent not in synonyms:
                    synonyms.append(desc_ent)
        
        synonym_var = f"?synonym_{primary_var_name}"
        if synonyms:
            where_clauses.append(f"""    VALUES {synonym_var} {{ {' '.join(f'"{syn}"' for syn in synonyms)} }}""")
            where_clauses.append(f"    OPTIONAL {{ {primary_entity_var} :Wordnet {synonym_var} }}")
            if primary_class in ["PhysicalObject", "Animal", "Person"]:
                where_clauses.append(f"    {primary_entity_var} :ObjectName {synonym_var} .")
            elif primary_class == "Context":
                where_clauses.append(f"    {primary_entity_var} :ContextName {synonym_var} .")

        # MODIFIED: Xử lý khi primary_class là Person để thêm 50% điều kiện bắt buộc
        if primary_class == "Person":
            # Thu thập tất cả thuộc tính, hành động, bối cảnh
            attributes = []
            actions = []
            contexts = []

            # Thuộc tính
            for ent in [primary_entity] + [desc_ent for main_ent, (desc_ent, _) in pair_similarities.items()]:
                if ent in data["classified_attributes"]:
                    for attr, values in data["classified_attributes"][ent].items():
                        for value in values:
                            attributes.append((attr, value))

            # Hành động
            for relation in data["classified_relations"]:
                if relation["subject"] in [primary_entity] + [desc_ent for main_ent, (desc_ent, _) in pair_similarities.items()]:
                    actions.append(relation)

            # Bối cảnh
            for context, context_class in desc_entities.items():
                if context_class == "Context" and context not in entity_vars:
                    contexts.append(context)

            # Chọn 50% ngẫu nhiên từ mỗi danh sách
            def select_half(lst):
                half_len = max(1, len(lst) // 2)  # Đảm bảo chọn ít nhất 1 nếu danh sách không rỗng
                return random.sample(lst, min(half_len, len(lst))) if lst else []

            mandatory_attributes = select_half(attributes)
            mandatory_actions = select_half(actions)
            mandatory_contexts = select_half(contexts)

            # Thêm điều kiện bắt buộc cho thuộc tính
            for attr, value in mandatory_attributes:
                attr_synonyms = get_synonyms(value, max_synonyms=5)
                if value not in attr_synonyms:
                    attr_synonyms.append(value)
                attr_synonym_var = f"?synonym_{attr}_{value.replace(' ', '_')}"
                where_clauses.append(f"""    VALUES {attr_synonym_var} {{ {' '.join(f'"{syn}"' for syn in attr_synonyms)} }}""")
                where_clauses.append(f"    {primary_entity_var} :{attr} {attr_synonym_var} .")  # Bắt buộc, không dùng OPTIONAL

            # Thêm điều kiện bắt buộc cho hành động
            for relation in mandatory_actions:
                action_name = relation["predicate"].replace(" ", "_")
                action_var = f"?ActionName_{action_name}"
                where_clauses.append(f"    {action_var} a :Action .")
                synonyms = get_synonyms(relation["predicate"], max_synonyms=5)
                if synonyms:
                    if relation["predicate"] not in synonyms:
                        synonyms.append(relation["predicate"])
                    synonym_var = f"?synonym_ActionName_{action_name}"
                    where_clauses.append(f"""    VALUES {synonym_var} {{ {' '.join(f'"{syn}"' for syn in synonyms)} }}""")
                    where_clauses.append(f"    OPTIONAL {{ {action_var} :Wordnet {synonym_var} }}")
                    where_clauses.append(f"    {action_var} :ActionName {synonym_var} .")
                else:
                    where_clauses.append(f"    {action_var} :ActionName \"{relation['predicate']}\" .")
                where_clauses.append(f"    {action_var} :hasAgent {primary_entity_var} .")
                obj_var = entity_vars.get(relation["object"]) if relation["object"] != "-" else None
                if not obj_var and relation["object"] != "-":
                    obj_var = f"?{relation['object'].replace(' ', '_')}"
                    entity_vars[relation["object"]] = obj_var
                if obj_var:
                    where_clauses.append(f"    {action_var} :hasObject {obj_var} .")

            # Thêm điều kiện bắt buộc cho bối cảnh
            for context in mandatory_contexts:
                var_name = f"ContextName_{context.replace(' ', '_')}"
                context_var = f'?{var_name}'
                entity_vars[context] = context_var
                where_clauses.append(f"    {context_var} a :Context .")
                where_clauses.append(f"    ?image :hasContext {context_var} .")
                synonyms = get_synonyms(context, max_synonyms=5)
                synonym_var = f"?synonym_{var_name}"
                if synonyms:
                    if context not in synonyms:
                        synonyms.append(context)
                    where_clauses.append(f"""    VALUES {synonym_var} {{ {' '.join(f'"{syn}"' for syn in synonyms)} }}""")
                    where_clauses.append(f"    OPTIONAL {{ {context_var} :Wordnet {synonym_var} }}")
                    where_clauses.append(f"    {context_var} :ContextName {synonym_var} .")

            # Thêm điều kiện tùy chọn (OPTIONAL) cho 50% còn lại
            optional_attributes = [x for x in attributes if x not in mandatory_attributes]
            optional_actions = [x for x in actions if x not in mandatory_actions]
            optional_contexts = [x for x in contexts if x not in mandatory_contexts]

            # Thuộc tính tùy chọn
            for attr, value in optional_attributes:
                attr_synonyms = get_synonyms(value, max_synonyms=5)
                if value not in attr_synonyms:
                    attr_synonyms.append(value)
                attr_synonym_var = f"?synonym_{attr}_{value.replace(' ', '_')}"
                where_clauses.append(f"""    VALUES {attr_synonym_var} {{ {' '.join(f'"{syn}"' for syn in attr_synonyms)} }}""")
                where_clauses.append(f"    OPTIONAL {{ {primary_entity_var} :{attr} {attr_synonym_var} }}")

            # Hành động tùy chọn
            for relation in optional_actions:
                action_name = relation["predicate"].replace(" ", "_")
                action_var = f"?ActionName_{action_name}"
                where_clauses.append(f"    OPTIONAL {{")
                where_clauses.append(f"        {action_var} a :Action .")
                synonyms = get_synonyms(relation["predicate"], max_synonyms=5)
                if synonyms:
                    if relation["predicate"] not in synonyms:
                        synonyms.append(relation["predicate"])
                    synonym_var = f"?synonym_ActionName_{action_name}"
                    where_clauses.append(f"""        VALUES {synonym_var} {{ {' '.join(f'"{syn}"' for syn in synonyms)} }}""")
                    where_clauses.append(f"        OPTIONAL {{ {action_var} :Wordnet {synonym_var} }}")
                    where_clauses.append(f"        {action_var} :ActionName {synonym_var} .")
                else:
                    where_clauses.append(f"        {action_var} :ActionName \"{relation['predicate']}\" .")
                where_clauses.append(f"        {action_var} :hasAgent {primary_entity_var} .")
                obj_var = entity_vars.get(relation["object"]) if relation["object"] != "-" else None
                if not obj_var and relation["object"] != "-":
                    obj_var = f"?{relation['object'].replace(' ', '_')}"
                    entity_vars[relation["object"]] = obj_var
                if obj_var:
                    where_clauses.append(f"        {action_var} :hasObject {obj_var} .")
                where_clauses.append(f"    }}")

            # Bối cảnh tùy chọn
            for context in optional_contexts:
                var_name = f"ContextName_{context.replace(' ', '_')}"
                context_var = f'?{var_name}'
                entity_vars[context] = context_var
                where_clauses.append(f"    OPTIONAL {{")
                where_clauses.append(f"        {context_var} a :Context .")
                where_clauses.append(f"        ?image :hasContext {context_var} .")
                synonyms = get_synonyms(context, max_synonyms=5)
                synonym_var = f"?synonym_{var_name}"
                if synonyms:
                    if context not in synonyms:
                        synonyms.append(context)
                    where_clauses.append(f"""        VALUES {synonym_var} {{ {' '.join(f'"{syn}"' for syn in synonyms)} }}""")
                    where_clauses.append(f"        OPTIONAL {{ {context_var} :Wordnet {synonym_var} }}")
                    where_clauses.append(f"        {context_var} :ContextName {synonym_var} .")
                where_clauses.append(f"    }}")
        else:
            # Giữ nguyên logic gốc cho thuộc tính, hành động, bối cảnh khi không phải Person
            for ent in [primary_entity] + [desc_ent for main_ent, (desc_ent, _) in pair_similarities.items()]:
                if ent in data["classified_attributes"]:
                    for attr, values in data["classified_attributes"][ent].items():
                        for value in values:
                            attr_synonyms = get_synonyms(value, max_synonyms=5)
                            if value not in attr_synonyms:
                                attr_synonyms.append(value)
                            attr_synonym_var = f"?synonym_{attr}_{value.replace(' ', '_')}"
                            where_clauses.append(f"""    VALUES {attr_synonym_var} {{ {' '.join(f'"{syn}"' for syn in attr_synonyms)} }}""")
                            where_clauses.append(f"    OPTIONAL {{ {primary_entity_var} :{attr} {attr_synonym_var} }}")

                for relation in data["classified_relations"]:
                    if relation["subject"] == ent:
                        action_name = relation["predicate"].replace(" ", "_")
                        action_var = f"?ActionName_{action_name}"
                        where_clauses.append(f"    OPTIONAL {{")
                        where_clauses.append(f"        {action_var} a :Action .")
                        synonyms = get_synonyms(relation["predicate"], max_synonyms=5)
                        if synonyms:
                            if relation["predicate"] not in synonyms:
                                synonyms.append(relation["predicate"])
                            synonym_var = f"?synonym_ActionName_{action_name}"
                            where_clauses.append(f"""        VALUES {synonym_var} {{ {' '.join(f'"{syn}"' for syn in synonyms)} }}""")
                            where_clauses.append(f"        OPTIONAL {{ {action_var} :Wordnet {synonym_var} }}")
                            where_clauses.append(f"        {action_var} :ActionName {synonym_var} .")
                        else:
                            where_clauses.append(f"        {action_var} :ActionName \"{relation['predicate']}\" .")
                        where_clauses.append(f"        {action_var} :hasAgent {primary_entity_var} .")
                        obj_var = entity_vars.get(relation["object"]) if relation["object"] != "-" else None
                        if not obj_var and relation["object"] != "-":
                            obj_var = f"?{relation['object'].replace(' ', '_')}"
                            entity_vars[relation["object"]] = obj_var
                        if obj_var:
                            where_clauses.append(f"        {action_var} :hasObject {obj_var} .")
                        where_clauses.append(f"    }}")

            for context, context_class in desc_entities.items():
                if context_class == "Context" and context not in entity_vars:
                    var_name = f"ContextName_{context.replace(' ', '_')}"
                    context_var = f'?{var_name}'
                    entity_vars[context] = context_var
                    where_clauses.append(f"    OPTIONAL {{")
                    where_clauses.append(f"        {context_var} a :Context .")
                    where_clauses.append(f"        ?image :hasContext {context_var} .")
                    synonyms = get_synonyms(context, max_synonyms=5)
                    synonym_var = f"?synonym_{var_name}"
                    if synonyms:
                        if context not in synonyms:
                            synonyms.append(context)
                        where_clauses.append(f"""        VALUES {synonym_var} {{ {' '.join(f'"{syn}"' for syn in synonyms)} }}""")
                        where_clauses.append(f"        OPTIONAL {{ {context_var} :Wordnet {synonym_var} }}")
                        where_clauses.append(f"        {context_var} :ContextName {synonym_var} .")
                    where_clauses.append(f"    }}")

        # Xử lý các thực thể phụ (non-primary)
        for entity, entity_class in main_entities.items():
            if entity != primary_entity:
                var_name = f"{entity_class}Name_{entity.replace(' ', '_')}"
                entity_var = f'?{var_name}'
                entity_vars[entity] = entity_var
                where_clauses.append(f"    OPTIONAL {{")
                where_clauses.append(f"        {entity_var} a :{entity_class} .")
                relation = ":hasContext" if entity_class == "Context" else ":contains"
                where_clauses.append(f"        ?image {relation} {entity_var} .")
                synonyms = get_synonyms(entity, max_synonyms=5)
                non_primary_similarities = len([e for e in pair_similarities if e != primary_entity])
                if entity in pair_similarities:
                    desc_ent = pair_similarities[entity][0]
                    synonyms.extend(get_synonyms(desc_ent, max_synonyms=5))
                    if desc_ent not in synonyms:
                        synonyms.append(desc_ent)
                elif non_primary_similarities == 0:
                    for desc_ent, desc_class in desc_entities.items():
                        if desc_class == entity_class:
                            synonyms.extend(get_synonyms(desc_ent, max_synonyms=5))
                            if desc_ent not in synonyms:
                                synonyms.append(desc_ent)
                synonym_var = f"?synonym_{var_name}"
                if synonyms:
                    if entity not in synonyms:
                        synonyms.append(entity)
                    where_clauses.append(f"""        VALUES {synonym_var} {{ {' '.join(f'"{syn}"' for syn in synonyms)} }}""")
                    where_clauses.append(f"        OPTIONAL {{ {entity_var} :Wordnet {synonym_var} }}")
                    if entity_class in ["PhysicalObject", "Animal", "Person"]:
                        where_clauses.append(f"        {entity_var} :ObjectName {synonym_var} .")
                    elif entity_class == "Context":
                        where_clauses.append(f"        {entity_var} :ContextName {synonym_var} .")
                where_clauses.append(f"    }}")

        where_clauses.append("    ?image a :Image ;")
        where_clauses.append("           :ImageURL ?url .")

    else:
        for entity, entity_class in data["classified_entities"].items():
            var_name = f"{entity_class}Name_{entity.replace(' ', '_')}"
            entity_var = f'?{var_name}'
            entity_vars[entity] = entity_var
            where_clauses.append(f"    {entity_var} a :{entity_class} .")
            relation = ":hasContext" if entity_class == "Context" else ":contains"
            where_clauses.append(f"    ?image {relation} {entity_var} .")
            synonyms = get_synonyms(entity, max_synonyms=10)
            synonym_var = f"?synonym_{var_name}"
            if synonyms:
                if entity not in synonyms:
                    synonyms.append(entity)
                where_clauses.append(f"""    VALUES {synonym_var} {{ {' '.join(f'"{syn}"' for syn in synonyms)} }}""")
                where_clauses.append(f"    OPTIONAL {{ {entity_var} :Wordnet {synonym_var} }}")
            if entity_class in ["PhysicalObject", "Animal", "Person"]:
                where_clauses.append(f"    {entity_var} :ObjectName {synonym_var} .")
            elif entity_class == "Context":
                where_clauses.append(f"    {entity_var} :ContextName {synonym_var} .")

        for entity, attributes in data["classified_attributes"].items():
            if entity in entity_vars:
                entity_var = entity_vars[entity]
                for attr, values in attributes.items():
                    for value in values:
                        attr_synonyms = get_synonyms(value, max_synonyms=5)
                        attr_synonym_var = f"?synonym_{attr}_{value.replace(' ', '_')}"
                        if attr_synonyms:
                            if value not in attr_synonyms:
                                attr_synonyms.append(value)
                            where_clauses.append(f"""    VALUES {attr_synonym_var} {{ {' '.join(f'"{syn}"' for syn in attr_synonyms)} }}""")
                            where_clauses.append(f"    OPTIONAL {{ {entity_var} :{attr} {attr_synonym_var} }}")
                            where_clauses.append(f"    {entity_var} :{attr} {attr_synonym_var} .")
                        else:
                            where_clauses.append(f"    {entity_var} :{attr} \"{value}\" .")

        for relation in data["classified_relations"]:
            subj_var = entity_vars.get(relation["subject"])
            obj_var = entity_vars.get(relation["object"]) if relation["object"] != "-" else None
            if not subj_var:
                continue
            action_name = relation["predicate"].replace(" ", "_")
            action_var = f"?ActionName_{action_name}"
            where_clauses.append(f"    {action_var} a :Action .")
            synonyms = get_synonyms(relation["predicate"], max_synonyms=5)
            if synonyms:
                if relation["predicate"] not in synonyms:
                    synonyms.append(action_name)
                synonym_var = f"?synonym_ActionName_{action_name}"
                where_clauses.append(f"""    VALUES {synonym_var} {{ {' '.join(f'"{syn}"' for syn in synonyms)} }}""")
                where_clauses.append(f"    OPTIONAL {{ {action_var} :Wordnet {synonym_var} }}")
                where_clauses.append(f"    {action_var} :ActionName {synonym_var} .")
            else:
                where_clauses.append(f"    {action_var} :ActionName \"{relation['predicate']}\" .")
            where_clauses.append(f"    {action_var} :hasAgent {subj_var} .")
            if obj_var:
                where_clauses.append(f"    {action_var} :hasObject {obj_var} .")

        where_clauses.append("    ?image a :Image ;")
        where_clauses.append("           :ImageURL ?url .")

    return select_clause + "\n".join(where_clauses) + "\n}"

@app.post("/generate_sparql")
def generate_sparql(
    request: QueryRequest, 
    from_image: bool = Query(default=False, description="Indicates if the input is from an image"),
    main_entity: str = Query(default=None, description="Main entity recognized from image")):
    """API nhận câu truy vấn, xử lý NLP và sinh SPARQL"""
    parsed_data = processor.preprocess_text(request.text)
    classification_input = {
        "entities": list(parsed_data["entities"]),
        "attributes": parsed_data["attributes"],
        "relations": parsed_data["relations"],
    }
    
    classified_data = classifier.classify_input(classification_input)
    sparql_query = "PREFIX : <http://www.semanticweb.org/asus/ontologies/2025/1/untitled-ontology-26#> " + generate_sparql_query(classified_data, from_image, main_entity)

    return {
        "sparql_query": sparql_query,
    }