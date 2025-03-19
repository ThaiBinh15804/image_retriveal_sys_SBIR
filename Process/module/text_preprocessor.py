import spacy
import json

class TextPreprocessor:
    def __init__(self):
        """Khởi tạo mô hình ngôn ngữ spaCy."""
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text):
        """
        Xử lý văn bản đầu vào và trích xuất thực thể, thuộc tính, quan hệ.
        
        :param text: Câu mô tả cần xử lý
        :return: Dictionary chứa entities, attributes, relations
        """
        doc = self.nlp(text.lower())  # Chuẩn hóa văn bản về chữ thường
        entities = set()
        attributes = {}
        relations = []
        noun_heads = {}  # Lưu danh từ để ánh xạ thuộc tính
        compound_map = {}  # Lưu danh từ ghép hoàn chỉnh

        # Xác định danh từ và danh từ ghép hợp lệ
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                lemma = token.lemma_
                compound_parts = [child.lemma_ for child in token.children if child.dep_ == "compound"]
                if compound_parts:
                    full_noun = " ".join(compound_parts + [lemma])
                    compound_map[lemma] = full_noun
                    for part in compound_parts:
                        entities.discard(part)
                    entities.discard(lemma)
                    entities.add(full_noun)
                else:
                    if lemma not in compound_map:
                        entities.add(lemma)

                noun_heads[token] = compound_map.get(lemma, lemma)

        # Xác định thuộc tính (ADJ, NUM)
        for token in doc:
            if token.pos_ in ["ADJ", "NUM"] and token.dep_ in ["amod", "nummod"]:
                noun_head = token.head
                if noun_head in noun_heads:
                    noun_lemma = noun_heads[noun_head]
                    attributes.setdefault(noun_lemma, []).append(token.lemma_)

        # Xác định quan hệ động từ
        for token in doc:
            if token.pos_ in ["VERB", "AUX"]:
                subject, obj = None, None
                if token.dep_ == "acl":
                    root_noun = token.head
                    while root_noun.dep_ not in ["nsubj", "nsubjpass", "ROOT"] and root_noun.head != root_noun:
                        root_noun = root_noun.head
                    if root_noun in noun_heads:
                        subject = noun_heads[root_noun]

                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"] and child in noun_heads:
                        subject = noun_heads[child]
                    elif child.dep_ in ["dobj"] and child in noun_heads:
                        obj = noun_heads[child]
                    elif child.dep_ == "prep":
                        for pobj in child.children:
                            if pobj.dep_ == "pobj" and pobj in noun_heads:
                                obj = noun_heads[pobj]

                if subject:
                    relations.append({"subject": subject, "predicate": token.lemma_, "object": obj if obj else "-"})

        # Xác định quan hệ từ giới từ
        for token in doc:
            if token.pos_ == "ADP":
                prep = token.lemma_
                pobj = None
                head_noun = None
                for child in token.children:
                    if child.dep_ == "pobj" and child in noun_heads:
                        pobj = noun_heads[child]

                if token.head in noun_heads:
                    head_noun = noun_heads[token.head]

                if head_noun and pobj:
                    relations.append({"subject": head_noun, "predicate": f"has_{prep}", "object": pobj})

        return {
            "entities": list(entities),
            "attributes": attributes,
            "relations": relations
        }
