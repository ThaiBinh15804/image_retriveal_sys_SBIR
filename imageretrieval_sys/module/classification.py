import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

class EntityClassifier:
    def __init__(self, entity_file, attribute_file, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.entities = self.load_txt_dictionary(entity_file)
        self.attribute_labels = self.load_txt_dictionary(attribute_file)
        self.labels = []
        self.faiss_index = None
        self.build_faiss_index()

    def load_txt_dictionary(self, file_path):
        """Đọc từ điển từ file txt."""
        dictionary = {}
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                category, words = line.strip().split(":")
                dictionary[category] = words.split(", ")
        return dictionary

    def build_faiss_index(self):
        """Tạo FAISS index từ danh sách thực thể."""
        words, vectors, mappings = [], [], {}
        for category, word_list in self.entities.items():
            for word in word_list:
                words.append(word)
                vectors.append(self.model.encode(word))
                mappings[word] = category  # Lưu mapping từ về danh mục
        
        self.word_to_category = mappings  # Lưu lại mapping
        word_embeddings = np.array(vectors, dtype=np.float32)
        
        embedding_dim = word_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.faiss_index.add(word_embeddings)
        self.labels = np.array(words)  # Chỉ dùng để truy xuất từ gần nhất

    def classify_entity(self, word, top_k=1):
        """Phân loại danh từ dựa vào FAISS."""
        word_vector = np.array([self.model.encode(word)]).astype("float32")
        _, indices = self.faiss_index.search(word_vector, top_k)
        closest_word = self.labels[indices[0][0]]  # Lấy từ gần nhất
        return self.word_to_category[closest_word]  # Trả về danh mục thay vì từ

    def classify_entities(self, entities):
        """Phân loại toàn bộ entities."""
        if not entities:
            return {}
        
        word_vectors = np.array(self.model.encode(entities)).astype("float32")
        _, indices = self.faiss_index.search(word_vectors, 1)
        
        return {word: self.word_to_category[self.labels[idx]] for word, idx in zip(entities, indices.flatten())}

    def classify_attributes(self, attributes):
        """Phân loại attributes vào nhóm thuộc tính."""
        if not attributes:
            return {}
        
        ref_words = [word for words in self.attribute_labels.values() for word in words]
        ref_embeddings = np.array(self.model.encode(ref_words)).astype("float32")
        
        classified_attributes = {}
        for entity, attr_list in attributes.items():
            classified_attributes[entity] = {}
            attr_embeddings = np.array(self.model.encode(attr_list)).astype("float32")
            similarity_matrix = np.dot(attr_embeddings, ref_embeddings.T) / (
                np.linalg.norm(attr_embeddings, axis=1, keepdims=True) * np.linalg.norm(ref_embeddings, axis=1)
            )
            
            idx_max = similarity_matrix.argmax(axis=1)
            word_to_label = {word: label for label, words in self.attribute_labels.items() for word in words}
            
            for attr, idx in zip(attr_list, idx_max):
                best_label = word_to_label[ref_words[idx]]
                if best_label not in classified_attributes[entity]:
                    classified_attributes[entity][best_label] = []
                classified_attributes[entity][best_label].append(attr)
        
        return classified_attributes

    def classify_relations(self, relations, classified_entities):
        """Gán quan hệ vào lớp action và xử lý quan hệ giữa Context và Object."""
        updated_entities = classified_entities.copy()
        new_physical_objects = {}

        updated_relations = []
        
        for r in relations:
            subject = r["subject"]
            object_ = r["object"]

            subject_class = classified_entities.get(subject, "Unknown")
            object_class = classified_entities.get(object_, "Unknown")

            # Nếu subject là Context và object là Object → tạo PhysicalObject mới
            if subject_class == "Context" and object_class == "Object":
                new_physical_object_name = f"{subject}_object"
                new_physical_objects[new_physical_object_name] = "PhysicalObject"

                # Thay subject trong quan hệ bằng PhysicalObject mới
                r["subject"] = new_physical_object_name

            # Nếu object là Context và subject là Object → tạo PhysicalObject mới
            elif object_class == "Context" and subject_class == "Object":
                new_physical_object_name = f"{object_}_object"
                new_physical_objects[new_physical_object_name] = "PhysicalObject"

                # Thay object trong quan hệ bằng PhysicalObject mới
                r["object"] = new_physical_object_name

            # Gán lớp action cho quan hệ
            r["class"] = "action"
            updated_relations.append(r)

        # Cập nhật danh sách thực thể với các PhysicalObject mới
        updated_entities.update(new_physical_objects)

        return updated_relations, updated_entities
    
    def classify_input(self, input_data):
        """Chạy toàn bộ quy trình phân loại."""
        classified_entities = self.classify_entities(input_data["entities"])
        classified_attributes = self.classify_attributes(input_data["attributes"])

        # Phân loại quan hệ và cập nhật danh sách thực thể nếu có PhysicalObject mới
        classified_relations, updated_entities = self.classify_relations(
            input_data["relations"], classified_entities
        )

        return {
            "classified_entities": updated_entities,
            "classified_attributes": classified_attributes,
            "classified_relations": classified_relations,
        }