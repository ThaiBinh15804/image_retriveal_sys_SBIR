from owlready2 import *
from nltk.corpus import wordnet  # Ensure NLTK is installed and WordNet is downloaded
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class OntologyHandler:
    def __init__(self, ontology_path):
        """Khởi tạo handler và tải ontology từ file OWL."""
        self.ontology = get_ontology(ontology_path).load()
        self.ontology_path = ontology_path

    def find_wordnet(self, name, context):
        """Tìm các WordNet phù hợp nhất liên quan đến tên và ngữ cảnh."""
        synsets = wordnet.synsets(name)
        if not synsets:
            return None

        # Tokenize and clean the context description
        context_tokens = set(word_tokenize(context.lower()))
        stop_words = set(stopwords.words("english"))
        context_tokens = context_tokens - stop_words

        # Rank synsets based on overlap with context
        ranked_synsets = []
        for synset in synsets:
            definition_tokens = set(word_tokenize(synset.definition().lower()))
            overlap = len(context_tokens & definition_tokens)
            if overlap > 0:
                ranked_synsets.append((synset, overlap))

        # Sort by overlap score in descending order
        ranked_synsets.sort(key=lambda x: x[1], reverse=True)

        # Return the top definitions
        return [synset.definition() for synset, _ in ranked_synsets[:3]]  # Top 3 definitions
        
    def add_image_entity(self, image_name, image_url, image_description):
        """Thêm thực thể Image vào ontology."""
        if not image_url:
            raise ValueError("The image_url cannot be None or empty!")

        with self.ontology:
            ImageClass = self.ontology.Image
            new_image = ImageClass(image_name)
            new_image.ImageName = [image_name]
            new_image.ImageURL = [image_url]  # Ensure image_url is not None
            new_image.ImageDescription = [image_description]
    
    def process_and_add_entities(self, image_id, classified_entities, classified_attributes, classified_relations):
        """Xử lý và thêm thực thể, thuộc tính, quan hệ vào ontology với ID ảnh làm định danh."""
        entity_mapping = {}  # Lưu ánh xạ tên thực thể gốc -> tên trong ontology

        with self.ontology:
            # Thêm thực thể vào ontology
            for entity, entity_class in classified_entities.items():
                class_name = entity_class.replace(" ", "")  # "Physical Object" → "PhysicalObject"
                entity_name = f"{image_id}_{class_name}_{entity.replace(' ', '_')}"  # Định danh thực thể

                # Tạo thực thể trong ontology
                ontology_entity = getattr(self.ontology, class_name)(entity_name)

                # Gán thuộc tính name cho thực thể theo lớp
                if class_name in ["Person", "Animal", "PhysicalObject"]:
                    ontology_entity.ObjectName = [entity]
                elif class_name == "Context":
                    ontology_entity.ContextName = [entity]
                elif class_name == "Action":
                    ontology_entity.ActionName = [entity]

                # Tìm và gán WordNet
                wordnet_definitions = self.find_wordnet(entity, classified_attributes.get("image_description", ""))
                if wordnet_definitions:
                    ontology_entity.Wordnet = wordnet_definitions

                # Lưu ánh xạ để dùng cho thuộc tính & quan hệ
                entity_mapping[entity] = entity_name

                # Tìm thực thể ảnh để gán quan hệ
                image_entity = self.ontology.search_one(iri=f"*{image_id}")
                if image_entity:
                    if class_name in ["Person", "Animal", "PhysicalObject"]:  # Thuộc lớp Object
                        image_entity.contains.append(ontology_entity)
                    elif class_name == "Context":  # Thuộc lớp Context
                        image_entity.hasContext.append(ontology_entity)

            # Thêm thuộc tính vào thực thể
            for entity, attributes in classified_attributes.items():
                if entity in entity_mapping:
                    entity_instance = self.ontology.search_one(iri=f"*{entity_mapping[entity]}")
                    if entity_instance:
                        for attr_name, attr_value in attributes.items():
                            if hasattr(entity_instance, attr_name):
                                setattr(entity_instance, attr_name, attr_value)

            # Thêm quan hệ giữa các thực thể
            for relation in classified_relations:
                subject = entity_mapping.get(relation["subject"])
                object_ = entity_mapping.get(relation["object"])
                predicate = relation["predicate"]

                if subject and object_:
                    subject_instance = self.ontology.search_one(iri=f"*{subject}")
                    object_instance = self.ontology.search_one(iri=f"*{object_}")

                    if subject_instance and object_instance:
                        # Tạo thực thể mới thuộc lớp Action
                        action_name = f"{image_id}_Action_{predicate.replace(' ', '_')}"
                        action_instance = self.ontology.Action(action_name)

                        # Gán ActionName
                        action_instance.ActionName = [predicate]

                        # Thiết lập quan hệ với subject và object
                        action_instance.hasAgent.append(subject_instance)
                        action_instance.hasObject.append(object_instance)


    
    def save_ontology(self, output_file=None):
        """Lưu ontology vào file OWL."""
        self.ontology.save(file=output_file or self.ontology_path)

def main():
    # Khởi tạo ontology handler
    ontology_path = "flickr_onto.owl"
    handler = OntologyHandler(ontology_path)

    # Dữ liệu đầu vào
    image_id = "12345.jpg"
    image_url = "http://example.com/image.jpg"
    image_description = "A man operating a pulley system while wearing a hat."

    classified_entities = {
        "hat": "Physical Object",
        "pulley system": "Physical Object",
        "man": "Person"
    }

    classified_attributes = {
        "man": {"Quantity": ["several"]},
        "hat": {"Texture": ["hard"]},
        "pulley system": {"Size": ["giant"]}
    }

    classified_relations = [
        {"subject": "man", "predicate": "operate", "object": "pulley system", "class": "action"},
        {"subject": "man", "predicate": "has_in", "object": "hat", "class": "action"}
    ]

    # Thêm thực thể ảnh
    handler.add_image_entity(image_id, image_url, image_description)
    
    # Xử lý và thêm thực thể, thuộc tính, quan hệ
    handler.process_and_add_entities(image_id, classified_entities, classified_attributes, classified_relations)
    
    # Lưu ontology
    handler.save_ontology("output_ontology.owl")
    
    print("Ontology updated and saved as output_ontology.owl")

if __name__ == "__main__":
    main()
