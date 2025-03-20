import os
import time
import json
import spacy
import pandas as pd
from owlready2 import *

from module.ontology_handler import OntologyHandler
from module.data_loader import read_comments_csv, read_links_csv, merge_dicts
from module.text_preprocessor import TextPreprocessor
from module.classification import EntityClassifier  # Import lớp EntityClassifier

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Đo thời gian bắt đầu
    total_start_time = time.time()

    # Khởi tạo bộ phân loại thực thể
    start_time = time.time()
    classifier = EntityClassifier(
        os.path.join(current_dir, "module", "classicalNoun.txt"),
        os.path.join(current_dir, "module", "attributes.txt")
    )
    print(f"Initialized EntityClassifier in {time.time() - start_time:.2f} seconds")

    # Khởi tạo xử lý NLP
    start_time = time.time()
    processor = TextPreprocessor()
    print(f"Initialized NLP processor in {time.time() - start_time:.2f} seconds")

    # Đọc file OWL (Ontology)
    start_time = time.time()
    # owl_file = os.path.join(current_dir, "module", "flickr_onto.owl")
    owl_file = "D:/Learn/InSchool/NCKH/image_retriveal_sys/output_ontology.owl"
    ontology_handler = OntologyHandler(owl_file)
    print(f"Loaded ontology in {time.time() - start_time:.2f} seconds")

    # Đọc dữ liệu từ CSV
    start_time = time.time()
    comments_file = os.path.join(current_dir, "data", "3","comments.csv")
    links_file = os.path.join(current_dir, "data", "3","images.csv")

    comments_data = read_comments_csv(comments_file)
    links_data = read_links_csv(links_file)
    unified_data = merge_dicts(comments_data, links_data)
    print(unified_data)
    print(f"Loaded CSV data in {time.time() - start_time:.2f} seconds")

    # Xử lý từng ảnh
    for image_name, data in unified_data.items():
        image_url = data["link"]
        image_description = " ".join(data["comments"])

        all_entities = set()
        all_attributes = {}
        all_relations = []

        for comment in data["comments"]:
            # Xử lý NLP
            start_time = time.time()
            parsed_data = processor.preprocess_text(comment)
            print(f"Processed NLP for comment in {time.time() - start_time:.2f} seconds")

            all_entities.update(parsed_data["entities"])
            for entity, attrs in parsed_data["attributes"].items():
                all_attributes.setdefault(entity, []).extend(attrs)
            all_relations.extend(parsed_data["relations"])

        # Phân loại
        start_time = time.time()
        classified_data = classifier.classify_input(
            {
                "entities": list(all_entities),
                "attributes": all_attributes,
                "relations": all_relations,
            }
        )
        print(f"Classified entities in {time.time() - start_time:.2f} seconds")

        classified_entities = classified_data["classified_entities"]
        classified_attributes = classified_data["classified_attributes"]
        classified_relations = classified_data["classified_relations"]

        # Thêm thực thể vào ontology
        start_time = time.time()
        ontology_handler.add_image_entity(image_name, image_url, image_description)
        ontology_handler.process_and_add_entities(image_name, classified_entities, classified_attributes, classified_relations)
        print(f"Added entities to ontology in {time.time() - start_time:.2f} seconds")

    # Lưu ontology
    start_time = time.time()
    ontology_handler.save_ontology("output_ontology.owl")
    print(f"Saved ontology in {time.time() - start_time:.2f} seconds")

    # Tổng thời gian thực thi
    print(f"Total execution time: {time.time() - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
