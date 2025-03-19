import unittest
import os
from classification import EntityClassifier

def main():
    entity_file = "classicalNoun.txt"
    attribute_file = "attributes.txt"

    with open(entity_file, "w", encoding="utf-8") as ef:
        ef.write("Object: car, tree, house\n")
        ef.write("Context: city, forest, village\n")

    with open(attribute_file, "w", encoding="utf-8") as af:
        af.write("Color: red, blue, green\n")
        af.write("Material: metal, wood, plastic\n")

    classifier = EntityClassifier(entity_file, attribute_file)
    
    # Kiểm tra classify_relations
    input_data = {
        "entities": ["city", "car"],
        "attributes": {},
        "relations": [{"subject": "city", "predicate": "contains", "object": "car"}]
    }
    
    classified_data = classifier.classify_input(input_data)
    print("Classified Relations:", classified_data)
    

if __name__ == "__main__":
    main()
