import json  
import pdb
import os
import cv2  
import numpy as np  
  
def visualize_relationship(image_path, relationship):  
    image = cv2.imread(image_path)  
  
    object_bbox = relationship['object']  
    subject_bbox = relationship['subject']  
  
    cv2.rectangle(image, (object_bbox['x'], object_bbox['y']),  
                  (object_bbox['x'] + object_bbox['w'], object_bbox['y'] + object_bbox['h']),  
                  (0, 255, 0), 2)  
    cv2.rectangle(image, (subject_bbox['x'], subject_bbox['y']),  
                  (subject_bbox['x'] + subject_bbox['w'], subject_bbox['y'] + subject_bbox['h']),  
                  (255, 0, 0), 2)  
    
    subject = relationship["subject"]["name"] if "name" in relationship["subject"] else relationship["subject"]["names"][0]
    obj = relationship["object"]["name"]  if "name" in relationship["object"] else relationship["object"]["names"][0]

    cv2.putText(image, obj, (object_bbox['x'], object_bbox['y'] - 10),  
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  
    cv2.putText(image, subject, (subject_bbox['x'], subject_bbox['y'] - 10),  
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  
    
    dirpath = os.path.dirname(image_path)
    save_name =f"{os.path.basename(image_path).split('.')[0]}_{subject}_{relationship['predicate']}_{obj}.jpg" 
    cv2.imwrite(os.path.join(dirpath, save_name), image)  

    
# with open("/mnt/msranlp/zliang/data/vg/attribute_synsets.json/attribute_synsets.json", "r") as f:  
#     attribute_synsets = json.load(f)  
  
# with open("/mnt/msranlp/zliang/data/vg/object_alias.txt", "r") as f:  
#     object_alias = f.readlines()  
  
# with open("/mnt/msranlp/zliang/data/vg/object_synsets.json/object_synsets.json", "r") as f:  
#     object_synsets = json.load(f)  
  
# with open("/mnt/msranlp/zliang/data/vg/objects.json/objects.json", "r") as f:  
#     objects = json.load(f)  
  
# with open("/mnt/msranlp/zliang/data/vg/relationship_alias.txt", "r") as f:  
#     relationship_alias = f.readlines()  
  
# with open("/mnt/msranlp/zliang/data/vg/relationship_synsets.json/relationship_synsets.json", "r") as f:  
#     relationship_synsets = json.load(f)  
  
with open("/mnt/msranlp/zliang/data/vg/relationships.json/relationships.json", "r") as f:  
    relationships = json.load(f)  
  
image_relationships = relationships[0]["relationships"]  
image_path = '/home/v-zpeng/obj-sam-prev/output/1.jpg'

natural_language_descriptions = []  
for relationship in image_relationships:
    # pdb.set_trace()  
    visualize_relationship(image_path, relationship)
    subject = relationship["subject"]["name"] if "name" in relationship["subject"] else relationship["subject"]["names"][0]
    subject_obj_id = relationship["subject"]['object_id']
    predicate = relationship["predicate"] 
    obj = relationship["object"]["name"]  if "name" in relationship["object"] else relationship["object"]["names"][0]
    object_obj_id = relationship["object"]['object_id']
    
    # description = f"{subject}({subject_obj_id}) {predicate} {obj}({object_obj_id})"
    # description = f"{subject} {predicate} {obj} "
    subject_box = f"{relationship['subject']['x']},{relationship['subject']['y']},{relationship['subject']['w']+relationship['subject']['x']},{relationship['subject']['h']+relationship['subject']['y']}"
    object_box = f"{relationship['object']['x']},{relationship['object']['y']},{relationship['object']['w']+relationship['object']['x']},{relationship['object']['h']+relationship['object']['y']}"
    description = f"{subject}({subject_box}) {predicate} {obj}({object_box})"  
    print(description)

