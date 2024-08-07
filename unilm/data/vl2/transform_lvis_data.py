import json  
import base64  
import os  
from PIL import Image  
from tqdm import tqdm
import json  
from pycocotools.coco import COCO  
from lvis import LVIS
import inflect  
import re 

import pdb  
  
def pluralize_noun(noun):  
    p = inflect.engine()  
    return p.plural_noun(noun) 

def remove_brackets(category_name):  
    return re.sub(r'\s?\(.*?\)', '', category_name) 
  
def process_lvis_json_file(json_file_name, lvis_img_root_path, tsv_save_root_path):      
    lvis = LVIS(json_file_name)    
      
    file_counter = 0    
    line_counter = 0    
    basename = os.path.basename(json_file_name)[:-5]
    save_name = os.path.join(tsv_save_root_path, f"{basename}_{file_counter}.tsv")  
    output_file = open(save_name, "w")    
    print(f"Writing to {save_name}")    
    for img_id in tqdm(lvis.imgs.keys()):    
        img_info = lvis.load_imgs([img_id])[0]    
        ann_ids = lvis.get_ann_ids(img_ids=[img_info['id']])    
        anns = lvis.load_anns(ann_ids)    
        
        # Encode the image to base64    
        image_name = '/'.join(img_info['coco_url'].split('/')[-2:])
        image_path = os.path.join(lvis_img_root_path, image_name)  
          
        with open(image_path, "rb") as image_file:    
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")    
          
        # Extract annotations    
        captions = []  
        convert_boxes_dict = {}  
        image_h = img_info['height']  
        image_w = img_info['width']  
          
        for ann in anns:  
            category_id = ann["category_id"]  
            category_name = remove_brackets(lvis.cats[category_id]["name"].replace('_', ' '))
            if category_name not in captions:  
                captions.append(category_name)  
              
            box = ann["bbox"]  
            convert_box = [  
                0, len(category_name),  
                box[0] / image_w, box[1] / image_h, (box[0]+box[2]) / image_w, (box[1]+box[3]) / image_h,  
                1.0  
            ]  
            if category_name not in convert_boxes_dict:  
                convert_boxes_dict[category_name] = []  
            convert_boxes_dict[category_name].append(convert_box)  
          
        plur_captions = list(map(pluralize_noun, captions))
        neg_captions = list(map(lambda category_id: remove_brackets(lvis.cats[category_id]["name"].replace('_', ' ')), img_info['neg_category_ids']))
        
        # pdb.set_trace()
        caption = str([captions, plur_captions, neg_captions])
        # Write the data to the TSV file    
        output_file.write(f"{image_name}\t{encoded_image}\t{caption}\t{str(convert_boxes_dict)}\n")    
          
        # Check if the current file has reached 1000 lines    
        if line_counter == 999:    
            output_file.close()    
            line_counter = 0    
            file_counter += 1    
            save_name = os.path.join(tsv_save_root_path, f"{basename}_{file_counter}.tsv")  
            output_file = open(f"{save_name}", "w")    
            print(f"Writing to {save_name}")    
        else:    
            line_counter += 1    
    
    output_file.close()  


  
  
def main():  
    file_root_path = '/mnt/msranlp/zliang/data/coco/annotations'
    coco_img_root_path = '/mnt/msranlp/zliang/data/coco'
    tsv_save_root_path = '/mnt/msranlp/zliang/data/tuning/lvis'
    json_files = [
        # f"{file_root_path}/finetune_refcoco_train.json", # 120624
        # f"{file_root_path}/finetune_refcoco+_train.json", # 120191
        # f"{file_root_path}/finetune_refcocog_train.json", # 80512
        f"{file_root_path}/lvis_v1_train.json", # 100170
    ]   
  
    for json_file_name in json_files:  
        process_lvis_json_file(json_file_name, coco_img_root_path, tsv_save_root_path)  
    
    # source_files = []
    # for json_file_name in json_files:  
    #     for file_counter in range(100):
    #         file_path = f"{json_file_name[:-5]}_{file_counter}.tsv"
    #         if os.path.isfile(file_path):
    #             normed_file_path = file_path.split('/')[-1]
    #             normed_file_path = "../" + normed_file_path
    #             source_files.append(normed_file_path)
                
    # file_list = {  
    #     "source": source_files,  
    #     "source_lang": "LLaVA-Instruct-150K",  
    #     "weight": 1.0,  
    #     "name": "laion"  
    # }
    
    # with open("/mnt/msranlp/zliang/data/tuning/LLaVA-Instruct-150K/json/train.json", "w") as file_list_file:  
    #     json.dump([file_list], file_list_file, indent=4)
        
  
if __name__ == "__main__":  
    main()  
