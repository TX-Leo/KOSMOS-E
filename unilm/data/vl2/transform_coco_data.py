import json  
import base64  
import os  
from PIL import Image  
from tqdm import tqdm
import json  
from pycocotools.coco import COCO  
import inflect  

import pdb  


def process_json_file(json_file_name, coco_img_root_path, tsv_save_root_path):  
    # with open(json_file_name, "r") as file:  
    #     data = json.load(file)  
  
    coco = COCO(json_file_name)  
    # pluralize = inflect.engine()
    
    file_counter = 0  
    line_counter = 0  
    basename = os.path.basename(json_file_name)[len("finetune_"):-5]
    save_name = os.path.join(tsv_save_root_path, f"{basename}_{file_counter}.tsv")
    output_file = open(save_name, "w")  
    print(f"Writing to {save_name}")  
    for img_id in tqdm(coco.imgs.keys()):  
        img_info = coco.loadImgs(img_id)[0]  
        ann_ids = coco.getAnnIds(imgIds=img_info['id'])  
        anns = coco.loadAnns(ann_ids)  
        
        # Encode the image to base64  
        image_name = img_info["file_name"]  
        if 'coco' in img_info['dataset_name']:
            if 'train2014' in image_name:
                image_path = os.path.join(coco_img_root_path, 'train2014', image_name)  
            else:
                image_path = os.path.join(coco_img_root_path, 'val2014', image_name)
        # elif 'lvis' in img_info['dataset_name']:
        #     image_path = os.path.join(lvis_img_root_path, image_name)
        else:
            raise ValueError(f"Unkonwn dataset {img_info['dataset_name']}")
        
        with open(image_path, "rb") as image_file:  
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")  
        
        # Extract annotations  
        caption = img_info['caption']
        image_h = img_info['height']
        image_w = img_info['width']
        convert_boxes = []
        for ann in anns:
            box = ann["bbox"]
            convert_box = [
                0, len(caption),
                box[0] / image_w, box[1] / image_h, (box[0]+box[2]) / image_w, (box[1]+box[3]) / image_h,
                1.0
                ]
            convert_boxes.append(convert_box)
        # Write the data to the TSV file  
        output_file.write(f"{image_name}\t{encoded_image}\t{caption}\t{str(convert_boxes)}\n")  
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
    file_root_path = '/mnt/msranlp/zliang/data/mdetr_annotations'
    coco_img_root_path = '/mnt/msranlp/shaohanh/exp/unigpt_exp/data/COCO'
    tsv_save_root_path = '/mnt/msranlp/zliang/data/tuning/refcoco'
    json_files = [
        f"{file_root_path}/finetune_refcoco_train.json", # 120624
        f"{file_root_path}/finetune_refcoco+_train.json", # 120191
        f"{file_root_path}/finetune_refcocog_train.json", # 80512
        # f"{file_root_path}/finetune_lvis_train.json", # 360952
    ]   
  
    for json_file_name in json_files:  
        process_json_file(json_file_name, coco_img_root_path, tsv_save_root_path)  
    
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
