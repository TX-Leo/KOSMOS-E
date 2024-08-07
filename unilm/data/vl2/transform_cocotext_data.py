import json  
import base64  
import os  
from PIL import Image  
from tqdm import tqdm
import json  
from pycocotools.coco import COCO  

import pdb  

def filter_data(json_file, area_bar=-1, min_side_bar=32):
    with open(json_file, "r") as file:  
        data = json.load(file)  

    train_cnt = 0
    valid_train_cnt = 0
    for img_id in tqdm(data['imgs'].keys(), desc=f'filter the text gt label under the area bar {area_bar} or min  max_side bar {min_side_bar}'):
        # Image info: {'id': 390310, 'set': 'val', 'width': 640, 'file_name': 'COCO_train2014_000000390310.jpg', 'height': 640}
        # Annotaion info: {'mask': [154.7, 458.8, 154.4, 479.7, 243.4, 478.8, 246.3, 457.2], 'class': 'machine printed', 
        # 'bbox': [154.4, 457.2, 91.9, 22.5], 'image_id': 390310, 'id': 117099, 'language': 'english', 'area': 1916.88, 
        # 'utf8_string': 'ELVERT', 'legibility': 'legible'}
        image_info = data['imgs'][img_id]
        if image_info['set'] == 'val':
            continue
        image_h = image_info['height']
        image_w = image_info['width']
        ann_ids = data['imgToAnns'][img_id]
        anns = [data['anns'][str(ann_id)]for ann_id in ann_ids]
        train_cnt += len(anns)
        
        filterd_anns = []
        for ann in anns:
            if ann['language'] not in ['English', 'english']:
                continue
            if ann['legibility'] not in ['legible']:
                continue
            x, y, w, h = ann['bbox']
            if area_bar > 0 and w * h < area_bar:
                continue
            if min_side_bar > 0 and max(h, w) < min_side_bar:
                continue
            filterd_anns.append(ann)   
        valid_train_cnt += len(filterd_anns)
    print(f"{valid_train_cnt}/{train_cnt}={valid_train_cnt / train_cnt}")
        
def process_json_file(json_file, coco_img_root_path, tsv_save_root_path, area_bar=-1, min_side_bar=-1):  
    with open(json_file, "r") as file:  
        data = json.load(file)  
    
    file_counter = 0  
    line_counter = 0  
    basename = os.path.basename(json_file).split('.')[0] + '_v2_train'
    save_name = os.path.join(tsv_save_root_path, f"{basename}_{file_counter}.tsv")
    output_file = open(save_name, "w")  
    print(f"Writing to {save_name}")  
    for img_id in tqdm(data['imgs'].keys()):  
        image_info = data['imgs'][img_id]
        if image_info['set'] == 'val':
            continue  
        
        # Extract annotations 
        image_h = image_info['height']
        image_w = image_info['width']
        ann_ids = data['imgToAnns'][img_id]
        anns = [data['anns'][str(ann_id)]for ann_id in ann_ids]  
        convert_boxes = []
        for ann in anns:
            if ann['language'] not in ['English', 'english']:
                continue
            if ann['legibility'] not in ['Legible', 'legible']:
                continue
            x, y, w, h = ann['bbox']
            if area_bar > 0 and w * h < area_bar:
                continue
            if min_side_bar > 0 and max(h, w) < min_side_bar:
                continue
            
            word = ann['utf8_string']
            box = ann["bbox"]
            convert_box = [
                0, len(word),
                box[0] / image_w, box[1] / image_h, (box[0] + box[2]) / image_w, (box[1] + box[3]) / image_h,
                1.0
                ]
            convert_boxes.append({word: convert_box})
        if len(convert_boxes) == 0:
            continue
        
        # Encode the image to base64  
        image_name = image_info["file_name"]  
        image_path = os.path.join(coco_img_root_path, image_name.split('_')[1], image_name) 

        with open(image_path, "rb") as image_file:  
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8") 
            
        # Write the data to the TSV file  
        output_file.write(f"{image_path}\t{encoded_image}\t{str(convert_boxes)}\n")  
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
    json_file = '/mnt/msranlp/zliang/data/cocotext/cocotext.v2.json'
    coco_img_root_path = '/mnt/msranlp/shaohanh/exp/unigpt_exp/data/COCO'
    tsv_save_root_path = '/mnt/msranlp/zliang/data/tuning/cocotext'

    # process_json_file(json_file, coco_img_root_path, tsv_save_root_path, area_bar=1000)  
    # filter_data(json_file, area_bar=1000)
    # filter_data(json_file, area_bar=2000)
    # filter_data(json_file, area_bar=3000)
    # filter_data(json_file, area_bar=4000) 
    
    process_json_file(json_file, coco_img_root_path, tsv_save_root_path, min_side_bar=23) 
    # filter_data(json_file, min_side_bar=32)
    # filter_data(json_file, min_side_bar=64)
    # filter_data(json_file, min_side_bar=96)
    # filter_data(json_file, min_side_bar=128)       
  
if __name__ == "__main__":  
    main()  
