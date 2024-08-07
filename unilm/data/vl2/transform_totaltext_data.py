import json  
import base64  
import os  
from PIL import Image  
from tqdm import tqdm
import json  
import re

import pdb  
 
from scipy.io import loadmat  

def contains_only_symbols(s):  
    pattern = r'^[!@#$%^&*(),.?":{}|<>_\-=+`~/\\;\'\[\]]+$'  
    return bool(re.match(pattern, s))

def load_mat_files(directory):  
    mat_files = {} 
    for filename in tqdm(os.listdir(directory), desc='Load and convert .mat annotations'):  
        if filename.endswith(".mat"):  
            file_path = os.path.join(directory, filename)  
            mat_file = loadmat(file_path)  
            # mat_files.append(mat_file) 
            boxes = mat_file['gt']
            convert_boxes = []
            for box in boxes:
                x1 = int(min(box[1][0]))
                y1 = int(min(box[3][0]))
                x2 = int(max(box[1][0]))
                y2 = int(max(box[3][0]))
                word = str(box[4][0])
                if contains_only_symbols(word):
                    continue
                
                convert_boxes.append([word, x1, y1, x2, y2])
            if len(convert_boxes) > 0:
                mat_files[filename.split('.')[0].split('_')[-1]] = convert_boxes
    # pdb.set_trace()
    return mat_files  
        
def process_json_file(ann_dir, coco_img_root_path, tsv_save_root_path, area_bar=-1, min_side_bar=-1):  
    data = load_mat_files(ann_dir)
    
    file_counter = 0  
    line_counter = 0  
    basename = 'totaltext_train'
    save_name = os.path.join(tsv_save_root_path, f"{basename}_{file_counter}.tsv")
    output_file = open(save_name, "w")  
    print(f"Writing to {save_name}")  
    for img_id in tqdm(data.keys()):  
        image_name = img_id + '.jpg'
        image_path = os.path.join(coco_img_root_path, image_name) 
        try:
            pil_img = Image.open(image_path)
        except FileNotFoundError as e:
            image_name = img_id + '.JPG'
            image_path = os.path.join(coco_img_root_path, image_name)
            pil_img = Image.open(image_path)
        image_h = pil_img.height
        image_w = pil_img.width
        
        with open(image_path, "rb") as image_file:  
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            
        # Extract annotations 
        anns = data[img_id] 
        convert_boxes = []
        for ann in anns:
            word, x1, y1, x2, y2 = ann
            if area_bar > 0 and (x2 - x1) * (y2 - y1) < area_bar:
                continue
            if min_side_bar > 0 and max(y2 - y1, x2 - x1) < min_side_bar:
                continue

            convert_box = [
                0, len(word),
                x1 / image_w, y1 / image_h, x2 / image_w, y2 / image_h,
                1.0
                ]
            convert_boxes.append({word: convert_box})
        if len(convert_boxes) == 0:
            continue 
            
        for each_box in convert_boxes:
            # Write the data to the TSV file  
            output_file.write(f"{image_path}\t{encoded_image}\t{str([each_box])}\n")  
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
    json_file = '/mnt/msranlp/zliang/data/totaltext/Train'
    coco_img_root_path = '/mnt/msranlp/zliang/data/totaltext/Images/Train'
    tsv_save_root_path = '/mnt/msranlp/zliang/data/tuning/totaltext'

    # load_mat_files('/mnt/msranlp/zliang/data/totaltext/Train')
    process_json_file(json_file, coco_img_root_path, tsv_save_root_path)
  
if __name__ == "__main__":  
    main()  
