import json  
import base64  
import os  
from PIL import Image  
from tqdm import tqdm

import pdb

def process_json_file(json_file_name, img_root_path, filter_file_path):  
    with open(json_file_name, "r") as file:  
        data = json.load(file)  
    with open(filter_file_path, "r") as file:  
        filtered_data = file.readlines()
    filtered_data = [f.strip().split('_')[-1] for f in filtered_data]
    file_counter = 0  
    line_counter = 0  
    output_file = open(f"{json_file_name[:-5]}_{file_counter}.tsv", "w")  
    print(f"Writing to {json_file_name[:-5]}_{file_counter}.tsv")
    filter_cnt = 0
    for item in tqdm(data):  
        # Encode the image to base64  
        image_name = item["image"]
        # pdb.set_trace()
        if image_name in filtered_data:
            # print("the image is the val/test for refcoco, we filter it")
            filter_cnt += 1
            continue
        
        if os.path.isfile(os.path.join(img_root_path, "train2014", "COCO_train2014_" + image_name)):
            image_path = os.path.join(img_root_path, "train2014", "COCO_train2014_" + image_name)
        elif os.path.isfile(os.path.join(img_root_path, "val2014", "COCO_val2014_" + image_name)):
            image_path = os.path.join(img_root_path, "val2014", "COCO_val2014_" + image_name)
        else:
            raise ValueError(f"can not find the image {image_name} in {img_root_path}")
        
        with open(image_path, "rb") as image_file:  
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")  
  
        # Extract conversations  
        # conversations = "\n".join([conv["value"] for conv in item["conversations"]])
        conversations = str(item["conversations"])  

        # pdb.set_trace()
        # Write the data to the TSV file  
        output_file.write(f"{item['id']}\t{encoded_image}\t{conversations}\n")  
  
        # Check if the current file has reached 1000 lines  
        if line_counter == 999:  
            output_file.close()  
            line_counter = 0  
            file_counter += 1  
            output_file = open(f"{json_file_name[:-5]}_{file_counter}.tsv", "w")   
            print(f"Writing to {json_file_name[:-5]}_{file_counter}.tsv")
        else:  
            line_counter += 1  
  
    output_file.close()  
    print(f"Total filtered in {json_file_name}: {filter_cnt}")
  
def main():  
    file_root_path = '/mnt/msranlp/zliang/data/tuning/LLaVA-Instruct-150K-filtered'
    img_root_path = '/mnt/msranlp/shaohanh/exp/unigpt_exp/data/COCO'
    filter_file_path = '/mnt/msranlp/wenwan/data/mdetr_annotations/merged_val_test.images'
    json_files = [
        f"{file_root_path}/complex_reasoning_77k.json",
        f"{file_root_path}/conversation_58k.json", 
        f"{file_root_path}/detail_23k.json"
    ]   
  
    for json_file_name in json_files:  
        process_json_file(json_file_name, img_root_path, filter_file_path)  
    
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
