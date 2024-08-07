import json  
import base64  
import os  
from PIL import Image  
from tqdm import tqdm
import json  
import numpy as np
import random

import pdb 

def replace_index_to_name(index_list, name_list, bbox_list, height, width, bin_size):
    re_order_string = ""
    for i, item in enumerate(index_list):
        if isinstance(item, list):
            # may be multi object here
            if len(item) == 1:
                idx = item[0]
                phrase = name_list[idx]
                bbox = bbox_list[idx]
                converted_bbox = [
                    bbox[0] / width,
                    bbox[1] / height,
                    bbox[2] / width,
                    bbox[3] / height,
                ]
                ul_idx, lr_idx = get_box_coords_index(bin_size, converted_bbox)
                ul_idx_name = f"<patch_index_{str(ul_idx).zfill(4)}>"
                lr_idx_name = f"<patch_index_{str(lr_idx).zfill(4)}>"
                phrase_box_name = f"<phrase>{phrase}</phrase><object>{ul_idx_name}{lr_idx_name}</object>"
                re_order_string += f" {phrase_box_name}"
            else:
                # pdb.set_trace()
                phrase_box_names = ""
                for idx_ind, idx in enumerate(item):
                    phrase = name_list[idx]
                    bbox = bbox_list[idx]
                    converted_bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
                    ul_idx, lr_idx = get_box_coords_index(bin_size, converted_bbox)
                    ul_idx_name = f"<patch_index_{str(ul_idx).zfill(4)}>"
                    lr_idx_name = f"<patch_index_{str(lr_idx).zfill(4)}>"
                    phrase_box_name = f"<phrase>{phrase}</phrase><object>{ul_idx_name}{lr_idx_name}</object>"
                    if idx_ind + 1 == len(item) - 1:
                        phrase_box_names = phrase_box_names + phrase_box_name + " and "
                    elif idx_ind + 1 == len(item):
                        # the last one
                        phrase_box_names += f"{phrase_box_name}"
                    else:
                        phrase_box_names += f"{phrase_box_name}, "
                re_order_string += f" {phrase_box_names}"
        elif '[SEP]' in item:
            continue
        elif item in ['.', "'", "?", "!", ",", ":"]:
            re_order_string += f"{item}"
        elif len(item) == 1 and i != 0 and index_list[i-1] == "'":
            re_order_string += f"{item}"
        else:
            # pdb.set_trace()
            re_order_string += f" {item}"
    # print(re_order_string, '\t from \t', index_list)
    return re_order_string.strip()

            
def get_box_coords_index(P, box_coords):
    """
    Generate by ChatGPT ^_^
    
    Given a grid of length P and the coordinates of a bounding box, returns the indices of the grid cells that
    correspond to the upper-left and lower-right corners of the bounding box.
    
    Args:
    - P (int): the length of the grid
    - box_coords (np.array of shape (4,)): the normalized coordinates of the bounding box, in the form [x1, y1, x2, y2]
    
    Returns:
    - ul_idx (int): the index of the grid cell that corresponds to the upper-left corner of the bounding box
    - lr_idx (int): the index of the grid cell that corresponds to the lower-right corner of the bounding box
    """
    # pdb.set_trace()
    # Compute the size of each cell in the grid
    cell_size = 1.0 / P
    
    # Compute the indices of the grid cells that correspond to the upper-left and lower-right corners of the bounding box
    ul_x = int(np.floor(max(box_coords[0], 0) / cell_size))
    ul_y = int(np.floor(max(box_coords[1], 0) / cell_size))
    ul_idx = ul_x + ul_y * P
    
    lr_x = int(np.floor(min(box_coords[2], 0.99999) / cell_size))
    lr_y = int(np.floor(min(box_coords[3], 0.99999) / cell_size))
    lr_idx = lr_x + lr_y * P
    
    return ul_idx, lr_idx

def process_json_file(json_file, img_root_path, tsv_save_root_path, bin_size=32):  
    with open(json_file, "r") as file:  
        qar_data = json.load(file) 
    qar_data = [qar for qar in qar_data if qar['label'] == 1]
    random.shuffle(qar_data)
    
    print(f"Load {len(qar_data)} question-answer pairs")
    file_counter = 0  
    line_counter = 0  
    basename = os.path.basename(json_file)[len("pevl_"):-len("_QAR_data.json")]
    save_name = os.path.join(tsv_save_root_path, f"filtered_{basename}_{file_counter}.tsv")
    save_path_name = os.path.join(tsv_save_root_path, f"filtered_{basename}_{file_counter}.locout")
    output_file = open(save_name, "w")  
    output_path_file = open(save_path_name, "w")  
    print(f"Writing to {save_name}")  
    for item in tqdm(qar_data): 
        # pdb.set_trace() 
        # get the basic information
        file_name = item["file_name"]
        image_height = item["height"]
        image_width = item["width"]
        instance_names = item["names"]
        instance_bbox = item["bbox_list"]
        question = item["question"]
        # answer
        answer_choices = item["answer_choices"]  
        answer_label = item["answer_label"] 
        # right_answer = item["right_answer"]
               
        # rationale
        ration_choices = item['rationale_choices']
        ration_label = item['rationale_label']
        # with_ration = item['with_rationale']
        # wrong_ration = item.get('wrong_rationale', None)
        # right_ration = item['right_rationale']
        
        # encode the image
        image_path = os.path.join(img_root_path, file_name)  
        if not os.path.exists(f"{image_path}"):
            print(f"Image path {image_path} not exist, continue")
            continue  
        
        with open(image_path, "rb") as image_file:  
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")  
        
        # pdb.set_trace()
        question_string = replace_index_to_name(question, instance_names, instance_bbox, image_height, image_width, bin_size)
        answer_string_list = [replace_index_to_name(answer, instance_names, instance_bbox, image_height, image_width, bin_size) for answer in answer_choices]
        ration_string_list = [replace_index_to_name(ration, instance_names, instance_bbox, image_height, image_width, bin_size) for ration in ration_choices]
        
        collect_info = {}
        collect_info['question'] = question
        collect_info['question_string'] = question_string
        collect_info['answer_choices'] = answer_choices
        collect_info['answer_choices_string'] = answer_string_list
        collect_info['answer_label'] = answer_label
        
        collect_info['ration_choices'] = ration_choices
        collect_info['ration_label'] = ration_label
        collect_info['ration_choices_string'] = ration_string_list
        
        collect_info['instance_names'] = instance_names
        collect_info['instance_bbox'] = instance_bbox
        
        # some rules to filter the data        
        if '<phrase>' in question_string and '<phrase>' in answer_string_list[answer_label] and '<phrase>' in ration_string_list[ration_label]:
            pass
        else:
            continue
        
        # Write the data to the TSV file  
        output_file.write(f"{image_path}\t{encoded_image}\t{str(collect_info)}\n")  
        output_path_file.write(f"[image]{image_path}<tab><grounding>Question: {question_string} Answer: {answer_string_list[answer_label]} Ration: {ration_string_list[ration_label]}\n")
        # Check if the current file has reached 1000 lines  
        if line_counter == 999:  
            output_file.close()  
            output_path_file.close()
            line_counter = 0  
            file_counter += 1  
            save_name = os.path.join(tsv_save_root_path, f"filtered_{basename}_{file_counter}.tsv")
            save_path_name = os.path.join(tsv_save_root_path, f"filtered_{basename}_{file_counter}.locout")
            output_file = open(f"{save_name}", "w")  
            output_path_file = open(save_path_name, "w")  
            print(f"Writing to {save_name}")  
        else:  
            line_counter += 1  
  
    output_file.close()  
    output_path_file.close()
  
def main():  
    # json_file = '/mnt/msranlp/zliang/data/vcr/pevl_vcr/pevl_vcr_train_QA_data.json'
    json_file = '/mnt/msranlp/zliang/data/vcr/pevl_vcr/pevl_vcr_train_QAR_data.json'
    
    
    img_root_path = '/mnt/msranlp/zliang/data/vcr'
    tsv_save_root_path = '/mnt/msranlp/zliang/data/tuning/vcr'
    
    process_json_file(json_file, img_root_path, tsv_save_root_path) 
        
  
if __name__ == "__main__":  
    main()  
    
    
# QA evaluation
# {
# 'question': ['How', 'is', [0], 'feeling', '?'], 
# 'file_name': 'vcr1images/lsmdc_1054_Harry_Potter_and_the_prisoner_of_azkaban/1054_Harry_Potter_and_the_prisoner_of_azkaban_00.01.46.736-00.01.50.168@0.jpg', 
# 'height': 797, 
# 'width': 1920, 
# 'bbox_list': [[955.7418212890625, 52.329559326171875, 1551.5677490234375, 789.0325927734375, 0.9993261098861694], 
#               [20.42578125, 32.5933837890625, 916.5200805664062, 787.1964721679688, 0.9991201758384705], 
#               [902.0066528320312, 111.65625, 1035.2879638671875, 701.32861328125, 0.9876241683959961], 
#               [1403.113037109375, 282.20465087890625, 1542.410888671875, 557.3927612304688, 0.7517433762550354], 
#               [785.5349731445312, 527.7738647460938, 841.3390502929688, 657.4290161132812, 0.8897306323051453], 
#               [366.7726745605469, 0.0, 487.1645812988281, 79.29119873046875, 0.9390438199043274]], 
# 'names': ['person', 'person', 'person', 'car', 'cellphone', 'clock'], 
# 'answer_choices': [
    # [[0], 'is', 'feeling', 'amused', '.'], 
    # [[0], 'is', 'upset', 'and', 'disgusted', '.'], 
    # [[0], 'is', 'feeling', 'very', 'scared', '.'], 
    # [[0], 'is', 'feeling', 'uncomfortable', 'with', [2], '.']
    # ], 
# 'annot_id': 'val-0', 
# 'img_id': 'val-0', 
# 'answer_label': 1
# }

# QAR evaluation
# {'question': 
#   ['How', 'is', [0], 'feeling', '?', '[SEP]', [0], 'is', 'upset', 'and', 'disgusted', '.'], 
#   'file_name': 'vcr1images/lsmdc_1054_Harry_Potter_and_the_prisoner_of_azkaban/1054_Harry_Potter_and_the_prisoner_of_azkaban_00.01.46.736-00.01.50.168@0.jpg', 
#   'height': 797, 
#   'width': 1920, 
#   'bbox_list': [
    #   [955.7418212890625, 52.329559326171875, 1551.5677490234375, 789.0325927734375, 0.9993261098861694], 
    #   [20.42578125, 32.5933837890625, 916.5200805664062, 787.1964721679688, 0.9991201758384705], 
    #   [902.0066528320312, 111.65625, 1035.2879638671875, 701.32861328125, 0.9876241683959961], 
    #   [1403.113037109375, 282.20465087890625, 1542.410888671875, 557.3927612304688, 0.7517433762550354], 
    #   [785.5349731445312, 527.7738647460938, 841.3390502929688, 657.4290161132812, 0.8897306323051453], 
    #   [366.7726745605469, 0.0, 487.1645812988281, 79.29119873046875, 0.9390438199043274]
#   ], 
#   'names': ['person', 'person', 'person', 'car', 'cellphone', 'clock'], 
#   'rationale_choices': [
    #   [[0], "'", 's', 'mouth', 'has', 'wide', 'eyes', 'and', 'an', 'open', 'mouth', '.'], 
    #   ['When', 'people', 'have', 'their', 'mouth', 'back', 'like', 'that', 'and', 'their', 'eyebrows', 'lowered', 'they', 'are', 'usually', 'disgusted', 'by', 'what', 'they', 'see', '.'], 
    #   [[2, 1, 0], 'are', 'seated', 'at', 'a', 'dining', 'table', 'where', 'food', 'would', 'be', 'served', 'to', 'them', '.', 'people', 'unaccustomed', 'to', 'odd', 'or', 'foreign', 'dishes', 'may', 'make', 'disgusted', 'looks', 'at', 'the', 'thought', 'of', 'eating', 'it', '.'], 
    #   [[0], "'", 's', 'expression', 'is', 'twisted', 'in', 'disgust', '.']
    # ], 
#   'img_id': 'val-0', 
#   'annot_id': 'val-0', 
#   'rationale_label': 3
# }

