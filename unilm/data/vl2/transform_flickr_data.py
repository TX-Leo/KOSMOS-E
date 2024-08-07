# import sentencepiece as spm
import json
import random
import numpy as np
from functools import partial
from pycocotools.coco import COCO 
from tqdm import tqdm
import os
from copy import deepcopy
import base64 

import pdb

try:
    import spacy  
except:
    print("If need to prepare the data, install spacy please")
    print("pip install spacy")
    print("python -m spacy download en_core_web_trf ")

def process_box_ann(ann, img_info):
    image_h = float(img_info['height'])
    image_w = float(img_info['width'])
    box = ann["bbox"]
    tokens_positive = ann['tokens_positive'][0]
    convert_box = [
        tokens_positive[0], tokens_positive[1],
        box[0] / image_w, box[1] / image_h, (box[0] + box[2]) / image_w, (box[1] + box[3]) / image_h,
        1.0
        ]
    return convert_box

# image infomation
# {'file_name': '3359636318.jpg', 'height': '334', 'width': '500', 'id': 0, 
#  'caption': 'Two people are talking outside of the video game shop next door to the mobile phone store .', 
# 'dataset_name': 'flickr', 'tokens_negative': [[0, 91]], 'sentence_id': 0, 'original_img_id': 3359636318, 
# 'tokens_positive_eval': [[[0, 10]], [[34, 53]], [[67, 89]]]}

# ann
# {'area': 10752.0, 'iscrowd': 0, 'image_id': 0, 'category_id': 1, 'id': 0, 
# 'bbox': [144.0, 166.0, 64.0, 168.0], 'tokens_positive': [[0, 10]]},
        
def extract_expression(json_file, nlp, output_file_path):
    
    # data = json.load(open(json_file, 'r', encoding='utf-8'))
    coco = COCO(json_file)
    
    processed_dataset = []
    # for item in tqdm(data['images'], desc='Extracting expression for flicker dataset'):
    for img_id in tqdm(coco.imgs.keys(), desc='Extracting expression for flicker dataset'):

        img_info = coco.loadImgs(img_id)[0]  
        ann_ids = coco.getAnnIds(imgIds=img_info['id'])  
        anns = coco.loadAnns(ann_ids)  
        processed_anns = list(map(partial(process_box_ann, img_info=img_info), anns))
        
        phrases_list = []
        expressions_list = []
        nums_list = []
        extension_to_whole = []
        
        caption = img_info['caption'][:-1].strip() if img_info['caption'][-1] == '.' else img_info['caption'].strip()
        phrases_pos = [p[0] for p in img_info['tokens_positive_eval']]
        phrases = [img_info['caption'][p[0]:p[1]] for p in phrases_pos]
        
        if len(phrases) == 0:
            continue
        
        # parsing the caption sentence
        doc = nlp(caption)
        dep = [token.dep_ for token in doc]
        pos = [token.pos_ for token in doc]
        
        matched_phrase_idx = []
        # try to extending the given phrase
        for nc in doc.noun_chunks:
            # detect the matched phrase
            nc_start = nc.start_char
            nc_end = nc.end_char
            matched_phrase_i = -1
            for phrase_i in range(len(phrases_pos)):
                if phrases_pos[phrase_i][0] <= nc_start and phrases_pos[phrase_i][1] >= nc_end:
                    matched_phrase_i = phrase_i
            if matched_phrase_i == -1:
                # pdb.set_trace()
                print(f"Note, a noun ({nc.text}) is not detected in the phrase list {phrases}")
                continue
            phrase_i = matched_phrase_i
            matched_phrase_idx.append(phrase_i)
            
            # extending
            if nc.text in phrases[phrase_i]:
                # determine the matched anns according to the phrase name
                phrase_start_pos = phrases_pos[phrase_i][0]
                phrase_end_pos = phrases_pos[phrase_i][1]
                phrase = phrases[phrase_i]
                matched_anns = [p_ann for p_ann in processed_anns if p_ann[0] == phrase_start_pos and p_ann[1] == phrase_end_pos]
                phrases_list.extend(matched_anns)
                nums_list.append(len(matched_anns))
                
                extend_nc = doc[nc.root.left_edge.i:nc.root.right_edge.i+1]
                extend_nc_start_pos = extend_nc.start_char
                extend_nc_end_pos = extend_nc.end_char
                
                phrase_i += 1
                
                if len(nc.root.conjuncts) == 0 and len(extend_nc.text) > len(phrase):
                    print(f"{phrase} -> {extend_nc.text}")
                    matched_anns = deepcopy(matched_anns)
                    for m_ann in matched_anns:
                        m_ann[0] = extend_nc_start_pos
                        m_ann[1] = extend_nc_end_pos
                    expressions_list.extend(matched_anns)
                    extension_to_whole.extend([True if extend_nc_start_pos == 0 and extend_nc_end_pos == len(caption) else False,]*len(matched_anns))
                else:
                    expressions_list.extend(matched_anns)
                    extension_to_whole.extend([False,]*len(matched_anns))
        
        if len(phrases_list) < len(processed_anns):
            dropped_phrase_idx = [di for di in range(len(phrases)) if di not in matched_phrase_idx]
            dropped_phrase_list = [phrases_pos[di] for di in dropped_phrase_idx]
            extend_pos = [[e[0], e[1]] for e in expressions_list]
            for new_list in dropped_phrase_list:
                if all(  
                    (new_list[1] <= sublist[0]) or (new_list[0] >= sublist[1]) or  
                    (new_list[0] >= sublist[0] and new_list[1] <= sublist[1])  
                    for sublist in extend_pos  
                ):
                    matched_anns = [p_ann for p_ann in processed_anns if p_ann[0] == new_list[0] and p_ann[1] == new_list[1]]
                    phrases_list.extend(matched_anns)
                    nums_list.append(len(matched_anns))
                    expressions_list.extend(matched_anns)
                    extension_to_whole.extend([False,]*len(matched_anns))
            
            if len(phrases_list) < len(processed_anns):
                print("+++++++++++++++++++++++++++")
                print("NOTE: Some phrases dropped!")
                print(caption)
                print(phrases_list)
                print(processed_anns)
                print("+++++++++++++++++++++++++++")
        # print('\n')
        # if any(extension_to_whole):
        #     pdb.set_trace()
        processed_dataset.append(
            {
                'file_name': img_info['file_name'],
                'height': int(img_info['height']),
                'width': int(img_info['width']),
                'image_info': img_info,
                'ori_anns': anns,
                'processed_anns': processed_anns,
                'caption': caption,
                'spacy_dep': dep,
                'spacy_pos': pos,
                'phrases_list': phrases_list,
                'expressions_list': expressions_list,
                'nums_list': nums_list,
                'extension_to_whole': extension_to_whole,
            }
        )
        
    with open(output_file_path, 'w') as output_file:  
        json.dump(processed_dataset, output_file)  
  
    print(f"Saved the JSON file to {output_file_path}")    

def process_json_file(json_file, flickr_img_root_path, tsv_save_root_path):  
    with open(json_file, "r") as file:  
        data = json.load(file)  
    
    file_counter = 0  
    line_counter = 0  
    basename = os.path.basename(json_file)[len("final_"):-5]
    save_name = os.path.join(tsv_save_root_path, f"{basename}_{file_counter}.tsv")
    output_file = open(save_name, "w")  
    print(f"Writing to {save_name}")  
    for item in tqdm(data, desc="saving to tsv"):  
        
        # img_info = item['image_info']
        # Encode the image to base64  
        image_name = item["file_name"]  
        image_path = os.path.join(flickr_img_root_path, image_name)
        
        with open(image_path, "rb") as image_file:  
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")  
        
        # Extract annotations  
        caption = item['caption']
        extra_info = str(item)
        
        # Write the data to the TSV file  
        output_file.write(f"{image_path}\t{encoded_image}\t{caption}\t{str(extra_info)}\n")  
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

def stat_json_file(json_file):  
    with open(json_file, "r") as file:  
        data = json.load(file) 
    
    total_num = len(data)
    exist_extend_cnt = 0
    complete_sentence_cnt = 0
    for item in tqdm(data, desc="Stat the json"): 
        phrases_list = item['phrases_list']
        expressions_list = item['expressions_list']
        is_complete_sentence = not any(item['extension_to_whole']) # the caption is a complete sentence or not
        exist_extend = [p[0]!=e[0] or p[1] != e[1] for p, e in zip(phrases_list, expressions_list)]
        
        if is_complete_sentence:
            complete_sentence_cnt += 1
        if any(exist_extend):
            exist_extend_cnt += 1
    
    print(f"Existing extension sentence: {exist_extend_cnt}/{total_num} = {exist_extend_cnt/total_num}")
    print(f"Complete sentence: {complete_sentence_cnt}/{total_num} = {complete_sentence_cnt/total_num}")
        
        
if __name__ == '__main__':
    # nlp = spacy.load('en_core_web_trf')
    # output_file_path = "/mnt/msranlp/zliang/data/mdetr_annotations/final_flickr_separateGT_train_extend.json"
    # extract_expression("/mnt/msranlp/zliang/data/mdetr_annotations/final_flickr_separateGT_train.json", nlp, output_file_path)
    
    json_file = '/mnt/msranlp/zliang/data/mdetr_annotations/final_flickr_separateGT_train_extend.json'
    # flickr_img_root_path = '/mnt/msranlp/shaohanh/exp/unigpt_exp/data/flickr30k/flickr30k-images'
    # tsv_save_root_path = '/mnt/msranlp/zliang/data/tuning/flickr'
    # process_json_file(json_file, flickr_img_root_path, tsv_save_root_path)
    stat_json_file(json_file)