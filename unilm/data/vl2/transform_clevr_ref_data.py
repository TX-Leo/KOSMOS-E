import json  
import base64  
import os  
from PIL import Image  
from tqdm import tqdm
import json  
import re
import numpy as np
import cv2
import random
from collections import defaultdict

import pdb  

class clevr_ref_util:
    def __init__(self, refexp_path, scene_file, num_refexp=-1):
        self.scene_file = scene_file
        self.refexp_path = refexp_path
        self.num_refexp = num_refexp
        self.scenes = None
        self.exps = None
        self.load_scene_refexp()

    def load_scene_refexp(self):
        print('loading scene.json...')
        scenes = json.load(open(self.scene_file))
        self.scenes = scenes['scenes']
        print('loading refexp.json...')
        if self.num_refexp != -1:
            self.exps = json.load(open(self.refexp_path))['refexps'][:self.num_refexp]
        else:
            self.exps = json.load(open(self.refexp_path))['refexps'][:]
        print('loading json done')

        self.imgid_scenes={}
        for sce in self.scenes:
          img_id = sce['image_index']
          self.imgid_scenes[img_id] = sce

    def get_mask_from_refexp(self, refexp, height=-1, width=-1):
        sce = self.get_scene_of_refexp(refexp)
        obj_list = self.get_refexp_output_objectlist(refexp)
        
        heatmap = np.zeros((320,480))

        def from_imgdensestr_to_imgarray(imgstr):
            img = []
            cur = 0
            for num in imgstr.split(','):
                num = int(num)
                img += [cur]*num;
                cur = 1-cur
            img = np.asarray(img).reshape((320,480))
            return img

        for objid in obj_list:
            obj_mask = sce['obj_mask'][str(objid+1)]
            mask_img = from_imgdensestr_to_imgarray(obj_mask)
            heatmap += mask_img
        if height !=-1 and width !=-1:
            heatmap = cv2.resize(heatmap, (width, height))
        return heatmap
        

    def get_scene_of_refexp(self, exp):
        image_index = exp['image_index']
        sce = self.imgid_scenes[image_index]
        return sce


    def get_refexp_output_objectlist(self, exp):
        prog = exp['program']
        image_filename = exp['image_filename']
        last = prog[-1]
        obj_list = last['_output']
        return obj_list


def process_json_file(ref_json_file, sec_json_file, img_root_path, tsv_save_root_path, sample_img_num=-1):  
    clevr_ref = clevr_ref_util(ref_json_file, sec_json_file)
    
    file_counter = 0  
    line_counter = 0  
    valid_cnt = 0
    valid_box_cnt = 0
    template_stat = defaultdict(int)
    basename = 'clevr_ref_sample5k'
    save_name = os.path.join(tsv_save_root_path, f"{basename}_{file_counter}.tsv")
    output_file = open(save_name, "w")  
    print(f"Writing to {save_name}")  
    for refexp in tqdm(clevr_ref.exps):  
        caption = refexp['refexp']
        sce = clevr_ref.get_scene_of_refexp(refexp)
        obj_list = clevr_ref.get_refexp_output_objectlist(refexp)
        
        # pdb.set_trace()
        # {'two_hop.json', 'single_and.json', 'same_relate.json', 
        # 'one_hop.json', 'three_hop.json', 'zero_hop.json', 'single_or.json'}
        template_filename = refexp['template_filename']
        template_stat[template_filename] += 1
        
        if template_filename not in ['zero_hop.json']:
            continue
        
        if len(refexp['refexp'].split(' ')) < 5:
            continue
        # print(refexp['refexp'])
        
        if random.random() < 0.5:
            continue
        
        obj_box = []
        for objid in obj_list:
            obj_box.append(sce['obj_bbox'][str(objid+1)])
        
        valid_cnt += 1
        valid_box_cnt += len(obj_box)
        
        image_name = refexp['image_filename']
        image_path = os.path.join(img_root_path, image_name) 
        try:
            pil_img = Image.open(image_path)
        except FileNotFoundError as e:
            print("No image found at ", image_path)
            continue
        image_h = pil_img.height
        image_w = pil_img.width
        
        with open(image_path, "rb") as image_file:  
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # pdb.set_trace()
        convert_boxes = []
        for ann in obj_box:
            x1, y1, w, h = ann
            x2 = x1 + w
            y2 = y1 + h
            convert_box = [
                0, len(caption),
                x1 / image_w, y1 / image_h, x2 / image_w, y2 / image_h,
                1.0
                ]
            convert_boxes.append(convert_box)
        if len(convert_boxes) == 0:
            continue 
            
        # Write the data to the TSV file  
        output_file.write(f"{image_path}\t{encoded_image}\t{caption}\t{str(convert_boxes)}\n")  
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

        if sample_img_num > 0 and valid_cnt >= sample_img_num:
            break
        
    output_file.close()  
    
    print("valid count", valid_cnt, valid_box_cnt)
    print("template", '\n', template_stat)

  
  
def main():  
    ref_json_file = '/mnt/msranlp/zliang/data/clevr_ref+/clevr_ref+_1.0/refexps/clevr_ref+_train_refexps.json'
    sec_json_file = '/mnt/msranlp/zliang/data/clevr_ref+/clevr_ref+_1.0/scenes/clevr_ref+_train_scenes.json'
    img_root_path = '/mnt/msranlp/zliang/data/clevr_ref+/clevr_ref+_1.0/images/train'
    tsv_save_root_path = '/mnt/msranlp/zliang/data/tuning/clevr_ref'

    # load_mat_files('/mnt/msranlp/zliang/data/totaltext/Train')
    process_json_file(ref_json_file, sec_json_file, img_root_path, tsv_save_root_path, sample_img_num=5000)
  
if __name__ == "__main__":  
    main()  
