'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-02-10 16:23:52
Version: v1
File: 
Brief: 
'''
'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-02-03 21:48:21
Version: v1
File: 
Brief: 
'''
from utils.dataset_processing import grasp, image
from .grasp_data import GraspDatasetBase
class CornellDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """
    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        import glob
        import os
        import re
        super(CornellDataset, self).__init__(**kwargs)

        # dataset_root
        self.dataset_root = file_path
        # grasp_rectangle_files and pc_txt_files
        self.grasp_rectangle_files = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        if len(self.grasp_rectangle_files) == 0:
            self.grasp_rectangle_files = glob.glob(os.path.join(file_path, '*', '*', 'pcd*cpos.txt'))
            self.pc_txt_files = [file for file in glob.glob(os.path.join(file_path, '*', '*', 'pcd*.txt')) if re.compile(r"pcd\d{4}\.txt").match(os.path.basename(file))]
            if len(self.grasp_rectangle_files) == 0:
                self.grasp_rectangle_files = glob.glob(os.path.join(file_path, '*', '*', 'pcd*grasp_xywha.txt'))
                self.grasp_rectangle_files = [f.replace('grasp_xywha.txt', 'cpos.txt') for f in self.grasp_rectangle_files]
                # raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
        else:
            self.pc_txt_files = [file for file in glob.glob(os.path.join(file_path, '*', 'pcd*.txt')) if re.compile(r"pcd\d{4}\.txt").match(os.path.basename(file))]
        self.grasp_rectangle_files = [f.replace("\\", "/") for f in self.grasp_rectangle_files]
        self.pc_txt_files = [f.replace("\\", "/") for f in self.pc_txt_files]
        # image_sum
        self.image_sum = len(self.grasp_rectangle_files)
        # for ds_rotate
        if ds_rotate:
            self.grasp_rectangle_files = self.grasp_rectangle_files[int(self.image_sum * ds_rotate):] + self.grasp_rectangle_files[:int(self.image_sum * ds_rotate)]
        # sort grasp_rectangle_files
        self.grasp_rectangle_files.sort()
        # rgb_files
        self.rgb_files = [f.replace('cpos.txt', 'r.png') for f in self.grasp_rectangle_files]
        # rgb_image_height and rgb_image_width
        (self.rgb_image_height,self.rgb_image_width,_) = image.Image.from_file(self.rgb_files[0]).img.shape # height:480 width:640
        # depth_files
        self.depth_files = [f.replace('cpos.txt', 'd.tiff') for f in self.grasp_rectangle_files]
        # # pointer
        # self.pointer = []
        # for idx in range(self.image_sum):
        #     if idx == 0:
        #         self.pointer.append(len(grasp.GraspRectangles.load_from_cornell_file(self.grasp_rectangle_files[idx])))
        #     self.pointer.append(self.pointer[-1]+len(grasp.GraspRectangles.load_from_cornell_file(self.grasp_rectangle_files[idx])))
        # self.grasp_sum = pointer[-1]
        # grasp_sum
        self.grasp_sum = len(self.grasp_rectangle_files)
        
        # ====== for cornell dataset augmentation ======
        # pc_npy_files
        self.pc_npy_files = [re.sub(r"(?<=pcd\d{4})_\d{2}_\d{2}(?=pc)", "", f.replace('cpos.txt', 'pc.npy')) for f in self.grasp_rectangle_files]
        # pc_pcd_files
        self.pc_pcd_files = [re.sub(r"(?<=pcd\d{4})_\d{2}_\d{2}(?=pc)", "", f.replace('cpos.txt', 'pc.pcd')) for f in self.grasp_rectangle_files]
        # rgb_grasp_files
        self.rgb_grasp_files = [f.replace('cpos.txt', 'rgrasp.png') for f in self.grasp_rectangle_files]
        # grasp_xywha_files
        self.grasp_xywha_files = [f.replace('cpos.txt', 'grasp_xywha.txt') for f in self.grasp_rectangle_files]
        # grasp_tlbra_files
        self.grasp_tlbra_files = [f.replace('cpos.txt', 'grasp_tlbra.txt') for f in self.grasp_rectangle_files]
        # grasp_tsv_files
        self.grasp_tsv_files = [f.replace('cpos.txt', 'grasp.tsv') for f in self.grasp_rectangle_files]
        # predicted_files
        self.predicted_files = [f.replace('cpos.txt', 'predicted.txt') for f in self.grasp_rectangle_files]
        # rgb_predicted_files
        self.rgb_predicted_files = [f.replace('cpos.txt', 'rpredicted.png') for f in self.grasp_rectangle_files]
        # result_files
        self.result_files = [f.replace('cpos.txt', 'result.json') for f in self.grasp_rectangle_files]
        # instructions
        self.instructions,self.obj_names,self.instructions_name,self.instructions_shape,self.instructions_purpose,self.instructions_strategy,self.instructions_color,self.instructions_part,self.instructions_position,self.instructions_angle = self.init_instructions()

    def init_instructions(self):
        import json
        import os
        instructions_path = self.dataset_root+'/else/instructions.json'
        if os.path.exists(instructions_path):
            obj_names = []
            instructions_name = []
            instructions_shape = []
            instructions_purpose = []
            instructions_strategy = []
            instructions_color = []
            instructions_part = []
            instructions_position = []
            instructions_angle = []
        
            with open(instructions_path,'r') as json_file:
                instructions = json.loads(json_file.read())
                for idx in range(self.image_sum):
                    instruction = instructions[str(idx)]
                    if "obj_name" in instruction:
                        obj_names.append(instructions[str(idx)]["obj_name"])
                    if "instruction_name" in instruction:
                        instructions_name.append(instructions[str(idx)]["instruction_name"])
                    if "instruction_shape" in instruction:
                        instructions_shape.append(instructions[str(idx)]["instruction_shape"])
                    if "instruction_purpose" in instruction:
                        instructions_purpose.append(instructions[str(idx)]["instruction_purpose"])
                    if "instruction_strategy" in instruction:
                        instructions_strategy.append(instructions[str(idx)]["instruction_strategy"])
                    if "instruction_color" in instruction:
                        instructions_color.append(instructions[str(idx)]["instruction_color"])
                    if "instruction_part" in instruction:
                        instructions_part.append(instructions[str(idx)]["instruction_part"])
                    if "instruction_position" in instruction:
                        instructions_position.append(instructions[str(idx)]["instruction_position"])
                    if "instruction_angle" in instruction:
                        instructions_angle.append(instructions[str(idx)]["instruction_angle"])

            return instructions,obj_names,instructions_name,instructions_shape,instructions_purpose,instructions_strategy,instructions_color,instructions_part,instructions_position,instructions_angle
        else:
            return [[] for _ in range(10)]
# =================== for generate instructions =============================
    def init_obj_names(self):
        import os
        obj_names_file_path = self.dataset_root+'/else/obj_names.txt'
        if os.path.exists(obj_names_file_path):
            obj_names = self.read_txt_file(self.dataset_root+'/else/obj_names.txt')
        else:
            obj_names = []
        return obj_names
    
    def add_names(self):
        def add_element_to_dict(ordered_dict, key, value):
            from collections import OrderedDict
            new_ordered_dict = OrderedDict([(key, value)] + list(ordered_dict.items()))
            return new_ordered_dict
        import json
        with open(self.dataset_root+'/else/instructions.json','r') as json_file:
            json_data = json.loads(json_file.read())
        for idx in range(self.image_sum):
            json_data[str(idx)] = add_element_to_dict(json_data[str(idx)], "obj_name", self.obj_names[idx])

        with open(self.dataset_root+'/else/instructions.json', "w") as file:
            json.dump(json_data, file,indent=4)
    
    def count_names(self):
        import json
        from random import shuffle
        name_counts = {}
        names_num = 0
        for name in self.obj_names:
            names_num += 1
            if name in name_counts:
                name_counts[name] += 1
            else:
                name_counts[name] = 1
        keys = list(name_counts.keys())
        shuffle(keys)
        print(f'num:{len(keys)}')
        print(f'names_num:{names_num}')
        shuffled_name_counts = {}
        for key in keys:
            shuffled_name_counts[key] = name_counts[key]
        with open(self.dataset_root+'/else/obj_names_counts.json','w') as json_file:
            json.dump(shuffled_name_counts,json_file,indent=4)
        return shuffled_name_counts

    def _distribute_names(self,names_dict, num_groups):
        import math
        total_count = sum(names_dict.values())
        target_count = math.ceil(total_count / num_groups)
        
        groups = [[] for _ in range(num_groups)]
        group_counts = [0] * num_groups
        
        sorted_names = sorted(names_dict.items(), key=lambda x: x[1], reverse=True)
        
        for name, count in sorted_names:
            min_count = min(group_counts)
            min_index = group_counts.index(min_count)
        
            if min_count + count <= target_count:
                groups[min_index].append(name)
                group_counts[min_index] += count
            else:
                for i in range(num_groups):
                    if group_counts[i] + count <= target_count:
                        groups[i].append(name)
                        group_counts[i] += count
                        break
        return groups

    def distribute_names(self):
        import json
        with open(self.dataset_root+'/else/obj_names_counts.json','r') as file:
            names_dict = json.load(file)
        num_groups = 5
        name_groups = self._distribute_names(names_dict, num_groups)
        results = {}

        names = list(names_dict.keys())
        keys_to_remove = []
        
        for i, group in enumerate(name_groups):
            group_data = {}
            group_data['obj_names'] = group
            
            
            group_data['names_counts'] = [names_dict[name] for name in group]
            group_data['total_names_count'] = sum(group_data['names_counts'])

            group_key = f'{i + 1:02}'
            results[group_key] = group_data

            for key in group:
                if key in names:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            names.remove(key)
        print(f'====={names}')

        with open(self.dataset_root+'/else/name_groups.json', 'w') as file:
            json.dump(results, file, indent=4)

    def init_instructions_name_shape_purpose_strategy_color_part(self):
        import json
        # instructions_name/shape/purpose/strategy
        instructions_name = []
        instructions_shape = []
        instructions_purpose = []
        instructions_strategy = []
        with open(self.dataset_root+'/else/instructions_name_shape_purpose_strategy.json','r') as json_file:
            json_data = json.load(json_file)
            for obj_name in self.obj_names:
                instructions_name.append(json_data[obj_name]["instruction_name"])
                instructions_shape.append(json_data[obj_name]["instruction_shape"])
                instructions_purpose.append(json_data[obj_name]["instruction_purpose"])
                instructions_strategy.append(json_data[obj_name]["instruction_strategy"])

        # instructions_color
        instructions_color = []
        with open(self.dataset_root+'/else/instructions_color.json','r') as json_file:
            json_data = json.loads(json_file)
            for idx in range(self.image_sum):
                instructions_color.append(json_data[idx])

        # instructions_part
        instructions_part = []
        with open(self.dataset_root+'/else/instructions_part.json','r') as json_file:
            json_data = json.loads(json_file)
            for idx in range(self.image_sum):
                instructions_part.append(json_data[idx])

        return instructions_name, instructions_shape, instructions_purpose, instructions_strategy, instructions_color, instructions_part
    
    def generate_instructions(self):
        import json
        instructions = {}
        for idx in self.image_sum:
            obj_name = self.obj_names[idx]
            instruction_name = self.instructions_name[idx]
            instruction_shape = self.instructions_shape[idx]
            instruction_purpose = self.instructions_purpose[idx]
            instruction_strategy = self.instructions_strategy[idx]
            instruction_color = self.instructions_color[idx]
            instruction_part = self.instructions_part[idx]
            instructions[idx] = {"obj_name":obj_name,
                                 "instruction_name":instruction_name,
                                 "instruction_shape":instruction_shape,
                                 "instruction_purpose":instruction_purpose,
                                 "instruction_strategy":instruction_strategy,
                                 "instruction_color":instruction_color,
                                 "instruction_part":instruction_part
                                 }
        with open(self.dataset_root+'/else/instructions.json','w') as json_file:
            json.dump(instructions, json_file,indent=4)
    
    def generate_instruction_color(self,idx):
        img_path = self.rgb_files[idx]
        prompt_color = """As an expert in color recognition, your task is to analyze a series of images featuring objects positioned on a white table. 
Your goal is to provide a comprehensive description of the color(s) exhibited by each object, using precise color nouns such as black, white, or green. 
If an object displays multiple colors, please provide an accurate description for each of them. 
The final response should be formatted as "Grasp the [color] object," where "[color]" represents the color of the object. You must begin with "Grasp the object. And you don't need to describe that kind of object it is."
Analyze the image and determine the color(s) of the object placed on the white table. 
"""
        response = self.GPT4V(prompt_color,img_path)
        print(f'idx_{idx}, response: {response}')
        return response

    def generate_instructions_color(self,idxs):
        import json
        json_temp_file_path = self.dataset_root + '/else/instructions_color_temp_9.json'
        data = {}
        for idx in idxs:
            response = self.generate_instruction_color(idx)
            data[idx] = response
            print('============================')
            print(f'data: {data}')
            with open(json_temp_file_path, 'w') as json_temp_file:
                json.dump(data, json_temp_file,indent=4)

    def generate_instruction_part(self,idx):
        img_path = self.rgb_files[idx]
        instruction_part = {}
        grasp_Rectangles = grasp.GraspRectangles.load_from_cornell_file(self.grasp_rectangle_files[idx])
        obj_name = self.obj_names[idx]
        for i in range(len(grasp_Rectangles)):
            grasp_Rectangle = grasp_Rectangles[i]
            point1 = grasp_Rectangle.points[0]
            point2 = grasp_Rectangle.points[1]
            point3 = grasp_Rectangle.points[2]
            point4 = grasp_Rectangle.points[3]
            prompt_part = f"""Now, I would like you to specialize in providing detailed descriptions of rectangular grasps on objects depicted in images.
I will provide you with an image featuring an object placed on a white table. Additionally, I will share with you the name of the object and the coordinates of the four corner points that define a rectangular grasp. Each corner point is represented by a pair of coordinates (x, y).
Your objective is to deliver a comprehensive description of the relationship between the grasp and the object, with a particular emphasis on identifying the specific part of the object that will be grasped.
You should carefully consider the object's characteristics, such as the mouth or handle of a water bottle, or the front, middle, or back end of a remote control, or even the edge or middle of a hat, to determine the precise location of the grasp on the object.
The four provided coordinates represent the positions of the rectangle's vertices within the image (pixel coordinates).
Your final response should be formatted as "Grasp the [part] of the object," where "[part]" denotes the specific part of the object being grasped.
Analyze the image and based on the given coordinates of the grasp rectangle corners: {point1}, {point2}, {point3}, {point4}, determine which part of the object ({obj_name}) will be grasped."""
            response = self.GPT4V(prompt_part,img_path)
            instruction_part[i] = response
            # print(prompt_part)
            print(response)
        return instruction_part

    def generate_instructions_part(self,idxs):
        import json
        json_temp_file_path = self.dataset_root + '/else/instructions_part_temp_9.json'
        data = {}
        for idx in idxs:
            response = self.generate_instruction_part(idx)
            data[idx] = response
            print('============================')
            print(f'data: {data}')
            with open(json_temp_file_path, 'w') as json_temp_file:
                json.dump(data, json_temp_file,indent=4)

#  ======================================================

    def get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_rectangle_files[idx])
        center = gtbbs.center
        left = max(0, min(center[0] - self.output_size // 2, self.rgb_image_width - self.output_size))
        top = max(0, min(center[1] - self.output_size // 2, self.rgb_image_height - self.output_size))
        return center, left, top

    def get_grasp_Rectangles(self, idx, rot=0, zoom=1.0):
        grasp_Rectangles = grasp.GraspRectangles.load_from_cornell_file(self.grasp_rectangle_files[idx])
        center, left, top = self.get_crop_attrs(idx)
        grasp_Rectangles.rotate(rot, center)
        grasp_Rectangles.offset((-left, -top))
        grasp_Rectangles.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return grasp_Rectangles

    def get_rgb_Image(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_Image = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self.get_crop_attrs(idx)
        rgb_Image.rotate(rot, center)
        rgb_Image.crop((top, left), (min(self.rgb_image_height, top + self.output_size), min(self.rgb_image_width, left + self.output_size)))
        rgb_Image.zoom(zoom)
        rgb_Image.resize((self.output_size, self.output_size))
        if normalise:
            rgb_Image.normalise()
            rgb_Image.img = rgb_Image.img.transpose((2, 0, 1)) #640*480*3-->3*640*480
        return rgb_Image
    
    def generate_depth_Image(self,idx):    
        depth_Image = image.DepthImage.from_pcd(self.pc_txt_files[idx],(self.rgb_image_height,self.rgb_image_width))
        depth_Image.inpaint()
        depth_Image.save(self.depth_files[idx])
    
    def get_depth_Image(self, idx, rot=0, zoom=1.0):
        depth_Image = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self.get_crop_attrs(idx)
        depth_Image.rotate(rot, center)
        depth_Image.crop((top, left), (min(self.rgb_image_height, top + self.output_size), min(self.rgb_image_width, left + self.output_size)))
        depth_Image.normalise()
        depth_Image.zoom(zoom)
        depth_Image.resize((self.output_size, self.output_size))
        return depth_Image

    def rotate_rgb_Image_and_grasp_Rectangles(self,Image,grasp_Rectangles,angle,center):
        # rotate image
        Image_rotated = Image.copy()
        Image_rotated.rotate(angle=angle)

        # rotate grasp
        grasp_Rectangles_rotated = grasp_Rectangles.copy()
        grasp_Rectangles_rotated.rotate(angle,center)
        
        return Image_rotated,grasp_Rectangles_rotated
    
    def translate_rgb_Image_and_grasp_Rectangles(self,Image,grasp_Rectangles,x_shift,y_shift):
        # translate image
        Image_translated = Image.copy()
        Image_translated.translate(x_shift,y_shift)
        
        # translate grasp
        grasp_Rectangles_translated = grasp_Rectangles.copy()
        grasp_Rectangles_translated.translate(x_shift,y_shift)
        
        return Image_translated,grasp_Rectangles_translated

    def crop_rgb_Image_and_grasp_Rectangles(self,Image,grasp_Rectangles,original_size,cropped_size):
        # crop image
        Image_cropped = Image.copy()
        Image_cropped.crop(top_left=((original_size-cropped_size)//2,(original_size-cropped_size)//2),bottom_right=(cropped_size+(original_size-cropped_size)//2,cropped_size+(original_size-cropped_size)//2))
        Image_cropped.resize((cropped_size, cropped_size))
        
        # test crop grasp
        grasp_Rectangles_cropped = grasp_Rectangles.copy()
        grasp_Rectangles_cropped.offset(offset=(-(original_size-cropped_size)//2, -(original_size-cropped_size)//2))

        return Image_cropped,grasp_Rectangles_cropped
    
    @classmethod
    def show_rgb_Image_and_grasp_Rectangles(self,rgb_Image,grasp_Rectangles,vis=True,save_path=None):
        import cv2
        # show all rect grasp in an image
        if vis or save_path is not None:
            if isinstance(rgb_Image,image.Image):
                img = rgb_Image.img
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = rgb_Image
            for grasp_rectangle in grasp_Rectangles:
                points = grasp_rectangle.points
                cv2.line(img,(round(points[0][0]),round(points[0][1])),(round(points[1][0]),round(points[1][1])),(0,0,0),1)
                cv2.line(img,(round(points[1][0]),round(points[1][1])),(round(points[2][0]),round(points[2][1])),(255,0,0),1)
                cv2.line(img,(round(points[2][0]),round(points[2][1])),(round(points[3][0]),round(points[3][1])),(0,0,0),1)
                cv2.line(img,(round(points[3][0]),round(points[3][1])),(round(points[0][0]),round(points[0][1])),(255,0,0),1)
            if save_path is not None:
                cv2.imwrite(save_path,img)
            if vis:
                cv2.imshow('img',img)
                cv2.waitKey(0)
    
    @classmethod
    def show_rgb_Image_and_Grasps(self,rgb_Image,Grasps,vis=True,save_path=None,line_weight=2,color='blue'):
        import cv2
        # show all grasp in an image
        if vis or save_path is not None:
            if isinstance(rgb_Image,image.Image):
                img = rgb_Image.img
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = rgb_Image
            for grasp_rectangle in Grasps.as_grs:
                points = grasp_rectangle.points
                cv2.line(img,(round(points[0][0]),round(points[0][1])),(round(points[1][0]),round(points[1][1])),(0,0,0),1)
                if color=='blue':
                    cv2.line(img,(round(points[1][0]),round(points[1][1])),(round(points[2][0]),round(points[2][1])),(255,0,0),line_weight)
                if color=='red':
                    cv2.line(img,(round(points[1][0]),round(points[1][1])),(round(points[2][0]),round(points[2][1])),(0,0,255),line_weight)
                cv2.line(img,(round(points[2][0]),round(points[2][1])),(round(points[3][0]),round(points[3][1])),(0,0,0),1)
                if color=='blue':
                    cv2.line(img,(round(points[3][0]),round(points[3][1])),(round(points[0][0]),round(points[0][1])),(255,0,0),line_weight)
                if color=='red':
                    cv2.line(img,(round(points[3][0]),round(points[3][1])),(round(points[0][0]),round(points[0][1])),(0,0,255),line_weight)
            if save_path is not None:
                cv2.imwrite(save_path,img)
            if vis:
                cv2.imshow('img',img)
                cv2.waitKey(0)

    @classmethod
    def show_rgb_Image_and_predicted_Grasps_and_real_Grasps(self,rgb_Image,predicted_Grasps,real_Grasps,vis=True,save_path=None,real_line_weight=1,predicted_line_weight=2):
        import cv2
        # show all grasp in an image
        if vis or save_path is not None:
            if isinstance(rgb_Image,image.Image):
                img = rgb_Image.img
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = rgb_Image
            for real_grasp_rectangle in real_Grasps.as_grs:
                points = real_grasp_rectangle.points
                cv2.line(img,(round(points[0][0]),round(points[0][1])),(round(points[1][0]),round(points[1][1])),(0,0,0),1)
                cv2.line(img,(round(points[1][0]),round(points[1][1])),(round(points[2][0]),round(points[2][1])),(255,0,0),real_line_weight)
                cv2.line(img,(round(points[2][0]),round(points[2][1])),(round(points[3][0]),round(points[3][1])),(0,0,0),1)
                cv2.line(img,(round(points[3][0]),round(points[3][1])),(round(points[0][0]),round(points[0][1])),(255,0,0),real_line_weight)
            for predicted_grasp_rectangle in predicted_Grasps.as_grs:
                points = predicted_grasp_rectangle.points
                cv2.line(img,(round(points[0][0]),round(points[0][1])),(round(points[1][0]),round(points[1][1])),(0,0,0),1)
                cv2.line(img,(round(points[1][0]),round(points[1][1])),(round(points[2][0]),round(points[2][1])),(0,0,255),predicted_line_weight)
                cv2.line(img,(round(points[2][0]),round(points[2][1])),(round(points[3][0]),round(points[3][1])),(0,0,0),1)
                cv2.line(img,(round(points[3][0]),round(points[3][1])),(round(points[0][0]),round(points[0][1])),(0,0,255),predicted_line_weight)
            if save_path is not None:
                cv2.imwrite(save_path,img)
            if vis:
                cv2.imshow('img',img)
                cv2.waitKey(0)
    
    @classmethod
    def add_obj_name_to_img(self, rgb_Image, grasp_Rectangles, obj_name, vis=True, save_path=None):
        import cv2
        # show all rect grasp in an image
        if vis or save_path is not None:
            if isinstance(rgb_Image,image.Image):
                img = rgb_Image.img
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = rgb_Image
            for grasp_rectangle in grasp_Rectangles:
                points = grasp_rectangle.points
                cv2.line(img, (round(points[0][0]), round(points[0][1])), (round(points[1][0]), round(points[1][1])), (0, 0, 0), 1)
                cv2.line(img, (round(points[1][0]), round(points[1][1])), (round(points[2][0]), round(points[2][1])), (255, 0, 0), 1)
                cv2.line(img, (round(points[2][0]), round(points[2][1])), (round(points[3][0]), round(points[3][1])), (0, 0, 0), 1)
                cv2.line(img, (round(points[3][0]), round(points[3][1])), (round(points[0][0]), round(points[0][1])), (255, 0, 0), 1)
            font = cv2.FONT_HERSHEY_COMPLEX
            bottom_left_corner = (10, img.shape[0] - 10) 
            font_scale = 0.5
            font_color = (0, 0, 0) 
            line_type = 1
            font_thickness = 1
            cv2.putText(img, obj_name, bottom_left_corner, font, font_scale, font_color, font_thickness, line_type)
            if save_path is not None:
                cv2.imwrite(save_path, img)
            if vis:
                cv2.imshow('img', img)
                cv2.waitKey(0)
    
    def show_original_rgb_Image_and_grasp_Rectangles(self,idx): 
        rgb_Image = image.Image.from_file(self.rgb_files[idx])      
        grasp_Rectangles = grasp.GraspRectangles.load_from_cornell_file(self.grasp_rectangle_files[idx])
        self.show_rgb_Image_and_grasp_Rectangles(rgb_Image,grasp_Rectangles,vis=True)
    
    def show_cropped_rgb_Image_and_grasp_Rectangles(self,idx): 
        rgb_Image = self.get_rgb_Image(idx=idx,normalise=False) # 640*480-->351*351
        grasp_Rectangles = self.get_grasp_Rectangles(idx=idx)
        self.show_rgb_Image_and_grasp_Rectangles(rgb_Image,grasp_Rectangles,vis=True)

    def show_original_rgb_Image_and_Grasps(self,idx):
        rgb_Image = image.Image.from_file(self.rgb_files[idx])      
        Grasps = grasp.Grasps.load_from_cornell_file(self.grasp_xywha_files[idx])
        self.show_rgb_Image_and_Grasps(rgb_Image,Grasps,vis=True)
    
    def generate_pc_npy(self,idx,save_path):
        import numpy as np
        data = np.loadtxt(self.pc_txt_files[idx], skiprows=10)  # load the pcd file and skip the first 11 rows
        points = data[:, 0:3] # only save xyz
        np.save(save_path, points)
    
    def generate_pc_pcd(self,idx,save_path):
        import open3d as o3d
        import numpy as np
        data = np.loadtxt(self.pc_txt_files[idx], skiprows=10)  # load the pcd file and skip the first 11 rows
        points = data[:, :3]  # only save xyz
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(save_path, point_cloud)
    
    def show_pc_txt(self,idx):
        import numpy as np
        import open3d as o3d
        point_cloud = np.loadtxt(self.pc_txt_files[idx], skiprows=10, usecols=(0, 1, 2))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([pcd])
        
    def show_pc_npy(self,idx):
        import numpy as np
        import open3d as o3d
        point_cloud = np.load(self.pc_npy_files[idx])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([pcd])
        
    def show_pc_pcd(self,idx):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(self.pc_pcd_files[idx])
        o3d.visualization.draw_geometries([pcd])
    
    def image_numpy_to_base64(self,img_numpy):
        import base64
        from PIL import Image
        import io
        # Numpy to PIL.Image
        img_pil = Image.fromarray(img_numpy)
        # PIL.Image to BytesIO
        img_byte_array = io.BytesIO()
        img_pil.save(img_byte_array, format='PNG')
        img_bytes = img_byte_array.getvalue()
        # BytesIO to base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64
    
    def generate_grasp_xywha_tsv(self,rgb_Image,Grasps,tsv_file_path,relative_pc_npy_file_path=None,xywha='xywha',encoded=False,decimals=None,instruction=None):
        import csv
        # get rgb image base64
        if isinstance(rgb_Image, image.Image):
            rgb_image_base64 = self.image_numpy_to_base64(rgb_Image.img)
            (rgb_image_height,rgb_image_width,_) = rgb_Image.img.shape
        if isinstance(rgb_Image, str):
            rgb_image_base64 = rgb_Image
            (rgb_image_height,rgb_image_width) = (224,224)
        
        data_list = []
        # for every grasp
        for i in range(len(Grasps)):
            grasp = Grasps[i]
            text = ''
            # instruction
            if isinstance(instruction, str):
                text += instruction
                text += ' '
            elif isinstance(instruction,dict):
                text += instruction[str(i)]
                text += ' '

            # get grasp data
            x,y,w,h,a = grasp.get_data
            
            # int
            x = int(x)
            y = int(y)

            # get text
            if 'xy' in xywha:
                text += f'Grasp center point coordinates: x_{x},y_{y} '
            if 'w' in xywha:
                if decimals is not None:
                    w = round(w,decimals)
                text += f'Grasp width: w_{w} '
            if 'h' in xywha:
                if decimals is not None:
                    h = round(h,decimals)
                text += f'Grasp height: h_{h} '
            if 'a' in xywha:
                if encoded:
                    a_encoded = grasp.a_encoded
                    text += f'Grasp rotation angle in radians: a_{a_encoded} '
                else:
                    if decimals is not None:
                        a = round(a,decimals)
                    text += f'Grasp rotation angle in radians: a_{a} '

            # delete all blankspace at the end
            text = text.rstrip()
            # get data
            if relative_pc_npy_file_path:
                data = ['00000000000',text,rgb_image_base64,rgb_image_width,rgb_image_height,relative_pc_npy_file_path]
            else:
                data = ['00000000000',text,rgb_image_base64,rgb_image_width,rgb_image_height]
            data_list.append(data)
            # print(f'text:\n{text}\nrgb_image_width:\n{rgb_image_width}\nrgb_image_height:\n{rgb_image_height}\nrelative_pc_npy_file_path:{relative_pc_npy_file_path}')
                
        # write data_list to tsv file
        with open(tsv_file_path,'w',newline='') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')
            for data in data_list:
                writer.writerow(data)
    
    def generate_grasp_xywha_tsv_with_instruction_angle(self,rgb_Image,Grasps,tsv_file_path,relative_pc_npy_file_path=None,xywha='xywha',encoded=False,decimals=None):
        import csv
        # get rgb image base64
        if isinstance(rgb_Image, image.Image):
            rgb_image_base64 = self.image_numpy_to_base64(rgb_Image.img)
            (rgb_image_height,rgb_image_width,_) = rgb_Image.img.shape
        if isinstance(rgb_Image, str):
            rgb_image_base64 = rgb_Image
            (rgb_image_height,rgb_image_width) = (224,224)
        
        data_list = []
        # for every grasp
        for grasp in Grasps:
            text = ''
            # get grasp data
            x,y,w,h,a = grasp.get_data
            
            # int
            x = int(x)
            y = int(y)

            # get text
            if 'a' in xywha:
                if encoded:
                    a_encoded = grasp.a_encoded
                    text += f'Grasp the object with a rotation angle of a_{a_encoded} radians. '
                else:
                    if decimals is not None:
                        a = round(a,decimals)
                    text += f'Grasp the object with a rotation angle of a_{a} radians. '
            if 'xy' in xywha:
                text += f'Grasp center point coordinates: x_{x},y_{y} '
            if 'w' in xywha:
                if decimals is not None:
                    w = round(w,decimals)
                text += f'Grasp width: w_{w} '
            if 'h' in xywha:
                if decimals is not None:
                    h = round(h,decimals)
                text += f'Grasp height: h_{h} '
            
            # delete all blankspace at the end
            text = text.rstrip()
            # get data
            if relative_pc_npy_file_path:
                data = ['00000000000',text,rgb_image_base64,rgb_image_width,rgb_image_height,relative_pc_npy_file_path]
            else:
                data = ['00000000000',text,rgb_image_base64,rgb_image_width,rgb_image_height]
            data_list.append(data)
            # print(f'text:\n{text}\nrgb_image_width:\n{rgb_image_width}\nrgb_image_height:\n{rgb_image_height}\nrelative_pc_npy_file_path:{relative_pc_npy_file_path}')
                
        # write data_list to tsv file
        with open(tsv_file_path,'w',newline='') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')
            for data in data_list:
                writer.writerow(data)

    def generate_instruction_tsv(self,rgb_Image,instruction,tsv_file_path,relative_pc_npy_file_path=None):
        import csv
        # get rgb image base64
        if isinstance(rgb_Image, image.Image):
            rgb_image_base64 = self.image_numpy_to_base64(rgb_Image.img)
            (rgb_image_height,rgb_image_width,_) = rgb_Image.img.shape
        if isinstance(rgb_Image, str):
            rgb_image_base64 = rgb_Image
            (rgb_image_height,rgb_image_width) = (224,224)

        # get data
        if relative_pc_npy_file_path:
            data = ['00000000000',instruction,rgb_image_base64,rgb_image_width,rgb_image_height,relative_pc_npy_file_path]
        else:
            data = ['00000000000',instruction,rgb_image_base64,rgb_image_width,rgb_image_height]
        # write data_list to tsv file
        with open(tsv_file_path,'w',newline='') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')
            writer.writerow(data)
    
    def get_instruction_position_for_single_object(self, i, j, img_width=224, img_height=224):
        section_width = img_width // 3
        section_height = img_height // 3
        
        row = i // section_height
        col = j // section_width

        text = "Grasp the object in the "
        if row == 0:
            if col == 0:
                text += "top left"
            elif col == 1:
                text += "top middle"
            elif col == 2:
                text += "top right"
        elif row == 1:
            if col == 0:
                text += "middle left"
            elif col == 1:
                text += "center"
            elif col == 2:
                text += "middle right"
        elif row == 2:
            if col == 0:
                text += "bottom left"
            elif col == 1:
                text += "bottom middle"
            elif col == 2:
                text += "bottom right"
        text += " section of this image."
        return text

    def get_instruction_position_for_multiobject(self, i):
        text = "Grasp the object in the "
        if i == 0:
            text += "top left"
        elif i ==1:
            text += "top right"
        elif i ==2:
            text += "bottom left"
        elif i ==3:
            text += "bottom right"
        text += " section of this image."
        return text

    def get_instruction_angle(self,Grasps,encoded=False):
        instruction_angle = {}
        for i in range(len(Grasps)):
            grasp = Grasps[i]
            if encoded:
                a_encoded = grasp.a_encoded
                text = f'Grasp the object with a rotation angle of a_{a_encoded} radians.'
            else:
                _,_,_,_,a = grasp.get_data
                text = f'Grasp the object with a rotation angle of a_{a} radians.'
            instruction_angle[i] = text
        return instruction_angle


    def generate_grasp_tlbra_tsv(self,rgb_Image,TLBRAGrasps,tsv_file_path,relative_pc_npy_file_path=None,tlbra='tlbra',encoded=False,decimals=None):
        import csv
        # get rgb image base64
        if isinstance(rgb_Image, image.Image):
            rgb_image_base64 = self.image_numpy_to_base64(rgb_Image.img)
            (rgb_image_height,rgb_image_width,_) = rgb_Image.img.shape
        if isinstance(rgb_Image, str):
            rgb_image_base64 = rgb_Image
            (rgb_image_height,rgb_image_width) = (224,224)
            
        data_list = []

        # for every tlbra_grasp
        for tlbra_grasp in TLBRAGrasps:
            # get data and text
            text = ''
            if 'tl' in tlbra:
                if encoded:
                    tl_encoded = tlbra_grasp.tl_encoded
                    text += f'Grasp top-left point coordinates: tl_{tl_encoded} '
                else:
                    tl = tlbra_grasp.tl
                    text += f'Grasp top-left point coordinates: tlx_{tl[0]},tly_{tl[1]} '
            if 'br' in tlbra:
                if encoded:
                    br_encoded = tlbra_grasp.br_encoded
                    text += f'Grasp bottom-right point coordinates: br_{br_encoded} '
                else:
                    br = tlbra_grasp.br
                    text += f'Grasp bottom-right point coordinates: brx_{br[0]},bry_{br[1]} '
            if 'a' in tlbra:
                if encoded:
                    a_encoded = tlbra_grasp.a_encoded
                    text += f'Grasp rotation angle in radians: a_{a_encoded} '
                else:
                    a = tlbra_grasp.angle
                    if decimals is not None:
                        a = round(a,decimals)
                    text += f'Grasp rotation angle in radians: a_{a} '
            
            # delete all blankspace at the end
            text = text.rstrip()
            # get data
            if relative_pc_npy_file_path:
                data = ['00000000000',text,rgb_image_base64,rgb_image_width,rgb_image_height,relative_pc_npy_file_path]
            else:
                data = ['00000000000',text,rgb_image_base64,rgb_image_width,rgb_image_height]
            data_list.append(data)
            # print(f'text:\n{text}\nrgb_image_width:\n{rgb_image_width}\nrgb_image_height:\n{rgb_image_height}\nrelative_pc_npy_file_path:{relative_pc_npy_file_path}')
        
        # write data_list to tsv file
        with open(tsv_file_path,'w',newline='') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')
            for data in data_list:
                writer.writerow(data)
    
    def write_to_txt_file(self,txt_file_path,text):
        with open(txt_file_path, 'w') as txt_file:
            txt_file.write(text)
    
    def read_txt_file(self,txt_file_path):
        with open(txt_file_path, "r") as txt_file:
            lines = txt_file.readlines()
        lines = [line.strip() for line in lines]
        return lines

    def dataset_augment(self,new_dataset_path,output_size=224,idx_start=0,idx_end=885,rotate_num=3,rotate_range=360,translate_num=3,translate_range=40,vis=False,gen_pc=False,instructions_file_name='instructions_temp_1'):
        import os
        import numpy as np
        import json

        image_num = 0
        grasp_sum = 0
        instructions = {}
        
        for idx in range(idx_start,idx_end):
            # get num1 and num2
            num_1 = os.path.basename(self.grasp_rectangle_files[idx])[3:5]  # pcd0100cpos.txt --> 01
            num_2 = os.path.basename(self.grasp_rectangle_files[idx])[5:7]  # pcd0100cpos.txt --> 00
            
            # creat dataset subfolder
            dataset_subfolder = new_dataset_path+'/'+num_1+f'/pcd{num_1}{num_2}/'
            if not os.path.exists(dataset_subfolder):
                os.makedirs(dataset_subfolder)
            
            if gen_pc:
                # create point cloud in npy file and save
                self.generate_pc_npy(idx,dataset_subfolder+f'pcd{num_1}{num_2}pc.npy')
                self.generate_pc_pcd(idx,dataset_subfolder+f'pcd{num_1}{num_2}pc.pcd')
            
            # orignal image(640*480) to 351*351
            rgb_Image = self.get_rgb_Image(idx=idx,normalise=False)
            grasp_Rectangles = self.get_grasp_Rectangles(idx=idx)
            self.show_rgb_Image_and_grasp_Rectangles(rgb_Image,grasp_Rectangles,vis)
            
            for i in range(rotate_num): 
                # rotate image and grasp
                angle = np.deg2rad(np.random.randint(0, rotate_range))
                rgb_Image_rotated,grasp_Rectangles_rotated = self.rotate_rgb_Image_and_grasp_Rectangles(rgb_Image,grasp_Rectangles,angle,center=[self.output_size//2,self.output_size//2])#[175,175]
                self.show_rgb_Image_and_grasp_Rectangles(rgb_Image_rotated,grasp_Rectangles_rotated,vis)
                
                for j in range(translate_num):
                    # get the file number: pcd0100_00_00
                    file_number = f'pcd{num_1}{num_2}'+'_{:02d}_{:02d}'.format(i, j)
                    
                    # translate image and grasp
                    x_shift,y_shift = (np.random.randint(-translate_range, translate_range+1),np.random.randint(-translate_range, translate_range+1))
                    rgb_Image_translated,grasp_Rectangles_translated = self.translate_rgb_Image_and_grasp_Rectangles(rgb_Image_rotated,grasp_Rectangles_rotated,x_shift,y_shift)
                    self.show_rgb_Image_and_grasp_Rectangles(rgb_Image_translated,grasp_Rectangles_translated,vis)
                    
                    # crop image and grasp(351*351-->224*224)
                    rgb_Image_cropped,grasp_Rectangles_cropped = self.crop_rgb_Image_and_grasp_Rectangles(rgb_Image_translated,grasp_Rectangles_translated,original_size=self.output_size,cropped_size=output_size)
                    
                    # save image
                    rgb_Image_cropped.save(dataset_subfolder+file_number+'r.png')

                    # save image with grasp
                    self.show_rgb_Image_and_grasp_Rectangles(rgb_Image_cropped,grasp_Rectangles_cropped,vis,save_path=dataset_subfolder+file_number+'rgrasp.png')
                    
                    # save image with grasp and obj_name
                    # self.add_obj_name_to_img(rgb_Image_cropped,grasp_Rectangles_cropped,obj_name=self.obj_names[idx],vis=vis,save_path=dataset_subfolder+file_number+'rgrasp_obj.png')

                    # save grasp_rectangles
                    grasp_Rectangles_cropped.save_txt(dataset_subfolder+file_number+'cpos.txt')
                    
                    # save grasps
                    Grasps = grasp_Rectangles_cropped.as_grasps
                    Grasps.save_txt(dataset_subfolder+file_number+'grasp_xywha.txt')
                    
                    # # save tlbra_grasps
                    # TLBRAGrasps = grasp_Rectangles_cropped.as_tlbra_grasps
                    # TLBRAGrasps.save_txt(dataset_subfolder+file_number+'grasp_tlbra.txt')

                    # # save tlbra_grasps_encoded
                    # TLBRAGrasps.save_txt_encoded(dataset_subfolder+file_number+'grasp_tlbra_encoded.txt')
                    
                    # relative_pc_npy_file_path
                    if gen_pc:
                        relative_pc_npy_file_path = f'/{num_1}/pcd{num_1}{num_2}/pcd{num_1}{num_2}pc.npy'
                    else:
                        relative_pc_npy_file_path = None
                        
                    # get rgb image base64
                    rgb_image_base64 = self.image_numpy_to_base64(rgb_Image_cropped.img)
                    
                    # instruction
                    # name
                    instruction_name = self.instructions_name[idx]
                    self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_encoded_with_instruction_name.tsv',relative_pc_npy_file_path,'xya',encoded=True,decimals=3,instruction=instruction_name)
                    # shape
                    instruction_shape = self.instructions_shape[idx]
                    self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_encoded_with_instruction_shape.tsv',relative_pc_npy_file_path,'xya',encoded=True,decimals=3,instruction=instruction_shape)
                    # purpose
                    instruction_purpose = self.instructions_purpose[idx]
                    self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_encoded_with_instruction_purpose.tsv',relative_pc_npy_file_path,'xya',encoded=True,decimals=3,instruction=instruction_purpose)
                    # strategy
                    instruction_strategy = self.instructions_strategy[idx]
                    self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_encoded_with_instruction_strategy.tsv',relative_pc_npy_file_path,'xya',encoded=True,decimals=3,instruction=instruction_strategy)
                    # color
                    instruction_color = self.instructions_color[idx]
                    self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_encoded_with_instruction_color.tsv',relative_pc_npy_file_path,'xya',encoded=True,decimals=3,instruction=instruction_color)
                    # part
                    instruction_part = self.instructions_part[idx]
                    self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_encoded_with_instruction_part.tsv',relative_pc_npy_file_path,'xya',encoded=True,decimals=3,instruction=instruction_part)
                    # position
                    instruction_position = self.get_instruction_position_for_single_object(grasp_Rectangles_cropped.center[0],grasp_Rectangles_cropped.center[1],224,224)
                    self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_encoded_with_instruction_position.tsv',relative_pc_npy_file_path,'xya',encoded=True,decimals=3,instruction=instruction_position)
                    # angle
                    instruction_angle = self.get_instruction_angle(Grasps,encoded=True)
                    self.generate_grasp_xywha_tsv_with_instruction_angle(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_encoded_with_instruction_angle.tsv',relative_pc_npy_file_path,'xya',encoded=True,decimals=3)

                    # save pcdxxxx_xx_xxgrasp_xywha/xywa/xyha/xya/xya_encoded/xy/w/h/a/a_encoded.tsv （without instruction)
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xywha.tsv',relative_pc_npy_file_path,'xywha',encoded=False,decimals=3)
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xywa.tsv',relative_pc_npy_file_path,'xywa',encoded=False,decimals=3)
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xyha.tsv',relative_pc_npy_file_path,'xyha',encoded=False,decimals=3)
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya.tsv',relative_pc_npy_file_path,'xya',encoded=False,decimals=3)
                    self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_encoded.tsv',relative_pc_npy_file_path,'xya',encoded=True,decimals=3)
                    self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xy.tsv',relative_pc_npy_file_path,'xy',encoded=False,decimals=3)
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_w.tsv',relative_pc_npy_file_path,'w',encoded=False,decimals=3)
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_h.tsv',relative_pc_npy_file_path,'h',encoded=False,decimals=3)
                    self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_a.tsv',relative_pc_npy_file_path,'a',encoded=False,decimals=3)
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_a_encoded.tsv',relative_pc_npy_file_path,'a',encoded=True,decimals=3)
                    
                    # save instructions_name/instructions_strategy
                    # self.generate_instruction_tsv(rgb_image_base64,instruction_name,dataset_subfolder+file_number+'grasp_instruction_name.tsv',relative_pc_npy_file_path)
                    # self.generate_instruction_tsv(rgb_image_base64,instruction_color,dataset_subfolder+file_number+'grasp_instruction_color.tsv',relative_pc_npy_file_path)
                    # self.generate_instruction_tsv(rgb_image_base64,instruction_shape,dataset_subfolder+file_number+'grasp_instruction_shape.tsv',relative_pc_npy_file_path)
                    # self.generate_instruction_tsv(rgb_image_base64,instruction_purpose,dataset_subfolder+file_number+'grasp_instruction_purpose.tsv',relative_pc_npy_file_path)
                    # self.generate_instruction_tsv(rgb_image_base64,instruction_position,dataset_subfolder+file_number+'grasp_instruction_position.tsv',relative_pc_npy_file_path)
                    # self.generate_instruction_tsv(rgb_image_base64,instruction_strategy,dataset_subfolder+file_number+'grasp_instruction_strategy.tsv',relative_pc_npy_file_path)


                    instruction = {"obj_name":self.obj_names[idx],
                            "instruction_name":instruction_name,
                            "instruction_shape":instruction_shape,
                            "instruction_purpose":instruction_purpose,
                            "instruction_position":instruction_position,
                            "instruction_strategy":instruction_strategy,
                            "instruction_color":instruction_color,
                            "instruction_part":instruction_part,
                            "instruction_position":instruction_position,
                            "instruction_angle":instruction_angle
                    }
                    instructions[image_num] = instruction

                    # save pcdxxxx_xx_xxgrasp_xywha/xywa/xyha/xya/xya_encoded/xy/w/h/a/a_encoded.tsv (with instruction_name)
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xywha_with_instruction_name.tsv',relative_pc_npy_file_path,'xywha',encoded=False,decimals=3,instruction=self.instructions_name[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xywa_with_instruction_name.tsv',relative_pc_npy_file_path,'xywa',encoded=False,decimals=3,instruction=self.instructions_name[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xyha_with_instruction_name.tsv',relative_pc_npy_file_path,'xyha',encoded=False,decimals=3,instruction=self.instructions_name[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_with_instruction_name.tsv',relative_pc_npy_file_path,'xya',encoded=False,decimals=3,instruction=self.instructions_name[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_encoded_with_instruction_name.tsv',relative_pc_npy_file_path,'xya',encoded=True,decimals=3,instruction=self.instructions_name[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xy_with_instruction_name.tsv',relative_pc_npy_file_path,'xy',encoded=False,decimals=3,instruction=self.instructions_name[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_w_with_instruction_name.tsv',relative_pc_npy_file_path,'w',encoded=False,decimals=3,instruction=self.instructions_name[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_h_with_instruction_name.tsv',relative_pc_npy_file_path,'h',encoded=False,decimals=3,instruction=self.instructions_name[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_a_with_instruction_name.tsv',relative_pc_npy_file_path,'a',encoded=False,decimals=3,instruction=self.instructions_name[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_a_encoded_with_instruction_name.tsv',relative_pc_npy_file_path,'a',encoded=True,decimals=3,instruction=self.instructions_name[idx])
                    
                    # save pcdxxxx_xx_xxgrasp_xywha/xywa/xyha/xya/xya_encoded/xy/w/h/a/a_encoded.tsv (with instruction_name)
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xywha_with_instruction_strategy.tsv',relative_pc_npy_file_path,'xywha',encoded=False,decimals=3,instruction=self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xywa_with_instruction_strategy.tsv',relative_pc_npy_file_path,'xywa',encoded=False,decimals=3,instruction=self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xyha_with_instruction_strategy.tsv',relative_pc_npy_file_path,'xyha',encoded=False,decimals=3,instruction=self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_with_instruction_strategy.tsv',relative_pc_npy_file_path,'xya',encoded=False,decimals=3,instruction=self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_encoded_with_instruction_strategy.tsv',relative_pc_npy_file_path,'xya',encoded=True,decimals=3,instruction=self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xy_with_instruction_strategy.tsv',relative_pc_npy_file_path,'xy',encoded=False,decimals=3,instruction=self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_w_with_instruction_strategy.tsv',relative_pc_npy_file_path,'w',encoded=False,decimals=3,instruction=self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_h_with_instruction_strategy.tsv',relative_pc_npy_file_path,'h',encoded=False,decimals=3,instruction=self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_a_with_instruction_strategy.tsv',relative_pc_npy_file_path,'a',encoded=False,decimals=3,instruction=self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_a_encoded_with_instruction_strategy.tsv',relative_pc_npy_file_path,'a',encoded=True,decimals=3,instruction=self.instructions_strategy[idx])
                    
                    # save pcdxxxx_xx_xxgrasp_xywha/xywa/xyha/xya/xya_encoded/xy/w/h/a/a_encoded.tsv (with instruction_name and instruction_strategy)
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xywha_with_instruction_name_and_strategy.tsv',relative_pc_npy_file_path,'xywha',encoded=False,decimals=3,instruction=self.instructions_name[idx]+' '+self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xywa_with_instruction_name_and_strategy.tsv',relative_pc_npy_file_path,'xywa',encoded=False,decimals=3,instruction=self.instructions_name[idx]+' '+self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xyha_with_instruction_name_and_strategy.tsv',relative_pc_npy_file_path,'xyha',encoded=False,decimals=3,instruction=self.instructions_name[idx]+' '+self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_with_instruction_name_and_strategy.tsv',relative_pc_npy_file_path,'xya',encoded=False,decimals=3,instruction=self.instructions_name[idx]+' '+self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xya_encoded_with_instruction_name_and_strategy.tsv',relative_pc_npy_file_path,'xya',encoded=True,decimals=3,instruction=self.instructions_name[idx]+' '+self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_xy_with_instruction_name_and_strategy.tsv',relative_pc_npy_file_path,'xy',encoded=False,decimals=3,instruction=self.instructions_name[idx]+' '+self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_w_with_instruction_name_and_strategy.tsv',relative_pc_npy_file_path,'w',encoded=False,decimals=3,instruction=self.instructions_name[idx]+' '+self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_h_with_instruction_name_and_strategy.tsv',relative_pc_npy_file_path,'h',encoded=False,decimals=3,instruction=self.instructions_name[idx]+' '+self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_a_with_instruction_name_and_strategy.tsv',relative_pc_npy_file_path,'a',encoded=False,decimals=3,instruction=self.instructions_name[idx]+' '+self.instructions_strategy[idx])
                    # self.generate_grasp_xywha_tsv(rgb_image_base64,Grasps,dataset_subfolder+file_number+'grasp_a_encoded_with_instruction_name_and_strategy.tsv',relative_pc_npy_file_path,'a',encoded=True,decimals=3,instruction=self.instructions_name[idx]+' '+self.instructions_strategy[idx])
                    
                    # # save pcdxxxx_xx_xxgrasp_tlbra/tlbra_encoded/tl/br/a/tl_encoded/br_encoded/a_encoded.tsv
                    # self.generate_grasp_tlbra_tsv(rgb_image_base64,TLBRAGrasps,dataset_subfolder+file_number+'grasp_tlbra.tsv',relative_pc_npy_file_path,'tlbra',encoded=False,decimals=3)
                    # self.generate_grasp_tlbra_tsv(rgb_image_base64,TLBRAGrasps,dataset_subfolder+file_number+'grasp_tlbra_encoded.tsv',relative_pc_npy_file_path,'tlbra',encoded=True,decimals=3)
                    # self.generate_grasp_tlbra_tsv(rgb_image_base64,TLBRAGrasps,dataset_subfolder+file_number+'grasp_tl.tsv',relative_pc_npy_file_path,'tl',encoded=False,decimals=3)
                    # self.generate_grasp_tlbra_tsv(rgb_image_base64,TLBRAGrasps,dataset_subfolder+file_number+'grasp_br.tsv',relative_pc_npy_file_path,'br',encoded=False,decimals=3)
                    # self.generate_grasp_tlbra_tsv(rgb_image_base64,TLBRAGrasps,dataset_subfolder+file_number+'grasp_a.tsv',relative_pc_npy_file_path,'a',encoded=False,decimals=3)
                    # self.generate_grasp_tlbra_tsv(rgb_image_base64,TLBRAGrasps,dataset_subfolder+file_number+'grasp_tl_encoded.tsv',relative_pc_npy_file_path,'tl',encoded=True,decimals=3)
                    # self.generate_grasp_tlbra_tsv(rgb_image_base64,TLBRAGrasps,dataset_subfolder+file_number+'grasp_br_encoded.tsv',relative_pc_npy_file_path,'br',encoded=True,decimals=3)
                    # self.generate_grasp_tlbra_tsv(rgb_image_base64,TLBRAGrasps,dataset_subfolder+file_number+'grasp_a_encoded.tsv',relative_pc_npy_file_path,'a',encoded=True,decimals=3)
                    
                    # update the image_num and grasp_sum
                    image_num += 1
                    grasp_sum += len(grasp_Rectangles_cropped)
                    
                    # print(f'====== [pcd{num_1}{num_2}_{i}_{j},image_num_{image_num},grasp_sum_{grasp_sum}] is generated successfully!!! ======')
                
                # print(f'====== [pcd{num_1}{num_2}_{i}_**,image_num_{image_num},grasp_sum_{grasp_sum}] is generated successfully!!! ======')
            
            print(f'====== [pcd{num_1}{num_2}_**_**,image_num_{image_num},grasp_sum_{grasp_sum}] is generated successfully!!! ======')


        new_instructions_file_dir = new_dataset_path+'/else/'
        if not os.path.exists(new_instructions_file_dir):
            os.makedirs(new_instructions_file_dir)
        with open(new_instructions_file_dir+f'{instructions_file_name}.json', 'w') as json_file:
            json.dump(instructions, json_file,indent=4)

    def merge_and_resize_imgs(self,img1_path, img2_path, img3_path, img4_path, vis=False, save_path=None,overlap_width=30):
        import cv2
        import numpy as np
        def blend_images(image1, image2, axis, overlap_width):  
            if axis == 0:
                alpha = np.linspace(0, 1, overlap_width).reshape(-1, 1, 1)  
                alpha = np.repeat(alpha, image1.shape[1], axis=1) 
            else: 
                alpha = np.linspace(0, 1, overlap_width).reshape(1, -1, 1)  
                alpha = np.repeat(alpha, image1.shape[0], axis=0)
            beta = 1 - alpha  
            overlap1 = image1[-overlap_width:] if axis == 0 else image1[:, -overlap_width:]  
            overlap2 = image2[:overlap_width] if axis == 0 else image2[:, :overlap_width]  
            blended_overlap = overlap1 * beta + overlap2 * alpha  
            if axis == 0:  
                result = np.vstack((image1[:-overlap_width], blended_overlap, image2[overlap_width:]))  
            else:  
                result = np.hstack((image1[:, :-overlap_width], blended_overlap, image2[:, overlap_width:]))  
            return result.astype(np.uint8)
        
        top_left = cv2.imread(img1_path)  
        top_right = cv2.imread(img2_path)  
        bottom_left = cv2.imread(img3_path)  
        bottom_right = cv2.imread(img4_path)  
        top_blend = blend_images(top_left, top_right, axis=1, overlap_width=overlap_width)  
        bottom_blend = blend_images(bottom_left, bottom_right, axis=1, overlap_width=overlap_width)  
        final_blend = blend_images(top_blend, bottom_blend, axis=0, overlap_width=overlap_width)  
        # resize
        final_blend = cv2.resize(final_blend, (224, 224))
        if vis:
            cv2.imshow('Blended Image', final_blend) 
            cv2.waitKey(0)  
            cv2.destroyAllWindows()  
        if save_path:
            cv2.imwrite(save_path, final_blend)
        return final_blend

    def map_coordinates(self,point,img_index):
        x = point[0]
        y = point[1]
        if img_index == 0:
            return [int(x//2),int(y//2)]
        elif img_index == 1:
            return [int(x//2)+112,int(y//2)]
        elif img_index == 2:
            return [int(x//2),int(y//2)+112]
        elif img_index == 3:
            return [int(x//2)+112,int(y//2)+112]
    
    def map_grasp_Rectangle(self,grasp_Rectangle,img_index):
        point1 = self.map_coordinates(grasp_Rectangle.points[0],img_index)
        point2 = self.map_coordinates(grasp_Rectangle.points[1],img_index)
        point3 = self.map_coordinates(grasp_Rectangle.points[2],img_index)
        point4 = self.map_coordinates(grasp_Rectangle.points[3],img_index)
        new_grasp_Rectangle = grasp.GraspRectangle([point1,point2,point3,point4])
        return new_grasp_Rectangle

    def map_grasp_Rectangles(self,grasp_Rectangles,img_index,save_path=None):
        import os
        new_grasp_Rectangles = grasp.GraspRectangles()
        for grasp_Rectangle in grasp_Rectangles:
            new_grasp_Rectangle = self.map_grasp_Rectangle(grasp_Rectangle,img_index)
            new_grasp_Rectangles.append(new_grasp_Rectangle)
        if save_path:
            folder_path = os.path.dirname(save_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            new_grasp_Rectangles.save_txt(save_path)
        return new_grasp_Rectangles
    
    def map_Grasp(self,Grasp,img_index):
        new_center = self.map_coordinates(Grasp.center,img_index)
        new_width = Grasp.width / 2
        new_height = Grasp.height / 2
        new_angle = Grasp.angle
        new_grasp = grasp.Grasp(new_center,new_width,new_height,new_angle)
        return new_grasp
    
    def map_Grasps(self,Grasps,img_index,save_path=None):
        import os
        # from utils.dataset_processing import grasp, image
        new_Grasps = grasp.Grasps()
        for Grasp in Grasps:
            new_Grasp = self.map_Grasp(Grasp,img_index)
            new_Grasps.append(new_Grasp)
        if save_path:
            folder_path = os.path.dirname(save_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            new_Grasps.save_txt(save_path)
        return new_Grasps

    def dataset_multiobject(self,new_dataset_path,idx_start=0,idx_end=14160,instructions_file_name='instructions_temp_1',read_idxs_from_file=False,overlap_width=15):
        import numpy as np
        import cv2
        import json
        import os

        if read_idxs_from_file:
            idxs_random_txt_path = f'{self.dataset_root}/else/idxs_random.txt'
            with open(idxs_random_txt_path, 'r') as txt_file:
                lines = txt_file.readlines()
                idxs = [int(line.strip()) for line in lines]
        else:
            # shuffle the idxs randomly
            idxs = list(range(self.image_sum))
            np.random.seed(123)
            np.random.shuffle(idxs)
        print(f'idxs size: {len(idxs)}')

        instructions = {}
        # every four idxs
        flag = 0
        idxs = idxs[idx_start:min(self.image_sum,idx_end)]
        for i in range(0, len(idxs), 4):
            # merge four images
            idx1,idx2,idx3,idx4 = idxs[i:i+4]
            merged_img = self.merge_and_resize_imgs(self.rgb_files[idx1], self.rgb_files[idx2], self.rgb_files[idx3], self.rgb_files[idx4],overlap_width=overlap_width)
            # rgb_image_base64 = self.image_numpy_to_base64(merged_img)

            for j in range(4):
                idx = idxs[i:i+4][j]
                img_path = self.rgb_files[idx]
                new_img_path = img_path.replace(self.dataset_root,new_dataset_path)
                folder_path = os.path.dirname(new_img_path) 
                if not os.path.exists(folder_path): 
                    os.makedirs(folder_path) 
                Grasps_path = self.grasp_xywha_files[idx]
                new_Grasps_path = Grasps_path.replace(self.dataset_root,new_dataset_path)
                # while not os.path.exists(new_img_path):
                cv2.imwrite(new_img_path, merged_img)
                rgb_Image = image.Image.from_file(new_img_path)
                rgb_image_base64 = self.image_numpy_to_base64(rgb_Image.img)
                Grasps = grasp.Grasps.load_from_cornell_file(Grasps_path)
                new_Grasps = self.map_Grasps(Grasps,img_index=j,save_path=new_Grasps_path)
                self.show_rgb_Image_and_Grasps(rgb_Image,new_Grasps,vis=False,save_path=new_img_path.replace('r.png','rgrasp.png'))
                
                # instruction
                # name
                instruction_name = self.instructions_name[idx]
                self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_name.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_name)
                # shape
                instruction_shape = self.instructions_shape[idx]
                self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_shape.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_shape)
                # purpose
                instruction_purpose = self.instructions_purpose[idx]
                self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_purpose.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_purpose)
                # strategy
                instruction_strategy = self.instructions_strategy[idx]
                self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_strategy.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_strategy)
                # color
                instruction_color = self.instructions_color[idx]
                self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_color.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_color)
                # part
                instruction_part = self.instructions_part[idx]
                self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_part.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_part)
                # position
                instruction_position = self.get_instruction_position_for_multiobject(j)
                self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_position.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_position)
                # angle
                instruction_angle = self.instructions_angle[idx]
                self.generate_grasp_xywha_tsv_with_instruction_angle(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_angle.tsv'),None,'xya',encoded=True,decimals=3)

                instruction = {"obj_name":self.obj_names[idx],
                            "instruction_name":instruction_name,
                            "instruction_shape":instruction_shape,
                            "instruction_purpose":instruction_purpose,
                            "instruction_position":instruction_position,
                            "instruction_strategy":instruction_strategy,
                            "instruction_color":instruction_color,
                            "instruction_part":instruction_part,
                            "instruction_position":instruction_position,
                            "instruction_angle":instruction_angle
                    }
                instructions[idx] = instruction
                
                flag += 1
                print(f'====== [progress]:{flag}/{self.image_sum} ======')
                
        new_instructions_file_dir = new_dataset_path+'/else/'
        if not os.path.exists(new_instructions_file_dir):
            os.makedirs(new_instructions_file_dir)
        with open(new_instructions_file_dir+f'{instructions_file_name}.json', 'w') as json_file:
            json.dump(instructions, json_file,indent=4)
    
    def idx_original_to_augmented(self,idx,rotate_num=4,translate_num=4):
        return list(range(idx*(rotate_num*translate_num),idx*(rotate_num*translate_num)+15))
    
    def idx_augmented_to_original(self,idx,rotate_num=4,translate_num=4):
        return idx//(rotate_num*translate_num)

    def new_dataset_multiobject(self,new_dataset_path,idx_start=0,idx_end=14160,overlap_width=15,rotate_num=4,translate_num=4,instructions_file_name='instructions',group_json_file_path=None):
        import cv2
        import json
        import random
        import os

        instructions = {}
        idxs = list(range(0,self.image_sum))
        idxs = idxs[idx_start:min(self.image_sum,idx_end)]

        # get groups
        with open(group_json_file_path,'r') as json_file:
            groups = json.loads(json_file.read())
            groups = list(groups.values())
        # print(f'groups:{groups}')
        for idx in idxs:
            # get idxs_set(4 idxs)
            idx_original = self.idx_augmented_to_original(idx,rotate_num,translate_num)
            # print(f'idx_original:{idx_original}')
            for group in groups:
                # print(f'group["item_idxs"]:{group["item_idxs"]}')
                if idx_original in group["item_idxs"]: #  10 in [1,2,30,20]
                    item_group = group["item_group"]   #  [black,white]
                    # get item_group_idxs
                    item_group_idxs_original = []
                    item_group_idxs = []
                    for i in item_group:
                        for group in groups:
                            if i == group["item"]:
                                item_group_idxs_original.extend(group["item_idxs"])
                                break
                    for j in item_group_idxs_original:
                        item_group_idxs.extend(self.idx_original_to_augmented(j,rotate_num,translate_num))
                    break
            item_group_idxs = list(set(item_group_idxs))
            # print(f"item_group_idxs:{item_group_idxs}")
            print(f"len item_group_idxs:{len(item_group_idxs)}")
            idxs_set = random.sample(item_group_idxs, 3)
            flag = random.randint(0, 3)
            idxs_set.insert(flag, idx)
            print(f'idxs_set:{idxs_set}')

            for k in range(len(idxs_set)):
                if idxs_set[k]<0 or idxs_set[k]>=self.image_sum:
                    idxs_set[k] = random.randint(0, self.image_sum-1)
                    print(f'idxs_set:{idxs_set}')
            
            # merge four images
            merged_img = self.merge_and_resize_imgs(self.rgb_files[idxs_set[0]], self.rgb_files[idxs_set[1]], self.rgb_files[idxs_set[2]], self.rgb_files[idxs_set[3]],overlap_width=overlap_width)
            
            img_path = self.rgb_files[idx]
            new_img_path = img_path.replace(self.dataset_root,new_dataset_path)
            folder_path = os.path.dirname(new_img_path) 
            if not os.path.exists(folder_path): 
                os.makedirs(folder_path) 
            Grasps_path = self.grasp_xywha_files[idx]
            new_Grasps_path = Grasps_path.replace(self.dataset_root,new_dataset_path)
            # save merged_img
            cv2.imwrite(new_img_path, merged_img)
            rgb_Image = image.Image.from_file(new_img_path)
            rgb_image_base64 = self.image_numpy_to_base64(rgb_Image.img)
            # save new_Grasps
            Grasps = grasp.Grasps.load_from_cornell_file(Grasps_path)
            new_Grasps = self.map_Grasps(Grasps,img_index=flag,save_path=new_Grasps_path)
            # save merged_img with new_Grasps
            self.show_rgb_Image_and_Grasps(rgb_Image,new_Grasps,vis=False,save_path=new_img_path.replace('r.png','rgrasp.png'))

            # instruction
            # name
            instruction_name = self.instructions_name[idx]
            self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_name.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_name)
            # color
            instruction_color = self.instructions_color[idx]
            self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_color.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_color)
            # shape
            instruction_shape = self.instructions_shape[idx]
            self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_shape.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_shape)
            # purpose
            instruction_purpose = self.instructions_purpose[idx]
            self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_purpose.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_purpose)
            # position
            instruction_position = self.get_instruction_position_for_multiobject(flag)
            self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_position.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_position)
            # strategy
            instruction_strategy = self.instructions_strategy[idx]
            self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_strategy.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_strategy)
            # angle
            instruction_angle = self.instructions_angle[idx]
            self.generate_grasp_xywha_tsv_with_instruction_angle(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_angle.tsv'),None,'xya',encoded=True,decimals=3)
            # part
            instruction_part = self.instructions_part[idx]
            self.generate_grasp_xywha_tsv(rgb_image_base64,new_Grasps,new_Grasps_path.replace('grasp_xywha.txt','grasp_xya_encoded_with_instruction_part.tsv'),None,'xya',encoded=True,decimals=3,instruction=instruction_part)
             
            instruction = {"obj_name":self.obj_names[idx],
                        "instruction_name":instruction_name,
                        "instruction_shape":instruction_shape,
                        "instruction_purpose":instruction_purpose,
                        "instruction_position":instruction_position,
                        "instruction_strategy":instruction_strategy,
                        "instruction_color":instruction_color,
                        "instruction_part":instruction_part,
                        "instruction_position":instruction_position,
                        "instruction_angle":instruction_angle
                }
            instructions[idx] = instruction
            print(f'====== [progress]:{idx}/{self.image_sum} ======')

        new_instructions_file_dir = new_dataset_path+'/else/'
        if not os.path.exists(new_instructions_file_dir):
            os.makedirs(new_instructions_file_dir)
        with open(new_instructions_file_dir+f'{instructions_file_name}.json', 'w') as json_file:
            json.dump(instructions, json_file,indent=4)
    
    # ====== for generate dataloader ======
    def generate_datalaoder_idxs(self,split,eval_format='IW'):
        import numpy as np
        
        # get all idxs
        idxs = list(range(self.image_sum))
        
        # normalize the split
        sum = split[0]+split[1]+split[2]
        split[0] = split[0]/sum
        split[1] = split[1]/sum
        split[2] = split[2]/sum
        
        # shuffle the idxs if IW
        if eval_format == 'IW':
            np.random.seed(123)
            np.random.shuffle(idxs)
        # get the training, validation, and test idxs
        train_idxs = idxs[:int(np.floor(split[0] * self.image_sum))]
        valid_idxs = idxs[int(np.floor(split[0] * self.image_sum)):int(np.floor((split[0]+split[1]) * self.image_sum))]
        test_idxs = idxs[int(np.floor((split[0]+split[1]) * self.image_sum)):]

        # print
        print(f'====== train_idxs size: {len(train_idxs)}')
        print(f'====== valid_idxs size: {len(valid_idxs)}')
        print(f'====== test_idxs size: {len(test_idxs)}')
        # print(f'====== test_idxs: {test_idxs}')
        
        return train_idxs,valid_idxs,test_idxs

    def generate_dataloader_json(self,idxs,dataloader_num,grasp_format='xywha',splited=False,encoded=False,type='train'): # 'xywha' or 'xywa' or 'xyha' or 'xya' or 'split'
        import json
        import os
        
        # add '_encoded' suffix if encoded
        encoded_suffix = '_encoded' if encoded else ''
        
        # get data source
        data_source = []
        for idx in idxs:
            if splited:
                if 'xy' in grasp_format:
                    # data_source.append(self.grasp_tsv_files[idx].replace('grasp.tsv',f'grasp_xy{encoded_suffix}.tsv').replace(self.dataset_root,'../../../'))
                    data_source.append(self.grasp_tsv_files[idx].replace('grasp.tsv',f'grasp_xy.tsv').replace(self.dataset_root,'../../../'))
                if 'w' in grasp_format:
                    data_source.append(self.grasp_tsv_files[idx].replace('grasp.tsv',f'grasp_w{encoded_suffix}.tsv').replace(self.dataset_root,'../../../'))
                if 'h' in grasp_format:
                    data_source.append(self.grasp_tsv_files[idx].replace('grasp.tsv',f'grasp_h{encoded_suffix}.tsv').replace(self.dataset_root,'../../../'))
                if 'tl' in grasp_format:
                    data_source.append(self.grasp_tsv_files[idx].replace('grasp.tsv',f'grasp_tl{encoded_suffix}.tsv').replace(self.dataset_root,'../../../'))
                if 'br' in grasp_format:
                    data_source.append(self.grasp_tsv_files[idx].replace('grasp.tsv',f'grasp_br{encoded_suffix}.tsv').replace(self.dataset_root,'../../../'))
                if 'a' in grasp_format:
                    data_source.append(self.grasp_tsv_files[idx].replace('grasp.tsv',f'grasp_a{encoded_suffix}.tsv').replace(self.dataset_root,'../../../'))
            else:
                data_source.append(self.grasp_tsv_files[idx].replace('grasp.tsv',f'grasp_{grasp_format}{encoded_suffix}.tsv').replace(self.dataset_root,'../../../'))
        
        # get data
        data = [{
            "source": data_source,
            "source_lang": "cornell",
            "weight": 1.0,
            "name": "cornell"
        }]
        
        # create json folder
        json_folder = f'{self.dataset_root}/dataloader/{dataloader_num}/dataloader_config/json/'
        if not os.path.exists(json_folder):
            os.makedirs(json_folder)
        
        # save dataloader json
        with open(f'{json_folder}/{type}.json','w') as json_file:
            json.dump(data,json_file,indent=4)
        
        # save dataloader idxs
        with open(f'{json_folder}/{type}.txt','w') as txt_file:
            for idx in idxs:
                txt_file.write(str(idx) + '\n')

    def generate_dataloader_jsons(self,train_idxs,valid_idxs,test_idxs,dataloader_num,grasp_format='xywha',splited=False,encoded=False):
        # train;valid;test
        self.generate_dataloader_json(train_idxs,dataloader_num,grasp_format,splited,encoded,type='train')
        self.generate_dataloader_json(valid_idxs,dataloader_num,grasp_format,splited,encoded,type='valid')
        self.generate_dataloader_json(test_idxs,dataloader_num,grasp_format,splited,encoded,type='test')

    def generate_dataloader_single(self,split,eval_format='IW',dataloader_num='01',grasp_format='xywha',splited=False,encoded=False):
        train_idxs,valid_idxs,test_idxs = self.generate_datalaoder_idxs(split,eval_format)
        self.generate_dataloader_jsons(train_idxs,valid_idxs,test_idxs,dataloader_num,grasp_format,splited,encoded)
    
    def generate_dataloader_five_fold_cross_validation(self,eval_format='IW',dataloader_num_list=['01','02','03','04','05'],grasp_format='xywha',splited=False,encoded=False):
        import numpy as np
        import json

        # get all idxs
        idxs = list(range(self.image_sum))
        idxs_sets = []
        
        if eval_format == 'IW':
            if not dataloader_num_list:
                dataloader_num_list = ['01','02','03','04','05']
            # shuffle the idxs if IW
            np.random.seed(123)
            np.random.shuffle(idxs)
            split = [0.2, 0.4, 0.6, 0.8]
            idxs_sets.append(idxs[:int(np.floor(split[0] * self.image_sum))])
            for i in range(len(split)-1):
                idxs_sets.append(idxs[int(np.floor(split[i] * self.image_sum)):int(np.floor(split[i+1] * self.image_sum))])
            idxs_sets.append(idxs[int(np.floor(split[-1] * self.image_sum)):])

        elif eval_format == 'OW':
            # with open(self.dataset_root+'/else/name_groups.json','r') as json_file:
            #     json_data = json.load(json_file)
            # for i in range(5):
            #     idxs = []
            #     names = json_data[str(i)]["obj_names"]
            #     for idx in range(len(self.obj_names)):
            #         if self.obj_names[idx] in names:
            #             idxs.append(idx)
            #     idxs_sets.append(idxs)
       
            if not dataloader_num_list:
                dataloader_num_list = ['06','07','08','09','10']
            split = [0.2, 0.4, 0.6, 0.8]
            idxs_sets.append(idxs[:int(np.floor(split[0] * self.image_sum))])
            for i in range(len(split)-1):
                idxs_sets.append(idxs[int(np.floor(split[i] * self.image_sum)):int(np.floor(split[i+1] * self.image_sum))])
            idxs_sets.append(idxs[int(np.floor(split[-1] * self.image_sum)):])
        
        # generate train_idxs and test_idxs(valid_idxs is empty list)
        for i in range(5):
            train_idxs = [idx for j in range(5) if j != i for idx in idxs_sets[j]]
            valid_idxs = []
            test_idxs  = idxs_sets[i]
            self.generate_dataloader_jsons(train_idxs,valid_idxs,test_idxs,dataloader_num_list[i],grasp_format,splited,encoded)

    def generate_dataloader(self,if_five_fold_cross_validation,split,eval_format='IW',dataloader_num='01',dataloader_num_list=['01','02','03','04','05'],grasp_format='xywha',splited=False,encoded=False):
        if if_five_fold_cross_validation:
            self.generate_dataloader_five_fold_cross_validation(eval_format,dataloader_num_list,grasp_format,splited,encoded)
        else:
            self.generate_dataloader_single(split,eval_format,dataloader_num,grasp_format,splited,encoded)
    
    # ====== for evaluation ======
    def generate_predicted_Grasps(self,idx,responses,grasp_format='xywha',encoded=False,suffix=None,save_dir=None):
        import re
        import os
        import numpy as np
        # for grasp_format:'xywha'
        if 'xy' in grasp_format:
            predicted_Grasps = grasp.Grasps()
            Grasps_real = grasp.Grasps.load_from_cornell_file(self.grasp_xywha_files[idx])
            
            # process response to get predicted value
            for response in responses:
                if 'xy' in grasp_format:
                    try:
                        x_predicted = int(re.findall(r'x_(\d+)', response)[0])
                        y_predicted = int(re.findall(r'y_(\d+)', response)[0])
                        if x_predicted>=224 or x_predicted<=0:
                            x_predicted = Grasps_real.get_mean_x()
                        if y_predicted>=224 or y_predicted<=0:
                            y_predicted = Grasps_real.get_mean_y()
                    except:
                        x_predicted = Grasps_real.get_mean_x()
                        y_predicted = Grasps_real.get_mean_y()
                else:
                    x_predicted = Grasps_real.get_mean_x()
                    y_predicted = Grasps_real.get_mean_y()
                if 'w' in grasp_format:
                    try:
                        w_predicted = float(re.findall(r'w_([\d.]+)', response)[0])
                        if w_predicted>=224 or w_predicted<=0:
                            w_predicted = Grasps_real.get_mean_w()
                    except:
                        w_predicted = Grasps_real.get_max_w()
                else:
                    w_predicted = Grasps_real.get_max_w()
                if 'h' in grasp_format:
                    try:
                        h_predicted = float(re.findall(r'h_([\d.]+)', response)[0])
                        if h_predicted>=224 or h_predicted<=0:
                            h_predicted = Grasps_real.get_mean_h()
                    except:
                        h_predicted = Grasps_real.get_max_h()
                else:
                    h_predicted = Grasps_real.get_max_h()
                if 'a' in grasp_format:
                    if encoded:
                        try:
                            a_predicted_encoded = int(re.findall(r'a_(\d+)', response)[0])
                            a_predicted = grasp.TLBRAGrasp.decode_value(a_predicted_encoded)
                            if a_predicted_encoded>=256 or a_predicted_encoded<=0:
                                a_predicted = Grasps_real.get_mean_a()
                        except:
                            a_predicted = Grasps_real.get_mean_a()
                    else:
                        try:
                            a_predicted = float(re.findall(r'a_(-?[\d.]+)', response)[0])
                            if a_predicted>=np.pi/2 or a_predicted<=-np.pi/2:
                                a_predicted = Grasps_real.get_mean_a()
                        except:
                            a_predicted = Grasps_real.get_mean_a()
                else:
                    a_predicted = Grasps_real.get_mean_a()
                
                # generate predicted_Grasps
                predicted_Grasps.append(grasp.Grasp([x_predicted,y_predicted],w_predicted,h_predicted,a_predicted))
            
            # save the predicted
            if save_dir:
                predicted_Grasps.save_txt(save_dir+os.path.basename(self.predicted_files[idx].replace('predicted.txt',f'predicted_xywha{suffix}.txt')))
            else:
                predicted_Grasps.save_txt(self.predicted_files[idx].replace('predicted.txt',f'predicted_xywha{suffix}.txt'))

        # for grasp_format:'tlbra'
        else:
            predicted_TLBRAGrasps = grasp.TLBRAGrasps()
            TLBRAGrasps_real = grasp.TLBRAGrasps.load_from_cornell_file(self.grasp_tlbra_files[idx])
            
            # process response to get predicted value
            for response in responses:
                if 'tl' in grasp_format:
                    if encoded:
                        try:
                            tl_predicted_encoded = int(re.findall(r'tl_(\d+)', response)[0])
                            tl_predicted = grasp.TLBRAGrasp.decode_point(tl_predicted_encoded)
                            if tl_predicted_encoded>=1024 or tl_predicted_encoded<=0:
                                tl_predicted = TLBRAGrasps_real.get_mean_tl()
                        except:
                            tl_predicted = TLBRAGrasps_real.get_mean_tl()
                    else:
                        try:
                            tl_predicted_x = int(re.findall(r'tlx_(\d+)', response)[0])
                            tl_predicted_y = int(re.findall(r'tly_(\d+)', response)[0])
                            tl_predicted = [tl_predicted_x,tl_predicted_y]
                            if tl_predicted_x>=224 or tl_predicted_x<=0 or tl_predicted_y>=224 or tl_predicted_y<=0:
                                tl_predicted = TLBRAGrasps_real.get_mean_tl()
                        except:
                            tl_predicted = TLBRAGrasps_real.get_mean_tl()
                else:
                    tl_predicted = TLBRAGrasps_real.get_mean_tl()
                if 'br' in grasp_format:
                    if encoded:
                        try:
                            br_predicted_encoded = int(re.findall(r'br_(\d+)', response)[0])
                            br_predicted = grasp.TLBRAGrasp.decode_point(br_predicted_encoded)
                            if br_predicted_encoded>=1024 or br_predicted_encoded<=0:
                                br_predicted = TLBRAGrasps_real.get_mean_br()
                        except:
                            br_predicted = TLBRAGrasps_real.get_mean_br()
                    else:
                        try:
                            br_predicted_x = int(re.findall(r'brx_(\d+)', response)[0])
                            br_predicted_y = int(re.findall(r'bry_(\d+)', response)[0])
                            br_predicted = [br_predicted_x,br_predicted_y]
                            if br_predicted_x>=224 or br_predicted_x<=0 or br_predicted_y>=224 or br_predicted_y<=0:
                                br_predicted = TLBRAGrasps_real.get_mean_br()
                        except:
                            br_predicted = TLBRAGrasps_real.get_mean_br()
                else:
                    br_predicted = TLBRAGrasps_real.get_mean_br()
                if 'a' in grasp_format:
                    if encoded:
                        try:
                            a_predicted_encoded = int(re.findall(r'a_(\d+)', response)[0])
                            a_predicted = grasp.TLBRAGrasp.decode_value(a_predicted_encoded)
                            if a_predicted_encoded>=256 or a_predicted_encoded<=0:
                                a_predicted = TLBRAGrasps_real.get_mean_a()
                        except:
                            a_predicted = TLBRAGrasps_real.get_mean_a()
                    else:
                        try:
                            a_predicted = float(re.findall(r'a_(-?[\d.]+)', response)[0])
                            if a_predicted>=np.pi/2 or a_predicted<=-np.pi/2:
                                a_predicted = TLBRAGrasps_real.get_mean_a()
                        except:
                            a_predicted = TLBRAGrasps_real.get_mean_a()
                else:
                    a_predicted = TLBRAGrasps_real.get_mean_a()
                
                # generate predicted_TLBRAGrasps
                predicted_TLBRAGrasps.append(grasp.TLBRAGrasp(tl_predicted,br_predicted,a_predicted))
            
            # TLBRAGrasps to Grasps
            predicted_Grasps = predicted_TLBRAGrasps.as_grasps
            
            # save the predicted    
            if save_dir:
                predicted_TLBRAGrasps.save_txt(save_dir+os.path.basename(self.predicted_files[idx].replace('predicted.txt',f'predicted_{grasp_format}{suffix}.txt')))
                predicted_Grasps.save_txt(save_dir+os.path.basename(self.predicted_files[idx].replace('predicted.txt',f'predicted_xywha{suffix}.txt')))
            else:
                predicted_TLBRAGrasps.save_txt(self.predicted_files[idx].replace('predicted.txt',f'predicted_{grasp_format}{suffix}.txt'))
                predicted_Grasps.save_txt(self.predicted_files[idx].replace('predicted.txt',f'predicted_xywha{suffix}.txt'))
            
        print(f'[idx:{idx}] predicted_Grasps is generated successfully!!!')
        return predicted_Grasps
    
    def eval_Grasps_iou(self,predicted_Grasps,real_Grasps,iou_threshold=0.25,save_results_json_path=None,save_predicted_image_path=None,vis=False,rgb_Image=None):
        import json
        
        # init the evaluation results
        accuracy = {'correct': 0, 'failed': 0, 'accuracy':0.0}
        accuracy_for_every_grasp = {'correct': 0, 'failed': 0, 'accuracy':0.0}
        all_metrics = {"first":0,"second":0,"both":0,"none":0}
        max_iou_list = []
        all_iou_list = []
        all_metrics_info_list = []
        
        # evaluate the predicted_Grasps
        for i in range(len(predicted_Grasps)):
            predicted_Grasp = predicted_Grasps[i]
            iou = predicted_Grasp.iou(real_Grasps[i])
            max_iou = predicted_Grasp.max_iou(real_Grasps)
            metrics = predicted_Grasp.get_metrics(real_Grasps,iou_threshold)
            all_iou = predicted_Grasp.get_all_iou(real_Grasps)
            all_metrics_info = predicted_Grasp.get_all_metrics_info(real_Grasps)
            if iou > iou_threshold:
                accuracy_for_every_grasp['correct'] += 1
            else:
                accuracy_for_every_grasp['failed'] += 1
            if max_iou > iou_threshold:
                accuracy['correct'] += 1
            else:
                accuracy['failed'] += 1
            all_metrics.update(metrics)
            max_iou_list.append(max_iou)
            all_iou_list.append(all_iou)    
            all_metrics_info_list.append(all_metrics_info)
        
        # grasp accuracy
        accuracy['accuracy'] = accuracy['correct'] / (accuracy['correct'] + accuracy['failed']) * 100 if accuracy['correct'] + accuracy['failed'] > 0 else 0.0
        accuracy_for_every_grasp['accuracy'] = accuracy_for_every_grasp['correct'] / (accuracy_for_every_grasp['correct'] + accuracy_for_every_grasp['failed']) * 100 if accuracy_for_every_grasp['correct'] + accuracy_for_every_grasp['failed'] > 0 else 0.0
        
        # print the evaluation results
        print(f'Accuracy for every grasp: %d/%d = %.2f%%' % (accuracy_for_every_grasp['correct'], accuracy_for_every_grasp['correct'] + accuracy_for_every_grasp['failed'], accuracy_for_every_grasp['accuracy']))
        print(f'Accuracy: %d/%d = %.2f%%' % (accuracy['correct'], accuracy['correct'] + accuracy['failed'], accuracy['accuracy']))
        print(f'List of max_iou: {max_iou_list}')
        print(f'List of all_iou: {all_iou_list}')
        print(f'all metrics: {all_metrics}')
        # print(f'List of all_metrics_info: {all_metrics_info_list}')
        
        # save result to the json file
        if save_results_json_path:
            with open(save_results_json_path,'w') as result_file:
                results = {'accuracy':accuracy,"accuracy_for_every_grasp":accuracy_for_every_grasp,"all_metrics":all_metrics,'max_iou_list':max_iou_list,"all_iou_list":all_iou_list,"all_metrics_info_list":all_metrics_info_list,'predicted_Grasps':predicted_Grasps.to_list,'real_Grasps':real_Grasps.to_list}
                json.dump(results,result_file,indent=4)
                print(f'Save eval results successfully')
        
        # save predicted grasp image
        if save_predicted_image_path:
            self.show_rgb_Image_and_predicted_Grasps_and_real_Grasps(rgb_Image,predicted_Grasps,real_Grasps,vis,save_predicted_image_path)
        return accuracy,accuracy_for_every_grasp,max_iou_list,all_iou_list,all_metrics
    
    def eval_image_iou(self,idx,predicted_Grasps=None,iou_threshold=0.25,grasp_format='xywha',suffix=None,vis=False,save_dir=None):
        import os
        
        # predicted.txt
        if not predicted_Grasps:
            predicted_Grasps = grasp.Grasps.load_from_cornell_file(self.predicted_files[idx].replace('predicted.txt',f'predicted_xywha{suffix}.txt'))
        
        # result.json and rpredicted.png
        if save_dir:
            save_results_json_path = save_dir+os.path.basename(self.result_files[idx].replace('result.json',f'result_{grasp_format}{suffix}.json'))
            save_predicted_image_path = save_dir+os.path.basename(self.rgb_predicted_files[idx].replace('rpredicted.png',f'rpredicted_{grasp_format}{suffix}.png'))
        else:
            save_results_json_path = self.result_files[idx].replace('result.json',f'result_{grasp_format}{suffix}.json')
            save_predicted_image_path = self.rgb_predicted_files[idx].replace('rpredicted.png',f'rpredicted_{grasp_format}{suffix}.png')
        
        # get the real grasps and the rgb_image for evaluation
        real_Grasps = grasp.Grasps.load_from_cornell_file(self.grasp_xywha_files[idx])
        rgb_Image=image.Image.from_file(self.rgb_files[idx])
        
        # evaluate
        accuracy,accuracy_for_every_grasp,max_iou_list,all_iou_list,all_metrics = self.eval_Grasps_iou(predicted_Grasps,
                                                                real_Grasps,
                                                                iou_threshold,
                                                                save_results_json_path,
                                                                save_predicted_image_path,
                                                                vis,
                                                                rgb_Image)
        return accuracy,accuracy_for_every_grasp,max_iou_list,all_iou_list,all_metrics
    
    def eval_images_iou(self,idxs,generate_predictions_func,iou_threshold=0.25,grasp_format='xywha',splited=False,encoded=False,suffix=None,vis=False,save_dir=None,instruction_type=None):
        import json
        import os
        
        # create save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # init the evaluation results
        final_accuracy = {'correct': 0, 'failed': 0, 'accuracy':0.0}
        final_accuracy_for_every_grasp = {'correct': 0, 'failed': 0, 'accuracy':0.0}
        final_max_iou_list = []
        final_all_metrics = {"first":0,"second":0,"both":0,"none":0}
        idx_num = 0
        final_accuracy_for_every_obj = {}
        final_accuracy_for_every_obj_every_grasp = {}
        obj_names_unique = list(set(self.obj_names))
        for obj_name in obj_names_unique:
            final_accuracy_for_every_obj[obj_name] = {'correct': 0, 'failed': 0, 'accuracy':0.0}
            final_accuracy_for_every_obj_every_grasp[obj_name] = {'correct': 0, 'failed': 0, 'accuracy':0.0}
        print(f'self.obj_names:{self.obj_names}')
        print(f'final_accuracy_for_every_obj:{final_accuracy_for_every_obj}')
        
        for idx in idxs:
            idx_num+=1
            print(f'\n====== [{idx_num}/{len(idxs)}]: idx_{idx}')
            # img_input
            img_input = self.rgb_files[idx]
            # get the input text and the corresponding responses
            responses = []
            # for splited
            if splited:
                response = ''
                if 'xy' in grasp_format:
                    text_input_xy = 'Grasp center point coordinates: x_'
                    _,_,_,_,_,response_xy = generate_predictions_func(img_input,text_input_xy,None)
                    response = response + text_input_xy + response_xy + ' '
                if 'w' in grasp_format:
                    text_input_w = 'Grasp width: w_'
                    _,_,_,_,_,response_w = generate_predictions_func(img_input,text_input_w,None)
                    response = response + text_input_w + response_w + ' '
                if 'h' in grasp_format:
                    text_input_h = 'Grasp height: h_'
                    _,_,_,_,_,response_h = generate_predictions_func(img_input,text_input_h,None)
                    response = response + text_input_h + response_h + ' '
                if 'tl' in grasp_format:
                    if encoded:
                        text_input_tl = 'Grasp top-left point coordinates: tl_'
                    else:
                        text_input_tl = 'Grasp top-left point coordinates: tlx_'
                    _,_,_,_,_,response_tl = generate_predictions_func(img_input,text_input_tl,None)
                    response = response + text_input_tl + response_tl + ' '
                if 'br' in grasp_format:
                    if encoded:
                        text_input_br = 'Grasp top-left point coordinates: br_'
                    else:
                        text_input_br = 'Grasp bottom-right point coordinates: brx_'
                    _,_,_,_,_,response_br = generate_predictions_func(img_input,text_input_br,None)
                    response = response + text_input_br + response_br + ' '
                if 'a' in grasp_format:
                    text_input_a = 'Grasp rotation angle in radians: a_'
                    _,_,_,_,_,response_a = generate_predictions_func(img_input,text_input_a,None)
                    response = response + text_input_a + response_a + ' '
                responses.append(response)
            # for not splited
            else:
                if 'xy' in grasp_format:
                    if instruction_type:
                        if instruction_type == 'angle' or instruction_type == 'part':
                            instructions = self.instructions[str(idx)][f'instruction_{instruction_type}']
                            for i in range(len(instructions)):
                                text_input = instructions[str(i)]
                                print(f'text_input:{text_input}')
                                _,_,_,_,_,response = generate_predictions_func(img_input,text_input,None)
                                response = text_input+response
                                responses.append(response)
                        else:
                            text_input = self.instructions[str(idx)][f'instruction_{instruction_type}']
                            text_input += ' Grasp center point coordinates: x_'
                            print(f'text_input:{text_input}')
                            _,_,_,_,_,response = generate_predictions_func(img_input,text_input,None)
                            response = text_input+response
                            responses.append(response)
                    else:
                        text_input = 'Grasp center point coordinates: x_'
                        _,_,_,_,_,response = generate_predictions_func(img_input,text_input,None)
                        response = text_input+response
                        responses.append(response)
                if 'tl' in grasp_format:
                    if encoded:
                        text_input = 'Grasp top-left point coordinates: tl_'
                    else:
                        text_input = 'Grasp top-left point coordinates: tlx_'
                    _,_,_,_,_,response = generate_predictions_func(img_input,text_input,None)
                    response = text_input+response
                    responses.append(response)
            
            print(f'responses:{responses}')
            # responses = ["Grasp the object with a rotation angle of a_71 radians. Grasp center point coordinates: x_10,y_20","Grasp the object with a rotation angle of a_90 radians. Grasp center point coordinates: x_40,y_50"]

            # generate predicted_files
            predicted_Grasps = self.generate_predicted_Grasps(idx,responses,grasp_format,encoded,suffix,save_dir)
            # print(f'predicted_Grasps:\n{predicted_Grasps[0].center,predicted_Grasps[0].angle}')
            
            # eval the predicted_files,generate rgb_predicted_files and result_files
            results,results_for_every_grasp,max_iou_list,all_iou_list,all_metrics = self.eval_image_iou(idx,predicted_Grasps,iou_threshold,grasp_format,suffix,vis,save_dir)

            # update the evaluation results
            final_accuracy['correct'] += results['correct']
            final_accuracy['failed'] += results['failed']
            final_accuracy_for_every_grasp['correct'] += results_for_every_grasp['correct']
            final_accuracy_for_every_grasp['failed'] += results_for_every_grasp['failed']
            final_accuracy_for_every_obj[self.instructions[str(idx)]["obj_name"]]['correct'] += results['correct']
            final_accuracy_for_every_obj[self.instructions[str(idx)]["obj_name"]]['failed'] += results['failed']
            final_accuracy_for_every_obj_every_grasp[self.instructions[str(idx)]["obj_name"]]['correct'] += results_for_every_grasp['correct']
            final_accuracy_for_every_obj_every_grasp[self.instructions[str(idx)]["obj_name"]]['failed'] += results_for_every_grasp['failed']
            
            final_max_iou_list.append(max_iou_list)
            final_all_metrics["first"]+=all_metrics["first"]
            final_all_metrics["second"]+=all_metrics["second"]
            final_all_metrics["both"]+=all_metrics["both"]
            final_all_metrics["none"]+=all_metrics["none"]
            
            final_accuracy['accuracy'] = final_accuracy['correct'] / (final_accuracy['correct'] + final_accuracy['failed']) * 100 if final_accuracy['correct'] + final_accuracy['failed'] > 0 else 0.0
            final_accuracy_for_every_grasp['accuracy'] = final_accuracy_for_every_grasp['correct'] / (final_accuracy_for_every_grasp['correct'] + final_accuracy_for_every_grasp['failed']) * 100 if final_accuracy_for_every_grasp['correct'] + final_accuracy_for_every_grasp['failed'] > 0 else 0.0
            print(f'@@@@@@ Final Accuracy: %d/%d = %.2f%%' % (final_accuracy['correct'], final_accuracy['correct'] + final_accuracy['failed'], final_accuracy['accuracy']))
            print(f'@@@@@@ Final Accuracy for every grasp: %d/%d = %.2f%%' % (final_accuracy_for_every_grasp['correct'], final_accuracy_for_every_grasp['correct'] + final_accuracy_for_every_grasp['failed'], final_accuracy_for_every_grasp['accuracy']))
            
        # final grasp accuracy 
        final_accuracy['accuracy'] = final_accuracy['correct'] / (final_accuracy['correct'] + final_accuracy['failed']) * 100 if final_accuracy['correct'] + final_accuracy['failed'] > 0 else 0.0
        final_accuracy_for_every_grasp['accuracy'] = final_accuracy_for_every_grasp['correct'] / (final_accuracy_for_every_grasp['correct'] + final_accuracy_for_every_grasp['failed']) * 100 if final_accuracy_for_every_grasp['correct'] + final_accuracy_for_every_grasp['failed'] > 0 else 0.0
        for obj_name in obj_names_unique:
            final_accuracy_for_every_obj[obj_name]['accuracy'] = final_accuracy_for_every_obj[obj_name]['correct'] / (final_accuracy_for_every_obj[obj_name]['correct'] + final_accuracy_for_every_obj[obj_name]['failed']) * 100 if final_accuracy_for_every_obj[obj_name]['correct'] + final_accuracy_for_every_obj[obj_name]['failed'] > 0 else 0.0
            final_accuracy_for_every_obj_every_grasp[obj_name]['accuracy'] = final_accuracy_for_every_obj_every_grasp[obj_name]['correct'] / (final_accuracy_for_every_obj_every_grasp[obj_name]['correct'] + final_accuracy_for_every_obj_every_grasp[obj_name]['failed']) * 100 if final_accuracy_for_every_obj_every_grasp[obj_name]['correct'] + final_accuracy_for_every_obj_every_grasp[obj_name]['failed'] > 0 else 0.0

        # print(f'final_all_metrics:{final_all_metrics}')
        print(f'@@@@@@ Final Accuracy: %d/%d = %.2f%%' % (final_accuracy['correct'], final_accuracy['correct'] + final_accuracy['failed'], final_accuracy['accuracy']))
        print(f'@@@@@@ Final Accuracy for every grasp: %d/%d = %.2f%%' % (final_accuracy_for_every_grasp['correct'], final_accuracy_for_every_grasp['correct'] + final_accuracy_for_every_grasp['failed'], final_accuracy_for_every_grasp['accuracy']))
        # print(f'final_max_iou_list:\n{final_max_iou_list}')
        
        # save evaluation results
        with open(f'{save_dir}/final_accuracy.json', 'w') as file:
            json.dump(final_accuracy, file, indent=4)
        with open(f'{save_dir}/final_accuracy_for_every_obj.json', 'w') as file:
            json.dump(final_accuracy_for_every_obj, file, indent=4)
        with open(f'{save_dir}/final_accuracy_for_every_grasp.json', 'w') as file:
            json.dump(final_accuracy_for_every_grasp, file, indent=4)
        with open(f'{save_dir}/final_accuracy_for_every_obj_every_grasp.json', 'w') as file:
            json.dump(final_accuracy_for_every_obj_every_grasp, file, indent=4)
        with open(f'{save_dir}/final_max_iou_list.txt', 'w') as file:
            list_str = '\n'.join(str(item) for item in final_max_iou_list)
            file.write(list_str)
        with open(f'{save_dir}/final_all_metrics.json', 'w') as file:
            json.dump(final_all_metrics, file, indent=4)
        return final_accuracy,final_accuracy_for_every_obj,final_accuracy_for_every_grasp,final_accuracy_for_every_obj_every_grasp,final_max_iou_list,final_all_metrics
    
    def evaluate(self,dataloader_num,generate_predictions_func,iou_threshold=0.25,grasp_format='xywha',splited=False,encoded=False,suffix=None,vis=False,save_dir=None,instruction_type=None):
        # get the test idxs
        test_idxs_txt_path = f'{self.dataset_root}/dataloader/{dataloader_num}/dataloader_config/json/test.txt'
        with open(test_idxs_txt_path, 'r') as txt_file:
            lines = txt_file.readlines()
            test_idxs = [int(line.strip()) for line in lines]
        print(f'test idxs size: {len(test_idxs)}')
        print(f'test idxs: {test_idxs}')
        # test_idxs = [0]
        
        # evaluate the test dataset
        final_accuracy,final_accuracy_for_every_obj,final_accuracy_for_every_grasp,final_accuracy_for_every_obj_every_grasp,final_max_iou_list,final_all_metrics = self.eval_images_iou(test_idxs,generate_predictions_func,iou_threshold,grasp_format,splited,encoded,suffix,vis,save_dir,instruction_type)
       

    # ======= for instructions
    def GPT4V(self,prompt,img_path,temp=0.7,top_p=0.95,max_tokens=800,response_num=1):
        import requests
        import base64
        GPT4V_KEY = "0e7ec4ffe8e44e4c8a14d6d052d69f6d"
        headers = {
            "Content-Type": "application/json",
            "api-key": GPT4V_KEY,
        }
        base64image = base64.b64encode(open(img_path, 'rb').read()).decode('ascii')
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/webp;base64,{base64image}"
                        },
                        },
                    ],
                }
            ],
            "temperature": temp,
            "top_p": top_p,
            "max_tokens": max_tokens,
            'n':response_num
        }
        GPT4V_ENDPOINT = "https://gpt4nlc1.openai.azure.com/openai/deployments/gpt4v/extensions/chat/completions?api-version=2023-07-01-preview"
        try:
            response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status() 
        except requests.RequestException as e:
            print(e)
        final_response = response.json()['choices'][0]['message']['content']
        # print(final_response)
        return final_response
    
    def get_obj_name(self,idx):
        import json
        with open('utils/data/language_templates_site.json','r') as json_file:
            data = json.load(json_file)
            all_obj_names_list = list(data["reasoning_tuning_templates"].keys())
            all_obj_names_str = ', '.join(all_obj_names_list)
        img_path = self.rgb_files[idx]
        prompt = f"""You are now an object recognition robot, and your task is to identify the names of the objects in the picture as accurately as possible. All the object names are as follows: {all_obj_names_str}. Let's identify the object on the white table in the picture given to you. Give me the name of the object directly. The name of the object should be one of the names of all the objects above.(Object names are all lowercase)"""
        # print(prompt)
        obj_name = self.GPT4V(prompt,img_path)
        print(f'idx_{idx}, obj_name: {obj_name}')
        return obj_name

    def get_obj_description(self,idx,vis=False,save_path=None):
        rgb_Image = image.Image.from_file(self.rgb_files[idx])
        grasp_Rectangles = grasp.GraspRectangles.load_from_cornell_file(self.grasp_rectangle_files[idx])
        obj_name = self.get_obj_name(idx)
        # TODO: save the obj name
        if not save_path:
            save_path = self.rgb_files[idx].replace("r.png","rgrasp_obj.png")
        self.add_obj_name_to_img(rgb_Image, grasp_Rectangles, obj_name, vis, save_path)
    
    def get_objs_description(self,idxs):
        for idx in idxs:
            self.get_obj_description(idx,vis=False)
            
    def generate_image_with_grasp(self,idxs):
        for idx in idxs:
            rgb_Image = image.Image.from_file(self.rgb_files[idx])      
            grasp_Rectangles = grasp.GraspRectangles.load_from_cornell_file(self.grasp_rectangle_files[idx])
            self.show_rgb_Image_and_grasp_Rectangles(rgb_Image,grasp_Rectangles,vis=False,save_path=self.rgb_files[idx].replace("r.png","rgrasp.png"))

    def get_idxs_from_dataloader_json(self,json_dir,grasp_format='xywha',splited=False,encoded=False): # 'xywha' or 'xywa' or 'xyha' or 'xya' or 'split'
        import json
        encoded_suffix = '_encoded' if encoded else ''
        with open(f'{json_dir}/test.json','r') as json_file:
            json_data = json.load(json_file)
        data_source = json_data[0]["source"]
        if splited:
            idxs = []
            for source in data_source:
                if 'xy' in grasp_format:
                    if source.replace('../../../',self.dataset_root).replace(f'grasp_xy.tsv','grasp.tsv') not in self.grasp_tsv_files:
                        continue
                    idx = self.grasp_tsv_files.index(source.replace('../../../',self.dataset_root).replace(f'grasp_xy.tsv','grasp.tsv'))
                    idxs.append(idx)
                    continue
                elif 'w' in grasp_format:
                    if source.replace('../../../',self.dataset_root).replace(f'grasp_w{encoded_suffix}.tsv','grasp.tsv') not in self.grasp_tsv_files:
                        continue
                    idx = self.grasp_tsv_files.index(source.replace('../../../',self.dataset_root).replace(f'grasp_w{encoded_suffix}.tsv','grasp.tsv'))
                    idxs.append(idx)
                    continue
                elif 'h' in grasp_format:
                    if source.replace('../../../',self.dataset_root).replace(f'grasp_h{encoded_suffix}.tsv','grasp.tsv') not in self.grasp_tsv_files:
                        continue
                    idx = self.grasp_tsv_files.index(source.replace('../../../',self.dataset_root).replace(f'grasp_h{encoded_suffix}.tsv','grasp.tsv'))
                    idxs.append(idx)
                    continue
                elif 'a' in grasp_format:
                    if source.replace('../../../',self.dataset_root).replace(f'grasp_a{encoded_suffix}.tsv','grasp.tsv') not in self.grasp_tsv_files:
                        continue
                    idx = self.grasp_tsv_files.index(source.replace('../../../',self.dataset_root).replace(f'grasp_a{encoded_suffix}.tsv','grasp.tsv'))
                    idxs.append(idx)
                    continue
        else:
            idxs = [self.grasp_tsv_files.index(source.replace('../../../',self.dataset_root).replace(f'grasp_{grasp_format}{encoded_suffix}.tsv','grasp.tsv')) for source in data_source]
        # detele the same element
        idxs = list(set(idxs))
        
        print(f'idxs size: {len(idxs)}')
        print(f'idxs: {idxs}')
        
        # save dataloader idxs
        with open(f'{json_dir}/test.txt','w') as txt_file:
            for idx in idxs:
                txt_file.write(str(idx) + '\n')
                
        return idxs

    # ====== for test
    @classmethod
    def test_plot_for_paper(self):
        import json
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/01/eval_savedir/pcd0823_00_00result_xya-05-01.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v12/08/pcd0823/pcd0823_00_00r.png'
        # name = 'angle0'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/01/eval_savedir/pcd0823_00_00result_xya-05-01.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v12/08/pcd0823/pcd0823_00_00r.png'
        # name = 'angle1'
        # grasp_num = 1
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/01/eval_savedir/pcd0823_00_00result_xya-05-01.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v12/08/pcd0823/pcd0823_00_00r.png'
        # name = 'angle2'
        # grasp_num = 2
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/01/eval_savedir/pcd0823_00_00result_xya-05-01.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v12/08/pcd0823/pcd0823_00_00r.png'
        # name = 'angle3'
        # grasp_num = 3
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/01/eval_savedir/pcd0823_00_00result_xya-05-01.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v12/08/pcd0823/pcd0823_00_00r.png'
        # name = 'angle4'
        # grasp_num = 4
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path =  '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/01/eval_savedir/pcd0187_01_00result_xya-05-01.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v12/01/pcd0187/pcd0187_01_00r.png'
        # name = 'part0'
        # grasp_num = 4
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path =  '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/01/eval_savedir/pcd0187_01_00result_xya-05-01.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v12/01/pcd0187/pcd0187_01_00r.png'
        # name = 'part1'
        # grasp_num = 1
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/11/eval_savedir/pcd0147_03_02result_xya-05-11.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v33/01/pcd0147/pcd0147_03_02r.png'
        # name = 'name0'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/11/eval_savedir/pcd0103_00_00result_xya-05-11.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v33/01/pcd0103/pcd0103_00_00r.png'
        # name = 'name1'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/11/eval_savedir/pcd0113_01_01result_xya-05-11.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v33/01/pcd0113/pcd0113_01_01r.png'
        # name = 'name2'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/12/eval_savedir/pcd0127_00_01result_xya-05-12.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v33/01/pcd0127/pcd0127_00_01r.png'
        # name = 'shape0'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/12/eval_savedir/pcd0896_00_01result_xya-05-12.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v33/08/pcd0896/pcd0896_00_01r.png'
        # name = 'shape1'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/13/eval_savedir/pcd0320_01_01result_xya-05-13.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v33/03/pcd0320/pcd0320_01_01r.png'
        # name = 'purpose0'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/13/eval_savedir/pcd0111_03_02result_xya-05-13.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v33/01/pcd0111/pcd0111_03_02r.png'
        # name = 'purpose1'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/14/eval_savedir/pcd0123_00_01result_xya-05-14.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v33/01/pcd0123/pcd0123_00_01r.png'
        # name = 'position0'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/14/eval_savedir/pcd0168_03_03result_xya-05-14.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v33/01/pcd0168/pcd0168_03_03r.png'
        # name = 'position1'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/15/eval_savedir/pcd0326_03_00result_xya-05-15.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v33/03/pcd0326/pcd0326_03_00r.png'
        # name = 'strategy0'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/15/eval_savedir/pcd0718_00_01result_xya-05-15.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v33/07/pcd0718/pcd0718_00_01r.png'
        # name = 'strategy1'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/15/eval_savedir/pcd0325_01_02result_xya-05-15.json'
        image_path = '/mnt/msranlpintern/dataset/cornell-v33/03/pcd0325/pcd0325_01_02r.png'
        name = 'strategy2'
        grasp_num = 0
        instruction = "It is usually round and flat. Grasping its edge provides a stable hold. Gripper rotation can be orthogonal to its curvature edge."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/16/eval_savedir/pcd1034_00_03result_xya-05-16.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v34/10/pcd1034/pcd1034_00_03r.png'
        # name = 'color0'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # # ======
        # result_xya_json_file_path = '/mnt/msranlpintern/dataset/cornell-instruction-v1/dataloader/05/train_output/16/eval_savedir/pcd0867_00_00result_xya-05-16.json'
        # image_path = '/mnt/msranlpintern/dataset/cornell-v34/08/pcd0867/pcd0867_00_00r.png'
        # name = 'color1'
        # grasp_num = 0
        # instruction = "Grasp the green object."
        # ======
        
        
        save_dir = '/home/shaohanhuang/kosmos-e/kosmos-e-cornell/result_images/'
        rgb_Image = image.Image.from_file(image_path)
        rgb_Image.save(save_dir+name+'_original.png')
        with open(result_xya_json_file_path,'r') as json_file:
            json_data = json.loads(json_file.read())
        predicted_Grasps = grasp.Grasps.load_from_list(json_data["predicted_Grasps"])
        predicted_Grasps =  grasp.Grasps.load_from_grasp(predicted_Grasps[grasp_num])
        real_Grasps = grasp.Grasps.load_from_list(json_data["real_Grasps"])
        
        self.show_rgb_Image_and_Grasps(rgb_Image,real_Grasps,vis=False,save_path=save_dir+name+'_grasp_real.png',line_weight=2,color='blue')
        self.show_rgb_Image_and_Grasps(rgb_Image,predicted_Grasps,vis=False,save_path=save_dir+name+'_grasp_predicted.png',line_weight=3,color='red')
        self.show_rgb_Image_and_predicted_Grasps_and_real_Grasps(rgb_Image,predicted_Grasps,real_Grasps,vis=False,save_path=save_dir+name+'_grasp_real_and_predicted.png',real_line_weight=2,predicted_line_weight=3)
        
        with open(save_dir+name+'_instruction.txt','w') as txt_file:
            txt_file.write(instruction)
        
        predicted_text = f'Grasp center point coordinates: ({predicted_Grasps[0].center[0]},{predicted_Grasps[0].center[1]}) Grasp rotation angle in radians: {predicted_Grasps[0].angle}'
        with open(save_dir+name+'_predicted_text.txt','w') as txt_file:
            txt_file.write(predicted_text)
        print(f'=== save to {save_dir}')