'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-02-06 16:16:42
Version: v1
File: 
Brief: 
'''
import argparse
from utils.data import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset augement')

    parser.add_argument('--dataset', type=str, default='cornell',
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, default='/mnt/msranlpintern/dataset/cornell-v12/',#'/mnt/msranlpintern/dataset/cornell-v12/' G:/dataset/cornell-v12/
                        help='Path to dataset')
    parser.add_argument('--input-size', type=int, default=351,
                        help='Input image size')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--augment', type=bool,default=False,
                        help='Whether data augmentation should be applied')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    
    parser.add_argument('--new-dataset-path', type=str, default='/mnt/msranlpintern/dataset/cornell-v33/', # G:/dataset/cornell-v13/ /mnt/msranlpintern/dataset/cornell-v13/
                        help='Path to the new dataset')
    parser.add_argument('--idx-start', type=int, default=0, # 0
                        help='the starting idx')
    parser.add_argument('--idx-end', type=int, default=14160, #0 3540 7080 10620 14160
                        help='the ending idx')
    
    parser.add_argument('--overlap-width', type=int,default=15,
                        help='')
    parser.add_argument('--rotate-num', type=int, default=4,
                        help='The number of rotation for data augementation')
    parser.add_argument('--translate-num', type=int, default=4,
                        help='The number of translation for data augementation')
    parser.add_argument('--instructions-file-name', type=str,default='instructions',
                        help='')
    parser.add_argument('--group-json-file-path', type=str,default='/mnt/msranlpintern/dataset/cornell-v12/else/group_name_shape_purpose_strategy.json',  #'/mnt/msranlpintern/dataset/cornell-v12/else/group_color.json'
                        help='')
    
    args = parser.parse_args()
    return args

def new_dataset_multiobject():
    args = parse_args()

    # Load Dataset
    print(f'Loading {args.dataset} dataset...... Path: {args.dataset_path}')
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(args.dataset_path,
                    output_size=args.input_size,
                    ds_rotate=args.ds_rotate,
                    random_rotate=args.augment,
                    random_zoom=args.augment,
                    include_depth=args.use_depth,
                    include_rgb=args.use_rgb
                    )
    
    print('the number of image is {}'.format(dataset.image_sum)) # 885
    print('the number of grasp is {}'.format(dataset.grasp_sum)) # 5110
    
    # dataset.show_original_rgb_Image_and_grasp_Rectangles(idx=0)
    # dataset.show_cropped_rgb_Image_and_grasp_Rectangles(idx=0)
    
    dataset.new_dataset_multiobject(args.new_dataset_path,
                                    args.idx_start,
                                    args.idx_end,
                                    args.overlap_width,
                                    args.rotate_num,
                                    args.translate_num,
                                    args.instructions_file_name,
                                    args.group_json_file_path
                                    )
if __name__ == '__main__':
    new_dataset_multiobject()