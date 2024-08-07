'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-02-03 21:48:20
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
    parser.add_argument('--dataset-path', type=str, default='/mnt/msranlpintern/dataset/cornell-v1/',#'G:/dataset/cornell-v1/　'
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
    
    parser.add_argument('--new-dataset-path', type=str, default='/mnt/msranlpintern/dataset/cornell-v12/', # D:/dataset/cornell-v3/
                        help='Path to the new dataset')
    parser.add_argument('--output-size', type=int, default=224,
                        help='Output image size')
    parser.add_argument('--idx-start', type=int, default=0, # 0
                        help='the starting idx')
    parser.add_argument('--idx-end', type=int, default=885, # 885
                        help='the ending idx')
    parser.add_argument('--rotate-num', type=int, default=4,
                        help='The number of rotation for data augementation')
    parser.add_argument('--rotate-range', type=int, default=360,
                        help='The number of rotation for data augementation')
    parser.add_argument('--translate-num', type=int, default=4,
                        help='The number of translation for data augementation')
    parser.add_argument('--translate-range', type=int, default=40,
                        help='The number of translation for data augementation')
    parser.add_argument('--vis', action='store_true',default=False,
                        help='Visualise the network output')
    parser.add_argument('--gen-pc', action='store_true',default=False,
                        help='generate pc')
    
    # for test
    parser.add_argument('--instructions-file-name', type=str,default='instructions_temp_9',
                        help='')
    args = parser.parse_args()
    return args

def dataset_augment():
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
    
    dataset.dataset_augment(args.new_dataset_path,
                            args.output_size,
                            args.idx_start,
                            args.idx_end,
                            args.rotate_num,
                            args.rotate_range,
                            args.translate_num,
                            args.translate_range,
                            args.vis,
                            args.gen_pc,
                            args.instructions_file_name)
        
if __name__ == '__main__':
    dataset_augment()