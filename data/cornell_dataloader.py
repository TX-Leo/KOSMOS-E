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
    parser = argparse.ArgumentParser(description='Evaluate networks')
    
    parser.add_argument('--dataset', type=str, default='cornell',
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, default='/mnt/msranlpintern/dataset/cornell-v12/',#'/mnt/msranlpintern/dataset/cornell-v3/' G:/dataset/cornell-v1/
                        help='Path to dataset')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--augment', type=bool,default=False,
                        help='Whether data augmentation should be applied')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    
    parser.add_argument('--eval-format', type=str, default='OW',
                        help='the format of the evaluation(IW or OW)')
    
    parser.add_argument('--five-fold-cross-validation', action='store_true', default=True,
                        help='')
    parser.add_argument('--dataloader-num-list', type=list, default=None,
                        help='')
    
    parser.add_argument('--split', type=list, default=[20,4,3], # [7785,90,90]
                        help='the proportion of the dataset(train:valid:test)')
    parser.add_argument('--dataloader-num', type=str, default='12',
                        help='the number of the dataloader folder')
    
    parser.add_argument('--grasp-format', type=str, default='xya',
                        help='')
    parser.add_argument('--splited',action='store_true', default=True,
                        help='')
    parser.add_argument('--encoded',action='store_true', default=True,
                        help='')
    
    args = parser.parse_args()
    return args

def generate_dataloader():
    
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
    print('The number of image is {}'.format(dataset.image_sum)) # 885 * 9 = 7965
    print('The number of grasp is {}'.format(dataset.grasp_sum)) # 5110 * 9 = 45990
    
    # idx = 0
    # dataset.show_original_rgb_Image_and_grasp_Rectangles(idx)
    # dataset.show_cropped_rgb_Image_and_grasp_Rectangles(idx)
    # dataset.show_original_rgb_Image_and_Grasps(idx)
    
    # for five-fold-cross-validation
    dataset.generate_dataloader(args.five_fold_cross_validation,
                                args.split,
                                args.eval_format,
                                args.dataloader_num,
                                args.dataloader_num_list,
                                args.grasp_format,
                                args.splited,
                                args.encoded)
    
    # for single
    # dataset.get_idxs_from_dataloader_json(f'{args.dataset_path}/dataloader/{args.dataloader_num}/dataloader_config/json/',
    #                                       args.grasp_format,
    #                                       args.splited,
    #                                       args.encoded)
if __name__ == "__main__":
    generate_dataloader()