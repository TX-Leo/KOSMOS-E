'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-02-03 21:48:21
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
    parser.add_argument('--dataset-path', type=str, default='/mnt/msranlpintern/dataset/cornell-v1/',#'/mnt/msranlpintern/dataset/cornell-v1-selected/'
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
    

    # dataset.get_objs_description(idxs=list(range(611,885)))
    # dataset.generate_image_with_grasp(idxs=list(range(0,885)))

    # dataset.generate_instructions_color(idxs=list(range(612,885)))
    dataset.generate_instructions_part(idxs=list(range(690,612,-1)))
    # dataset.test_multiobject()

if __name__ == "__main__":
    generate_dataloader()