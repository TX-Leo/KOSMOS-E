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
    parser.add_argument('--dataset-path', type=str, default='G:/dataset/cornell-v12/',#'/mnt/msranlpintern/dataset/cornell-v3/' G:/dataset/cornell-v12/
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
    
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    parser.add_argument('--dataloader-num', type=str, default='01',
                        help='Dataset workers')
    parser.add_argument('--train-output-num', type=str, default='01',
                        help='Dataset workers')
    parser.add_argument('--grasp-format', type=str, default='xya',
                        help='xywha/tlbra')
    parser.add_argument('--splited',action='store_true', default=False,
                        help='')
    parser.add_argument('--encoded',action='store_true', default=True,
                        help='')
    parser.add_argument('--vis', action='store_true',default=False,
                        help='Visualise the network output')
    parser.add_argument('--instruction-type', type=str, default='name',
                        help='name/angle/color/shape/purpose/position/strategy')
    args = parser.parse_args()
    return args
   
def test_evaluate(args,generate_predictions=None):

    # Load Dataset
    print(f'Loading {args.dataset} dataset...... Path: {args.dataset_path}')
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(args.dataset_path,
                           output_size=args.input_size,
                           ds_rotate=args.ds_rotate,
                           random_rotate=args.augment,
                           random_zoom=args.augment,
                           include_depth=args.use_depth,
                           include_rgb=args.use_rgb
                           )
    print('The number of image is {}'.format(test_dataset.image_sum)) # 885
    print('The number of grasp is {}'.format(test_dataset.grasp_sum)) # 5110
    
    # idx = 0
    # test_dataset.show_original_rgb_Image_and_grasp_Rectangles(idx)
    # test_dataset.show_cropped_rgb_Image_and_grasp_Rectangles(idx)
    # test_dataset.show_original_rgb_Image_and_Grasps(idx)
    
    test_dataset.evaluate(  args.dataloader_num,
                            generate_predictions,
                            args.iou_threshold,
                            args.grasp_format,
                            args.splited,
                            args.encoded,
                            f'-{args.dataloader_num}-{args.train_output_num}',
                            args.vis,
                            f'{args.dataset_path}/dataloader/{args.dataloader_num}/train_output/{args.train_output_num}/eval_savedir/',
                            args.instruction_type
                            )

def evaluate():
    args = parse_args()
    test_evaluate(args,generate_predictions=None)
    
if __name__ == '__main__':
    evaluate()