import sentencepiece as spm
import json


def cook_data():
    # read a json file
    # obj = json.load(open(input_file, 'r', encoding='utf-8'))
    # for i, template in enumerate(['▁summary ▁this ▁image :', '▁describe ▁this ▁image :', '▁a ▁image ▁of', '▁a ▁picture ▁of', '▁a ▁photo ▁of']):
    #     with open(input_file + f'.template_{i}.out', 'w', encoding='utf-8') as f:
    #         for item in obj:
    #             f.write(f'[image]/mnt/unilm/shaohanh/exp/unigpt_exp/data/COCO/{item["image"]}<tab>{template}\n') # 
    sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    sessions = ['Session1']
    writer1 = open('test.s1.out', 'w', encoding='utf-8')
    writer2 = open('test.s1.label', 'w', encoding='utf-8')
    for session in sessions:
        with open(f'D:\\temp\\meta_data\\{session}\\test_meta_data.json', 'r', encoding='utf-8') as f:
            obj = json.load(f)
            for item in obj['meta_data']:
                writer1.write(f'▁Speech ▁emotion ▁classification . ▁The ▁emotion ▁is ▁happy , ▁sad , ▁angry ▁or ▁neural .<tab>[image]/mnt/localdata/readincu/t-schen/data/s3prl_data/IEMOCAP_full_release/{item["path"]}<tab>▁This ▁emotion ▁is ▁happy\n')
                writer1.write(f'▁Speech ▁emotion ▁classification . ▁The ▁emotion ▁is ▁happy , ▁sad , ▁angry ▁or ▁neural .<tab>[image]/mnt/localdata/readincu/t-schen/data/s3prl_data/IEMOCAP_full_release/{item["path"]}<tab>▁This ▁emotion ▁is ▁sad\n')
                writer1.write(f'▁Speech ▁emotion ▁classification . ▁The ▁emotion ▁is ▁happy , ▁sad , ▁angry ▁or ▁neural .<tab>[image]/mnt/localdata/readincu/t-schen/data/s3prl_data/IEMOCAP_full_release/{item["path"]}<tab>▁This ▁emotion ▁is ▁angry\n')
                writer1.write(f'▁Speech ▁emotion ▁classification . ▁The ▁emotion ▁is ▁happy , ▁sad , ▁angry ▁or ▁neural .<tab>[image]/mnt/localdata/readincu/t-schen/data/s3prl_data/IEMOCAP_full_release/{item["path"]}<tab>▁This ▁emotion ▁is ▁neural\n')
                writer2.write(f'{item["path"]}\t{item["label"]}\n')

if __name__ == '__main__':
    # cook_data(r"C:\Users\shaohanh\Downloads\cloze_test_test__spring2016 - cloze_test_ALL_test.tsv")
    # import sys
    # eval_log(sys.argv[1], sys.argv[2], sys.argv[3])
    # eval_log('/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/data/gpt/story_cloze/test_data.txt', 'log.txt', '/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/data/gpt/story_cloze/cloze_test_test__spring2016 - cloze_test_ALL_test.tsv')
    # root_path = '/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp'
    # branchs = ['vl_base_20m_mtnlg_6e4_256-2048_deepnorm_fix','vl_base_20m_mtnlg_6e4_256-2048_no_deepnorm_fix', 'vl_base_20m_mtnlg_6e4_256-2048_deepnorm_1M_xcon', 'vl_base_20m_mtnlg_6e4_256-2048_deepnorm_1M']
    # ckpts = ['checkpoint_1_200000', 'checkpoint_1_100000', 'checkpoint_1_300000', 'checkpoint_1_400000', 'checkpoint_1_500000']
    # for branch in branchs:
    #     for ckpt in ckpts:
    #         print(branch, ckpt)
    #         try:
    #             eval_log(f'{root_path}/data/gpt/story_cloze/test_data.txt', f'{root_path}/{branch}/{ckpt}_text_zero-shot/story_cloze_log.txt', f'{root_path}/data/gpt/story_cloze/cloze_test.tsv')
    #         except:
    #             pass
    cook_data()