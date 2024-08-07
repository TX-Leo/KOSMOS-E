import pandas as pd
import sentencepiece as spm


def cook_data(input_file):
    # read file, split by \t, and drop the first column
    tokenizer = spm.SentencePieceProcessor(model_file=r"C:\Users\shaohanh\Desktop\sentencepiece.bpe.model")
    with open(input_file, 'r', encoding='utf-8') as f, open('data.txt', 'w', encoding='utf-8') as f2:
        lines = f.read().splitlines()
        lines = [line.split('\t')[1:] for line in lines[1:]]
        for line in lines:
            input_line = ' '.join(line[:4])
            tokenized_line = tokenizer.encode_as_pieces(input_line + ' ' + line[4])
            f2.write(' '.join(tokenized_line) + '\n')
            tokenized_line = tokenizer.encode_as_pieces(input_line + ' ' + line[5])
            f2.write(' '.join(tokenized_line) + '\n')

def eval_log(input_file, log_file, raw_file):
    scores = {}
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            if line.startswith('P-'):
                sentence_id = int(line.split('\t')[0][2:])
                scores[sentence_id] = [float(i) for i in line.split('\t')[1].split(' ')]
    
    # print(len(scores))

    res = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for i in range(0, len(lines), 2):
            # if i >= len(scores):
            #     break
            # find the same part
            j = 0
            while lines[i].split(' ')[j] == lines[i + 1].split(' ')[j]:
                j += 1
            if i in scores and i + 1 in scores:
                score_a = sum(scores[i][j:len(lines[i].split(' '))]) / (len(lines[i].split(' ')) - j)
                score_b = sum(scores[i + 1][j:len(lines[i + 1].split(' '))]) / (len(lines[i+1].split(' ')) - j)                
                if score_a > score_b:
                    res[i // 2] = 1
                else:
                    res[i // 2] = 2
    # print(len(res))
    labels = []
    with open(raw_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        lines = [line.split('\t')[1:] for line in lines[1:]]
        for line in lines:
            labels.append(int(line[-1]))
    # print(len(res), len(labels))
    num = 0
    correct = 0
    for i in range(len(labels)):
        if i in res:
            num += 1
            if res[i] == labels[i]:
                correct += 1
    print('acc:', correct / num)
    
if __name__ == '__main__':
    # cook_data(r"C:\Users\shaohanh\Downloads\cloze_test_test__spring2016 - cloze_test_ALL_test.tsv")
    # import sys
    # eval_log(sys.argv[1], sys.argv[2], sys.argv[3])
    # eval_log('/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/data/gpt/story_cloze/test_data.txt', 'log.txt', '/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp/data/gpt/story_cloze/cloze_test_test__spring2016 - cloze_test_ALL_test.tsv')
    root_path = '/mnt/localdata/msranlp/shaohanh/exp/unigpt_exp'
    branchs = ['vl_base_20m_mtnlg_6e4_256-2048_deepnorm_fix','vl_base_20m_mtnlg_6e4_256-2048_no_deepnorm_fix', 'vl_base_20m_mtnlg_6e4_256-2048_deepnorm_1M_xcon', 'vl_base_20m_mtnlg_6e4_256-2048_deepnorm_1M']
    branchs = ['speech-unigpt-medium-lr1e-3-bs512-reset-wd0.01-mlmdrop-warm500-fxeos-fxfinal-mask-gpt-scratch']
    ckpts = ['checkpoint_1_200000', 'checkpoint_1_100000', 'checkpoint_1_300000', 'checkpoint_1_400000', 'checkpoint_1_500000']
    ckpts = ['checkpoint_1_100000', 'checkpoint_1_135000']
    for branch in branchs:
        for ckpt in ckpts:
            print(branch, ckpt)
            try:
                eval_log(f'{root_path}/data/gpt/story_cloze/test_data.txt', f'{root_path}/{branch}/{ckpt}_text_zero-shot/story_cloze_log.txt', f'{root_path}/data/gpt/story_cloze/cloze_test.tsv')
            except:
                pass
