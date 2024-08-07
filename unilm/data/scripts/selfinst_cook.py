import json
import os
import random


def cook_instance(raw_data, templates, num):
    random.shuffle(raw_data)
    objs = []
    for example in raw_data[:num]:
        input_temp, answer_temp = random.sample(templates, 1)[0]
        keys = list(example.keys())
        for key in keys:
            if key == 'options_':
                continue
            match_str = "{" + key + "}"
            if match_str in input_temp:
                input_temp = input_temp.replace(match_str, example[key])
            if match_str in answer_temp:
                answer_temp = answer_temp.replace(match_str, example[key])
        doc = {
            'input': input_temp,
            'output': answer_temp
        }
        # if answer_temp == '{answer}':
        #     print(doc)
        #     import pudb; pu.db
        objs.append(doc)
    return objs

def cook_data(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    lines_per_file = 1000
    i = 0
    file_name = 'selfinst_{}.json'
    writer = open(os.path.join(output_folder, file_name.format(i // lines_per_file)), 'w', encoding='utf-8')
    objs = []

    input_file = r"D:\2023\git\self-instruct\data\finetuning\self_instruct_221203\gpt3_finetuning_data.jsonl"
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            new_ojb = {
                'input': obj['prompt'],
                'output': obj['completion'].replace('<|endoftext|>', '')
            }
            objs.append(new_ojb)

    random.shuffle(objs)
    for doc in objs:
        writer.write(json.dumps(doc) + '\n')
        i += 1
        if i % lines_per_file == 0:
            writer = open(os.path.join(output_folder, file_name.format(i // lines_per_file)), 'w', encoding='utf-8')


def cook_json():
    data = []
    item = {
        "source": [],
        "source_lang": "laion",
        "weight": 1.0,
        "name": "laion"
    }

    # total = 0
    # for i in range(69):
    #     shard_path = "../shard_data/core_{}.json".format(i)
    #     item['source'].append(shard_path)
    #     total += 1
    total = 0
    for i in range(83):
        shard_path = "../shard_data/selfinst_{}.json".format(i)
        item['source'].append(shard_path)
        total += 1
    
    data.append(item)
    json.dump(data, open('train.json', 'w', encoding='utf-8'), indent=2)

if __name__ == '__main__':
    # cook_data(r"C:\Users\shaohanh\Downloads\core_data\core_data.jsonl", r"C:\Users\shaohanh\Downloads\core_data\shard_data")
    # cook_data(r"C:\Users\shaohanh\Downloads\full_data\full_data.jsonl", r"C:\Users\shaohanh\Downloads\core_data\shard_data")
    # cook_data("selfinst_data")
    cook_json()
