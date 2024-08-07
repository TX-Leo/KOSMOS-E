import json
import os
import random


def cook_data(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    lines_per_file = 1000
    i = 0
    writer = open(os.path.join(output_folder, 'full_{}.json'.format(i // lines_per_file)), 'w', encoding='utf-8')
    objs = []
    with open(input_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            obj = json.loads(line)
            for ins in obj['instances']:
                doc = {
                    'input': ins['instruction_with_input'],
                    'output': ins['output']
                }
                objs.append(doc)
            if 'reformulations' in obj:
                for ins in obj['reformulations']:
                    doc = {
                        'input': ins['instruction_with_input'],
                        'output': ins['output']
                    }
                    objs.append(doc)

    random.shuffle(objs)
    for doc in objs:
        writer.write(json.dumps(doc) + '\n')
        i += 1
        if i % lines_per_file == 0:
            writer = open(os.path.join(output_folder, 'full_{}.json'.format(i // lines_per_file)), 'w', encoding='utf-8')

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
    for i in range(241):
        shard_path = "../shard_data/full_{}.json".format(i)
        item['source'].append(shard_path)
        total += 1
    
    data.append(item)
    json.dump(data, open('train.json', 'w', encoding='utf-8'), indent=2)

if __name__ == '__main__':
    # cook_data(r"C:\Users\shaohanh\Downloads\core_data\core_data.jsonl", r"C:\Users\shaohanh\Downloads\core_data\shard_data")
    # cook_data(r"C:\Users\shaohanh\Downloads\full_data\full_data.jsonl", r"C:\Users\shaohanh\Downloads\core_data\shard_data")
    cook_json()
