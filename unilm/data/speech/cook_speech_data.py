
"""
cook speech data to our data loader format, one doc per line
json line: [{"text": "text", "file": "file", "size": "size"}]
"""
import json


def cook_json_line(sent_file, path_file, output_file):
    with open(sent_file, 'r', encoding='utf8') as f:
        lines = f.read().strip().split('\n')
    with open(path_file, 'r', encoding='utf8') as f:
        paths = f.read().strip().split('\n')[1:]
    assert len(lines) == len(paths)

    # 100 doc per file
    doc_num_per_file = 100
    doc_count = 0
    writer = open(f"{doc_count // doc_num_per_file}.txt", 'w', encoding='utf8')
    last_doc_id = None
    doc = []
    for line, path in zip(lines, paths):
        doc_id = path.split('/')[2]
        # one doc per line, json line: [{"text": "text", "file": "file"}]
        if doc_id != last_doc_id:
            if doc:
                writer.write(json.dumps(doc) + '\n')
            doc = []
            doc_count += 1
            if doc_count % doc_num_per_file == 0:
                writer.close()
                writer = open(f"{doc_count // doc_num_per_file}.txt", 'w', encoding='utf8')
            last_doc_id = doc_id
        doc.append({"text": line, "file": path.split('\t')[0], "size": path.split('\t')[1]})


def cook_json_config_file():
    num = 55
    obj = [{
        "source": [f'{i}.txt' for i in range(num)],
        "source_lang": "en",
        "weight": 1.0,
        "name": "960h"
    }]
    with open('config.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(obj, indent=4))

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--sent_file", type=str, default="data/speech/sentences.txt")
    # parser.add_argument("--path_file", type=str, default="data/speech/paths.txt")
    # parser.add_argument("--output_file", type=str, default="data/speech/speech.json")
    # args = parser.parse_args()
    # cook_json_line(args.sent_file, args.path_file, args.output_file)

    cook_json_config_file()
