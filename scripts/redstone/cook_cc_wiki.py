import json

def cook_cc_wiki_config():
    config_file = r"C:\Users\shaohanh\Downloads\train-nogithub-noarvix-nopubmed-mtnlg.json"
    output_file = r"C:\Users\shaohanh\Downloads\train-cc-wiki.json"
    obj = json.load(open(config_file, 'r', encoding='utf-8'))
    new_obj = []
    for item in obj:
        if 'cc' in item['name'].lower() or 'wiki' in item['name'].lower():
            print(item['name'])
            new_obj.append(item)
    json.dump(new_obj, open(output_file, 'w', encoding='utf-8'), indent=4)


if __name__ == '__main__':
    cook_cc_wiki_config()