import json
import os

def cook_json():
    data = []
    item = {
        "source": [],
        "source_lang": "laion",
        "weight": 1.0,
        "name": "laion"
    }

    total = 0
    for i in range(128):
        # print("laion2b", i)
        for j in range(94):
            shard_path = "../laion2b_filtered_tsvs_v1/{:05d}/{:05d}_{:05d}.tsv".format(i, i, j)
            item['source'].append(shard_path)
            total += 1
            shard_abs_path = os.path.join("/mnt/unilm/wenwan/bvt/data/laion_dataloader_config", shard_path)
            # if not os.path.isfile(shard_abs_path):
            #     print("File not find", shard_abs_path)
    print("laion2b", total)
    
    
    for i in range(128):
        # print("coyo", i)
        for j in range(54):
            shard_path = "../coyo_filtered_tsvs_v1/{:05d}/{:05d}_{:05d}.tsv".format(i, i, j)
            # print(shard_path)
            item['source'].append(shard_path)
            total += 1
            shard_abs_path = os.path.join("/mnt/unilm/wenwan/bvt/data/laion_dataloader_config", shard_path)
            # if not os.path.isfile(shard_abs_path):
            #     print("File not find", shard_abs_path)
    print("laion2b + coyo", total)


    for i in range(1152):
        for j in range(5):
            shard_path = "../cc15m_filter_tsvs_smallchunk_v1/{:04d}_{}".format(i, j)
            # print(shard_path)
        
            shard_abs_path = os.path.join("/mnt/unilm/wenwan/bvt/data/laion_dataloader_config", shard_path)
            # if not os.path.isfile(shard_abs_path):
            #     print("File not find", shard_abs_path)
                # continue
            item['source'].append(shard_path)
            total += 1
    print("laion2b + coyo + cc15m", total)


    for i in range(3882):
        shard_path = "../laion400m_zw_filtered_tsvs_v1/{:04d}.tsv".format(i)
        # print(shard_path)
        shard_abs_path = os.path.join("/mnt/unilm/wenwan/bvt/data/laion_dataloader_config", shard_path)
        # if not os.path.isfile(shard_abs_path):
            # print("File not find", shard_abs_path)
            # continue
        item['source'].append(shard_path)
        total += 1
    print("laion2b + coyo + cc15m + laion400m", total)


    
    for i in range(8):
        for j in range(15):
            data_shard_index = "all_data_eng{:01d}_{:02d}".format(i, j)
            for k in range(150):
                shard_path = "../turing1b_filter_tsvs_v1_newformat/{}/{}_filtered_{:03d}".format(data_shard_index, data_shard_index, k)
                shard_abs_path = os.path.join("/mnt/unilm/wenwan/bvt/data/laion_dataloader_config", shard_path)
                # if not os.path.isfile(shard_abs_path):
                #     print("File not find", shard_abs_path)
                    # continue
                # print(shard_path)
                item['source'].append(shard_path)
                total += 1

    print("laion2b + coyo + cc15m + laion400m + turing1b", total)
    print(len(item['source']))


  
    data.append(item)
    json.dump(data, open('train.json', 'w', encoding='utf-8'), indent=2)

if __name__ == '__main__':
    # run()
    cook_json()
