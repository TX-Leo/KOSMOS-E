from azure.storage.blob import ContainerClient # pip install azure-storage-blob
import json, os

AZURE_URL="https://turingdata2.blob.core.windows.net/ita"
CREDENTIAL="sv=2020-08-04&st=2022-11-07T17%3A55%3A08Z&se=2023-06-08T16%3A55%3A00Z&sr=c&sp=rl&sig=LQwL3DhQLAGfZ7EIV3X8JhnnCRpAK2k1nNJHFPAOf1c%3D"
STEP=50

def convert(num):
    container_client = ContainerClient.from_container_url(
        container_url=AZURE_URL,
        credential=CREDENTIAL
    )
    for i in range(num * STEP, (num + 1) * STEP):
        print(i)
        local_path = f'/mnt/msranlp/shaohanh/bvl/raw_wild/partition.{i:03d}.ndjson'
        source_file = f"2022-05/turing-wild-2/merged_content/partition.{i:03d}.ndjson"
        download_file_path = 'temp.ndjson'

        if os.path.exists(local_path):
            continue

        if i >= 7190:
            break
        try_num = 0
        lines = []
        while True: # read until EOF
            try:
                merged_content = container_client.get_blob_client(source_file)
                print(f"Reading {source_file}")
                bytes = merged_content.download_blob().readall() 
                with open(download_file_path, "wb") as file:
                    file.write(bytes)
                with open(download_file_path, "r") as file:
                    lines += file.readlines()
                # lines = merged_content.download_blob().content_as_text(encoding='UTF-8').strip().split('\n')
                break
            except:
                print(f"Error reading {source_file}")
                try_num += 1
                if try_num > 10:
                    break

        print('Read {} lines'.format(len(lines)))
        writer = open(f'/mnt/msranlp/shaohanh/bvl/raw_wild/partition.{i:03d}.ndjson', 'w', encoding='utf-8')
        for doc_jsonstr in lines:
            json_obj = json.loads(doc_jsonstr)
            del json_obj['Content']
            writer.write(json.dumps(json_obj) + '\n')

import sys
num = int(sys.argv[1])
convert(num)
