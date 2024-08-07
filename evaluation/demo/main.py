import requests

from flask import Flask, request
from flask import render_template
import socket
import sentencepiece as spm


db_path = '/home/shaohanh/demo/interactive.db'
app = Flask(__name__, static_url_path='', static_folder='.', template_folder='.')
# ip_port = ('127.0.0.1', 7999)
# s = socket.socket()     
# s.connect(ip_port) 
from tinydb import TinyDB, Query
import datetime, time
db = TinyDB(db_path)     

@app.route('/query', methods=['GET', 'POST'])
def query_ans():
    if request.method == 'POST':
        query = request.form['query']
        image = request.form['image']
        # print(image)
        import base64
        postfix = image.split(';base64,')[0].split('/')[1]
        imgdata = base64.b64decode(image.split('base64,')[1])
        import uuid
        image_uuid = str(uuid.uuid4())
        filename = '/tmp/flam/images/' + image_uuid + '.' + postfix  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(imgdata)
        if '[image]' in query:
            final_query = '[cap]' + query.replace('[image]', '<tab>[image]' + filename + '<tab>').strip('<tab>')
        else:
            final_query = '[cap][image]' + filename + '<tab>' + query
        print(final_query)
        id = datetime.datetime.now().timestamp()
        db.insert({'id': id, 'input': final_query, 'output': ''})
        while True:
            local_db = TinyDB(db_path)
            q = Query()
            item = local_db.search(q.id == id)
            if item[0]['output'] != '':
                output_str = item[0]['output']
                output_str = output_str.split('</image>')[1]
                # return output_str
                return {
                    'output': output_str,
                    'image_path': f'show_box_on_{image_uuid}.jpg'
                }
            else:
                time.sleep(0.2)
    else:
        type = request.args.get('type')
        if type == 'image':
            query = request.args.get('query')
            image = request.args.get('image')
            image = image.split('/')[1]
            final_query = '[cap][image]/tmp/flam/images/' + image + '<tab>' + query
            print(final_query)
            # s.sendall(final_query.encode())
            # server_reply = s.recv(1024).decode()
            id = datetime.datetime.now().timestamp()
            # db.insert({'id': id, 'input': final_query, 'output': '', 'top_p': float(top_p), 'temperature': float(temperature)})
            db.insert({'id': id, 'input': final_query, 'output': ''})
            while True:
                local_db = TinyDB(db_path)
                q = Query()
                item = local_db.search(q.id == id)
                if item[0]['output'] != '':
                    output_str = item[0]['output']
                    output_str = output_str.split('</image>')[1]
                    return output_str
                else:
                    time.sleep(0.2)
        elif type == 'gpt':
            query = request.args.get('query')
            tokenized_line = tokenizer.EncodeAsPieces(query)
            print(' '.join(tokenized_line))
            # ▁a ▁man
            final_query = '[gpt]' +  ' '.join(tokenized_line)
            print(final_query)
            # s.sendall(final_query.encode())
            # server_reply = s.recv(1024).decode()
            # print(server_reply)
            # return server_reply
            id = datetime.datetime.now().timestamp()
            # db.insert({'id': id, 'input': final_query, 'output': '', 'top_p': float(top_p), 'temperature': float(temperature)})
            db.insert({'id': id, 'input': final_query, 'output': '', 'mdoel': 'base_gpt_laion_100k'})
            while True:
                local_db = TinyDB(db_path)
                q = Query()
                item = local_db.search(q.id == id)
                if item[0]['output'] != '':
                    # cut output at the first <eol>
                    # output_str = item[0]['output'].split('.')[0]
                    output_str = item[0]['output']

                    # eol_num_in_input = final_query.count('<eol>')
                    # output_str = '<br>'.join(item[0]['output'].split('<eol>')[:eol_num_in_input+1])
                    # if len(item[0]['output'].split('<eol>')) > eol_num_in_input+1:
                    #     output_str += '<span style=\"display: none;\"><br>' + '<br>'.join(item[0]['output'].split('<eol>')[eol_num_in_input+1:]) + '</span>'
                    return output_str
                else:
                    time.sleep(0.2)
        else:
            return 'error'
        
    # s.sendall(query.encode())
    # server_reply = s.recv(1024).decode()
    # print(server_reply)
    server_reply = query + 'test......'
    return server_reply


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=443)