import json  
import random  
  
def is_valid_conversation(conversation):  
    if len(conversation) % 2 != 0:  
        return False  
  
    for i, chat in enumerate(conversation):  
        if i % 2 == 0 and chat["from"] != "human":  
            return False  
        if i % 2 != 0 and chat["from"] != "gpt":  
            return False  
  
    return True  
  
def group_conversations(conversation):  
    groups = []  
    for i in range(0, len(conversation), 2):  
        groups.append([conversation[i], conversation[i + 1]])  
    return groups  
  
def get_random_groups(groups, num_groups):  
    return random.sample(groups, num_groups)  
  
def main():  
    conversation_json = """  
    [  
          {  
            "from": "human",  
            "value": "What are the colors of the bus in the image?\\n<image>"  
          },  
          {  
            "from": "gpt",  
            "value": "The bus in the image is white and red."  
          },  
          {  
            "from": "human",  
            "value": "What feature can be seen on the back of the bus?"  
          },  
          {  
            "from": "gpt",  
            "value": "The back of the bus features an advertisement."  
          },  
          {  
            "from": "human",  
            "value": "Is the bus driving down the street or pulled off to the side?"  
          },  
          {  
            "from": "gpt",  
            "value": "The bus is driving down the street, which is crowded with people and other vehicles."  
          }  
    ]  
    """  
  
    conversation = json.loads(conversation_json)  
  
    if is_valid_conversation(conversation):  
        groups = group_conversations(conversation)  
        num_groups = random.randint(1, len(groups))  
        random_groups = get_random_groups(groups, num_groups)  
        flattened_groups = [chat for group in random_groups for chat in group]  
        print(json.dumps(flattened_groups, indent=2))  
        
        human_input_string = ""
        for i in range(len(flattened_groups)-1):
            human_input_string = human_input_string + conversation[i]['value'].replace("<image>\n", "").replace("\n<image>", "") + " "
        gpt_output_string = conversation[len(flattened_groups)-1]['value'].replace("<image>\n", "").replace("\n<image>", "")
        
        print("Input: {}".format(human_input_string))
        print("Output: {}".format(gpt_output_string))     
    else:  
        print("Invalid conversation format.")  
  
if __name__ == "__main__":  
    main()