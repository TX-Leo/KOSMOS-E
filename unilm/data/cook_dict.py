with open('dict.txt', 'w', encoding='utf-8') as f:
    for i in range(50257):
        f.write(f"{i} 1\n")