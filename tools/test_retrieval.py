import json

filename = 'retrieved_target_valid'

with open(filename + '.json', encoding='utf-8') as f:
    retrieved = json.load(f)

with open(filename + '.ret', 'w', encoding='utf-8') as f:
    for item in retrieved:
        f.write(item['retrieved']['target'] + '\n')


for item in retrieved:
    print('Q: ', item['query']['target'])
    print('R: ', item['retrieved']['target'])
    print()
