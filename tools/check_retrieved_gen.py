import json

with open('tools/retrieved_valid.json', encoding='utf-8') as jsonf:
    retrieved = json.load(jsonf)

with open('testout/roto-ret-enc', encoding='utf-8') as genf:
    for idx, line in enumerate(genf):
        # print('GOLD: ', retrieved[idx]['query']['target'])
        # print('PRED: ', line.strip())
        print(retrieved[idx]['retrieved']['target'])
        # print('\n')
