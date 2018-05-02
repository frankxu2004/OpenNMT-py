import json

with open('retrieved_target_train.json', encoding='utf-8') as of:
    retrieved = json.load(of)

for item in retrieved:
    print('Q: ', item['query']['target'])
    print('R: ', item['retrieved']['target'])
    print()
