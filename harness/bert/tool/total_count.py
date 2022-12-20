import json
dataset = json.load(open('dev-v1.1.json'))['data']
total = 0
for article in dataset:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            total += 1
print(total)
