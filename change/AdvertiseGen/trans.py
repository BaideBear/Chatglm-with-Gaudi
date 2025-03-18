import pandas as pd
import json

data_list = []
with open('train.json', 'r', encoding='utf-8') as file:
    for line in file:
        data_list.append(json.loads(line))
df = pd.DataFrame(data_list)
def process_content(content):
    attributes = dict(item.split('#') for item in content.split('*'))
    input_str = ', '.join([f"{key}: {value}" for key, value in attributes.items()])
    return input_str
df['input'] = df['content'].apply(process_content)
df['output'] = df['summary']
final_df = df[['input', 'output']]
final_df.to_csv('train.csv', index=False, encoding='utf-8-sig')
