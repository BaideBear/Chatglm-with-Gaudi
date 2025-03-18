import json
import vllm
from vllm import LLM, SamplingParams
import os
import argparse
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest
import yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
model_path = config['configs']['model_path']
lora_path = config['lora_configs']['lora_path']
dev_path = config['configs']['dev_path']
output_path = config['configs']['output_path']
use_warmup = config['configs']['use_wramup']
use_lora = config['configs']['use_lora']
batch_size = config['configs']['batch_size']
temperature = config['sampling_configs']['temperature']
top_p = config['sampling_configs']['top_p']
max_tokens = config['sampling_configs']['max_tokens']

os.environ['PT_HPU_LAZY_MODE'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--enforce-eager', type=int, default=0, help='Set to 0 to enforce eager execution')
args = parser.parse_args()

if use_warmup == False:
    os.environ['VLLM_SKIP_WARMUP'] = "true"

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def extract_contents(data):
    return [item['content'] for item in data]

def vllm_inference(batchs):
    if use_lora == True:
        model = LLM(model=model_path, dtype='bfloat16' , 
                    trust_remote_code=True, 
                    enable_lora=True, 
                    )
    else: 
        model = LLM(model=model_path, dtype='bfloat16' , trust_remote_code=True)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)
    responses = []
    for prompt in batchs:
        if use_lora == True:
            response = model.generate(
                prompt, 
                sampling_params,
                lora_request=LoRARequest("adapter", 1, lora_path)
                )
        else:
            response = model.generate(prompt, sampling_params)
        responses.extend(response)
    return responses

def create_batches(data, batch_size):
    total_samples = len(data)
    batches = [data[i:i + batch_size] for i in range(0, total_samples, batch_size)]
    
    return batches



data = load_data(dev_path)
prompts = extract_contents(data)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompts = [f"<|user|>\n{msg}\n" for msg in prompts]
batchs = create_batches(prompts, batch_size)
responses = vllm_inference(batchs)

out = []
for item in responses:
    out.append(item.outputs[0].text)

with open(output_path, 'w', encoding='utf-8') as json_file:
    json.dump(out, json_file, ensure_ascii=False, indent=4)