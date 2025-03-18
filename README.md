# 基于gaudi2加速卡对chatglm3-6b进行微调与推理

## 项目介绍

本项目为DataFountain举办的“基于Intel Gaudi AI加速器的大语言模型微调与推理优化”赛题的参赛项目， 最终获决赛三等奖（排名4/942）

本项目基于Intel Gaudi AI加速平台，编写适用于LoRA微调和推理的脚本，并且对脚本的性能进行调优。本文将从三个方面详细介绍整个工作流程：LoRA微调实现、推理实现、以及性能分析；

## 项目部署

将lier_workload 解压至/root 环境下：

tar -xvf lier_workload.tar /root

## 下载模型

```bash
cd /data  
sudo apt update  
sudo apt install git-lfs   #安装fit-lfs安装工具  
git lfs install  
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git 
``` 

## Install gaudi Driver and Software

下载habanalabs-installer.sh来安装相关的驱动和软件：  

```bash
cd /root  
wget -nv https://vault.habana.ai/artifactory/gaudi-installer/1.18.0/habanalabs-installer.sh  
chmod +x habanalabs-installer.sh  
./habanalabs-installer.sh install --type base  
```

## Install Intel Gaudi PyTorch

安装gaudi pytorch环境：  

```bash
./habanalabs-installer.sh install -t dependencies  
./habanalabs-installer.sh install --type pytorch 
``` 

## Install gaudi DeepSpeed

```bash

export http_proxy=http://10.132.19.35:7890  
export https_proxy=http://10.132.19.35:7890  
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.18.0  
```

## 运行模型微调

```bash
cd /root/lier_workload/  
git clone https://github.com/huggingface/optimum-habana.git   
cd optimum-habana/  
sudo apt update
sudo apt install gh
gh auth login    <!--按照提示登陆github-->
gh pr checkout 1478   <!--merge包含chatglm兼容的分支-->

cd /root/lier_workload/change
cp run_lora_clm.py /root/lier_workload/optimum-habana/examples/language-modeling/  <!--对官方脚本进行了修改-->
cp -r AdvertiseGen /root/lier_workload/optimum-habana/examples/language-modeling/   <!--经过格式预处理的数据集-->
cp modeling_chatglm.py /data/chatglm3-6b  

cd /root/lier_workload/optimum-habana/examples/language-modeling/
pip install -r requirements.txt
pip install transformers==4.45.2
git clone https://github.com/huggingface/evaluate.git  <!--将evaluate库本地部署，将相应的路径改为了本地路径-->
pip install habana-frameworks --upgrade  
pip install optimum-habana==1.14.1  

python3 run_lora_clm.py \
    --model_name_or_path /data/chatglm3-6b \    <!--模型路径-->
    --dataset_name AdvertiseGen \               <!--数据集路径-->
    --bf16 True \  
    --output_dir ./model_lora_glm \    <!--输出lora checkpoint 路径-->
    --num_train_epochs 3 \           <!--epoch数量-->
    --per_device_train_batch_size 16 \  <!--batch size, 不建议更改，可能会爆显存-->
    --eval_strategy "no" \  
    --save_strategy "no" \  
    --learning_rate 1e-4 \  
    --warmup_ratio  0.03 \  
    --lr_scheduler_type "constant" \  
    --max_grad_norm  0.3 \  
    --logging_steps 1 \  
    --do_train \  
    --do_eval \  
    --use_habana \  
    --use_lazy_mode \  
    --throughput_warmup_steps 3 \  
    --lora_rank=8 \  
    --lora_alpha=16 \  
    --lora_dropout=0.05 \  
    --lora_target_modules "query_key_value" \  
    --dataset_concatenation \  
    --max_seq_length 512 \  
    --low_cpu_mem_usage True \  
    --validation_split_percentage 4 \  
    --adam_epsilon 1e-08  

```

如果遇到： TypeError: ChatGLMTokenizer._pad() got an unexpected keyword argument 'padding_side'， 可以注释掉  
/root/habanalabs-venv/lib/python3.10/site-packages/transformers/tokenization_utils_base.py 中的相应行：（有概率会出现这个问题）


```python
outputs = self._pad(  
                inputs,  
                max_length=max_length,  
                padding_strategy=padding_strategy,  
                pad_to_multiple_of=pad_to_multiple_of,  
                #padding_side=padding_side,      <!--注释这一行-->
                return_attention_mask=return_attention_mask,  
            )
```

### 模型推理

```
cd /root/lier_workload/  
git clone https://github.com/HabanaAI/vllm-fork.git  
cd vllm-fork/  
git checkout eca9a83a2d6d60d4c5f577a993e1a95cd64d6a59  
pip install -r requirements-hpu.txt  
python setup.py develop  
cd ../vllm_workload
python inference.py
```

相关的推理参数可以在config.json中修改；



