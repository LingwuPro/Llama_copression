CUDA_VISIBLE_DEVICES=0 python collect.py 

CUDA_VISIBLE_DEVICES=0 python compression.py

# https://huggingface.co/decapoda-research/llama-7b-hf
# https://huggingface.co/datasets/yahma/alpaca-cleaned
# https://huggingface.co/tloen/alpaca-lora-7b


# python finetune.py \
#     --base_model 'decapoda-research/llama-7b-hf' \
#     --data_path 'yahma/alpaca-cleaned' \
#     --output_dir './lora-alpaca'

# yahma/llama-7b-hf

python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length

# python generate.py \
#     --load_8bit \
#     --base_model 'decapoda-research/llama-7b-hf' \
#     --lora_weights 'tloen/alpaca-lora-7b'

python generate.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './lora-alpaca'

# python finetune.py \
#     --base_model='decapoda-research/llama-7b-hf' \
#     --num_epochs=10 \
#     --cutoff_len=512 \
#     --group_by_length \
#     --output_dir='./lora-alpaca' \
#     --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
#     --lora_r=32 \
#     --micro_batch_size=8

# curl -X GET \
     # "https://datasets-server.huggingface.co/rows?dataset=boolq&config=default&split=train&offset=0&limit=100"