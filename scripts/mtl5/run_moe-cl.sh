# 杀掉占卡进程
ps -ef | grep '[t]raining_gpu' | awk '{print $2}' | xargs kill -9

SECONDS=0
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_IB_TIMEOUT=22

# 随机初始化
python mlora.py \
    --base_model ../model/Llama-2-7b-hf \
    --config configs/mtl5/moe-cl.json \
    --train_task dbpedia \
    --order rand_init \
    --rand_init

# 顺序 1
python mlora.py \
    --base_model ../model/Llama-2-7b-hf \
    --config configs/mtl5/moe-cl.json \
    --train_task dbpedia \
    --order order1

python mlora.py \
    --base_model ../model/Llama-2-7b-hf \
    --config configs/mtl5/moe-cl.json \
    --train_task amazon \
    --load_adapter_file dbpedia_finetuned \
    --order order1

python mlora.py \
    --base_model ../model/Llama-2-7b-hf \
    --config configs/mtl5/moe-cl.json \
    --train_task yahoo \
    --load_adapter_file amazon_finetuned \
    --order order1

python mlora.py \
    --base_model ../model/Llama-2-7b-hf \
    --config configs/mtl5/moe-cl.json \
    --train_task agnews \
    --load_adapter_file yahoo_finetuned \
    --order order1

# # 顺序 2
# python mlora.py \
#     --base_model ../model/Llama-2-7b-hf \
#     --config configs/mtl5/moe-cl.json \
#     --train_task dbpedia \
#     --order order2
    
# python mlora.py \
#     --base_model ../model/Llama-2-7b-hf \
#     --config configs/mtl5/moe-cl.json \
#     --train_task amazon \
#     --load_adapter_file dbpedia_finetuned \
#     --order order2

# python mlora.py \
#     --base_model ../model/Llama-2-7b-hf \
#     --config configs/mtl5/moe-cl.json \
#     --train_task agnews \
#     --load_adapter_file amazon_finetuned \
#     --order order2

# python mlora.py \
#     --base_model ../model/Llama-2-7b-hf \
#     --config configs/mtl5/moe-cl.json \
#     --train_task yahoo \
#     --load_adapter_file agnews_finetuned \
#     --order order2

# # 顺序 3
# python mlora.py \
#     --base_model ../model/Llama-2-7b-hf \
#     --config configs/mtl5/moe-cl.json \
#     --train_task yahoo \
#     --order order3

# python mlora.py \
#     --base_model ../model/Llama-2-7b-hf \
#     --config configs/mtl5/moe-cl.json \
#     --train_task amazon \
#     --load_adapter_file yahoo_finetuned \
#     --order order3
    
# python mlora.py \
#     --base_model ../model/Llama-2-7b-hf \
#     --config configs/mtl5/moe-cl.json \
#     --train_task agnews \
#     --load_adapter_file amazon_finetuned \
#     --order order3

# python mlora.py \
#     --base_model ../model/Llama-2-7b-hf \
#     --config configs/mtl5/moe-cl.json \
#     --train_task dbpedia \
#     --load_adapter_file agnews_finetuned \
#     --order order3

elapsed_time=$SECONDS
hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))

echo "Elapsed time: $hours hours, $minutes minutes, and $seconds seconds"

# 启动占卡进程
cd /apdcephfs_qy3/share_300998916/joyyyhuang/H20_occupied
sh run_mnist.sh
