# 杀掉占卡进程
ps -ef | grep '[t]raining_gpu' | awk '{print $2}' | xargs kill -9

SECONDS=0
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_IB_TIMEOUT=22

python mlora.py \
    --base_model ../model/Hunyuan-7B-Instruct \
    --config configs/mtl5/moe-cl.json \
    --train_task dbpedia \
    --order test

elapsed_time=$SECONDS
hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))

echo "Elapsed time: $hours hours, $minutes minutes, and $seconds seconds"

# 启动占卡进程
cd /apdcephfs_qy3/share_300998916/joyyyhuang/H20_occupied
sh run_mnist.sh
