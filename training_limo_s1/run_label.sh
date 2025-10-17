#!/bin/bash

# 区间列表
starts=(0 25 50 75 100 125 150 175)
ends=(25 50 75 100 125 150 175 200)

# GPU id
gpus=(0 1 2 3 4 5 6 7)

for i in $(seq 0 7); do
    s=${starts[$i]}
    e=${ends[$i]}
    g=${gpus[$i]}
    log="labeling_${s}_${e}.log"

    echo "Launching range $s-$e on GPU $g, log: $log"
    CUDA_VISIBLE_DEVICES=$g nohup python labeling.py \
        --start $s --end $e --gpu $g > $log 2>&1 &
done

wait   # 可选：等所有子进程结束再退出脚本
