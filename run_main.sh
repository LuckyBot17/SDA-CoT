#!/bin/bash

# 创建结果目录（如果不存在）
mkdir -p results

# 获取当前时间作为文件名
timestamp=$(date +"%m%d_%H%M%S")
#模型名称
model_name="deepseek-r1:32b"
dataset="csqa"
output_file="results/sda_cot_results_${timestamp}_${model_name}_${csqa}.txt"

# 运行main.py并将输出重定向到文件
echo "开始运行SDA-CoT，结果将保存到 ${output_file}"
python main.py > "${output_file}" 2>&1

# 运行完成后显示提示
echo "运行完成，结果已保存到 ${output_file}"
