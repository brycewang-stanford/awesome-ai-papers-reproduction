#!/bin/bash

# 查看xPatch实验结果脚本

echo "=========================================="
echo "xPatch实验结果汇总"
echo "=========================================="

# 切换到项目根目录
cd "$(dirname "$0")/.."

echo ""
echo "📊 Exchange Rate数据集结果:"
echo "----------------------------------------"
if [ -d "logs/ema" ]; then
    echo "EMA模式结果:"
    for file in logs/ema/xPatch_exchange_96_*.log; do
        if [ -f "$file" ]; then
            pred_len=$(echo $file | grep -o '_[0-9]\+\.log' | grep -o '[0-9]\+')
            result=$(grep "mse:" "$file" 2>/dev/null || echo "未完成")
            echo "  pred_len=$pred_len: $result"
        fi
    done
else
    echo "未找到EMA模式结果"
fi

echo ""
if [ -d "logs/reg" ]; then
    echo "REG模式结果:"
    for file in logs/reg/xPatch_exchange_96_*.log; do
        if [ -f "$file" ]; then
            pred_len=$(echo $file | grep -o '_[0-9]\+\.log' | grep -o '[0-9]\+')
            result=$(grep "mse:" "$file" 2>/dev/null || echo "未完成")
            echo "  pred_len=$pred_len: $result"
        fi
    done
else
    echo "未找到REG模式结果"
fi

echo ""
echo "📁 所有可用的日志文件:"
echo "----------------------------------------"
find logs -name "*.log" -type f | sort

echo ""
echo "=========================================="
echo "如需查看详细训练过程，使用:"
echo "cat logs/ema/文件名.log"
echo "==========================================" 