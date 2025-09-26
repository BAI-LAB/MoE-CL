import numpy as np
import pandas as pd


def calculate_lift_rate(data, target_precision=0.9):
    """
    计算剥离率
    :param data: DataFrame包含label, score_0, score_1列
    :param target_precision: 目标精确率阈值
    :return: 标签0和1的剥离率及对应阈值
    """
    results = {}
    
    # 分别处理标签0和1
    for label in [0,1]:
        score_col = f'score_{label}'
        
        # 将预测概率从高到低排序
        sorted_data = data.sort_values(by=score_col, ascending=False)
        
        thresholds = sorted_data[score_col].unique()
        best_threshold = None
        best_lift_rate = 0
        
        for threshold in thresholds:
            # 获取大于阈值的样本
            # threshold = 0.9934408088574697
            predicted_positive = sorted_data[sorted_data[score_col] >= threshold]
            
            if len(predicted_positive) == 0:
                continue
                
            # 计算精确率
            precision = sum(predicted_positive['label'] == label) / len(predicted_positive)
            
            # 如果精确率满足要求，计算剥离率
            if precision >= target_precision:
                lift_rate = len(predicted_positive) / len(data)
                
                if lift_rate > best_lift_rate:
                    best_lift_rate = lift_rate
                    best_threshold = threshold
        
        results[label] = {
            'threshold': best_threshold,
            'lift_rate': best_lift_rate
        }
    
    return results

# 示例使用
def main():
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成示例数据
    task = ["shipinhao", "xiaoshijie", "gongzhongpinglun"]
    for t in task:
        print(f"开始处理任务{t}")
        # data = pd.read_csv(f'/root/MoE-CL/results/pertask-ft/tencent/order1/{t}.csv')
        data = pd.read_csv(f'/root/MoE-CL/results/moe-cl/tencent/order1_weight003/{t}.csv')
        # data = pd.read_csv(f'/root/MoE-CL/results/sequential-ft-p/tencent/order1/{t}.csv')

        # 计算剥离率
        results = calculate_lift_rate(data, target_precision=0.8)
        print(results)
        
        # 打印结果
        for label, result in results.items():
            print(f"标签 {label}:")
            print(f"阈值: {result['threshold']:.4f}")
            print(f"剥离率: {result['lift_rate']:.4f}")
            print()

if __name__ == "__main__":
    main()
