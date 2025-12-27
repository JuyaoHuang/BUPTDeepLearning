"""
层数对比实验
实验步骤(8)：改变 Transformer 层数，对比不同深度模型的性能
"""
import os
import torch
from train import train


def compare_layers():
    """对比不同层数的 Transformer 模型"""

    # 测试不同的层数配置
    layer_configs = [1, 2, 4]

    results = []

    print("=" * 80)
    print("Transformer 层数对比实验")
    print("=" * 80)

    for num_layers in layer_configs:
        print(f"\n{'=' * 80}")
        print(f"训练 {num_layers} 层 Transformer 模型")
        print(f"{'=' * 80}\n")

        # 配置
        config = {
            'data_path': 'dataset',
            'batch_size': 64,
            'max_len': 64,
            'epochs': 30,
            'lr': 1e-3,
            'weight_decay': 0.01,
            'model_dim': 128,
            'num_layers': num_layers,  # 改变层数
            'num_heads': 8,
            'ffn_dim': 512,
            'dropout': 0.1,
            'log_dir': f'runs/semantic_matching_{num_layers}layers',
            'save_path': f'best_model_{num_layers}layers.pth',
            'early_stop_patience': 10
        }

        # 创建日志目录
        os.makedirs(config['log_dir'], exist_ok=True)

        # 训练
        try:
            model = train(config)

            # 加载最佳模型并记录结果
            checkpoint = torch.load(config['save_path'])
            best_acc = checkpoint['val_acc']

            results.append({
                'num_layers': num_layers,
                'best_val_acc': best_acc,
                'final_epoch': checkpoint['epoch'] + 1,
                'save_path': config['save_path']
            })

            print(f"\n{num_layers} 层模型训练完成")
            print(f"  最佳验证准确率: {best_acc:.2f}%")
            print(f"  训练轮次: {checkpoint['epoch'] + 1}")

        except Exception as e:
            print(f"\n{num_layers} 层模型训练失败: {e}")
            results.append({
                'num_layers': num_layers,
                'best_val_acc': 0.0,
                'final_epoch': 0,
                'error': str(e)
            })

    # 打印对比结果
    print("\n" + "=" * 80)
    print("实验结果对比")
    print("=" * 80)
    print(f"{'层数':<10} {'最佳验证准确率':<20} {'训练轮次':<15} {'模型文件':<30}")
    print("-" * 80)

    for result in results:
        if 'error' in result:
            print(f"{result['num_layers']:<10} {'训练失败':<20} {'-':<15} {'-':<30}")
        else:
            print(f"{result['num_layers']:<10} {result['best_val_acc']:.2f}%{'':<14} "
                  f"{result['final_epoch']:<15} {result['save_path']:<30}")

    # 找出最佳配置
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['best_val_acc'])
        print("\n" + "=" * 80)
        print(f"最佳配置: {best_result['num_layers']} 层")
        print(f"最佳验证准确率: {best_result['best_val_acc']:.2f}%")
        print("=" * 80)

    # 保存结果到文件
    with open('layer_comparison_results.txt', 'w', encoding='utf-8') as f:
        f.write("Transformer 层数对比实验结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'层数':<10} {'最佳验证准确率':<20} {'训练轮次':<15}\n")
        f.write("-" * 80 + "\n")
        for result in results:
            if 'error' not in result:
                f.write(f"{result['num_layers']:<10} {result['best_val_acc']:.2f}%{'':<14} "
                        f"{result['final_epoch']:<15}\n")
        if valid_results:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"最佳配置: {best_result['num_layers']} 层\n")
            f.write(f"最佳验证准确率: {best_result['best_val_acc']:.2f}%\n")

    print("\n对比结果已保存到: layer_comparison_results.txt")

    return results


if __name__ == '__main__':
    compare_layers()
