"""
注意力权重可视化
实验步骤(7)：提取多头注意力权重，绘制热力图并分析
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dataloader import get_dataloader
from Net import SemanticMatchingModel


def load_model(checkpoint_path, vocab_size, config, device):
    """加载训练好的模型"""
    model = SemanticMatchingModel(
        vocab_size=vocab_size,
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ffn_dim=config['ffn_dim'],
        max_len=config['max_len'],
        num_classes=2,
        dropout=0.0
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def visualize_attention(attention_weights, tokens, layer_idx=0, head_idx=0, save_path='attention_heatmap.png'):
    """
    可视化注意力权重
    :param attention_weights: 注意力权重 [num_heads, seq_len, seq_len]
    :param tokens: token 列表
    :param layer_idx: 第几层
    :param head_idx: 第几个头
    :param save_path: 保存路径
    """
    # 提取指定头的注意力权重
    attn = attention_weights[head_idx].cpu().numpy()  # [seq_len, seq_len]

    # 只显示有效的 token（去除 padding）
    valid_len = len([t for t in tokens if t != '[PAD]'])
    attn = attn[:valid_len, :valid_len]
    tokens = tokens[:valid_len]

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd',
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        vmin=0,
        vmax=attn.max()
    )

    plt.title(f'Attention Heatmap - Layer {layer_idx + 1}, Head {head_idx + 1}', fontsize=14, pad=20)
    plt.xlabel('Key Position', fontsize=12)
    plt.ylabel('Query Position', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"注意力热力图已保存到: {save_path}")
    plt.close()


def analyze_attention(attention_weights, tokens, layer_idx=0):
    """
    分析注意力权重
    :param attention_weights: 所有头的注意力权重 [num_heads, seq_len, seq_len]
    :param tokens: token 列表
    :param layer_idx: 第几层
    """
    num_heads = attention_weights.shape[0]
    valid_len = len([t for t in tokens if t != '[PAD]'])

    print(f"\n【Layer {layer_idx + 1} 注意力分析】")
    print(f"有效 token 数量: {valid_len}")
    print(f"注意力头数: {num_heads}")

    # 找到特殊 token 的位置
    cls_idx = tokens.index('[CLS]') if '[CLS]' in tokens else -1
    sep_indices = [i for i, t in enumerate(tokens) if t == '[SEP]']

    if cls_idx >= 0:
        print(f"\n[CLS] 位置 (索引 {cls_idx}) 的注意力分布:")
        for head_idx in range(min(4, num_heads)):  # 只分析前4个头
            attn = attention_weights[head_idx, cls_idx, :valid_len].cpu().numpy()
            top_k = 5
            top_indices = np.argsort(attn)[-top_k:][::-1]

            print(f"  Head {head_idx + 1} 最关注的 {top_k} 个位置:")
            for rank, idx in enumerate(top_indices, 1):
                print(f"    {rank}. 位置 {idx} ({tokens[idx]}): {attn[idx]:.4f}")

    if len(sep_indices) >= 2:
        print(f"\n句子边界分析:")
        print(f"  第一个 [SEP] 位置: {sep_indices[0]}")
        print(f"  第二个 [SEP] 位置: {sep_indices[1]}")
        print(f"  句子1 长度: {sep_indices[0] - 1}")
        print(f"  句子2 长度: {sep_indices[1] - sep_indices[0] - 1}")


def visualize_sample(model, sample, vocab, device, save_prefix='attention'):
    """
    对单个样本进行注意力可视化
    """
    model.eval()

    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    token_type_ids = sample['token_type_ids'].unsqueeze(0).to(device)
    attention_mask = sample['attention_mask'].unsqueeze(0).to(device)

    # 前向传播，获取注意力权重
    with torch.no_grad():
        logits, all_attention_weights = model(input_ids, token_type_ids, attention_mask)

    # 解码 tokens
    tokens = [vocab.id_to_str.get(i, '[UNK]') for i in input_ids[0].cpu().numpy()]

    predicted = logits.argmax(dim=1).item()
    true_label = sample['label'].item()

    print("\n" + "=" * 80)
    print("样本信息:")
    print(f"  Tokens: {' '.join(tokens[:30])}...")
    print(f"  真实标签: {true_label} ({'相似' if true_label == 1 else '不相似'})")
    print(f"  预测标签: {predicted} ({'相似' if predicted == 1 else '不相似'})")
    print(f"  预测正确: {'✓' if predicted == true_label else '✗'}")

    # 可视化每一层的注意力
    # num_layers = len(all_attention_weights)
    # for layer_idx in range(num_layers):
    #     attn = all_attention_weights[layer_idx][0]  # [num_heads, seq_len, seq_len]
    #
    #     # 分析注意力
    #     analyze_attention(attn, tokens, layer_idx)
    #
    #     # 绘制多个头的热力图
    #     for head_idx in range(min(4, attn.shape[0])):  # 最多绘制前4个头
    #         save_path = f'{save_prefix}_layer{layer_idx + 1}_head{head_idx + 1}.png'
    #         visualize_attention(attn, tokens, layer_idx, head_idx, save_path)

    # 绘制单图
    layer_idx = 0
    attn = all_attention_weights[layer_idx][0]  # [num_heads, seq_len, seq_len]

    analyze_attention(attn, tokens, layer_idx)

    # 绘制单头的热力图
    save_path = f'{save_prefix}_layer{layer_idx + 1}_head{0 + 1}.png'
    visualize_attention(attn, tokens, layer_idx, 0, save_path)

    print("\n" + "=" * 80)


if __name__ == '__main__':
    config = {
        'data_path': 'dataset',
        'batch_size': 64,
        'max_len': 64,
        'model_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'ffn_dim': 1024,
        'checkpoint_path': 'best_model_6layers_v2.pth'
    }

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    _, val_loader, _, vocab = get_dataloader(
        data_path=config['data_path'],
        batch_size=config['batch_size'],
        max_len=config['max_len']
    )
    val_dataset = val_loader.dataset

    model = load_model(config['checkpoint_path'], len(vocab), config, device)
    print(f"模型加载成功")

    # 选择一个样本进行可视化（可以修改索引选择不同样本）
    sample_idx = 10
    sample = val_dataset[sample_idx]

    print(f"\n对样本 {sample_idx} 进行注意力可视化...")
    visualize_sample(model, sample, vocab, device, save_prefix=f'attention_sample{sample_idx}')

    print("\n可视化完成!")
