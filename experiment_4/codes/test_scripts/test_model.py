"""
测试完整的语义匹配模型
验证实验步骤(4)：完整模型架构
"""
import torch
from Net import SemanticMatchingModel
from dataloader import get_dataloader

if __name__ == '__main__':
    data_path = '../dataset'
    train_loader, _, _, vocab = get_dataloader(data_path, batch_size=2, max_len=64)

    # 模型配置
    vocab_size = len(vocab)
    model_dim = 128
    num_layers = 2
    num_heads = 8
    ffn_dim = 512
    max_len = 64
    num_classes = 2


    # 创建完整模型
    model = SemanticMatchingModel(
        vocab_size=vocab_size,
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        max_len=max_len,
        num_classes=num_classes,
        dropout=0.1
    )

    print("\n模型架构：")
    print(f"词表大小: {vocab_size}")
    print(f"模型维度: {model_dim}")
    print(f"编码器层数: {num_layers}")
    print(f"注意力头数: {num_heads}")
    print(f"前馈网络维度: {ffn_dim}")
    print(f"输出类别数: {num_classes}")

    # 测试前向传播
    print("\n前向传播测试")
    for batch in train_loader:
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        print(f"\n输入数据:")
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Token Type IDs shape: {token_type_ids.shape}")
        print(f"  Attention Mask shape: {attention_mask.shape}")
        print(f"  Labels: {labels}")

        with torch.no_grad():
            logits, all_attention_weights = model(input_ids, token_type_ids, attention_mask)

        print(f"\n输出结果:")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits (第1个样本): {logits[0]}")
        print(f"  预测类别: {torch.argmax(logits, dim=-1)}")
        print(f"  真实类别: {labels}")
        print(f"  注意力权重层数: {len(all_attention_weights)} 层")
        print(f"  每层注意力权重 shape: {all_attention_weights[0].shape}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型参数统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

        break
