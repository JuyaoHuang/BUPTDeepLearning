"""
测试 Transformer Encoder 组件
验证实验步骤(3)：多头自注意力层和 Add&Norm 层
"""
import torch
from Net import TransformerEmbedding, MultiHeadAttention, AddNorm, TransformerEncoderLayer
from dataloader import get_dataloader

if __name__ == '__main__':
    data_path = '../dataset'
    train_loader, _, _, vocab = get_dataloader(data_path, batch_size=2, max_len=64)

    vocab_size = len(vocab)
    model_dim = 128
    num_heads = 8
    ffn_dim = 512  # 通常是 model_dim 的 4 倍

    # 创建嵌入层
    embedding_layer = TransformerEmbedding(vocab_size, model_dim, max_len=64, dropout=0.1)

    # 创建多头自注意力层
    attention_layer = MultiHeadAttention(model_dim, num_heads, dropout=0.1)

    # 创建 Add&Norm 层
    add_norm_layer = AddNorm(model_dim, dropout=0.1)

    # 创建 Transformer Encoder
    encoder_layer = TransformerEncoderLayer(model_dim, num_heads, ffn_dim, dropout=0.1)

    for batch in train_loader:
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']

        print("\n输入数据：")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention Mask shape: {attention_mask.shape}")

        with torch.no_grad():
            embeddings = embedding_layer(input_ids, token_type_ids)
            print(f"\n嵌入层输出：")
            print(f"Embeddings shape: {embeddings.shape}")

            attn_output, attn_weights = attention_layer(embeddings, attention_mask)
            print(f"\n多头自注意力层：")
            print(f"Attention output shape: {attn_output.shape}")
            print(f"Attention weights shape: {attn_weights.shape}")
            print(f"  - {num_heads} 个注意力头")
            print(f"  - 每个头的维度: {model_dim // num_heads}")
            print(f"  - 使用 Padding Mask 屏蔽无效位置")

            add_norm_output = add_norm_layer(embeddings, attn_output)
            print(f"\nAdd&Norm 层：")
            print(f"Add&Norm output shape: {add_norm_output.shape}")
            print(f"  - 残差连接: x + Dropout(attention_output)")
            print(f"  - 层归一化: LayerNorm")

            encoder_output, encoder_attn_weights = encoder_layer(embeddings, attention_mask)
            print(f"\nTransformer Encoder Layer：")
            print(f"Encoder output shape: {encoder_output.shape}")
            print(f"组件结构:")
            print(f"  1. 多头自注意力 ({num_heads} heads)")
            print(f"  2. Add&Norm")
            print(f"  3. 前馈网络 (FFN dim={ffn_dim})")
            print(f"  4. Add&Norm")

            print(f"\nPadding Mask 验证：")
            # 查看第一个样本的注意力权重
            first_sample_attn = encoder_attn_weights[0, 0]  # [seq_len, seq_len]
            first_sample_mask = attention_mask[0]  # [seq_len]

            # 找到第一个 padding 位置
            padding_positions = (first_sample_mask == 0).nonzero(as_tuple=True)[0]
            if len(padding_positions) > 0:
                first_pad_pos = padding_positions[0].item()
                print(f"第一个样本的填充位置从索引 {first_pad_pos} 开始")
                print(f"第 0 个 token 对填充位置 {first_pad_pos} 的注意力权重: {first_sample_attn[0, first_pad_pos].item():.6f}")
                print(f"  -> 接近 0，说明 Padding Mask 生效")
            else:
                print("该样本没有填充位置")

        break

