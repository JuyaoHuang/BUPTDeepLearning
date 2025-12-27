"""
测试 TransformerEmbedding 层
验证实验步骤(2)：打印嵌入层的输入输出
"""
import torch
from Net import TransformerEmbedding
from dataloader import get_dataloader

if __name__ == '__main__':
    data_path = '../dataset'
    train_loader, _, _, vocab = get_dataloader(data_path, batch_size=2, max_len=64)

    vocab_size = len(vocab)
    model_dim = 128
    embedding_layer = TransformerEmbedding(
        vocab_size=vocab_size,
        model_dim=model_dim,
        max_len=64,
        dropout=0.1
    )
    for batch in train_loader:
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']

        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Token Type IDs shape: {token_type_ids.shape}")
        print(f"\nInput IDs (第1个样本前15个token):\n{input_ids[0][:15]}")
        print(f"\nToken Type IDs (第1个样本前15个token):\n{token_type_ids[0][:15]}")

        with torch.no_grad():
            embeddings = embedding_layer(input_ids, token_type_ids)

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"预期形状: [batch_size={input_ids.size(0)}, seq_len={input_ids.size(1)}, model_dim={model_dim}]")
        print(f"\nEmbeddings (第1个样本, 第1个token的前10维):\n{embeddings[0, 0, :10]}")
        break
