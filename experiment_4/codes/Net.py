import torch
import torch.nn as nn
import math

class TransformerEmbedding(nn.Module):
    """
    Transformer 嵌入层：
    Token Embedding + Segment Embedding + Position Embedding
    """
    def __init__(self, vocab_size, model_dim, max_len=64, dropout=0.1):
        """
        :param vocab_size: 词表大小
        :param model_dim: 嵌入向量维度 (例如 128, 256, 768)
        :param max_len: 序列最大长度
        :param dropout: Dropout 比率
        """
        super(TransformerEmbedding, self).__init__()

        # Token Embedding: 将字符 ID 映射为词向量
        self.token_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)

        # Segment Embedding: 区分句子1 (0) 和句子2 (1)
        self.segment_embedding = nn.Embedding(2, model_dim)

        # Position Embedding: 可学习的位置编码
        self.position_embedding = nn.Embedding(max_len, model_dim)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(model_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids):
        """
        前向传播
        :param input_ids: [batch_size, seq_len] 输入的 token IDs
        :param token_type_ids: [batch_size, seq_len] 分段 IDs (0 或 1)
        :return: [batch_size, seq_len, model_dim] 嵌入向量
        """
        batch_size, seq_len = input_ids.size()

        # 生成位置索引 [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)  # [batch_size, seq_len]

        # 计算三种嵌入
        token_embed = self.token_embedding(input_ids)      # [batch_size, seq_len, model_dim]
        segment_embed = self.segment_embedding(token_type_ids)  # [batch_size, seq_len, model_dim]
        position_embed = self.position_embedding(position_ids)  # [batch_size, seq_len, model_dim]

        # 三者相加
        embeddings = token_embed + segment_embed + position_embed

        # LayerNorm + Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, model_dim, num_heads, dropout=0.1):
        """
        :param model_dim: 模型维度
        :param num_heads: 注意力头数
        :param dropout: Dropout 比率
        """
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "model_dim 必须能被 num_heads 整除"

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads  # 每个头的维度

        # Q, K, V 的线性变换
        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)

        # 输出的线性变换
        self.W_o = nn.Linear(model_dim, model_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)  # 缩放因子

    def forward(self, x, attention_mask=None):
        """
        前向传播
        :param x: [batch_size, seq_len, model_dim] 输入向量
        :param attention_mask: [batch_size, seq_len] Padding mask，1 表示有效位置，0 表示填充位置
        :return: [batch_size, seq_len, model_dim] 输出向量, attention_weights
        """
        batch_size, seq_len, _ = x.size()

        # 1. 线性变换得到 Q, K, V
        Q = self.W_q(x)  # [batch_size, seq_len, model_dim]
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. 拆分成多头 [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数 (Scaled Dot-Product Attention)
        # scores: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 4. 应用 Padding Mask（将填充位置的注意力分数设为负无穷）
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # 5. Softmax 得到注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 6. 加权求和
        # context: [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attention_weights, V)

        # 7. 拼接多头 [batch_size, seq_len, model_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)

        # 8. 最后的线性变换
        output = self.W_o(context)

        return output, attention_weights


class AddNorm(nn.Module):
    """
    残差连接 + 层归一化
    """
    def __init__(self, model_dim, dropout=0.1):
        """
        :param model_dim: 模型维度
        :param dropout: Dropout 比率
        """
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        """
        前向传播
        :param x: 残差连接的输入
        :param sublayer_output: 子层的输出
        :return: LayerNorm(x + Dropout(sublayer_output))
        """
        return self.layer_norm(x + self.dropout(sublayer_output))


class FeedForward(nn.Module):
    """
    前馈神经网络（两层全连接 + 激活函数）
    """
    def __init__(self, model_dim, ffn_dim, dropout=0.1):
        """
        :param model_dim: 模型维度
        :param ffn_dim: 前馈网络的隐藏层维度（通常是 model_dim 的 4 倍）
        :param dropout: Dropout 比率
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用 GELU 激活函数

    def forward(self, x):
        """
        前向传播
        :param x: [batch_size, seq_len, model_dim]
        :return: [batch_size, seq_len, model_dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层
    包含：多头自注意力 + Add&Norm + 前馈网络 + Add&Norm
    """
    def __init__(self, model_dim, num_heads, ffn_dim, dropout=0.1):
        """
        :param model_dim: 模型维度
        :param num_heads: 注意力头数
        :param ffn_dim: 前馈网络隐藏层维度
        :param dropout: Dropout 比率
        """
        super(TransformerEncoderLayer, self).__init__()

        # 多头自注意力
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.add_norm1 = AddNorm(model_dim, dropout)

        # 前馈网络
        self.feed_forward = FeedForward(model_dim, ffn_dim, dropout)
        self.add_norm2 = AddNorm(model_dim, dropout)

    def forward(self, x, attention_mask=None):
        """
        前向传播
        :param x: [batch_size, seq_len, model_dim] 输入向量
        :param attention_mask: [batch_size, seq_len] Padding mask
        :return: [batch_size, seq_len, model_dim] 输出向量, attention_weights
        """
        # 多头自注意力 + Add&Norm
        attn_output, attention_weights = self.attention(x, attention_mask)
        x = self.add_norm1(x, attn_output)

        # 前馈网络 + Add&Norm
        ffn_output = self.feed_forward(x)
        x = self.add_norm2(x, ffn_output)

        return x, attention_weights


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器（堆叠多层 TransformerEncoderLayer）
    """
    def __init__(self, num_layers, model_dim, num_heads, ffn_dim, dropout=0.1):
        """
        :param num_layers: 编码器层数
        :param model_dim: 模型维度
        :param num_heads: 注意力头数
        :param ffn_dim: 前馈网络隐藏层维度
        :param dropout: Dropout 比率
        """
        super(TransformerEncoder, self).__init__()

        # 堆叠多层 Encoder Layer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, attention_mask=None):
        """
        前向传播
        :param x: [batch_size, seq_len, model_dim] 输入向量
        :param attention_mask: [batch_size, seq_len] Padding mask
        :return: [batch_size, seq_len, model_dim] 输出向量, all_attention_weights
        """
        all_attention_weights = []

        for layer in self.layers:
            x, attention_weights = layer(x, attention_mask)
            all_attention_weights.append(attention_weights)

        return x, all_attention_weights


class SemanticMatchingModel(nn.Module):
    """
    语义匹配模型
    完整架构：Embedding + Transformer Encoder + Classifier
    """
    def __init__(self, vocab_size, model_dim=128, num_layers=2, num_heads=8,
                 ffn_dim=512, max_len=64, num_classes=2, dropout=0.1):
        """
        :param vocab_size: 词表大小
        :param model_dim: 模型维度
        :param num_layers: Transformer 编码器层数
        :param num_heads: 注意力头数
        :param ffn_dim: 前馈网络隐藏层维度
        :param max_len: 序列最大长度
        :param num_classes: 分类类别数（2：相似/不相似）
        :param dropout: Dropout 比率
        """
        super(SemanticMatchingModel, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size, model_dim, max_len, dropout)

        self.encoder = TransformerEncoder(num_layers, model_dim, num_heads, ffn_dim, dropout)

        self.classifier = nn.Linear(model_dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        前向传播
        :param input_ids: [batch_size, seq_len] Token IDs
        :param token_type_ids: [batch_size, seq_len] Segment IDs
        :param attention_mask: [batch_size, seq_len] Attention mask
        :return: logits [batch_size, num_classes], all_attention_weights
        """
        # 嵌入层
        embeddings = self.embedding(input_ids, token_type_ids)  # [batch_size, seq_len, model_dim]

        # Transformer 编码器
        encoder_output, all_attention_weights = self.encoder(embeddings, attention_mask)  # [batch_size, seq_len, model_dim]

        # 提取 [CLS] 位置的向量（索引为 0）
        cls_output = encoder_output[:, 0, :]  # [batch_size, model_dim]

        cls_output = self.dropout(cls_output)

        # 分类器
        logits = self.classifier(cls_output)  # [batch_size, num_classes]

        return logits, all_attention_weights
