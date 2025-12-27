import json
import torch
import os
from torch.utils.data import Dataset, DataLoader

# 定义特殊符号
PAD_TOKEN = '[PAD]' # 补全占位符，ID=0
UNK_TOKEN = '[UNK]' # 未知字符，ID=1
CLS_TOKEN = '[CLS]' # 句首分类标记，ID=2
SEP_TOKEN = '[SEP]' # 句子分隔标记，ID=3

def get_num_workers():
    """根据操作系统返回合适的 num_workers 值"""
    return 4


class Vocab:
    """
    词表类：负责将字符转换为 ID，或者将 ID 转换为字符
    """
    def __init__(self, data_path):
        self.str_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1, CLS_TOKEN: 2, SEP_TOKEN: 3}
        self.id_to_str = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: CLS_TOKEN, 3: SEP_TOKEN}
        self.build_vocab(data_path)
    def build_vocab(self, data_path):
        """遍历训练集构建词表"""
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 将句子 1 和 2 的所有字都加入词表
                text = data['sentence1'] + data['sentence2']
                for char in text:
                    if char not in self.str_to_id:
                        idx = len(self.str_to_id)
                        self.str_to_id[char] = idx
                        self.id_to_str[idx] = char

            print(f"词表构建完成，词汇量大小：{len(self.str_to_id)}")

    def __len__(self):
        return len(self.str_to_id)

    def convert_tokens_to_ids(self, tokens):
        """将字符列表转换为 ID 列表"""
        return [self.str_to_id.get(token, self.str_to_id[UNK_TOKEN]) for token in tokens]

class AFQMCDataset(Dataset):
    """ADQMC 语义匹配数据集"""
    def __init__(self, data_path, vocab, max_len=64):
        self.data = []
        self.vocab = vocab
        self.max_len = max_len

        # 加载 JSON 数据
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        s1 = item['sentence1']
        s2 = item['sentence2']
        label = int(item['label']) if 'label' in item else 0

        # 分词
        tokens1 = list(s1)
        tokens2 = list(s2)

        # 构建 BERT 输入格式：[CLS] s1 [SEP] s2 [SEP]
        # 为保证不超过 max_len，需要进行截断
        # 预留 3 个位置给特殊符号
        max_content_len = self.max_len - 3
        if len(tokens1) + len(tokens2) > max_content_len:
            # 字数超限，则各截断一半
            # 先截断 s2 保证 s1 完整，若 s1 也太长则都截
            if len(tokens1) > max_content_len //2:
                tokens1 = tokens1[:max_content_len // 2]
                tokens2 = tokens2[:max_content_len - len(tokens1)]
            else:
                tokens2 = tokens2[:max_content_len - len(tokens1)]

        # 拼接 token
        tokens = [CLS_TOKEN] + tokens1 + [SEP_TOKEN] + tokens2 + [SEP_TOKEN]

        # 转换为 ID ，作为 Token Embedding 输入
        input_ids = self.vocab.convert_tokens_to_ids(tokens)

        #  生成 Segment IDs ，作为 Segment Embedding 输入
        # s1 部分(含CLS和第一个SEP)为 0，s2 部分(含第二个SEP)为 1
        len_s1 = len(tokens1) + 2 # CLS + SEP
        len_s2 = len(tokens2) + 1 # SEP
        token_type_ids = [0]*len_s1 + [1]*len_s2

        # 生成 Attention Mask (实义字符为 1，PAD 为 0)
        attention_mask = [1]*len(input_ids)

        # padding 填充对齐
        cur_len = len(input_ids)
        padding_len = self.max_len - cur_len

        input_ids += [0]*padding_len # 0 是 PAD 的 ID
        token_type_ids += [0]*padding_len # PAD 部分属于段落 0
        attention_mask += [0]*padding_len # PAD 部分不计算注意力

        # 转为 tensor
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_dataloader(data_path, batch_size=64, max_len=64):
    """
    创建训练集、验证集和测试集的 DataLoader
    :param data_path: 数据集根目录
    :param batch_size: 批次大小
    :param max_len: 最大字数
    :return: train_loader, val_loader, test_loader
    """
    train_file = os.path.join(data_path, 'train.json')
    dev_file = os.path.join(data_path, 'dev.json')
    test_file = os.path.join(data_path, 'test.json')

    vocab = Vocab(train_file)

    train_dataset = AFQMCDataset(train_file, vocab, max_len=max_len)
    dev_dataset = AFQMCDataset(dev_file, vocab, max_len=max_len)
    test_dataset = AFQMCDataset(test_file, vocab, max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=get_num_workers(),
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get_num_workers(),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get_num_workers(),
        pin_memory=True
    )

    return train_loader, dev_loader, test_loader, vocab


if __name__ == '__main__':
    data_path = 'dataset'

    # 获取 dataloader
    train_loader, dev_loader, _, vocab = get_dataloader(data_path, batch_size=2)

    print("-" * 20)
    print("实验步骤(1) 验证结果：")

    # 打印一条 mini-batch 数据
    for batch in train_loader:
        print("Input IDs Shape:", batch['input_ids'].shape)
        print("Attention Mask Shape:", batch['attention_mask'].shape)
        print("Labels:", batch['label'])

        print("\n样本 1 解码:")
        ids = batch['input_ids'][0].numpy()
        tokens = [vocab.id_to_str[i] for i in ids]
        print("Token IDs:", ids[:15], "...")
        print("Tokens:", tokens[:15], "...")
        break