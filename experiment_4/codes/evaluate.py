"""
评估语义匹配模型
实验步骤(6)：在验证集上随机选取50条数据进行测试
"""
import torch
import random
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
        dropout=0.0  # 测试时不使用 dropout
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"成功加载模型: {checkpoint_path}")
    print(f"训练轮次: {checkpoint['epoch'] + 1}")
    print(f"验证准确率: {checkpoint['val_acc']:.2f}%")

    return model


def evaluate_full(model, val_loader, device, vocab):
    """在完整验证集上评估"""
    model.eval()
    correct = 0
    total = 0

    print("\n在完整验证集上评估...")
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits, _ = model(input_ids, token_type_ids, attention_mask)
            _, predicted = logits.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"验证集准确率: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def test_random_samples(model, val_dataset, device, vocab, num_samples=50):
    """随机抽取样本进行测试"""
    model.eval()

    # 随机选择样本
    total_samples = len(val_dataset)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))

    print(f"\n随机测试 {len(indices)} 条样本:")
    print("=" * 100)

    correct = 0
    results = []

    with torch.no_grad():
        for idx in indices:
            sample = val_dataset[idx]

            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            token_type_ids = sample['token_type_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            label = sample['label'].item()

            logits, _ = model(input_ids, token_type_ids, attention_mask)
            predicted = logits.argmax(dim=1).item()

            # 解码句子
            tokens = [vocab.id_to_str.get(i, '[UNK]') for i in input_ids[0].cpu().numpy()]
            # 找到两个句子的分隔位置
            sep_indices = [i for i, token in enumerate(tokens) if token == '[SEP]']

            if len(sep_indices) >= 2:
                sentence1_tokens = tokens[1:sep_indices[0]]  # 跳过 [CLS]
                sentence2_tokens = tokens[sep_indices[0] + 1:sep_indices[1]]
                sentence1 = ''.join(sentence1_tokens)
                sentence2 = ''.join(sentence2_tokens)
            else:
                sentence1 = ''.join(tokens[1:10])
                sentence2 = '未能解析'

            # 记录结果
            is_correct = (predicted == label)
            if is_correct:
                correct += 1

            result = {
                'idx': idx,
                'sentence1': sentence1,
                'sentence2': sentence2,
                'true_label': label,
                'pred_label': predicted,
                'correct': is_correct
            }
            results.append(result)

    # 打印结果
    for i, result in enumerate(results[:50], 1):
        status = "✓" if result['correct'] else "✗"
        label_text = {0: "不相似", 1: "相似"}

        print(f"\n样本 {i} (ID: {result['idx']}) {status}")
        print(f"  句子1: {result['sentence1']}")
        print(f"  句子2: {result['sentence2']}")
        print(f"  真实标签: {label_text[result['true_label']]}")
        print(f"  预测标签: {label_text[result['pred_label']]}")

    print("\n" + "=" * 100)
    accuracy = 100.0 * correct / len(results)
    print(f"随机样本准确率: {accuracy:.2f}% ({correct}/{len(results)})")

    return results, accuracy


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    _, val_loader, _, vocab = get_dataloader(
        data_path=config['data_path'],
        batch_size=config['batch_size'],
        max_len=config['max_len']
    )
    val_dataset = val_loader.dataset

    print(f"验证集大小: {len(val_dataset)}")

    model = load_model(config['checkpoint_path'], len(vocab), config, device)

    full_acc = evaluate_full(model, val_loader, device, vocab)

    results, sample_acc = test_random_samples(model, val_dataset, device, vocab, num_samples=50)

    print("\n评估完成!")
