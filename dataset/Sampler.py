import torch
from torch.utils.data import WeightedRandomSampler

def get_sampler(dataset):
    # クラスのサンプル数[healthy,multiple_diseases,rust,scab]
    class_counts = [516, 91, 622, 592]
    # 各クラスの重みを計算
    weights = [1.0 / x for x in class_counts]
    class_weights = torch.FloatTensor(weights)

    # サンプルの重みを設定
    sample_weights = [0] * len(dataset)
    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    # WeightedRandomSamplerを作成
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
