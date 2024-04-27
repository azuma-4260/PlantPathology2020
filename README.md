# Plant Pathology 2020 コンペティション

## プロジェクトの概要
このリポジトリは、Kaggleのコンペティション「[Plant Pathology 2020](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/overview)」に関するコードを含む。このコンペティションでは、植物の病気を画像から識別するタスクに取り組む。

## データの準備
データはコンペティションのページからダウンロードする。ダウンロードしたデータは以下のディレクトリ構造で配置する。

```markdown
data/
  - images/
    - Train_x.jpg
    - Test_x.jpg
  - train.csv
  - test.csv
```

## 必要なライブラリのインストール

### 仮想環境の作成
```bash
conda create -n pp2020 python=3.10
conda activate pp2020
```

### CUDA Toolkit 11.8 のインストール

```bash
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
```

### PyTorch と関連ライブラリのインストール

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Pandas と Matplotlib のインストール

```bash
pip install pandas matplotlib scikit-learn seaborn
```

## データの分割
Train:Val:Test = 16:4:5になるように分割した

## モデルの評価
- 正解率: 0.8110
- 適合率: 0.6688
- 再現率: 0.6606
- F1値: 0.6474
- 混同行列: 
