# Plant Pathology 2020 コンペティション

## プロジェクトの概要
このリポジトリは、Kaggleのコンペティション「[Plant Pathology 2020](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/overview)」に関するコードを含んでいます。このコンペティションでは、植物の病気を画像から識別するタスクに取り組みます。

## データの準備
データはコンペティションのページからダウンロードしてください。ダウンロードしたデータは以下のディレクトリ構造で配置してください。

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
