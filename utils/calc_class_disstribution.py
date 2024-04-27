import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('data/train.csv')

# 各分類の画像総数を計算
class_counts = df.sum(axis=0)

# 'image_id'列はカウントから除外
class_counts = class_counts.drop('image_id')

# 結果をテキストファイルに出力
with open('utils/class_counts.txt', 'w') as f:
    for class_name, count in class_counts.items():
        f.write(f"{class_name}: {count}\n")
