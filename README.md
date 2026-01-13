# PointNet / PointNet++ による点群分割の比較実装

[English](README_EN.md) | 日本語

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](https://pytorch.org/)

## 📋 目次

- [🎯 概要](#-概要)
- [🔍 背景・動機](#-背景動機)
- [🔬 手法](#-手法)
- [💻 実装環境](#-実装環境)
- [🚀 インストール](#-インストール)
- [📦 データセット準備](#-データセット準備)
- [📁 プロジェクト構造](#-プロジェクト構造)
- [🎮 使用方法](#-使用方法)
- [📊 実験結果](#-実験結果)
- [🎨 可視化](#-可視化)
- [🔮 今後の課題](#-今後の課題)
- [📚 参考文献](#-参考文献)
- [📄 ライセンス](#-ライセンス)
- [🙏 謝辞](#-謝辞)

---

## 🎯 概要

本リポジトリでは，点群深層学習の代表的手法である **PointNet** および **PointNet++** を対象とし，点群分割タスクにおける両モデルの実装および性能比較を行います。

### PointNet
PointNet は，点群を直接入力として扱う革新的なモデルであり，point-wise MLP とグローバルプーリングにより点群全体の特徴を学習する，点群深層学習分野の基礎となる手法です。一方で，局所的な幾何構造の表現には課題があります。

**詳細解説**: [アルゴリズム詳細解説（PointNet）](https://maviccc123456-del.github.io/pointnet/pointnet_explain/)（筆者執筆）

### PointNet++
PointNet++ は，上記の課題を解決するため，サンプリングおよびグルーピングによる階層的な局所特徴学習を導入したモデルです。これにより，局所領域ごとの幾何構造を考慮した特徴表現が可能となっています。

**詳細解説**: [アルゴリズム詳細解説（PointNet++）](YOUR_POINTNETPP_LINK)（筆者執筆）

本プロジェクトでは，両手法を同一条件下で実装・評価し，点群分割における性能および特徴表現の違いを実験的に検証します。

---

## 🔍 背景・動機

### PointNet の特徴と課題
- ✅ Point-wise MLP により各点を独立に処理するシンプルな構造
- ✅ 点の順序に依存しない permutation invariance を実現
- ❌ グローバルプーリングに強く依存するため，局所構造情報の表現が限定的

### PointNet++ の改善点
- ✅ Farthest Point Sampling (FPS) による効率的な代表点選択
- ✅ Ball Query / kNN による局所領域の構成
- ✅ Set Abstraction による階層的特徴学習

### 本プロジェクトの目的
**局所構造のモデリング能力の違いが点群分割性能に与える影響を明らかにする**

---

## 🔬 手法

### PointNet

```
Input Points → Point-wise MLP → Max Pooling → Global Feature
                     ↓                              ↓
              Local Features ← Concatenate ← Global Feature
                     ↓
              Point-wise MLP → Segmentation Labels
```

**特徴**
- ✅ 点の順序に依存しない
- ✅ 実装がシンプルで高速
- ❌ 局所構造の表現能力が限定的

**アーキテクチャ詳細**
- Input: (B, N, 3) - Batch size, Number of points, XYZ coordinates
- Feature extraction: MLP(64, 128, 1024)
- Segmentation head: MLP(512, 256, num_classes)

---

### PointNet++

```
Input Points → FPS Sampling → Ball Query / kNN → PointNet (SA)
                    ↓                                    ↓
            Layer 1 Features                    Layer 2 Features
                    ↓                                    ↓
            Feature Propagation ← Interpolation ← Higher Layer
                    ↓
         Point-wise MLP → Segmentation Labels
```

**特徴**
- ✅ 局所的な幾何構造を考慮可能
- ✅ 階層的特徴学習により高い表現力
- ⚠️ 計算コストが PointNet より高い

**アーキテクチャ詳細**
- Set Abstraction Layers: 3 layers
- Sampling ratio: [512, 128, None]
- Radius: [0.2, 0.4, None]
- Feature Propagation: 3 layers with skip connections

---

## 💻 実装環境

### System
- **OS**: Windows 10
- **Python**: 3.10.11

### Hardware
- **GPU**: NVIDIA GeForce RTX 4070 SUPER
- **GPU Memory**: 12 GB
- **CUDA Cores**: 7,168

### Deep Learning Framework
- **PyTorch**: 2.5.1（CUDA 12.1 対応）
- **CUDA (PyTorch)**: 12.1
- **CUDA (Driver)**: 12.7
- **cuDNN**: 9.x

### NVIDIA Driver
- **Driver Version**: 566.03

---

## 🚀 インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/YOUR_USERNAME/pointnet-comparison.git
cd pointnet-comparison
```

### 2. 仮想環境の作成（推奨）

```bash
# Anaconda を使用する場合
conda create -n pointnet python=3.10
conda activate pointnet

# または venv を使用する場合
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### requirements.txt の内容

```txt
torch==2.5.1
torchvision==0.20.1
numpy==1.24.3
matplotlib==3.7.1
tqdm==4.65.0
h5py==3.9.0
scikit-learn==1.3.0
tensorboard==2.14.0
open3d==0.17.0  # 可視化用（オプション）
```

---

## 📦 データセット準備

### ShapeNet Part Segmentation Dataset

1. **データセットのダウンロード**

```bash
# 自動ダウンロードスクリプトを実行
python download_data.py

# または手動でダウンロード
# https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```

2. **データセット構造**

```
data/
└── shapenetcore_partanno_segmentation_benchmark_v0_normal/
    ├── 02691156/  # Airplane
    ├── 02773838/  # Bag
    ├── 02954340/  # Cap
    └── ...
    ├── synsetoffset2category.txt
    └── train_test_split/
```

3. **データセット情報**
- **カテゴリ数**: 16
- **パーツ総数**: 50
- **点群数**: 約 16,000
- **各点群のポイント数**: 2,048

---

## 📁 プロジェクト構造

```
pointnet-comparison/
├── data/                          # データセット
│   └── shapenetcore_partanno_segmentation_benchmark_v0_normal/
├── models/                        # モデル定義
│   ├── __init__.py
│   ├── pointnet.py               # PointNet 実装
│   ├── pointnet2.py              # PointNet++ 実装
│   └── layers.py                 # 共通レイヤー
├── utils/                         # ユーティリティ
│   ├── __init__.py
│   ├── dataset.py                # データローダー
│   ├── metrics.py                # 評価指標
│   └── visualization.py          # 可視化ツール
├── checkpoints/                   # 学習済みモデル
│   ├── pointnet_best.pth
│   └── pointnet2_best.pth
├── logs/                          # TensorBoard ログ
│   ├── pointnet/
│   └── pointnet2/
├── results/                       # 実験結果
│   ├── images/                   # 可視化画像
│   └── metrics/                  # 評価結果
├── train_pointnet.py             # PointNet 学習スクリプト
├── train_pointnet2.py            # PointNet++ 学習スクリプト
├── test.py                       # 評価スクリプト
├── visualize.py                  # 可視化スクリプト
├── download_data.py              # データダウンロード
├── requirements.txt              # 依存パッケージ
├── config.yaml                   # 設定ファイル
└── README.md                     # 本ファイル
```

---

## 🎮 使用方法

### 学習

#### PointNet の学習

```bash
python train_pointnet.py \
    --batch_size 32 \
    --epochs 200 \
    --lr 0.001 \
    --num_points 2048 \
    --use_normals \
    --log_dir logs/pointnet
```

#### PointNet++ の学習

```bash
python train_pointnet2.py \
    --batch_size 16 \
    --epochs 200 \
    --lr 0.001 \
    --num_points 2048 \
    --use_normals \
    --log_dir logs/pointnet2
```

### 主要なハイパーパラメータ

| パラメータ | PointNet | PointNet++ | 説明 |
|----------|----------|------------|------|
| `batch_size` | 32 | 16 | バッチサイズ |
| `learning_rate` | 0.001 | 0.001 | 学習率 |
| `optimizer` | Adam | Adam | 最適化手法 |
| `weight_decay` | 0.0001 | 0.0001 | 重み減衰 |
| `lr_scheduler` | StepLR | StepLR | 学習率スケジューラ |
| `step_size` | 20 | 20 | LR減衰ステップ |
| `gamma` | 0.7 | 0.7 | LR減衰率 |

### 評価

```bash
# PointNet の評価
python test.py \
    --model pointnet \
    --checkpoint checkpoints/pointnet_best.pth \
    --num_points 2048

# PointNet++ の評価
python test.py \
    --model pointnet2 \
    --checkpoint checkpoints/pointnet2_best.pth \
    --num_points 2048
```

### 可視化

```bash
# 分割結果の可視化
python visualize.py \
    --model pointnet2 \
    --checkpoint checkpoints/pointnet2_best.pth \
    --sample_idx 100 \
    --output results/images/
```

### TensorBoard での学習監視

```bash
tensorboard --logdir logs/
```

ブラウザで `http://localhost:6006` を開く

---

## 📊 実験結果

### 実験設定

- **Task**: Part Segmentation
- **Dataset**: ShapeNet Part Segmentation
- **Number of points**: 2,048
- **Training epochs**: 200
- **Data augmentation**: Random rotation, jittering, scaling

### 定量評価

| Model | Overall Accuracy | Class mIoU | Instance mIoU | Parameters | Training Time |
|-------|-----------------|------------|---------------|------------|---------------|
| PointNet | XX.XX % | XX.XX % | XX.XX % | 3.5M | ~X hours |
| PointNet++ (SSG) | XX.XX % | XX.XX % | XX.XX % | 1.5M | ~X hours |
| PointNet++ (MSG) | XX.XX % | XX.XX % | XX.XX % | 1.7M | ~X hours |

### カテゴリ別 IoU

| Category | PointNet | PointNet++ (SSG) | PointNet++ (MSG) |
|----------|----------|------------------|------------------|
| Airplane | XX.XX | XX.XX | XX.XX |
| Bag | XX.XX | XX.XX | XX.XX |
| Cap | XX.XX | XX.XX | XX.XX |
| Car | XX.XX | XX.XX | XX.XX |
| Chair | XX.XX | XX.XX | XX.XX |
| Earphone | XX.XX | XX.XX | XX.XX |
| Guitar | XX.XX | XX.XX | XX.XX |
| Knife | XX.XX | XX.XX | XX.XX |
| Lamp | XX.XX | XX.XX | XX.XX |
| Laptop | XX.XX | XX.XX | XX.XX |
| Motorbike | XX.XX | XX.XX | XX.XX |
| Mug | XX.XX | XX.XX | XX.XX |
| Pistol | XX.XX | XX.XX | XX.XX |
| Rocket | XX.XX | XX.XX | XX.XX |
| Skateboard | XX.XX | XX.XX | XX.XX |
| Table | XX.XX | XX.XX | XX.XX |

### 学習曲線

![Training Curves](results/images/training_curves.png)

*学習損失とバリデーション精度の推移*

---

## 🎨 可視化

### 分割結果の比較

![Segmentation Comparison](results/images/segmentation_comparison.png)

*左: 入力点群, 中央: PointNet 予測, 右: PointNet++ 予測*

### カテゴリ別の分割例

| Category | Input | Ground Truth | PointNet | PointNet++ |
|----------|-------|--------------|----------|------------|
| Airplane | ![](results/images/airplane_input.png) | ![](results/images/airplane_gt.png) | ![](results/images/airplane_pn.png) | ![](results/images/airplane_pn2.png) |
| Chair | ![](results/images/chair_input.png) | ![](results/images/chair_gt.png) | ![](results/images/chair_pn.png) | ![](results/images/chair_pn2.png) |

### 混同行列

![Confusion Matrix](results/images/confusion_matrix.png)

---

## 💡 考察

### PointNet++ の優位性
- ✅ 局所構造を考慮できるため，全体的に高い分割精度を示した
- ✅ 特に境界付近や細部構造において性能差が顕著に現れた
- ✅ 小さなパーツ（例: 飛行機のエンジン）の分割精度が向上

### PointNet の特性
- ✅ シンプルな構造により学習が安定
- ✅ 推論速度が PointNet++ より約 XX% 高速
- ❌ 複雑な形状や細部の分割において課題

### 計算コストとのトレードオフ
- PointNet++: 高精度だが計算コストが高い（約 X 倍）
- PointNet: 精度は劣るが推論が高速
- 用途に応じた使い分けが重要

---

## 🔮 今後の課題

- [ ] **マルチスケール特徴統合**: PointNet++ のマルチスケール版（MSG）の実装
- [ ] **データ拡張の改善**: より効果的な augmentation 手法の検証
- [ ] **他のデータセット**: S3DIS, ScanNet などでの評価
- [ ] **軽量化**: モバイル環境向けの軽量版モデルの開発
- [ ] **リアルタイム処理**: 推論速度の最適化
- [ ] **Transformer との比較**: Point Transformer などとの性能比較

---

## 📚 参考文献

1. **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation**  
   Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas  
   CVPR 2017  
   [[Paper]](https://arxiv.org/abs/1612.00593) [[Code]](https://github.com/charlesq34/pointnet)

2. **PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space**  
   Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas  
   NeurIPS 2017  
   [[Paper]](https://arxiv.org/abs/1706.02413) [[Code]](https://github.com/charlesq34/pointnet2)

3. **ShapeNet: An Information-Rich 3D Model Repository**  
   Angel X. Chang, Thomas Funkhouser, et al.  
   arXiv 2015  
   [[Paper]](https://arxiv.org/abs/1512.03012) [[Website]](https://shapenet.org/)

---

## 📄 ライセンス

本プロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 謝辞

- **PointNet / PointNet++ の原著者**: Charles R. Qi 氏をはじめとする研究チームに感謝
- **ShapeNet Dataset**: Stanford University および Princeton University のデータセット提供に感謝
- **PyTorch コミュニティ**: 優れた深層学習フレームワークの開発に感謝
- **参考実装**: [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) 他

---

## 📧 連絡先

質問や提案がありましたら，お気軽にお問い合わせください。

- **GitHub**: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- **Email**: your.email@example.com
- **Blog**: [アルゴリズム詳細解説](https://maviccc123456-del.github.io/pointnet/pointnet_explain/)

---

## 📈 更新履歴

- **2025-01-XX**: プロジェクト開始，基本実装完了
- **2025-01-XX**: PointNet 学習完了
- **2025-01-XX**: PointNet++ 学習完了
- **2025-01-XX**: 実験結果まとめ，README 整備

---

**⭐ このプロジェクトが役に立ったら，Star をいただけると嬉しいです！**
