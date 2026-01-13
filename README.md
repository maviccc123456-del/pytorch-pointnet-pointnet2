# PointNet / PointNet++ による点群分割の比較実装

## 1.概要
本リポジトリでは，点群深層学習の代表的手法である **PointNet** および **PointNet++** を対象とし，
点群分割タスクにおける両モデルの実装および性能比較を行う。

PointNet は，点群を直接入力として扱う革新的なモデルであり，
点群深層学習分野の基礎となる手法である。
一方で，局所的な幾何構造の表現には課題がある。

PointNet++ は，階層的な局所特徴学習を導入することで，
PointNet の課題を改善したモデルである。

本プロジェクトでは，両手法を同一条件下で実装・評価し，
点群分割における性能および特徴表現の違いを実験的に検証する。

---

## 2.背景・動機

- PointNet は point-wise MLP により，各点を独立に処理するシンプルな構造を持つ
- グローバルプーリングに強く依存するため，局所構造情報の表現が限定的である
- PointNet++ はサンプリング・グルーピングによる階層的特徴学習を導入している

以上を踏まえ，本プロジェクトでは  
**局所構造のモデリング能力の違いが点群分割性能に与える影響**  
を明らかにすることを目的とする。

---

## 3.手法

### PointNet
- 各点に対して point-wise MLP を適用
- Max Pooling によりグローバル特徴を生成
- グローバル特徴と点特徴を結合し，各点の分割ラベルを予測

**特徴**
- 点の順序に依存しない
- 実装がシンプル
- 局所構造の表現能力が限定的

---

### PointNet++
- Farthest Point Sampling (FPS) による代表点の選択
- Ball Query / kNN による局所領域の構成
- 各局所領域に PointNet を適用（Set Abstraction）
- 階層的に特徴を学習

**特徴**
- 局所的な幾何構造を考慮可能
- 分割タスクにおいて高い表現力を持つ

---

## 4.実験設定

- Task: 点群分割（Semantic / Part Segmentation）
- Dataset:
  - ModelNet / ShapeNet Part / Custom Dataset
- Number of points: XXXX
- Evaluation metrics:
  - Overall Accuracy
  - Mean IoU

※ 学習条件（epoch 数，batch size，optimizer 等）は両モデルで統一

---

## 5.実験結果

| Model       | Accuracy | mIoU |
|------------|----------|------|
| PointNet   | XX.XX %  | XX.XX |
| PointNet++ | XX.XX %  | XX.XX |

**考察**
- PointNet++ は局所構造を考慮できるため，全体的に高い分割精度を示した
- 特に境界付近や細部構造において性能差が顕著に現れた

※ 可視化結果をここに追加すると効果的

---

## 実装環境

- Python: 3.x
- PyTorch: x.x.x
- CUDA: x.x
- OS: Windows / Linux

---

## 実行方法

```bash
# Training
python train_pointnet.py
python train_pointnet2.py

# Evaluation / Visualization
python test.py --model pointnet
python test.py --model pointnet2




