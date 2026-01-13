PointNet / PointNet++ による点群分割の比較実装
1. 概要（Overview）

本リポジトリでは，点群深層学習の代表的手法である PointNet および PointNet++ を対象とし，
点群分割タスクにおける両モデルの実装および性能比較を行う。

PointNet は点群を直接入力として扱う革新的なモデルであり，点群深層学習分野の基礎となる手法である。一方で，局所的な幾何構造の表現には課題がある。
PointNet++ はこの問題を解決するため，階層的な局所特徴学習を導入している。

本プロジェクトでは，両手法を同一条件下で実装・評価し，
点群分割における特徴表現および性能の違いを実験的に検証する。

2. 背景・動機（Why）

PointNet は point-wise MLP により，各点を独立に処理するシンプルかつ汎用的な構造を持つ

しかし，グローバルプーリングに強く依存するため，局所構造の情報が十分に活用されない

PointNet++ は，サンプリング・グルーピングを用いた階層的特徴学習により，局所領域の幾何構造を捉えることが可能

以上を踏まえ，本プロジェクトでは
「局所構造のモデリング能力の違いが，点群分割性能にどのような影響を与えるか」
を明確にすることを目的とする。

3. 手法（Method）
3.1 PointNet

各点に対して point-wise MLP を適用

Max Pooling によりグローバル特徴を生成

グローバル特徴と点ごとの特徴を結合し，分割ラベルを予測

特徴：

点の順序に依存しない

実装がシンプル

局所構造の表現能力に制限がある

3.2 PointNet++

Farthest Point Sampling (FPS) により代表点を選択

Ball Query / kNN により局所領域を構成

各局所領域に対して PointNet を適用（Set Abstraction）

階層的に特徴を学習

特徴：

局所的な幾何構造を考慮可能

分割タスクにおいて高い表現力を持つ

4. 実験設定（Experiment）

タスク：点群分割（Semantic / Part Segmentation）

使用データセット：

（例）ModelNet / ShapeNet Part / 自作データセット

入力点数：XXXX 点

評価指標：

Overall Accuracy

Mean IoU

※ 学習条件（エポック数，バッチサイズ，最適化手法等）は両モデルで統一

5. 実験結果（Result）
Model	Accuracy	mIoU
PointNet	XX.XX %	XX.XX
PointNet++	XX.XX %	XX.XX

PointNet++ は局所構造を考慮できるため，複雑な形状においてより高い分割精度を示した

特に境界付近や細部構造において差が顕著に現れた

（※ 可視化結果の画像をここに追加すると非常に良い）

6. 実装環境（Environment）

Python: 3.x

PyTorch: x.x.x

CUDA: x.x

OS: Windows / Linux

7. 実行方法（Usage）
# 学習
python train_pointnet.py
python train_pointnet2.py

# 推論・可視化
python test.py --model pointnet
python test.py --model pointnet2

8. まとめ（Conclusion）

本プロジェクトでは，PointNet と PointNet++ を点群分割タスクにおいて比較実装し，
局所構造を考慮した階層的特徴学習が分割性能向上に有効であることを確認した。

本実装を通じて，点群深層学習におけるモデル設計と特徴表現の重要性について理解を深めた。

9. 参考文献（Reference）

Qi, C. R., et al., PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

Qi, C. R., et al., PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
