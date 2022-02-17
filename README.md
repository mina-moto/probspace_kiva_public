# ProbSpace クラファンコンペ 1st place solution
[probSpace_kiva]: https://comp.probspace.com/competitions/kiva2021s
「[ProbSpace Kiva／クラウドファンディングの資金調達額予測](probSpace_kiva)」の 1st place solutionです。

## 画像による予測
- 画像データから学習済みモデルの転移学習によるLOAN_AMOUNTの予測を行う。
- それぞれの予測値をテーブルデータの入力とする。
- LOAN_AMOUNTをlog変換して学習した場合と、変換していない場合の2種類のモデルを用いた。
- 実行する前提として、プロジェクト直下の`.env`の`PROJECT_ROOT`にプロジェクト直下のフォルダを指定している。

データセットの作成

```
python src/image_predict/make_dataset.py -c config/image_predict/make_dataset/make_dataset001.yml
```

学習
```
python src/image_predict/trainer.py -c config/image_predict/trainer/trainer005.yaml
```

評価・推論
```
python src/image_predict/evaluator.py -c config/image_predict/evaluator/evaluator005.yaml
```

### log変換して学習する場合

データセットの作成

```
python src/image_predict/make_dataset.py -c config/image_predict/make_dataset/make_dataset002.yml
```

学習

```
python src/image_predict/trainer.py -c config/image_predict/trainer/trainer006.yaml
```

評価・推論
```
python src/image_predict/evaluator.py -c config/image_predict/evaluator/evaluator006.yaml
```

## テーブルデータの特徴量生成
- TAGSやカテゴリの統計量などのテーブルデータの特徴量の作成。
- 実行する前提として、プロジェクト直下の`.env`の`PROJECT_ROOT`にプロジェクト直下のフォルダを指定している。

TAGS、LOAN_USEに対してtf-idf、SVD
```
python src/table/nlp_vectorizer.py
```

DESCRIPTION_TRANSLATED、DESCRIPTION、LOAN_USEのカラムに基づく特徴量の生成
```
python src/table/sentence_feature.py
```

前処理などを含む上記以外の特徴量の作成
```
python src/table/make_features.py -c config/table/make_features/make_features012.yml 
```

画像・BERTの予測や上記結果の結合を行い、テーブルデータの特徴量の作成
```
python src/table/make_table_dataset.py -c config/table/make_table_dataset/009.yml
```