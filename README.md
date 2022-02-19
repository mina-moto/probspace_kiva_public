# ProbSpace クラファンコンペ 1st place solution

[probSpace_kiva]: https://comp.probspace.com/competitions/kiva2021s

「[ProbSpace Kiva／クラウドファンディングの資金調達額予測](probSpace_kiva)」の 1st place solutionです。

## 概要

- BERTによりDESCRIPTION_TRANSLATEDとSECTOR_NAME、COUNTRY_CODEを用いて予測する。
- SwinTransformerの転移学習による画像を用いて予測する。
- 上記結果をStackingしたテーブル特徴量を作成して、Lightgbm、CatBoostにより予測する。
- Lightgbm、CatBoostの予測結果を複数用意してアンサンブルする。

## 再現のための準備
- probspaceで提供されているデータを`input/`へ配置する。
- プロジェクトディレクトリ直下`.env`の`PROJECT_ROOT`で、このリポジトリまでのパスを指定する。

## BERTによる予測
- DESCRIPTION_TRANSLATEDとSECTOR_NAME、COUNTRY_CODEを用いてLOAN_AMOUNTの予測を行う。
  - SECTOR_NAME、COUNTRY_CODEはEmbedding層を用いてベクトルに変換する。
- それぞれの予測値をテーブルデータの入力とする。
- BERTの事前学習済みモデルは`roberta-base`と`microsoft/deberta-base`の2種類使用した。
- 条件を変更し、計8個のモデルを作成した。
  - LOAN_AMOUNTをlog変換
  - 損失関数の変更
  - BERTのカスタムヘッダーの変更

各モデルのnotebookまでのパス、ロスの種類などは下記テーブルを参照
| パス                                                               | ロス           | 備考                                                            |
| ------------------------------------------------------------------ | -------------- | --------------------------------------------------------------- |
| output/roberta_features/roberta_meanpooling_huber_20220209         | Huber          |                                                                 |
| output/roberta_features/roberta_cls_pooling_huber_20220210         | Huber          |                                                                 |
| output/roberta_features/roberta_clspooling_mse_20220209            | MSE            |                                                                 |
| output/roberta_features/deberta_clspooling_huber_20220210          | Huber          |                                                                 |
| output/roberta_features/deberta_meanpooling_huber_20220211         | Huber          | mean poolingを想定したが、誤ってcls poolingで学習推論している。 |
| output/roberta_features/roberta_clspooling_l1_target_log_20220211  | L1 target log  |                                                                 |
| output/roberta_features/roberta_clspooling_mse_target_log_20220211 | MSE target log |                                                                 |
| output/roberta_features/deberta_meanpooling_l1_target_log_20220212 | L1 target log  | mean poolingを想定したが、誤ってcls poolingで学習推論している。 |


順にnotebookを実行する。
- 前処理
  - `preprocess/preprocess_v4/preprocess_v4.ipynb`
- 各モデルの学習と推論
  - `output/roberta_features/deberta_clspooling_huber_20220210/deberta_clspooling_huber.ipynb`
  - `output/roberta_features/deberta_meanpooling_huber_20220211/deberta_meanpooling_huber.ipynb`
  - `output/roberta_features/deberta_meanpooling_l1_target_log_20220212/deberta_meanpooling_l1_target_log.ipynb`
  - `output/roberta_features/roberta_cls_pooling_huber_20220210/roberta_cls_pooling_huber.ipynb`
  - `output/roberta_features/roberta_clspooling_l1_target_log_20220211/roberta_clspooling_l1_target_log.ipynb`
  - `output/roberta_features/roberta_clspooling_mse_20220209/roberta_clspooling_mse.ipynb`
  - `output/roberta_features/roberta_clspooling_mse_target_log_20220211/roberta_clspooling_mse_target_log.ipynb`
  - `output/roberta_features/roberta_meanpooling_huber_20220209/roberta_meanpooling_huber.ipynb`

## 画像による予測

- 画像データから学習済みモデルの転移学習によるLOAN_AMOUNTの予測を行う。
- それぞれの予測値をテーブルデータの入力とする。
- LOAN_AMOUNTをlog変換して学習した場合と、変換していない場合の2種類のモデルを用いた。

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

tf-idfの特徴量を作成する。
- `notebook/preprocess_v1.ipynb`を実行する。
- tf-idf以外のことも行っていますが、最終的に私用しているのはtf-idfの部分のみです。

## LightGBM及びCatBoostによる予測

最終のアンサンブルに用いたモデルの学習、推論を行う。
- LightGBMは一つのファイルで学習から推論までを行っている。
- CatBoostは学習と推論でスクリプトが分かれている。
```
cd src/
python seed_change_exec_lgbm_train.py -c ../config/table/seed_change_exec_lgbm_train/011.yml
python exec_cb_train.py -c ../config/110.yml
python exec_cb_pred.py -c ../config/110.yml
python exec_cb_train.py -c ../config/325.yml
python exec_cb_pred.py -c ../config/325.yml
```

## アンサンブル

- Lightgbm、CatBoostの予測結果をアンサンブル（平均）する。
- 予測値は25-10000で離散化する。
- `output/ensemble/{now datetime}/`に予測結果を出力する。
- アンサンブルするモデルはスクリプト内のconfigで指定しているが、**Lightgbm及びCatboostの予測結果ファイル名には実行時刻が含まれるため、再現のためには変更する必要がある**。

```
python src/ensemble.py
```

## 参考にしたコード
一部のコードは下記を参考に作成しました。
- https://signate.jp/competitions/281/discussions/bert-5-folds-baseline-model-1
- https://github.com/senkin13/kaggle/tree/master/probspace_youtube
- https://github.com/MitsuruFujiwara/SantanderValuePredictionChallenge