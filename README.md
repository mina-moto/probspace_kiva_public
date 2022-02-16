# probspace_kiva_public

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
