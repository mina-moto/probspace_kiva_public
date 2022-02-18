"""
tableデータのデータセットの作成
feature_dir_pathで指定したtrain、testにその他の特徴量を結合する。


Examples
    python src/table/make_table_dataset.py -c config/table/make_table_dataset/009.yml

"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()  # noqa
sys.path.append(f"{os.getenv('PROJECT_ROOT')}src/")  # noqa
ROOT = Path(os.getenv('PROJECT_ROOT'))


class MakeTableDataset:
    def __init__(
        self,
        feature_dir_path,
        output_dir_path,
        is_sentence_feature,
        sentence_feature_dir_path,
        is_tfidf_svd_vec,
        tfidf_svd_vec_dir_path,
        stacking_prediction_list,
    ):
        """

        Args:
            feature_dir_path:追加先となるtrain、testの元の特徴量
            output_dir_path:出力先
            sentence_feature_dir_path:文章系の特徴量
            is_sentence_feature:文章系の特徴量を追加するかどうか
            tfidf_svd_vec_dir_path:TAGSとLOAN_IDのtfidf_svd
            is_tfidf_svd_vec:TAGSとLOAN_IDのtfidf_svdを追加するかどうか
            stacking_prediction_list:stackingする予測のリスト
        """
        self.feature_dir_path = Path(feature_dir_path)
        self.output_dir_path = Path(output_dir_path)
        self.is_sentence_feature = is_sentence_feature
        self.sentence_feature_dir_path = Path(sentence_feature_dir_path)
        self.is_tfidf_svd_vec = is_tfidf_svd_vec
        self.tfidf_svd_vec_dir_path = Path(tfidf_svd_vec_dir_path)
        self.stacking_prediction_list = stacking_prediction_list

    def merge_sentence_basic_feature(self, train_df, test_df):
        """
        sentence_basic_featureをtrain、testに結合する。
        word_count、word_ave_length、char_countのみ残す

        Returns:

        """
        train_df_ = train_df.copy()
        test_df_ = test_df.copy()
        # word_count、word_ave_length、char_countのみ残す
        cols = [
            "LOAN_ID",
            "DESCRIPTION_TRANSLATED_word_count",
            "DESCRIPTION_word_count",
            "LOAN_USE_word_count",
            "DESCRIPTION_TRANSLATED_word_ave_length",
            "DESCRIPTION_word_ave_length",
            "LOAN_USE_word_ave_length",
            "DESCRIPTION_TRANSLATED_char_count",
            "DESCRIPTION_char_count",
            "LOAN_USE_char_count",
        ]
        for c in ["DESCRIPTION", "DESCRIPTION_TRANSLATED", "LOAN_USE"]:
            train_sentence_basic_feature = pd.read_csv(
                self.sentence_feature_dir_path / c / "train_sentence_basic_feature.csv")
            test_sentence_basic_feature = pd.read_csv(
                self.sentence_feature_dir_path / c / "test_sentence_basic_feature.csv")

            # uppercase、paragraph_countは全て同じ値になるため削除する
            # train_sentence_basic_feature = train_sentence_basic_feature.drop(
            #     [f"{c}_uppercase", f"{c}_paragraph_count"], axis=1)
            # test_sentence_basic_feature = test_sentence_basic_feature.drop(
            #     [f"{c}_uppercase", f"{c}_paragraph_count"], axis=1)
            train_sentence_basic_feature = train_sentence_basic_feature.filter(items=cols)
            test_sentence_basic_feature = test_sentence_basic_feature.filter(items=cols)
            train_df_ = pd.merge(train_df_, train_sentence_basic_feature, on='LOAN_ID', how="left")
            test_df_ = pd.merge(test_df_, test_sentence_basic_feature, on='LOAN_ID', how="left")

        return train_df_, test_df_

    def merge_tfidf_svd_vec(self, train_df, test_df):
        """
        TAGSとLOAN_IDのtf-idf後、svdした値をtrain、testにmerge

        Returns:

        """
        train_df_ = train_df.copy()
        test_df_ = test_df.copy()
        for c in ["TAGS", "LOAN_USE"]:
            train_tf_idf_svd_vec = pd.read_csv(self.tfidf_svd_vec_dir_path / c / "train_tf_idf_svd_vec.csv")
            test_tf_idf_svd_vec = pd.read_csv(self.tfidf_svd_vec_dir_path / c / "test_tf_idf_svd_vec.csv")
            train_df_ = pd.merge(train_df_, train_tf_idf_svd_vec, on='LOAN_ID', how="left")
            test_df_ = pd.merge(test_df_, test_tf_idf_svd_vec, on='LOAN_ID', how="left")
        return train_df_, test_df_

    def stacking(self, train_df, test_df, stacking_prediction_list):
        """
        stacking_prediction_listのモデルの出力をstacking。
        train_df、test_dfに結合して返す。

        Args:
            train_df:
            test_df:
            stacking_prediction_list:

        Returns:
            stackingしたtrain_df、test_df
        """
        train_df_ = train_df.copy()
        test_df_ = test_df.copy()
    
        def merge_pred(df, pred_df, pred_name, new_pred_name):
            """
            dfにpred_dfのpred_nameを'LOAN_ID'軸で、new_pred_nameとしてmerge
            Args:
                df:
                pred_df:
                pred_name:
                new_pred_name:

            Returns:

            """
            df = pd.merge(df, pred_df[["LOAN_ID", pred_name]], how='left', on='LOAN_ID').rename(columns={pred_name: new_pred_name})
            return df
    
        for prediction in stacking_prediction_list:
            train_prediction = pd.read_csv(prediction["train_path"])
            test_prediction = pd.read_csv(prediction["test_path"])
            train_df_ = merge_pred(train_df_, train_prediction, prediction["train_pred_name"], prediction["new_pred_name"])
            test_df_ = merge_pred(test_df_, test_prediction, prediction["test_pred_name"], prediction["new_pred_name"])
        return train_df_, test_df_
    
    def run(self):
        os.makedirs(self.output_dir_path, exist_ok=True)
        train_feature_df = pd.read_csv(self.feature_dir_path / "train.csv")
        test_feature_df = pd.read_csv(self.feature_dir_path / "test.csv")

        if self.is_sentence_feature:
            train_feature_df, test_feature_df = self.merge_sentence_basic_feature(train_feature_df, test_feature_df)
        if self.is_tfidf_svd_vec:
            train_feature_df, test_feature_df = self.merge_tfidf_svd_vec(train_feature_df, test_feature_df)
        train_feature_df, test_feature_df = self.stacking(train_feature_df, test_feature_df, self.stacking_prediction_list)
        train_feature_df.to_csv(self.output_dir_path / "train.csv", index=False)
        test_feature_df.to_csv(self.output_dir_path / "test.csv", index=False)
        print("train_df.shape: ", train_feature_df.shape)
        print("test_df.shape: ", test_feature_df.shape)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='config/table/make_table_dataset/009.yml',
        help='config path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    MakeTableDataset(**config).run()


if __name__ == "__main__":
    main()
