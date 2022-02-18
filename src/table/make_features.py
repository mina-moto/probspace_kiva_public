import argparse
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import category_encoders as ce
import yaml
import sys
from dotenv import load_dotenv

# from src.table import sentence_feature

load_dotenv()  # noqa
sys.path.append(f"{os.getenv('PROJECT_ROOT')}src/")  # noqa
ROOT = Path(os.getenv('PROJECT_ROOT'))


def count_encoding(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    引数カラムのカテゴリごとの数

    Args:
        df:
        columns:

    Returns:

    """
    df_ = df.copy()
    count_encoder = ce.CountEncoder(cols=columns)
    df_result = count_encoder.fit_transform(df_[columns])
    df_result = df_result.add_suffix("_COUNT")
    df_ = pd.concat([df, df_result], axis=1)
    return df_


def make_feat_join_minor_cate_activity_name(df):
    df_ = df.copy()
    df_["activity_name_cnt"] = df_.groupby("ACTIVITY_NAME")["LOAN_ID"].transform("count")
    df_result = df_[["ACTIVITY_NAME", "activity_name_cnt"]].apply(
        lambda x: x["ACTIVITY_NAME"] if x["activity_name_cnt"] > 500 else "Others", axis=1
    )
    return df_result


def make_feat_join_minor_cate_country_code(df):
    df_ = df.copy()
    df_["country_code_cnt"] = df_.groupby("COUNTRY_CODE")["LOAN_ID"].transform("count")
    df_result = df_[["COUNTRY_CODE", "country_code_cnt"]].apply(
        lambda x: x["COUNTRY_CODE"] if x["country_code_cnt"] > 500 else "Others", axis=1
    )
    return df_result


def make_feat_join_minor_cate_currency(df):
    df_ = df.copy()
    df_["currency_cnt"] = df_.groupby("CURRENCY")["LOAN_ID"].transform("count")
    df_result = df_[["CURRENCY", "currency_cnt"]].apply(
        lambda x: x["CURRENCY"] if x["currency_cnt"] > 500 else "Others", axis=1
    )
    return df_result


def make_feat_is_loan_amount_outlier(df):
    df_ = df.copy()
    df_result = df_["LOAN_AMOUNT"].map(lambda col: 1 if col >= 2000 else 0)
    return df_result


def make_feat_concat_country_code_sector_name(df):
    df_ = df.copy()
    df_result = df_[["COUNTRY_CODE", "SECTOR_NAME"]].apply(lambda x: f"{x['COUNTRY_CODE']}_{x['SECTOR_NAME']}", axis=1)
    return df_result


def make_loan_use_first_word_feature(df: pd.DataFrame, skip_list: list = None) -> pd.Series:
    """
    df["LOAN_USE"]の先頭の単語を特徴量として返す。
    skip_list中の単語は飛ばして次の単語を対象とする。

    Args:
        df:
        skip_list:

    Returns:

    """
    if skip_list is None:
        skip_list = ["to", "To", "a", ""]
    df_ = df.copy()

    def extract_loan_use_first_word_skip(loan_use_str):
        loan_use_str_split = loan_use_str.split(" ")
        for word in loan_use_str_split:
            if word not in skip_list:
                return word
        return ""

    return df_["LOAN_USE"].apply(extract_loan_use_first_word_skip)


def make_feat_join_minor_loan_use_first_word(df):
    """
    LOAN_USE_first_wordのカテゴリで100以下のものをOthersとする。

    Args:
        df:

    Returns:

    """
    df_ = df.copy()
    df_["cnt"] = df_.groupby("LOAN_USE_first_word")["LOAN_ID"].transform("count")
    df_result = df_[["LOAN_USE_first_word", "cnt"]].apply(
        lambda x: x["LOAN_USE_first_word"] if x["cnt"] > 100 else "Others", axis=1
    )
    return df_result


def make_feat_join_minor_town_name(df):
    df_ = df.copy()
    df_["cnt"] = df_.groupby("TOWN_NAME")["LOAN_ID"].transform("count")
    df_result = df_[["TOWN_NAME", "cnt"]].apply(
        lambda x: x["TOWN_NAME"] if x["cnt"] > 500 else "Others", axis=1
    )
    return df_result


def agg_category_group(df: pd.DataFrame, category: str, target: str, agg_list: list = None) -> pd.DataFrame:
    """
    dfのcategoryカラムのグループごとにtargetカラムの統計量を取ってdf更新

    Args:
        df:
        category:
        target:
        agg_list:用いる統計量のリスト

    Returns:

    """
    df_ = df.copy()
    if agg_list is None:
        agg_list = ["mean", "std", "min", "max"]
    for agg_type in agg_list:
        group_agg = df_[[category, target]].groupby(category).agg(agg_type).reset_index().rename(
            {target: f"{category}_{agg_type}_{target}"}, axis=1
        )
        df_ = df_.merge(group_agg, how='left', on=[category])
    return df_


def make_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    特徴量生成処理

    Args:
        train_df:
        test_df:

    Returns:
        前処理後のtrain、test
    """
    train_df_ = train_df.copy()
    test_df_ = test_df.copy()
    df_ = pd.concat([train_df_, test_df_], axis=0)

    # train test 共通処理
    df_['TAGS_LENGTH'] = df_['TAGS'].astype(str).apply(lambda x: len(x))
    df_['TAGS_WORD_NUM'] = df_['TAGS'].astype(str).apply(lambda x: len(x.split(',')))
    df_["LOAN_USE_first_word"] = make_loan_use_first_word_feature(df_)

    count_encoding_columns = [
        "ORIGINAL_LANGUAGE",
        "ACTIVITY_NAME",
        "COUNTRY_CODE",
        "SECTOR_NAME",
        "CURRENCY",
        "TOWN_NAME",
        "LOAN_USE_first_word",
    ]
    df_ = count_encoding(df_, count_encoding_columns)

    # TAGS_LENGTHの統計量をとるカテゴリのリスト
    tags_length_agg_category_list = [
        "ORIGINAL_LANGUAGE",
        "ACTIVITY_NAME",
        "COUNTRY_CODE",
        "SECTOR_NAME",
        "CURRENCY",
        "TOWN_NAME",
        "LOAN_USE_first_word"]
    for c in tags_length_agg_category_list:
        df_ = agg_category_group(df_, category=c, target="TAGS_LENGTH")

    # minorカテゴリのOthers変換
    df_["LOAN_USE_first_word"] = make_feat_join_minor_loan_use_first_word(df_)
    df_["TOWN_NAME"] = make_feat_join_minor_town_name(df_)
    df_["ACTIVITY_NAME"] = make_feat_join_minor_cate_activity_name(df_)
    df_["COUNTRY_CODE"] = make_feat_join_minor_cate_country_code(df_)
    df_["CURRENCY"] = make_feat_join_minor_cate_currency(df_)

    #
    df_["country_code_sector_name"] = make_feat_concat_country_code_sector_name(df_)

    # train test 個別処理
    train_df_ = df_[:len(train_df_)]
    test_df_ = df_[len(train_df_):]
    test_df_ = test_df_.drop(columns='LOAN_AMOUNT')
    train_df_["is_loan_amount_outlier"] = make_feat_is_loan_amount_outlier(train_df_)

    return train_df_, test_df_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='config/table/make_features/make_features012.yml',
        help='config path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    OUTPUT = Path(config["output_dir"])
    os.makedirs(OUTPUT, exist_ok=True)
    train_df = pd.read_csv(ROOT / "input" / "train.csv")
    test_df = pd.read_csv(ROOT / "input" / "test.csv")

    train_df_, test_df_ = make_features(train_df, test_df)
    del_cols = [
        'COUNTRY_NAME',
        'DESCRIPTION',
        'DESCRIPTION_TRANSLATED',
        'IMAGE_ID',
        'LOAN_USE',
        'TAGS',
    ]
    train_del_cols = del_cols
    test_del_cols = del_cols
    train_df_ = train_df_.drop(columns=train_del_cols)
    test_df_ = test_df_.drop(columns=test_del_cols)
    train_df_.to_csv(OUTPUT / "train.csv", index=False)
    test_df_.to_csv(OUTPUT / "test.csv", index=False)
    print("train_df.shape: ", train_df_.shape)
    print("test_df.shape: ", test_df_.shape)
    print(f"train_df_.columns: {train_df_.columns}")
    print(f"test_df_.columns: {test_df_.columns}")


if __name__ == "__main__":
    main()
