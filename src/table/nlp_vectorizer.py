"""
TAGS、LOAN_USEにtf-idf、SVDを行う。

Examples:
    python src/table/nlp_vectorizer.py

Input:
    input/train.csv
    input/test.csv

Output:

    tf-idfのベクトル
    output/preprocess/table/tfidf_svd/TAGS/tfidf_vec.joblib
    output/preprocess/table/tfidf_svd/LOAN_USE/tfidf_vec.joblib

    svdにより次元圧縮したベクトル
    output/preprocess/table/tfidf_svd/TAGS/train_tf_idf_svd_vec
    output/preprocess/table/tfidf_svd/TAGS/test_tf_idf_svd_vec

    output/preprocess/table/tfidf_svd/LOAN_USE/train_tf_idf_svd_vec
    output/preprocess/table/tfidf_svd/LOAN_USE/test_tf_idf_svd_vec

"""
import re
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

import nltk
import numpy as np
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()  # noqa
sys.path.append(f"{os.getenv('PROJECT_ROOT')}src/")  # noqa
ROOT = Path(os.getenv('PROJECT_ROOT'))

nltk.download('stopwords')


def preprocess_sentence_tfidf(sentence: str):
    STOP_WORDS = set(stopwords.words("english"))
    stem = PorterStemmer()
    word_list = []
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)  # 句読点削除
    tokenized_word = word_tokenize(text)  # 単語
    for word in tokenized_word:
        if word not in STOP_WORDS:  # ストップワード削除
            word = stem.stem(word)  # ステミング
            word_list.append(word)
    return ' '.join(word_list)


def preprocess_tags_tfidf(tags: str):
    """
    tags用の前処理。
    ','で区切り、各tagから両端の連続する空白文字と'#'を削除する。

    Args:
        tags:

    Returns:

    """
    tag_list = tags.split(',')

    def extarct_tag_word(raw_tag):
        tag_word = raw_tag.strip()
        return tag_word.replace("#", "")
    return ' '.join(list(map(extarct_tag_word, tag_list)))


def main():
    train_df = pd.read_csv("input/train.csv")
    test_df = pd.read_csv("input/test.csv")

    df_ = pd.concat([train_df, test_df], axis=0)

    def save_tf_idf_svd_vec(df, train_len, save_dir):
        train_df = df[:train_len]
        test_df = df[train_len:]
        train_df.to_csv(f"{save_dir}train_tf_idf_svd_vec.csv", index=False)
        test_df.to_csv(f"{save_dir}test_tf_idf_svd_vec.csv", index=False)

    tfidf_vectorizer = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, use_idf=True, smooth_idf=True)
    svd = decomposition.TruncatedSVD(n_components=2, random_state=0)

    df_["TAGS"] = df_["TAGS"].astype(str).apply(preprocess_tags_tfidf)
    df_["LOAN_USE"] = df_["LOAN_USE"].astype(str).apply(preprocess_sentence_tfidf)
    # discriptionは既に存在
    for c in ["TAGS", "LOAN_USE"]:
        save_dir = f"output/preprocess/table/tfidf_svd/{c}/"

        os.makedirs(save_dir, exist_ok=True)

        tfidf_vec = tfidf_vectorizer.fit_transform(df_[c])
        joblib.dump(tfidf_vec, f'{save_dir}tfidf_vec.joblib')

        tfidf_svd_df = pd.DataFrame(svd.fit_transform(tfidf_vec), columns=[f'{c}_tfidf_svd1', f'{c}_tfidf_svd2'])
        tfidf_svd_df["LOAN_ID"] = df_["LOAN_ID"].to_list()
        save_tf_idf_svd_vec(tfidf_svd_df, len(train_df), save_dir)

        print(f"Success {c}")


if __name__ == "__main__":
    main()
