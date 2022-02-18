"""
"DESCRIPTION_TRANSLATED", "DESCRIPTION", "LOAN_USE"のカラムに基づく特徴量の生成

Output:
    output/preprocess/table/sentence_feature/{c}/

Examples:
    python src/table/sentence_feature.py

"""
import re
import string
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import nltk
import pandas as pd
import textstat
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
load_dotenv()  # noqa
sys.path.append(f"{os.getenv('PROJECT_ROOT')}src/")  # noqa
ROOT = Path(os.getenv('PROJECT_ROOT'))

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£',  # noqa
 '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\n', '\xa0', '\t',  # noqa
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',  # noqa
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',  # noqa
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]  # noqa

html_tags = ['<p>', '</p>', '<table>', '</table>', '<tr>', '</tr>', '<ul>', '<ol>', '<dl>', '</ul>', '</ol>',
             '</dl>', '<li>', '<dd>', '<dt>', '</li>', '</dd>', '</dt>', '<h1>', '</h1>',
             '<br>', '<br/>', '<br />', '<strong>', '</strong>', '<span>', '</span>', '<blockquote>', '</blockquote>',
             '<pre>', '</pre>', '<div>', '</div>', '<h2>', '</h2>', '<h3>', '</h3>', '<h4>', '</h4>', '<h5>', '</h5>',
             '<h6>', '</h6>', '<blck>', '<pr>', '<code>', '<th>', '</th>', '<td>', '</td>', '<em>', '</em>']

empty_expressions = ['&lt;', '&gt;', '&amp;', '&nbsp;',
                     '&emsp;', '&ndash;', '&mdash;', '&ensp;', '&quot;', '&#39;']


def create_basic_feature(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    dfのtarget_columnの文章に基づくの特徴量を作成する

    Args:
        df:
        target_column:

    Returns:
        dfに特徴量の列を追加したDataFrame

    """
    df_ = df.copy()
    # スペースのぞいた文字数
    df_[f'{target_column}_char_count'] = df_[target_column].apply(lambda t: len(t.replace(' ', '')))

    # 単語
    df_[f'{target_column}_word_count'] = df_[target_column].apply(lambda t: len(str(t).split()))
    df_[f'{target_column}_word_unique_count'] = df_[target_column].apply(lambda t: len(set(str(t).split())))
    df_[f'{target_column}_word_unique_ratio'] = df_[
        f'{target_column}_word_unique_count'] / (df_[f'{target_column}_word_count'] + 1)
    df_[f'{target_column}_word_ave_length'] = df_[target_column].apply(
        lambda t: sum([len(w) for w in t.split()]) / len(t.split()))

    # 句読点
    punctuations = string.punctuation
    df_[f'{target_column}_punc_count'] = df_[target_column].apply(lambda t: len([w for w in t if w in punctuations]))

    # 大文字、小文字
    df_[f'{target_column}_uppercase'] = df_[target_column].str.findall(r'[A-Z]').str.len() + 1
    df_[f'{target_column}_lowercase'] = df_[target_column].str.findall(r'[a-z]').str.len() + 1
    df_[f'{target_column}_up_low_ratio'] = df_[f'{target_column}_uppercase'] / (df_[f'{target_column}_lowercase'] + 1)

    # 段落
    df_[f'{target_column}_paragraph_count'] = df_[target_column].apply(lambda t: t.count('\n'))

    # 文章
    df_[f'{target_column}_sentence_count'] = df_[target_column].apply(lambda t: len(sent_tokenize(t)))
    df_[f'{target_column}_sentence_ave_length'] = df_[target_column].apply(
        lambda t: sum([len(s) for s in sent_tokenize(t)]) / len(sent_tokenize(t))
    )

    df_[f'{target_column}_sentence_max_length'] = df_[target_column].apply(
        lambda t: max([len(s) for s in sent_tokenize(t)])
    )
    df_[f'{target_column}_sentence_min_length'] = df_[target_column].apply(
        lambda t: min([len(s) for s in sent_tokenize(t)]))
    df_[f'{target_column}_word_per_sentence'] = df_[
        f'{target_column}_word_count'] / df_[f'{target_column}_sentence_count']

    # 母音
    df_[f'{target_column}_syllable_count'] = df_[target_column].apply(lambda t: textstat.syllable_count(t))
    df_[f'{target_column}_syllable_per_sentence'] = df_[
        f'{target_column}_syllable_count'] / df_[f'{target_column}_sentence_count']
    df_[f'{target_column}_syllable_per_word'] = df_[
        f'{target_column}_syllable_count'] / df_[f'{target_column}_word_count']

    return df_


def rm_spaces(text):
    spaces = ['\u200b', '\u200e', '\u202a', '\u2009', '\u2028', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u3000',
              '\x10', '\x7f', '\x9d', '\xad',
              '\x97', '\x9c', '\x8b', '\x81', '\x80', '\x8c', '\x85', '\x92', '\x88', '\x8d', '\x80', '\x8e', '\x9a',
              '\x94', '\xa0',
              '\x8f', '\x82', '\x8a', '\x93', '\x90', '\x83', '\x96', '\x9b', '\x9e', '\x99', '\x87', '\x84', '\x9f',
              ]  # noqa
    for space in spaces:
        text = text.replace(space, ' ')
    return text


def remove_urls(x):
    x = re.sub(r'(https?://[a-zA-Z0-9.-]*)', r'', x)

    # original
    x = re.sub(r'(quote=\w+\s?\w+;?\w+)', r'', x)
    return x


def clean_puncts(x):
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def clean_html_tags(x, stop_words=[]):
    for r in html_tags:
        x = x.replace(r, '')
    for r in empty_expressions:
        x = x.replace(r, ' ')
    for r in stop_words:
        x = x.replace(r, '')
    return x


def clean_sentences(series: pd.Series):
    """
    seriesの各要素の文章をclean

    Args:
        series:

    Returns:

    """
    series = series.apply(lambda x: str(x).lower())
    series = series.apply(lambda x: clean_html_tags(x))
    series = series.apply(lambda x: rm_spaces(x))
    series = series.apply(lambda x: remove_urls(x))
    series = series.apply(lambda x: clean_puncts(x))
    return series


def main():
    train_df = pd.read_csv(ROOT / "input" / "train.csv")
    test_df = pd.read_csv(ROOT / "input" / "test.csv")

    df_ = pd.concat([train_df, test_df], axis=0)

    def save_basic_feature(df, train_len, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        train_df = df[:train_len]
        test_df = df[train_len:]
        train_df.to_csv(f"{save_dir}train_sentence_basic_feature.csv", index=False)
        test_df.to_csv(f"{save_dir}test_sentence_basic_feature.csv", index=False)

    for c in ["DESCRIPTION_TRANSLATED", "DESCRIPTION", "LOAN_USE"]:
        df_[c] = clean_sentences(df_[c])
        basic_feature_df = create_basic_feature(df_[["LOAN_ID", c]], target_column=c)
        save_dir = f"output/preprocess/table/sentence_feature/{c}/"
        save_basic_feature(basic_feature_df, len(train_df), save_dir)
        print(f"Success {c}")


if __name__ == "__main__":
    main()
