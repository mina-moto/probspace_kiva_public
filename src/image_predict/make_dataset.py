"""
プロジェクトルートからの画像パスを持つデータセットの作成。
画像の存在しないデータは削除、LOAN_AMOUNTを25で除算してlog。
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import yaml


class MakeDataset:
    def __init__(
            self,
            train_path: str = "input/train.csv",
            test_path: str = "input/test.csv",
            train_images_dir_path: str = "input/train_images/train_images/",
            test_images_dir_path: str = "input/test_images/test_images/",
            train_output_path: str = "output/image_predict/dataset001/train.csv",
            test_output_path: str = "output/image_predict/dataset001/test.csv",
            is_log: bool = False
    ):
        """

        Args:
            train_path:
            test_path:
            train_images_dir_path:
            test_images_dir_path:
            train_output_path:
            test_output_path:
        """
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        self.train_images_dir_path = train_images_dir_path
        self.test_images_dir_path = test_images_dir_path
        self.train_output_path = train_output_path
        self.test_output_path = test_output_path
        self.is_log = is_log

    def run(self):
        train_image_dataset = self.train[["LOAN_ID", "IMAGE_ID", "LOAN_AMOUNT"]]
        test_image_dataset = self.test[["LOAN_ID", "IMAGE_ID"]]
        train_image_dataset["IMAGE_PATH"] = train_image_dataset['IMAGE_ID'].apply(
            lambda x: f"{self.train_images_dir_path}{x}.jpg")
        test_image_dataset["IMAGE_PATH"] = test_image_dataset['IMAGE_ID'].apply(
            lambda x: f"{self.test_images_dir_path}{x}.jpg")

        train_image_path_list = glob.glob(f"{self.train_images_dir_path}**/*.jpg", recursive=True)
        test_image_path_list = glob.glob(f"{self.test_images_dir_path}**/*.jpg", recursive=True)

        train_image_dataset = train_image_dataset.query("IMAGE_PATH in @train_image_path_list")
        test_image_dataset = test_image_dataset.query("IMAGE_PATH in @test_image_path_list")

        def transform(output, is_log=False):
            output = np.expm1(output) if is_log else output
            output = output / 25
            return output
        train_image_dataset["LOAN_AMOUNT"] = transform(train_image_dataset["LOAN_AMOUNT"], self.is_log)
        os.makedirs(os.path.dirname(self.train_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.test_output_path), exist_ok=True)
        train_image_dataset.to_csv(self.train_output_path, index=False)
        test_image_dataset.to_csv(self.test_output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='config/image_predict/make_dataset001.yml',
        help='config path')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    runner = MakeDataset(**config)
    runner.run()


if __name__ == '__main__':
    main()
