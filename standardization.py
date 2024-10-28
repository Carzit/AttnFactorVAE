import os
import sys
import pandas as pd
import numpy as np
import utils
import collections
import argparse

from typing import List, Literal, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Processor:
    def __init__(self) -> None:
        pass
    
    def process(self):
        raise NotImplementedError()

class StdProcessor(Processor):
    """标准化管线"""
    def __init__(self, 
                 FolderPath):
        self.FolderPath = FolderPath
        self.DataDict = {}

    def ReadFile(self,
                 FilePath,
                 FileFormat:Literal["csv", "pkl", "parquet", "feather"] = "pkl"):
        '''返回一个DataFrame。'''
        df = utils.load_dataframe(path = FilePath, format = FileFormat)
        df.set_index('stock_code', inplace = True)
        return df

    def LoadFile(self, FileFormat):
        '''读取并生成每日数据的总字典，其中keys为文件名（日期），values为dataframe。'''
        FileList = [files for files in os.listdir(self.FolderPath) if files.endswith(FileFormat)]

        for FileName in tqdm(FileList):
            FilePath = os.path.join(self.FolderPath, FileName)
            FileName_splitext = os.path.splitext(FileName)[0]
            DailyData = self.ReadFile(FilePath, FileFormat)
            self.DataDict[FileName_splitext] = DailyData

    def Standardization(self,
                        method:Literal["CSZScore", "CSRank", "ZScore", "MinMax", "RobustZScore"] = "CSZScore"):
        if method == "CSZScore":
            for dates in tqdm(self.DataDict.keys()):
                self.DataDict[dates].loc[:] = StandardScaler().fit_transform(self.DataDict[dates])
        elif method == "CSRank":
            for dates in tqdm(self.DataDict.keys()):
                self.DataDict[dates] = self.DataDict[dates].rank(method = 'min', ascending = False)
        elif method == "ZScore":
            self.DataDict = pd.DataFrame.from_dict(self.DataDict)
            train_index, test_index = train_test_split(self.DataDict.index, test_size = 0.3, random_state = 114514)
            scaler = StandardScaler()
            Data_train = self.DataDict.loc[train_index]
            Data_test = self.DataDict.loc[test_index]
            scaler.fit(Data_train)
            self.DataDict.loc[train_index] = scaler.transform(Data_train)
            self.DataDict.loc[test_index] = scaler.transform(Data_test)
        elif method == "MinMax":
            self.DataDict = pd.DataFrame.from_dict(self.DataDict)
            train_index, test_index = train_test_split(self.DataDict.index, test_size = 0.3, random_state = 114514)
            scaler = MinMaxScaler()
            Data_train = self.DataDict.loc[train_index]
            Data_test = self.DataDict.loc[test_index]
            scaler.fit(Data_train)
            self.DataDict.loc[train_index] = scaler.transform(Data_train)
            self.DataDict.loc[test_index] = scaler.transform(Data_test)
        elif method == "RobustZScore":
            self.DataDict = pd.DataFrame.from_dict(self.DataDict)
            train_index, test_index = train_test_split(self.DataDict.index, test_size = 0.3, random_state = 114514)
            scaler = RobustScaler()
            Data_train = self.DataDict.loc[train_index]
            Data_test = self.DataDict.loc[test_index]
            scaler.fit(Data_train)
            self.DataDict.loc[train_index] = scaler.transform(Data_train)
            self.DataDict.loc[test_index] = scaler.transform(Data_test)
        else:
            raise NotImplementedError()

    def SaveFile(self,
                save_folder:Optional[str] = os.curdir,
                save_format:Literal["csv", "pkl", "parquet", "feather"] = "pkl"):
        for dates in tqdm(self.DataDict.keys()):
            utils.save_dataframe(df = self.DataDict[dates],
                                 path = os.path.join(save_folder, f"{dates}.{save_format}"),
                                 format = save_format)

def parse_args():
    parser = argparse.ArgumentParser(description="Data Standardization.")

    parser.add_argument("-p", "--preprocess_data_folder", type = str, required = True,
                        help = "Path of the folder for preprocessed data. Make sure it contains labels, quantity prices and fundamental features.")
    parser.add_argument("-s", "--save_folder", type = str, required = True,
                        help = "Path of the folder for saving standardized data.")
    parser.add_argument("-m", "--standardization_method", type = str, default = "CSZScore",
                        help = "Method of Standardization, literally `CSZScore`, `CSRank`, `ZScore`, `MinMax` or `RobustZScore`. Drfault `CSZScore`.")
    parser.add_argument("--read_format", type=str, default="pkl",
                        choices=["csv", "pkl", "parquet", "feather"],
                        help="File format to read, literally `csv`, `pkl`, `parquet` or `feather`. Default `pkl`.")
    parser.add_argument("--save_format", type=str, default="pkl",
                        choices=["csv", "pkl", "parquet", "feather"],
                        help="File format to save, literally `csv`, `pkl`, `parquet` or `feather`. Default `pkl`.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        os.makedirs(os.path.join(args.save_folder, "quantity_price_feature"))
        os.makedirs(os.path.join(args.save_folder, "fundamental_feature"))
        os.makedirs(os.path.join(args.save_folder, "label"))

    LStd = StdProcessor(os.path.join(args.preprocess_data_folder, "label"))
    QStd = StdProcessor(os.path.join(args.preprocess_data_folder, "quantity_price_feature"))
    FStd = StdProcessor(os.path.join(args.preprocess_data_folder, "fundamental_feature"))

    print("Loading Label Data...")
    LStd.LoadFile(FileFormat = args.read_format)

    print("Loading Quantity Prices Data...")
    QStd.LoadFile(FileFormat = args.read_format)

    print("Loading Fundamental Features Data...")
    FStd.LoadFile(FileFormat = args.read_format)

    print("Standardizing Label Data...")
    LStd.Standardization(method = args.standardization_method)

    print("Standardizing Quantity Prices Data...")
    QStd.Standardization(method = args.standardization_method)

    print("Standardizing Fundamental Features Data...")
    FStd.Standardization(method = args.standardization_method)

    print("Saving Label Data...")
    LStd.SaveFile(save_folder = os.path.join(args.save_folder, "label"), save_format = args.save_format)

    print("Saving Quantity Prices Data...")
    QStd.SaveFile(save_folder = os.path.join(args.save_folder, "quantity_price_feature"), save_format = args.save_format)

    print("Saving Fundamental Features Data...")
    FStd.SaveFile(save_folder = os.path.join(args.save_folder, "fundamental_feature"), save_format = args.save_format)
    print(f'Standardized Data has been saved to `{args.save_folder}`')

#example: python standardization.py -p "data\preprocess" -s "data\standardization"