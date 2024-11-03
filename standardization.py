import os
import sys
import pandas as pd
import numpy as np
import dask.dataframe as dd
import utils
import collections
import argparse
import logging

from typing import List, Literal, Optional
from dask_ml.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from dask_ml.model_selection import train_test_split
from tqdm import tqdm
from preparers import LoggerPreparer

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
        self.methodmark = 'CSZScore'

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
        '''标准化核心，提供五种方案。'''
        if method == "CSZScore":
            for dates in tqdm(self.DataDict.keys()):
                self.DataDict[dates].loc[:] = StandardScaler().fit_transform(self.DataDict[dates])
        elif method == "CSRank":
            for dates in tqdm(self.DataDict.keys()):
                self.DataDict[dates] = self.DataDict[dates].rank(method = 'min', ascending = False)
        elif method == "ZScore":
            DataDict_list = []
            for dates in self.DataDict.keys():
                self.DataDict[dates]['dates'] = dates
                DataDict_list.append(self.DataDict[dates])
            self.DataDict = dd.concat(DataDict_list).compute()
            self.DataDict = self.DataDict.set_index(['dates', self.DataDict.index])
            train_index, test_index = train_test_split(self.DataDict.index, test_size = 0.3, shuffle = False)
            scaler = StandardScaler()
            Data_train = self.DataDict.loc[train_index]
            Data_test = self.DataDict.loc[test_index]
            scaler.fit(Data_train)
            self.DataDict.loc[train_index] = scaler.transform(Data_train)
            self.DataDict.loc[test_index] = scaler.transform(Data_test)
        elif method == "MinMax":
            DataDict_list = []
            for dates in self.DataDict.keys():
                self.DataDict[dates]['dates'] = dates
                DataDict_list.append(self.DataDict[dates])
            self.DataDict = dd.concat(DataDict_list).compute()
            self.DataDict = self.DataDict.set_index(['dates', self.DataDict.index])
            train_index, test_index = train_test_split(self.DataDict.index, test_size = 0.3, shuffle = False)
            scaler = MinMaxScaler()
            Data_train = self.DataDict.loc[train_index]
            Data_test = self.DataDict.loc[test_index]
            scaler.fit(Data_train)
            self.DataDict.loc[train_index] = scaler.transform(Data_train)
            self.DataDict.loc[test_index] = scaler.transform(Data_test)
        elif method == "RobustZScore":
            DataDict_list = []
            for dates in self.DataDict.keys():
                self.DataDict[dates]['dates'] = dates
                DataDict_list.append(self.DataDict[dates])
            self.DataDict = dd.concat(DataDict_list).compute()
            self.DataDict = self.DataDict.set_index(['dates', self.DataDict.index])
            train_index, test_index = train_test_split(self.DataDict.index, test_size = 0.3, shuffle = False)
            scaler = RobustScaler()
            Data_train = self.DataDict.loc[train_index]
            Data_test = self.DataDict.loc[test_index]
            scaler.fit(Data_train)
            self.DataDict.loc[train_index] = scaler.transform(Data_train)
            self.DataDict.loc[test_index] = scaler.transform(Data_test)
        else:
            raise NotImplementedError()
        self.methodmark = method

    def SaveFile(self,
                save_folder:Optional[str] = os.curdir,
                save_format:Literal["csv", "pkl", "parquet", "feather"] = "pkl"):
        if self.methodmark == 'CSZScore' or self.methodmark == 'CSRank':
            for dates in tqdm(self.DataDict.keys()):
                utils.save_dataframe(df = self.DataDict[dates],
                                     path = os.path.join(save_folder, f"{dates}.{save_format}"),
                                     format = save_format)
        else:
            grouped = self.DataDict.groupby(level = 0)
            for dates,df in tqdm(grouped):
                utils.save_dataframe(df = df.droplevel(level = 0),
                                     path = os.path.join(save_folder, f"{dates}.{save_format}"),
                                     format = save_format)
                
        del self.DataDict #释放内存给下一次标准化。

def parse_args():
    parser = argparse.ArgumentParser(description="Data Standardization.")

    parser.add_argument("--log_path", type=str, default="log/standardization.log",
                        help="Path of log file. Default `log/standardization.log`")
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

    logger = LoggerPreparer(name = "Standardization",
                            file_level = logging.INFO,
                            log_file = args.log_path).prepare()
    logger.debug(f"Command: {' '.join(sys.argv)}")

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        os.makedirs(os.path.join(args.save_folder, "quantity_price_feature"))
        os.makedirs(os.path.join(args.save_folder, "fundamental_feature"))
        os.makedirs(os.path.join(args.save_folder, "label"))

    LStd = StdProcessor(os.path.join(args.preprocess_data_folder, "label"))
    QStd = StdProcessor(os.path.join(args.preprocess_data_folder, "quantity_price_feature"))
    FStd = StdProcessor(os.path.join(args.preprocess_data_folder, "fundamental_feature"))

    logger.info("Loading Label Data...")
    LStd.LoadFile(FileFormat = args.read_format)

    logger.info("Loading Quantity Prices Data...")
    QStd.LoadFile(FileFormat = args.read_format)

    logger.info("Loading Fundamental Features Data...")
    FStd.LoadFile(FileFormat = args.read_format)

    logger.info("Standardizing Label Data...")
    LStd.Standardization(method = args.standardization_method)

    logger.info("Saving Label Data...")
    LStd.SaveFile(save_folder = os.path.join(args.save_folder, "label"), save_format = args.save_format)

    logger.info("Standardizing Quantity Prices Data...")
    QStd.Standardization(method = args.standardization_method)

    logger.info("Saving Quantity Prices Data...")
    QStd.SaveFile(save_folder = os.path.join(args.save_folder, "quantity_price_feature"), save_format = args.save_format)

    logger.info("Standardizing Fundamental Features Data...")
    FStd.Standardization(method = args.standardization_method)

    logger.info("Saving Fundamental Features Data...")
    FStd.SaveFile(save_folder = os.path.join(args.save_folder, "fundamental_feature"), save_format = args.save_format)
    logger.info(f'Standardized Data has been saved to `{args.save_folder}`')

    logger.info("Standardization Completed.")

#example: python standardization.py -p "data\preprocess" -s "data\standardization"
