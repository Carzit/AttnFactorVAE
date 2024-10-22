import os
import sys
import logging
import argparse
from numbers import Number
from typing import List, Literal, Optional
from dataclasses import dataclass
from functools import reduce

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from preparers import LoggerPreparer
import utils

# 定义一个FileData类，包含path和dataframe两个属性。作为基类，描述每个数据文件。
@dataclass
class FileData:
    path: str
    dataframe: pd.DataFrame

@dataclass
class FactorData(FileData):
    codes: List[str]
    dates: List[str]
    factor: str

@dataclass
class LabelData(FileData):
    codes: List[str]
    dates: List[str]
    label: str

class Processor:
    def __init__(self) -> None:
        pass
    
    def process(self):
        raise NotImplementedError()

class FactorProcessor(Processor):
    """
    因子处理管线
    """
    def __init__(self, 
                 folder_path:str):
        self.folder_path:str = folder_path #待处理数据所在文件夹的路径

        self.common_dates:List[str] = None #共有日期（用于后续做数据对齐）
        self.common_codes:List[str] = None #共有股票代码（用于后续做数据对齐）
        self.factor_names:List[str] = [] #因子名

        self.factor_data_list:List[FactorData] = []

    def read_and_process_file(self, 
                              file_path:str, 
                              file_format:Literal["csv", "pkl", "parquet", "feather"]="pkl") -> FileData:
        df = utils.load_dataframe(path=file_path, format=file_format)
        return FileData(path=file_path, dataframe=df)
    
    def load_data(self, file_format:str) -> None:
        # 读取指定文件夹中的所有pkl文件，处理数据并存储在 alpha_data_list 中，同时计算所有文件中共同的日期和股票代码。
        file_list = [f for f in os.listdir(self.folder_path) if f.endswith(file_format)]
        
        for file_name in tqdm(file_list):
            file_path = os.path.join(self.folder_path, file_name)
            factor_data = self.read_and_process_file(file_path, file_format)
            
            if self.common_dates is None:
                self.common_dates = pd.Index(factor_data.dates)
            else:
                self.common_dates = self.common_dates.intersection(factor_data.dates)

            if self.common_codes is None:
                self.common_codes = pd.Index(factor_data.codes)
            else:
                self.common_codes = self.common_codes.intersection(factor_data.codes)

            self.factor_data_list.append(factor_data)

        self.common_dates = list(self.common_dates)
        self.common_codes = list(self.common_codes)
    
    def common_filter(self) -> None:
        # 过滤数据，只保留所有文件中共同的日期和股票代码。
        for factor_data in tqdm(self.factor_data_list):
            factor_data.dataframe = factor_data.dataframe.loc[self.common_dates, self.common_codes]
            factor_data.codes = self.common_codes
            factor_data.dates = self.common_dates
    
    def merge_date(self, date:str) -> pd.DataFrame:
        # 根据不同的日期合并数据，并返回一个以股票代码为行名，以因子为列名的DataFrame。
        merged_df = pd.DataFrame(index=self.common_codes, columns=[])
        merged_df.index.name = 'stock_code'

        for factor_data in self.factor_data_list:
            if date in factor_data.dates:
                series = factor_data.dataframe.loc[date]
                series.name = factor_data.factor
                merged_df = merged_df.join(series, how='left')

        #merged_df.insert(0, 'date', date)
        merged_df.reset_index(inplace=True, drop=False)
        return merged_df

    def process(self, 
                save_folder:Optional[str]=os.curdir,
                save_format:Literal["csv", "pkl", "parquet", "feather"]="pkl") -> None:
        logging.debug("merging data...")
        for date in tqdm(self.common_dates):
            merged_df = self.merge_date(date)
            utils.save_dataframe(df=merged_df,
                           path=os.path.join(save_folder, f"{date}.{save_format}"),
                           format=save_format)

class QuantityPriceFeature_FactorProcessor(FactorProcessor):
    def __init__(self, folder_path: str):
        super().__init__(folder_path)
        
    def read_and_process_file(self, file_path:str, file_format:Literal["csv", "pkl", "parquet", "feather"]="pkl") -> FactorData:
        df = utils.load_dataframe(path=file_path, format=file_format)
        codes = df.columns.tolist()
        dates = df.index.tolist()
        factor_name = os.path.basename(file_path).replace("OHO_", "").replace(".pkl", "")
        self.factor_names.append(factor_name)

        return FactorData(path=file_path, 
                          dataframe=df, 
                          codes=codes, 
                          dates=dates, 
                          factor=factor_name)
    
class FundamentalFeature_FactorProcessor(FactorProcessor):
    def __init__(self, folder_path: str):
        super().__init__(folder_path)
        
    def read_and_process_file(self, file_path:str, file_format:Literal["csv", "pkl", "parquet", "feather"]="pkl") -> FactorData:
        df = utils.load_dataframe(path=file_path, format=file_format)
        codes = df.columns.tolist()
        dates = df.index.tolist()
        factor_name = os.path.basename(file_path).replace(".pkl", "")
        self.factor_names.append(factor_name)

        return FactorData(path=file_path, 
                          dataframe=df, 
                          codes=codes, 
                          dates=dates, 
                          factor=factor_name)
    
class LabelProcessor(FactorProcessor):
    def __init__(self, folder_path: str):
        super().__init__(folder_path)
        
    def read_and_process_file(self, file_path:str, file_format:Literal["csv", "pkl", "parquet", "feather"]="pkl") -> FactorData:
        df = utils.load_dataframe(path=file_path, format=file_format)
        codes = df.columns.tolist()
        dates = df.index.tolist()
        label_name = os.path.basename(file_path).replace("label_", "").replace(".pkl", "")
        self.factor_names.append(label_name)

        return FactorData(path=file_path, 
                          dataframe=df, 
                          codes=codes, 
                          dates=dates, 
                          factor=label_name)
        
class ProcessorAlignment:
    def __init__(self, 
                 processors:List[Processor],  
                 align_attr_name:str,
                 save_folder:Optional[str] = os.curdir) -> None:
        self.processors:List[Processor] = processors
        self.align_attr_name:str = align_attr_name
        self.save_folder:str = save_folder
        
    def align(self, lst1:List, lst2:List) -> List:
        set1 = set(lst1)
        set2 = set(lst2)
        common_set = set1.intersection(set2)
        return list(common_set)
    
    def process(self):
        attrs = [getattr(processor, self.align_attr_name) for processor in self.processors]
        align_attr = list(reduce(self.align, attrs))
        align_attr.sort()
        with open(os.path.join(self.save_folder, f"{self.align_attr_name}.json"), "w") as f:
            json.dump(align_attr, f)
        for processor in self.processors:
            setattr(processor, self.align_attr_name, align_attr)
        
            
def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocessor.")

    parser.add_argument("--log_path", type=str, default="log/data_construct.log", help="Path of log file. Default `log/data_construct.log`")

    parser.add_argument("-q", "--quantity_price_factor_folder", type=str, required=True, help="Path of folder for quantity-price factor files")
    parser.add_argument("-f", "--fundamental_factor_folder", type=str, required=True, help="Path of folder for fundamental factor files")
    parser.add_argument("-l", "--label_folder", type=str, required=True, help="Path of folder for label files")
    parser.add_argument("-s", "--save_folder", type=str, required=True, help="Path of folder for Processor to save processed result in subdir `alpha` and `label`")
    parser.add_argument("--read_format", type=str, default="pkl", help="File format to read, literally `csv`, `pkl`, `parquet` or `feather`. Default `pkl`")
    parser.add_argument("--save_format", type=str, default="pkl", help="File format to save, literally `csv`, `pkl`, `parquet` or `feather`. Default `pkl`")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger = LoggerPreparer(name="Preprocess", 
                            file_level=logging.INFO, 
                            log_file=args.log_path).prepare()

    logger.debug(f"Command: {' '.join(sys.argv)}")

    if not os.path.exists(args.save_folder):    
        os.makedirs(args.save_folder)
        os.makedirs(os.path.join(args.save_folder, "quantity_price_feature"))
        os.makedirs(os.path.join(args.save_folder, "fundamental_feature"))
        os.makedirs(os.path.join(args.save_folder, "label"))

    quantity_price_processor = QuantityPriceFeature_FactorProcessor(args.quantity_price_factor_folder)
    fundamental_processor = FundamentalFeature_FactorProcessor(args.fundamental_factor_folder)
    label_processor = LabelProcessor(args.label_folder)

    logger.info("Loading Quantity-Price Factor Data...")
    quantity_price_processor.load_data(file_format=args.read_format)
    
    logger.info("Loading Fundamental Factor data...")
    fundamental_processor.load_data(file_format=args.read_format)

    logger.info("Loading Label Data...")
    label_processor.load_data(file_format=args.read_format)

    logger.info("Doing Data Alignment...")
    date_aligner = ProcessorAlignment([quantity_price_processor, fundamental_processor, label_processor], 
                                      "common_dates",
                                      save_folder=args.save_folder)
    date_aligner.process()
    code_aligner = ProcessorAlignment([quantity_price_processor, fundamental_processor, label_processor], 
                                      "common_codes",
                                      save_folder=args.save_folder)
    code_aligner.process()

    quantity_price_processor.common_filter()
    fundamental_processor.common_filter()
    label_processor.common_filter()

    logger.info("Doing Quantity-Price Factor Data Split...")
    quantity_price_processor.process(save_folder=os.path.join(args.save_folder, "quantity_price_feature"), 
                                     save_format=args.save_format)
    logger.info(f"Quantity-Price Date Data Saved to `{os.path.join(args.save_folder, 'quantity_price_feature')}`")
    
    logger.info("Doing Fundamental Factor Data Split...")
    fundamental_processor.process(save_folder=os.path.join(args.save_folder, "fundamental_feature"), 
                                     save_format=args.save_format)
    logger.info(f"Fundamental Date Data Saved to `{os.path.join(args.save_folder, 'fundamental_feature')}`")
    
    logger.info("Doing Label Data Split...")
    label_processor.process(save_folder=os.path.join(args.save_folder, "label"), 
                            save_format=args.save_format)
    logger.info(f"Label Date Data Saved to `{os.path.join(args.save_folder, 'label')}`")
    
    logger.info("Data Construct Accomplished")

# example: python data_construct.py -q "data\raw\Alpha101" -f "data\raw\Fundamental" -l "data\raw\label" -s "data\preprocess"