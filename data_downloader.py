import os
import sys
import json
import datetime
import utils
import argparse
import pandas as pd
import numpy as np
import yfinance as yf

from typing import List, Literal, Optional
from tqdm import tqdm

class Downloader:
	def __init__(self):
		self.datelist = []
		self.stock_code = []
		self.trade_data = {}
		self.stock_indinfo = pd.DataFrame(columns = ['sector', 'industry', 'subindustry', 'cap'])

	def load_str(self,
				 input_stock_code,
				 input_date):
		#直接输入股票代码及所需日期（不推荐）。输入时均用逗号分隔。 Still under construction.
		pass

	def load_json(self,
				  file_path):
		#使用json批量载入股票代码及日期（推荐）。
		with open(file_path, "r") as fp:
			json_data = json.load(fp)
		self.datelist = json_data["trade_date"]
		self.stock_code = json_data["stock_code"]

	def yfinance_downloader(self):
		# 下载股票交易数据。
		start_date = datetime.datetime.strptime(self.datelist[0], '%Y%m%d')
		end_date = datetime.datetime.strptime(str(int(self.datelist[-1])+1), '%Y%m%d')
		tickers = yf.download(self.stock_code, start = start_date, end = end_date)
		tickers.index = self.datelist
		for dates, df in tickers.groupby(level = 0):
			df = df.T
			self.trade_data[dates] = df.pivot_table(index = df.index.get_level_values("Ticker"),
													columns = df.index.get_level_values("Price")).reset_index(names = "ts_code")
			self.trade_data[dates] = self.trade_data[dates].set_index("ts_code")
			self.trade_data[dates].columns = ["adj_close","close","high","low","open","vol"]
			self.trade_data[dates] = self.trade_data[dates].reindex(index = self.stock_code, fill_value = np.nan)

	def yfinance_indinfo(self):
		# 下载股票行业信息因子。
		for item in self.stock_code:
			try:
				ticker = yf.Ticker(item)
				self.stock_indinfo.loc[item] = [ticker.info['sectorKey'], ticker.info['industryKey'], ticker.info['industryKey'], ticker.info['marketCap']]
			except KeyError:
				pass
		self.stock_indinfo.rename_axis("ts_code")
		self.stock_indinfo = self.stock_indinfo.reindex(index = self.stock_code, fill_value = np.nan)
		for dates in self.trade_data.keys():
			self.trade_data[dates] = pd.concat([self.trade_data[dates], self.stock_indinfo], axis = 1)
			self.trade_data[dates].drop(["adj_close"])

	def save_file(self,
				  save_folder,
				  save_format):
		for dates in tqdm(self.trade_data.keys()):
			utils.save_dataframe(df = self.trade_data[dates],
								 path = os.path.join(save_folder, f"{dates}.{save_format}"),
								 format = save_format)

def parse_args():
	parser = argparse.ArgumentParser(description="Data download.")

	parser.add_argument("-f", "--file_path", type = str, required = True,
						help = "file path for json with stock codes and trade dates.")
	parser.add_argument("-s", "--save_folder", type = str, required = True,
                        help = "Path of the folder for saving standardized data.")
	parser.add_argument("--save_format", type=str, default="pkl",
                        choices=["csv", "pkl", "parquet", "feather"],
                        help="File format to save, literally `csv`, `pkl`, `parquet` or `feather`. Default `pkl`.")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	if not os.path.exists(args.save_folder):
		os.makedirs(args.save_folder)

	Downloader = Downloader()
	print("Loading JSON file...")
	Downloader.load_json(args.file_path)

	print("Downloading trade data...")
	Downloader.yfinance_downloader()
	Downloader.yfinance_indinfo()

	print("Saving file...")
	Downloader.save_file(args.save_folder, args.save_format)

#example: python data_downloader.py -f "configs\config_datadownload.json" -s "data\raw\raw"
