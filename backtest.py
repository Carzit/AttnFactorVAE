import pandas as pd
import logging
import backtrader as bt # 使用backtrader回测框架
from datetime import datetime as dt
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from preparers import LoggerPreparer

class BacktestData: # 数据读取器
    def __init__(self, path, robot, from_date=None, to_date=None, stock_list=None, logger=None):
        self.path = path # 数据路径
        self.robot = robot # 传入cerebro
        self.from_date = from_date # 开始日期
        self.to_date = to_date # 结束日期
        self.stock_list = stock_list # 股票列表
        self.data_list = os.listdir(path)
        self.raw_data = None
        self.logger = logger

    def data_reader(self): # 读取数据
        dates_list = []
        for name in tqdm(self.data_list):
            dates_list.append(pd.read_csv(self.path + "/" + name, parse_dates=['trade_date'], index_col=0))
            # print(f"Data {name} done!")
        self.raw_data = pd.concat(dates_list, ignore_index=True)
        # logger.info("Data done!")
        if not self.from_date:
            self.from_date = self.raw_data['trade_date'].min()
        else:
            self.from_date = pd.to_datetime(self.from_date)
        logger.info(f"From date: {self.from_date.strftime('%Y-%m-%d')}")
        if not self.to_date:
            self.to_date = self.raw_data['trade_date'].max()
        else:
            self.to_date = pd.to_datetime(self.to_date)
        logger.info(f"To date: {self.to_date.strftime('%Y-%m-%d')}")
        if not self.stock_list:
            self.stock_list = self.raw_data["ts_code"].unique()
        else:
            self.stock_list = [x.strip() for x in self.stock_list.split(",")]
        logger.info("Stock list done!")

    def data_processor(self): # 处理数据并传入cerebro
        for stock in tqdm(self.stock_list):
            dates = pd.DataFrame(self.raw_data['trade_date'].unique(), columns=['trade_date'])
            df = self.raw_data.query(f"ts_code=='{stock}'")[
                ['trade_date', 'open', 'high', 'low', 'close', 'vol']]
            if np.nan in df['close']:
                print(f"{stock} has NaN values, skip it!")
                continue
            df['openinterest'] = 0
            df.columns = ['trade_date', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
            data_ = pd.merge(dates, df, how='left', on='trade_date')
            data_ = data_.set_index("trade_date")
            data_.loc[:, ['volume', 'openinterest']] = data_.loc[:, ['volume', 'openinterest']].fillna(0)
            # data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close']].replace(0, np.nan)
            data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close']].ffill()
            data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close']].fillna(0)
            data_.loc[:, ['volume', 'openinterest']] = data_.loc[:, ['volume', 'openinterest']].fillna(0)
            datafeed = bt.feeds.PandasData(dataname=data_, fromdate=self.from_date, todate=self.to_date)
            self.robot.adddata(datafeed, name=stock)
            # print(f"{stock} Done! From {self.from_date.strftime('%Y-%m-%d')} to {self.to_date.strftime('%Y-%m-%d')}")

        logger.info("All stock Done!")


# 回测策略
class RiskOptimizedStrategy(bt.Strategy):
    params = (
        ('λ', 1),  # 风险厌恶系数，越大则风险厌恶程度越高
        ('κR', 1),  # 交易成本权重，越大则交易成本越高（无穷时新旧权重相同）
        ('κD', 1),  # 多样化权重，越大则多样化程度越高（无穷时所有股票权重为1/N）
        ('initial_num_stocks', 25),  # 初始持仓股票数量
        ('seed', 42),  # 随机种子
        ('w_minus', None),  # 当前权重（可以传入一个初始值）
        ('file_path', None),  # 传入预测的收益率数据
        ('save_path', None), # 保存结果的路径
    )
    
    def __init__(self):
        # 读取预测模型结果(收益率数据)并传入交易系统
        self.ret_data = pd.read_csv(self.params.file_path, index_col=0)
        self.ret_data.index = pd.to_datetime(self.ret_data.index, format='%Y%m%d')
        threshold = 0.1  # 缺失值比例阈值
        self.ret_data = self.ret_data.loc[:, self.ret_data.isnull().mean() < threshold]
        self.ret_data.fillna(0, inplace=True)
        # 获取数据集的所有股票
        cerebro_stocks = set([data._name for data in self.datas])
        ret_stocks_set = set(self.ret_data.columns)
        # 根据两个数据集股票池的交集构造self.stocks
        self.stocks = list(stock for stock in cerebro_stocks if stock in ret_stocks_set)
        self.num_stocks = len(self.stocks)
        self.result_list = [] # 保存每日的结果
        
        # 设置初始权重为均等权重（如果没有传入当前权重w_minus）
        if self.params.w_minus is None:
            weights = np.zeros(self.num_stocks)
            np.random.seed(self.params.seed)
            # 随机选择 initial_num_stocks 个位置，将这些位置的值设为 1/initial_num_stocks
            selected_indices = np.random.choice(self.num_stocks, self.params.initial_num_stocks, replace=False)
            weights[selected_indices] = 1 / self.params.initial_num_stocks
            self.w_minus = weights.flatten().tolist()
        else:
            self.w_minus = self.params.w_minus
        
        for i, stock in enumerate(self.stocks): # 初始交易
            target_weight = self.w_minus[i]
            if target_weight > 0:
                self.order_target_percent(data=stock, target=target_weight)
                # print(f'{self.ret_data.index[0]} {stock} 初始权重：{target_weight}')
        
        self.last_weights = self.get_weights_vector()  # 存储上一个时点的权重
        # print(f'初始权重：{self.w_minus}')

    def next(self):
        current_date = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')  # 获取当前的日期
        if current_date not in self.ret_data.index: # 如果当前日期没有收益率数据，则不进行交易
            return
        
        # 从预测的收益率数据中获取当前日期的收益率
        current_returns = self.ret_data.loc[current_date, self.stocks].values
        
        # 计算协方差矩阵 (这里用收益率的历史数据来计算协方差)
        cov_matrix = self.ret_data.loc[self.ret_data.index < current_date, self.stocks].cov()
        self.last_weights = self.last_weights.reshape(-1, 1)
        
        # 执行均值-方差优化，获取最优权重
        w_optimal = mean_variance_optimization(current_returns, cov_matrix, self.last_weights,
                                               λ=self.params.λ, κR=self.params.κR, κD=self.params.κD)
        
        
        # 使用 NumPy 检查
        if np.any(np.isinf(w_optimal)) or np.any(np.isnan(w_optimal)):
            # print(f"{current_date} 权重计算异常，跳过交易")
            return

        
        # 根据最优权重调整持仓
        for i, stock in enumerate(self.stocks):
            # 设置目标权重
            target_weight = w_optimal[i]

            if target_weight > 0 and cerebro.datasbyname[stock].close[0] > 0: # 调整该股票的目标权重
                self.order_target_percent(data=stock, target=target_weight)
                # print(f'{current_date} {stock} 目标权重：{target_weight/sum(w_optimal)}')
            elif target_weight <= 0:
                # 如果目标权重为 0，则平掉该仓位
                self.order_target_percent(data=stock, target=0)
                # print(f'{current_date} {stock} 目标权重：0，计算权重{target_weight}')
            else:
                # print(f'{current_date} {stock} 权重计算错误，算出为{target_weight}')
                pass

        # 更新当前的权重
        self.last_weights = self.get_weights_vector()
        fund_value = self.broker.getvalue()
        self.result_list.append([current_date, fund_value] + self.last_weights.flatten().tolist())
        
    def get_weights_vector(self):
        # 获取总资产价值
        total_value = self.broker.getvalue()
    
        # 初始化权重列表
        weights = []
    
        for stock in self.stocks:
            # 从 self.datas 中找到对应的股票数据
            data = cerebro.datasbyname[stock]
        
            # 获取持仓数量和当前价格
            position = self.broker.getposition(data).size
            current_price = data.close[0]
        
            # 计算股票市值
            stock_value = position * current_price
        
            # 计算权重
            weight = stock_value / total_value
            weights.append(weight)
    
        # 转换为 (-1, 1) 的 NumPy 矩阵
        weights_vector = np.array(weights).reshape(-1, 1)
        # print(f'当前权重：{weights_vector}')
        return weights_vector
    
    def stop(self):
        # 保存结果到 CSV 文件
        self.results = pd.DataFrame(self.result_list, columns=['date', 'fund_value'] + self.stocks)
        self.results.set_index('date', inplace=True)
        if self.params.file_path == "results.csv":
            if self.params.save_path is None:
                save_path = f'initial_num_stocks-{self.params.initial_num_stocks}_seed-{self.params.seed}_λ-{self.params.λ}_κR-{self.params.κR}_κD-{self.params.κD}_backtest_results.csv'.replace("λ", "lambda").replace("κ", "kappa")
            else:
                save_path = self.params.save_path + f'initial_num_stocks-{self.params.initial_num_stocks}_seed-{self.params.seed}_λ-{self.params.λ}_κR-{self.params.κR}_κD-{self.params.κD}_backtest_results.csv'.replace("λ", "lambda").replace("κ", "kappa")
        else:
            if self.params.save_path is None:
                save_path = f'{self.params.file_path.split(".")[0]}_initial_num_stocks-{self.params.initial_num_stocks}_seed-{self.params.seed}_λ-{self.params.λ}_κR-{self.params.κR}_κD-{self.params.κD}_backtest_results.csv'.replace("λ", "lambda").replace("κ", "kappa")
            else:
                save_path = self.params.save_path + f'{self.params.file_path.split(".")[0]}_initial_num_stocks-{self.params.initial_num_stocks}_seed-{self.params.seed}_λ-{self.params.λ}_κR-{self.params.κR}_κD-{self.params.κD}_backtest_results.csv'.replace("λ", "lambda").replace("κ", "kappa")
        self.results.to_csv(save_path)
        print(f'回测结果已保存到 {save_path}')


# 回测权重计算函数，使用Coqueret(2015)方法的一般化 
def mean_variance_optimization(returns, cov_matrix, w_minus, λ=1, κR=1, κD=1):
    N = len(returns)
    
    # 转换为矩阵形式
    mu = np.array(returns).reshape(-1, 1) # 预测的收益率
    Σ = np.array(cov_matrix) # 协方差矩阵，度量风险
    Λ = np.diag(np.ones(N))  # 用于构造约束条件2
    I_N = np.ones((N, 1)) # 单位向量
    
    # 计算权重
    M_inv = np.linalg.inv(λ * Σ + 2 * κR * Λ + 2 * κD * I_N)
    numerator = 1 - I_N.T @ M_inv @ (mu + 2 * κR * Λ @ w_minus) # 除数
    denominator = I_N.T @ M_inv @ I_N + 1e-6 # 被除数
    if denominator == 0:
        return w_minus.flatten().tolist() # 防止除数为0
    η = numerator / denominator # 系数，应当为常数
    
    # 最优权重
    w_star = M_inv @ (mu + η * I_N + 2 * κR * Λ @ w_minus) # 权重，形状应该为(N, 1)
    
    return w_star.flatten().tolist() # 转换为列表形式


class PortfolioMetrics:
    def __init__(self, backtest_data, weights_data, actual_signals=None, risk_free_rate=0.02):
        """
        初始化投资组合指标计算器
        参数：
        - backtest_data: 回测数据（包含 trade_date 和 fund_value 列）
        - weights_data: 权重数据（DataFrame，行对应日期，列对应股票代码）
        - actual_signals: 实际信号，用于计算准确率（DataFrame，每列为股票代码）
        - risk_free_rate: 无风险收益率，默认年化2%
        """
        self.backtest_data = backtest_data
        self.weights_data = weights_data
        self.actual_signals = actual_signals
        self.risk_free_rate = risk_free_rate / 252  # 转换为每日无风险收益率
        self.daily_returns = backtest_data['fund_value'].pct_change().dropna()  # 计算每日收益率
        self.portfolio_values = backtest_data['fund_value']

    def expected_excessive_returns(self):
        """
        计算预期超额收益
        """
        excessive_returns = self.daily_returns - self.risk_free_rate
        expected_excessive_return = excessive_returns.mean()
        return expected_excessive_return

    def information_ratio(self):
        """
        计算信息比率（IR）
        """
        excessive_returns = self.daily_returns - self.risk_free_rate
        tracking_error = excessive_returns.std()
        if tracking_error == 0:
            return np.nan  # 避免除以零
        ir = excessive_returns.mean() / tracking_error
        return ir

    def sharpe_ratio(self):
        """
        计算夏普比率
        """
        excessive_returns = self.daily_returns - self.risk_free_rate
        portfolio_volatility = self.daily_returns.std()
        if portfolio_volatility == 0:
            return np.nan  # 避免除以零
        sharpe = excessive_returns.mean() / portfolio_volatility
        return sharpe

    def accuracy_ratios(self):
        """
        计算准确率
        """
        if self.actual_signals is None:
            raise ValueError("Actual signals must be provided to calculate accuracy ratios.")

        # 使用权重数据推断信号
        signals = self.weights_data.diff().fillna(0)  # 差分计算权重变化
        signals = signals.applymap(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))  # 转换为信号：1（买入）、-1（卖出）、0（持仓不变）

        # 确保 actual_signals 和 signals 的形状和索引一致
        actual_signals = self.actual_signals.reindex(signals.index, fill_value=0)
        signals = signals.reindex(actual_signals.index).fillna(0)

        # 比较信号是否正确
        correct_predictions = (signals == actual_signals).sum().sum()  # 每日期每只股票的正确预测总数
        total_predictions = actual_signals.size  # 总的信号数
        if total_predictions == 0:
            return np.nan  # 避免除以零
        accuracy = correct_predictions / total_predictions
        return accuracy

    def maximum_drawdown(self):
        """
        计算最大回撤
        """
        cumulative_returns = self.portfolio_values / self.portfolio_values.iloc[0]
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        return max_drawdown

    def turnover(self):
        """
        计算换手率
        """
        turnover = self.weights_data.diff().abs().sum().sum() / len(self.weights_data)
        return turnover

    def calculate_all_metrics(self):
        """
        计算所有指标并返回一个字典
        """
        metrics = {
            "Expected Excessive Returns": self.expected_excessive_returns(),
            "Information Ratio": self.information_ratio(),
            "Sharpe Ratio": self.sharpe_ratio(),
            # "Maximum Drawdown": self.maximum_drawdown(),
            "Turnover": self.turnover(),
        }
        if self.actual_signals is not None:
            metrics["Accuracy Ratios"] = self.accuracy_ratios()
        return metrics


def calculate_metrics(data, csv_file_path, rate=0.02):
    # 获取交易数据
    trading_data = data.raw_data.copy()

    # 数据预处理
    trading_data['trade_date'] = pd.to_datetime(trading_data['trade_date'])

    # 确保 ts_code 没有前后空格
    trading_data['ts_code'] = trading_data['ts_code'].str.strip()

    # 限制日期范围到2023-05-05到2024-04-22
    start_date = pd.to_datetime('2023-05-05')
    end_date = pd.to_datetime('2024-04-22')
    trading_data = trading_data[(trading_data['trade_date'] >= start_date) & (trading_data['trade_date'] <= end_date)]

    # 读取第二个CSV文件（投资组合权重数据）
    if type(csv_file_path) == str:
        try:
            portfolio_df = pd.read_csv(csv_file_path, parse_dates=['date'])
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        except Exception as e:
            print(f"读取CSV文件 {csv_file_path} 时发生错误：{e}")
            return
        portfolio_df.rename(columns={'date': 'trade_date'}, inplace=True)
    elif isinstance(csv_file_path, pd.DataFrame):
        portfolio_df = csv_file_path.copy()
    else:
        print("第二个参数必须是字符串或pandas DataFrame。")

    # 限制日期范围
    portfolio_df = portfolio_df[(portfolio_df['trade_date'] >= start_date) & (portfolio_df['trade_date'] <= end_date)]

    # 提取 fund_value 列作为回测数据
    backtest_data = portfolio_df[['trade_date', 'fund_value']].copy()

    # 提取权重数据，宽格式 DataFrame，行对应日期，列对应股票代码
    weights_data = portfolio_df.drop(columns=['fund_value']).copy()

    # 确保列名一致并去除可能的空格
    weights_data.columns = weights_data.columns.str.strip()

    # 将 trade_date 设为索引
    weights_data.set_index('trade_date', inplace=True)

    # 填充缺失值为0（假设未持有的股票权重为0）
    weights_data = weights_data.fillna(0)

    # 对权重进行归一化（确保每个trade_date下权重之和为1）
    weights_data = weights_data.div(weights_data.sum(axis=1), axis=0).fillna(0)

    # 合并交易数据和权重数据（仅保留在权重数据日期范围内的交易数据）
    trading_data = trading_data[trading_data['trade_date'].isin(weights_data.index)]
    trading_data = trading_data[trading_data['ts_code'].isin(weights_data.columns)]

    # 计算实际信号（actual_signals）基于文件一的数据
    # 1. 比较 'close' 和 'open' 价格
    trading_data['signal'] = np.where(trading_data['close'] > trading_data['open'], 1,
                              np.where(trading_data['close'] < trading_data['open'], -1, 0))

    # 2. 创建宽格式的实际信号 DataFrame
    actual_signals = trading_data.pivot(index='trade_date', columns='ts_code', values='signal')

    # 3. 对实际信号进行重新索引以匹配权重数据的日期和股票代码
    actual_signals = actual_signals.reindex(index=weights_data.index, columns=weights_data.columns, fill_value=0)

    # 4. 确保没有缺失值
    actual_signals = actual_signals.fillna(0)

    # **重要**：确保 `actual_signals` 和 `weights_data` 完全对齐
    # 检查对齐情况
    if not actual_signals.index.equals(weights_data.index):
        print("警告：`actual_signals` 的日期索引与 `weights_data` 不一致。")
    if not actual_signals.columns.equals(weights_data.columns):
        print("警告：`actual_signals` 的股票代码列与 `weights_data` 不一致。")

    # 实例化 PortfolioMetrics 类
    portfolio_metrics = PortfolioMetrics(
        backtest_data=backtest_data,
        weights_data=weights_data,
        actual_signals=actual_signals,  # 传递实际信号数据
        risk_free_rate=rate  # 年化无风险利率
    )

    # 计算所有指标
    metrics = portfolio_metrics.calculate_all_metrics()

    # 打印指标结果
    # for key, value in metrics.items():
    #     print(f"{key}: {value}")
    return metrics


def convert_stkcd_to_exchange(stkcd):
    # 将 Stkcd 转换为 "数字.交易所" 格式
    # # 假设交易所信息根据股票代码的范围决定，例如：
    # # - 上海交易所：600000-699999 -> "SH"
    # # - 深圳交易所：000000-399999 -> "SZ"
    # # - 科创板：688000-689999 -> "KSH"
    # # 如果有其他定义，可以调整规则
    stkcd = int(stkcd)
    if 600000 <= stkcd <= 699999:
        return f"{stkcd:06}.SH"
    elif 0 <= stkcd <= 399999:
        return f"{stkcd:06}.SZ"
    elif 688000 <= stkcd <= 689999:
        return f"{stkcd:06}.KSH"
    else:
        return f"{stkcd:06}.OT"  # 其他未知交易所
    

def read_indices_data(file_path, indices_list):
    df = pd.read_csv(file_path)
    df["Indexcd"] = df["Indexcd"].apply(lambda x: f"{x:06}")
    df["Stkcd"] = df["Stkcd"].apply(convert_stkcd_to_exchange)
    stock_list = list(set(df["Stkcd"]))
    expanded_indexcd = list(set(df["Indexcd"]))
    df_expanded = pd.DataFrame()
    for index_cd in expanded_indexcd:
        temp_df = df.copy()
        temp_df["Indexcd"] = index_cd
        df_expanded = pd.concat([df_expanded, temp_df], ignore_index=True)

    # 初始化结果字典
    df_dict = {}

    # 遍历每一个 Indexcd，创建对应的 index_df
    for index_cd in indices_list:
        # 初始化 DataFrame，columns 为股票代码，index 为日期范围
        index_df = pd.DataFrame(index=pd.date_range(start="2023-05-05", end="2024-04-22", freq="D"), columns=stock_list)

        # 遍历日期范围
        for date in tqdm(index_df.index):
            # 筛选出符合条件的子 DataFrame
            date_df = df[(df["Indexcd"] == index_cd) & (df["Enddt"] == date)]

            # 遍历股票代码，将对应的 Weight 填充到 index_df 中
            for stock in stock_list:
                if stock in date_df["Stkcd"].values:
                    weight = date_df[date_df["Stkcd"] == stock]["Weight"].iloc[0]
                    index_df.at[date, stock] = weight

        # # 将构造好的 DataFrame 存入字典
        df_dict[index_cd] = index_df
        print(f"已完成{index_cd}")
    return df_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Backtestor.")

    parser.add_argument("--log_path", type=str, default="log/backtest.log", help="Path of log file. Default `log/backtest.log`")
    parser.add_argument("-m", "--market_data", type=str, required=True, help="Path of market data file") # 传入市场数据
    parser.add_argument("-i", "--indices_data", type=str, required=True, help="Path of indices data file") # 传入指数数据
    parser.add_argument("-p", "--prediction_data", type=str, default="csv", required=True, help="Path of prediction data file. Use `csv`") # 传入投资组合数据
    parser.add_argument("-s", "--save_folder", type=str, default="data//results//", required=True, help="Path of folder for Processor to save processed result in subdir `alpha` and `label`")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger = LoggerPreparer(name="Backtest", 
                            file_level=logging.INFO, 
                            log_file=args.log_path).prepare()
    # 输入参数
    param_list =[[0.5, 0.5, 0.5, 50, 123],
               [0, 0.5, 0.5, 50, 123],
               [3, 0.5, 0.5, 50, 123],
               [6, 0.5, 0.5, 50, 123],
               [12, 0.5, 0.5, 50, 123],
               [20, 0.5, 0.5, 50, 123],
               [60, 0.5, 0.5, 50, 123],
               [100, 0.5, 0.5, 50, 123],
            #    [0.5, 0, 0.5, 50, 123], 该组会报错，因为计算出η为0
               [0.5, 3, 0.5, 50, 123],
               [0.5, 6, 0.5, 50, 123],
               [0.5, 12, 0.5, 50, 123],
               [0.5, 20, 0.5, 50, 123],
               [0.5, 60, 0.5, 50, 123],
               [0.5, 100, 0.5, 50, 123],
               [0.5, 0.5, 0, 50, 123],
               [0.5, 0.5, 3, 50, 123],
               [0.5, 0.5, 6, 50, 123],
               [0.5, 0.5, 12, 50, 123],
               [0.5, 0.5, 20, 50, 123],
               [0.5, 0.5, 60, 50, 123],
               [0.5, 0.5, 100, 50, 123],
               [0.5, 0.5, 0.5, 10, 123],
               [0.5, 0.5, 0.5, 100, 123],
               [0.5, 0.5, 0.5, 300, 123],
               [0.5, 0.5, 0.5, 500, 123],
               [0.5, 0.5, 0.5, 50, 568],
               [0.5, 0.5, 0.5, 50, 565],
               [0.5, 0.5, 0.5, 50, 746],
               [0.5, 0.5, 0.5, 50, 195]]
    metrics_dict = {}
    market_data=args.market_data # 传入市场数据
    file_path=args.prediction_data # 传入模型预测结果
    save_path=args.save_folder # 回测结果保存路径
    for arg in param_list:
        logger.info(f"args: {arg}")
        market_data=args.market_data # 传入市场数据
        file_path=args.prediction_data # 传入模型预测结果
        save_path=args.save_folder # 回测结果保存路径
        λ=arg[0]
        κR=arg[1]
        κD=arg[2]
        initial_num_stocks=arg[3]
        seed=arg[4]
        #运行代码
        cerebro = bt.Cerebro() # 读取市场数据并传入交易系统
        data = BacktestData(market_data, cerebro, from_date="2023-05-05", to_date="2024-04-22", logger=logger)
        data.data_reader()
        data.data_processor()
        # logger.info("Market Data Loaded")
        cerebro.broker.set_cash(100000000)
        cerebro.broker.setcommission(commission=0.0005)
        cerebro.addstrategy(RiskOptimizedStrategy,
                    λ=λ, 
                    κR=κR, 
                    κD=κD, 
                    initial_num_stocks=initial_num_stocks, 
                    seed=seed, 
                    file_path=file_path, 
                    save_path=save_path) # 策略参数
        logger.info('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
        cerebro.run()
        logger.info('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
        if file_path == "results.csv":
            if save_path is None:
                file_save_path = f'initial_num_stocks-{initial_num_stocks}_seed-{seed}_λ-{λ}_κR-{κR}_κD-{κD}_backtest_results.csv'.replace("λ", "lambda").replace("κ", "kappa")
            else:
                file_save_path = os.path.join(save_path, f'initial_num_stocks-{initial_num_stocks}_seed-{seed}_λ-{λ}_κR-{κR}_κD-{κD}_backtest_results.csv'.replace("λ", "lambda").replace("κ", "kappa"))
        else:
            if save_path is None:
                file_save_path = f'{file_path.split(".")[0]}_initial_num_stocks-{initial_num_stocks}_seed-{seed}_λ-{λ}_κR-{κR}_κD-{κD}_backtest_results.csv'.replace("λ", "lambda").replace("κ", "kappa")
            else:
                file_save_path = os.path.join(save_path, f'{file_path.split(".")[0]}_initial_num_stocks-{initial_num_stocks}_seed-{seed}_λ-{λ}_κR-{κR}_κD-{κD}_backtest_results.csv'.replace("λ", "lambda").replace("κ", "kappa"))
        result = calculate_metrics(data, file_save_path) # 计算回测指标
        metrics_dict[file_save_path] = result
    logger.info("Backtest Model Experiments Finished")

    # 下面的代码为指数数据的相关部分
    cerebro = bt.Cerebro() # 读取市场数据并传入交易系统
    data = BacktestData(market_data, cerebro, from_date="2023-05-05", to_date="2024-04-22", logger=logger)
    data.data_reader()
    data.data_processor()
    df_indices = [pd.read_csv(os.path.join(args.indices_data, index_file)) for index_file in os.listdir(args.indices_data) if index_file.startswith("IDX_Idxtrd")]
    df_combined = pd.concat(df_indices, ignore_index=True)
    df_combined["Idxtrd01"] = pd.to_datetime(df_combined["Idxtrd01"])
    df_dict = read_indices_data(os.path.join(args.indices_data, "indices.csv"), ['000300','000016']) # 读取指数数据并保存到本地
    for index_cd, index_df in df_dict.items():
        # 过滤 df_market 获取当前 index_cd 的相关数据
        market_data = df_combined[df_combined["Indexcd"] == index_cd]
        market_data = market_data.set_index("Idxtrd01")["Idxtrd05"].reindex(index_df.index)
        # 将 Idxtrd05 的值加入到 index_df 的新列 fund_value
        index_df["fund_value"] = market_data
        index_df.index.name = 'date'
        if save_path is None:
            file_save_path = f"index_{index_cd}.csv"
        else:
            file_save_path = os.path.join(save_path, f"index_{index_cd}.csv")
        index_df.to_csv(file_save_path)
        result = calculate_metrics(data, file_save_path) # 计算回测指标
        metrics_dict[file_save_path] = result
    logger.info("Indices Experiments Finished")
    pd.to_pickle(metrics_dict, os.path.join(save_path, "metrics_dict.pkl")) # 保存回测指标

# example: python backtest.py -m "data/backtest_data/market_data" -i "data/backtest_data/indices_data" -p "data/backtest_data/predictions/results.csv" -s "data/backtest_data/backtest_results"