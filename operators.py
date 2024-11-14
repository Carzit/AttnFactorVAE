import os
import sys
import logging
import argparse
from numbers import Number
from typing import List, Tuple, Dict, Literal, Optional, Callable, Any, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from preparers import LoggerPreparer
import utils

class Operator:
    """
    算子基类，支持时间序列计算。
    """
    def __init__(self):
        self.wrapped_operator:Union["Operator", Tuple["Operator",...]] = None
        self.forward_days_required:int = 0
        self.backward_days_required:int = 0

    def calculate(self, *args, **kwds) -> pd.Series:
        pass

    def wrap(self, *operator:Union["Operator", Tuple["Operator","Operator"]]) -> None:
        operator = tuple([Constant(o) if isinstance(o, Number) else o for o in operator])
        self.forward_days_required:int = max([o.forward_days_required for o in operator])
        self.backward_days_required:int = max([o.backward_days_required for o in operator])
        if len(operator) == 1:
            operator = operator[0]
        self.wrapped_operator = operator

    def __add__(self, operator:Union["Operator", Number]):
        return Add(self, operator)
    
    def __radd__(self, operator:Union["Operator", Number]):
        return Add(operator, self)
    
    def __sub__(self, operator:Union["Operator", Number]):
        return Subtract(self, operator)
    
    def __rsub__(self, operator:Union["Operator", Number]):
        return Subtract(operator, self)

    def __mul__(self, operator:Union["Operator", Number]):
        return Multiply(self, operator)
    
    def __rsub__(self, operator:Union["Operator", Number]):
        return Multiply(operator, self)
    
    def __truediv__(self, operator:Union["Operator", Number]):
        return Divide(self, operator)
    
    def __rtruediv__(self, operator:Union["Operator", Number]):
        return Divide(operator, self)

    def __pow__(self, operator:Union["Operator", Number]):
        return Power(self, operator)

    def __rpow__(self, operator:Union["Operator", Number]):
        return Power(operator, self)

    def __lt__(self, operator:Union["Operator", Number]):
        return Lessthan(self, operator)

    def __le__(self, operator:Union["Operator", Number]):
        return LessEqual(self, operator)

    def __gt__(self, operator:Union["Operator", Number]):
        return Greaterthan(self, operator)

    def __ge__(self, operator:Union["Operator", Number]):
        return LessEqual(self, operator)

    def __eq__(self, operator:Union["Operator", Number]):
        return Equal(self, operator)

    def __neq__(self, operator:Union["Operator", Number]):
        return NotEqual(self, operator)
        
    def __call__(self, *args, **kwds)->pd.Series:
        return self.calculate(*args, **kwds)

class Constant(Operator):
    def __init__(self, value):
        super().__init__()
        self.v = value
    
    def calculate(self, day_index:int, file_list:List[str]):
        return self.v

class InfixOperator(Operator):
    def __init__(self, operator1:Union[Operator, Number], operator2:Union[Operator, Number]):
        super().__init__()
        self.wrap(operator1, operator2)

class Add(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str]):
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) + operator2.calculate(day_index, file_list)
    
class Subtract(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str]):
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) - operator2.calculate(day_index, file_list)
    
class Multiply(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str]):
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) * operator2.calculate(day_index, file_list)
    
class Divide(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str]):
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) / operator2.calculate(day_index, file_list)
    
class Power(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str]):
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) ** operator2.calculate(day_index, file_list)

class Greaterthan(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str]):
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) > operator2.calculate(day_index, file_list)

class GreaterEqual(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str]):
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) >= operator2.calculate(day_index, file_list)

class Lessthan(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str]):
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) < operator2.calculate(day_index, file_list)  

class LessEqual(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str]):
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) <= operator2.calculate(day_index, file_list)

class Equal(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str]):
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) == operator2.calculate(day_index, file_list)

class NotEqual(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str]):
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) != operator2.calculate(day_index, file_list)

class TernaryOperator(Operator):
    """三元运算符。类似c语言中的condition_expression ? true_expression : false_expression"""
    def __init__(self, condition_operator:Operator, true_operator:Operator, false_operator:Operator):
        super().__init__()
        self.wrap(condition_operator, true_operator, false_operator)

    def calculate(self, day_index:int, file_list:List[str]):
        condition_operator, true_operator, false_operator = self.wrapped_operator
        return pd.Series(np.where(condition_operator.calculate(day_index, file_list), 
                        true_operator.calculate(day_index, file_list), 
                        false_operator.calculate(day_index, file_list)))
    
class Open(Operator):
    def __init__(self):
        super().__init__()
    
    def calculate(self, day_index:int, file_list:List[str]):
        return utils.load_dataframe(path=file_list[day_index])["open"]
    
class Close(Operator):
    def __init__(self):
        super().__init__()
    
    def calculate(self, day_index:int, file_list:List[str]):
        return utils.load_dataframe(path=file_list[day_index])["close"]

class High(Operator):
    def __init__(self):
        super().__init__()
    
    def calculate(self, day_index:int, file_list:List[str]):
        return utils.load_dataframe(path=file_list[day_index])["high"]
    
class Low(Operator):
    def __init__(self):
        super().__init__()
    
    def calculate(self, day_index:int, file_list:List[str]):
        return utils.load_dataframe(path=file_list[day_index])["low"]

class Volumn(Operator):
    def __init__(self):
        super().__init__()
    
    def calculate(self, day_index:int, file_list:List[str]):
        return utils.load_dataframe(path=file_list[day_index])["vol"]

class Cap(Operator):
    def __init__(self):
        super().__init__()
    
    def calculate(self, day_index:int, file_list:List[str]):
        return utils.load_dataframe(path=file_list[day_index])["cap"]
    
class Rank(Operator):
    def __init__(self, operator:Operator, ascending=True):
        super().__init__()
        self.ascending=ascending
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]):
        _r = self.wrapped_operator.calculate(day_index, file_list).rank(method="average", na_option="keep", ascending=self.ascending)
        return _r / _r.sum(skipna=True)
    
class Abs(Operator):
    def __init__(self, operator:Operator):
        super().__init__()
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]):
        return self.wrapped_operator.calculate(day_index, file_list).abs()
    
class Log(Operator):
    def __init__(self, operator:Operator):
        super().__init__()
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]):
        return np.log(self.wrapped_operator.calculate(day_index, file_list))
    
class Sign(Operator):
    def __init__(self, operator:Operator):
        super().__init__()
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]):
        return np.sign(self.wrapped_operator.calculate(day_index, file_list))

class Scale(Operator):
    def __init__(self, operator:Operator, a:Number=1):
        super().__init__()
        self.a = a
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]):
        return self.wrapped_operator.calculate(day_index, file_list) / self.wrapped_operator.calculate(day_index, file_list).sum(skipna=True) * self.a

class Delay(Operator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__()
        self.d = d
        self.wrap(operator)
        self.forward_days_required += d

    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        return self.wrapped_operator.calculate(day_index-self.d, file_list)

class Delta(Operator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__()
        self.d = d
        self.wrap(operator)
        self.forward_days_required += d

    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        return self.wrapped_operator.calculate(day_index, file_list) - Delay(self.wrapped_operator, d=self.d).calculate(day_index, file_list)
    
class Ts_Corr(Operator):
    def __init__(self, operator1:Operator, operator2:Operator, d:int=1):
        super().__init__()
        self.d = int(d)
        self.wrap(operator1, operator2)
        self.forward_days_required += d
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        operator1, operator2 = self.wrapped_operator
        series_list1 = [operator1.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        series_list2 = [operator2.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_corr(series_list1, series_list2)

class Ts_Cov(Operator):
    def __init__(self, operator1:Operator, operator2:Operator, d:int=1):
        super().__init__()
        self.d = int(d)
        self.wrap(operator1, operator2)
        self.forward_days_required += d
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        operator1, operator2 = self.wrapped_operator
        series_list1 = [operator1.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        series_list2 = [operator2.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_cov(series_list1, series_list2)

class Ts_Max(Operator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__()
        self.d = int(d)
        self.wrap(operator)
        self.forward_days_required += d
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_max(series_list)

class Ts_Argmax(Operator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__()
        self.d = int(d)
        self.wrap(operator)
        self.forward_days_required += d
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_argmax(series_list)

class Ts_Argmin(Operator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__()
        self.d = int(d)
        self.wrap(operator)
        self.forward_days_required += d
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_argmin(series_list)

class Ts_Rank(Operator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__()
        self.d = int(d)
        self.wrap(operator)
        self.forward_days_required += d
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_rank(series_list)      

class Ts_Sum(Operator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__()
        self.d = int(d)
        self.wrap(operator)
        self.forward_days_required += d
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_sum(series_list)   

class Ts_Mean(Operator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__()
        self.d = int(d)
        self.wrap(operator)
        self.forward_days_required += d
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_mean(series_list) 

class Ts_Product(Operator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__()
        self.d = int(d)
        self.wrap(operator)
        self.forward_days_required += d
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_prod(series_list) 

class Ts_Stddev(Operator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__()
        self.d = int(d)
        self.wrap(operator)
        self.forward_days_required += d
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_std(series_list) 

def ts_max(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])):  # 样本数
        max_value = np.max([series_list[j][i] for j in range(d)])
        _r.append(max_value)
    
    return pd.Series(_r)

def ts_min(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])):  # 样本数
        min_value = np.min([series_list[j][i] for j in range(d)])
        _r.append(min_value)
    
    return pd.Series(_r)

def ts_argmax(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])):  # 样本数
        argmax_value = np.argmax([series_list[j][i] for j in range(d)])
        _r.append(argmax_value)
    
    return pd.Series(_r)

def ts_argmin(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])):  # 样本数
        argmin_value = np.argmin([series_list[j][i] for j in range(d)])
        _r.append(argmin_value)
    
    return pd.Series(_r)

def ts_rank(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])):  # 样本数
        value = pd.Series([series_list[j][i] for j in range(d)]).rank(method="average", na_option="keep", ascending=True)
        rank_value = (value / value.sum(skipna=True))[-1]
        _r.append(rank_value)
    
    return pd.Series(_r)

def ts_sum(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])):  # 样本数
        sum_value = np.sum([series_list[j][i] for j in range(d)])
        _r.append(sum_value)
    
    return pd.Series(_r)

def ts_mean(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])):  # 样本数
        mean_value = np.mean([series_list[j][i] for j in range(d)])
        _r.append(mean_value)
    
    return pd.Series(_r)

def ts_prod(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])):  # 样本数
        prod_value = np.prod([series_list[j][i] for j in range(d)])
        _r.append(prod_value)
    
    return pd.Series(_r)

def ts_std(series_list: List[pd.Series]):

    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])):  # 样本数
        std_value = np.std([series_list[j][i] for j in range(d)])
        _r.append(std_value)
    
    return pd.Series(_r)

def ts_corr(series_list1: List[pd.Series], series_list2: List[pd.Series]) -> pd.Series:
    """
    计算两个变量在d天上的相关性。
    
    :param series_list1: 第一个变量的时间序列列表，每个元素是一个pd.Series。
    :param series_list2: 第二个变量的时间序列列表，每个元素是一个pd.Series。
    :return: 每个样本在d天上的相关性，返回一个pd.Series。
    """
    # 确保两组series的长度一致
    if len(series_list1) != len(series_list2):
        raise ValueError("The length of both series lists must be the same.")
    series_list1 = align_series_list(series_list1)
    series_list2 = align_series_list(series_list2)
    
    _r = []
    d = len(series_list1)  # d天
    
    # 对每个样本，计算该样本在d天上的相关性
    for i in range(len(series_list1[0])):  # 样本数
        data_var1 = [series_list1[j][i] for j in range(d)]  # 提取第i个样本在d天上的数据
        data_var2 = [series_list2[j][i] for j in range(d)]  # 提取第i个样本在d天上的数据
        
        # 计算当前样本的相关性
        corr = pd.Series(data_var1).corr(pd.Series(data_var2))
        
        # 将结果存入列表
        _r.append(corr)
    
    # 将所有样本的相关性合并为一个Series并返回
    return pd.Series(_r)

def ts_cov(series_list1: List[pd.Series], series_list2: List[pd.Series]) -> pd.Series:

    if len(series_list1) != len(series_list2):
        raise ValueError("The length of both series lists must be the same.")
    series_list1 = align_series_list(series_list1)
    series_list2 = align_series_list(series_list2)
    
    _r = []
    d = len(series_list1)  # d天
    
    # 对每个样本，计算该样本在d天上的相关性
    for i in range(len(series_list1[0])):  # 样本数
        data_var1 = [series_list1[j][i] for j in range(d)]  # 提取第i个样本在d天上的数据
        data_var2 = [series_list2[j][i] for j in range(d)]  # 提取第i个样本在d天上的数据

        corr = pd.Series(data_var1).cov(pd.Series(data_var2))
        _r.append(corr)
    
    # 将所有样本的相关性合并为一个Series并返回
    return pd.Series(_r)

def align_series_list(series_list: List[pd.Series]) -> List[pd.Series]:
    """
    对齐series列表中的所有series，使得它们的长度一致，短的补NaN。
    
    :param series_list: 需要对齐的series列表。
    :return: 返回长度一致的series列表，短的series会被补充NaN。
    """
    # 找到最长的series长度
    max_len = max(len(series) for series in series_list)
    
    # 对齐所有series
    aligned_series_list = []
    for series in series_list:
        # 对每个series，使用reindex对齐，并补充NaN
        aligned_series = series.reindex(range(max_len))
        aligned_series_list.append(aligned_series)
    
    return aligned_series_list


o = TernaryOperator(Open()<Close(), Close(), Open())
print(o.forward_days_required)
print(o(3, [r"data\test\20130104.csv", r"data\test\20130107.csv", r"data\test\20130108.csv", r"data\test\20130109.csv", r"data\test\20130110.csv", r"data\test\20130111.csv"]))