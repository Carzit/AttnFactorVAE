__all__ = ["Operator", 
           "Constant", "Add", "Subtract", "Multiply", "Divide", "Power", "And", "Or", "Min", "Max", "TernaryOperator", 
           "Open", "Close", "High", "Low", "Vol", "Cap", "IndClass",
           "Returns", "ADV", "DecayLinear", "VWAP", "IndustryNeutralize",
           "Rank", "Abs", "Log", "Sign", "Scale", "Delay", "Delta", 
           "Ts_Corr", "Ts_Cov", "Ts_Max", "Ts_Min", "Ts_Argmax", "Ts_Argmin", "Ts_Rank", "Ts_Sum", "Ts_Mean", "Ts_Product", "Ts_Stddev"]

from numbers import Number
from typing import List, Tuple, Dict, Literal, Union, Optional, Callable, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import utils

class Operator:
    """
    算子基类，支持时间序列计算。
    """
    def __init__(self):
        self.wrapped_operator:Union["Operator", Tuple["Operator",...]] = None
        self.forward_days_required:int = 0

    def calculate(self, *args, **kwds) -> pd.Series:
        pass

    def wrap(self, *operator:Union["Operator", Tuple["Operator","Operator"]]) -> None:
        operator = tuple([Constant(o) if isinstance(o, Number) else o for o in operator])
        self.forward_days_required += max([o.forward_days_required for o in operator])
        if len(operator) == 1:
            operator = operator[0]
        self.wrapped_operator = operator

    def __add__(self, operator:Union["Operator", Number]) -> "Operator":
        return Add(self, operator)
    
    def __radd__(self, operator:Union["Operator", Number]) -> "Operator":
        return Add(operator, self)
    
    def __sub__(self, operator:Union["Operator", Number]) -> "Operator":
        return Subtract(self, operator)
    
    def __rsub__(self, operator:Union["Operator", Number]) -> "Operator":
        return Subtract(operator, self)

    def __mul__(self, operator:Union["Operator", Number]) -> "Operator":
        return Multiply(self, operator)
    
    def __rmul__(self, operator:Union["Operator", Number]) -> "Operator":
        return Multiply(operator, self)
    
    def __rsub__(self, operator:Union["Operator", Number]) -> "Operator":
        return Multiply(operator, self)
    
    def __truediv__(self, operator:Union["Operator", Number]) -> "Operator":
        return Divide(self, operator)
    
    def __rtruediv__(self, operator:Union["Operator", Number]) -> "Operator":
        return Divide(operator, self)

    def __pow__(self, operator:Union["Operator", Number]) -> "Operator":
        return Power(self, operator)

    def __rpow__(self, operator:Union["Operator", Number]) -> "Operator":
        return Power(operator, self)
    
    def __xor__(self, operator:Union["Operator", Number]) -> "Operator":
        return Power(self, operator)
    
    def __rxor__(self, operator:Union["Operator", Number]) -> "Operator":
        return Power(operator, self)

    def __lt__(self, operator:Union["Operator", Number]) -> "Operator":
        return Lessthan(self, operator)

    def __le__(self, operator:Union["Operator", Number]) -> "Operator":
        return LessEqual(self, operator)

    def __gt__(self, operator:Union["Operator", Number]) -> "Operator":
        return Greaterthan(self, operator)

    def __ge__(self, operator:Union["Operator", Number]) -> "Operator":
        return LessEqual(self, operator)

    def __eq__(self, operator:Union["Operator", Number]) -> "Operator":
        return Equal(self, operator)

    def __neq__(self, operator:Union["Operator", Number]) -> "Operator":
        return NotEqual(self, operator)
    
    def __and__(self, operator:Union["Operator", bool]) -> "Operator":
        return And(self, operator)
    
    def __rand__(self, operator:Union["Operator", bool]) -> "Operator":
        return And(operator, self)
    
    def __or__(self, operator:Union["Operator", bool]) -> "Operator":
        return Or(self, operator)
    
    def __ror__(self, operator:Union["Operator", bool]) -> "Operator":
        return Or(operator, self)
    
    def __lshift__(self, d:Number=1) -> "Operator":
        return Delay(self, d)
    
    def __rshift__(self, d:Number=1) -> "Operator":
        return Delay(self, -d)
        
    def __call__(self, *args, **kwds)->pd.Series:
        return self.calculate(*args, **kwds)
    
    def __repr__(self):
        if self.wrapped_operator is None:
            return f"{self.__class__.__name__}"
        elif isinstance(self.wrapped_operator, Operator):
            wrapped_repr = repr(self.wrapped_operator) 
        else:
            wrapped_repr = ", ".join(repr(operator) for operator in self.wrapped_operator) 
        return f"{self.__class__.__name__}({wrapped_repr})"

class Constant(Operator):
    def __init__(self, value):
        super().__init__()
        self.v = value
    
    def calculate(self, day_index:int, file_list:List[str])->Number:
        return self.v
    
    def __repr__(self):
        return str(self.v)

class InfixOperator(Operator):
    def __init__(self, operator1:Union[Operator, Number], operator2:Union[Operator, Number]):
        super().__init__()
        self.wrap(operator1, operator2)

class Add(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[Number, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) + operator2.calculate(day_index, file_list)
    
    def __repr__(self):
        return f"({" + ".join(repr(operator) for operator in self.wrapped_operator) })"
    
class Subtract(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[Number, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) - operator2.calculate(day_index, file_list)
    
    def __repr__(self):
        return f"({" - ".join(repr(operator) for operator in self.wrapped_operator) })"
    
class Multiply(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[Number, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) * operator2.calculate(day_index, file_list)
    
    def __repr__(self):
        return f"({" * ".join(repr(operator) for operator in self.wrapped_operator) })"
    
class Divide(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[Number, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) / operator2.calculate(day_index, file_list)
    
    def __repr__(self):
        return f"({" / ".join(repr(operator) for operator in self.wrapped_operator) })"
    
class Power(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[Number, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) ** operator2.calculate(day_index, file_list)
    
    def __repr__(self):
        return f"({" ** ".join(repr(operator) for operator in self.wrapped_operator) })"

class Greaterthan(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[bool, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) > operator2.calculate(day_index, file_list)
    
    def __repr__(self):
        return f"({" > ".join(repr(operator) for operator in self.wrapped_operator) })"

class GreaterEqual(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[bool, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) >= operator2.calculate(day_index, file_list)
    
    def __repr__(self):
        return f"({" >= ".join(repr(operator) for operator in self.wrapped_operator) })"

class Lessthan(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[bool, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) < operator2.calculate(day_index, file_list)  
    
    def __repr__(self):
        return f"({" < ".join(repr(operator) for operator in self.wrapped_operator) })"

class LessEqual(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[bool, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) <= operator2.calculate(day_index, file_list)
    
    def __repr__(self):
        return f"({" <= ".join(repr(operator) for operator in self.wrapped_operator) })"

class Equal(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[bool, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) == operator2.calculate(day_index, file_list)
    
    def __repr__(self):
        return f"({" == ".join(repr(operator) for operator in self.wrapped_operator) })"

class NotEqual(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[bool, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) != operator2.calculate(day_index, file_list)
    
    def __repr__(self):
        return f"({" != ".join(repr(operator) for operator in self.wrapped_operator) })"

class Or(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[bool, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) | operator2.calculate(day_index, file_list)
    
    def __repr__(self):
        return f"({" | ".join(repr(operator) for operator in self.wrapped_operator) })"
    
class And(InfixOperator):
    def calculate(self, day_index:int, file_list:List[str])->Union[bool, pd.Series]:
        operator1, operator2 = self.wrapped_operator
        return operator1.calculate(day_index, file_list) & operator2.calculate(day_index, file_list)
    
    def __repr__(self):
        return f"({" & ".join(repr(operator) for operator in self.wrapped_operator) })"

class Min(Operator):
    def __init__(self, *operators:Operator):
        super().__init__()
        self.wrap(*operators)

    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        _r = [o.calculate(day_index, file_list) for o in self.wrapped_operator]
        return pd.Series(np.min(_r, axis=0))
    
class Max(Operator):
    def __init__(self, *operators:Operator):
        super().__init__()
        self.wrap(*operators)

    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        _r = [o.calculate(day_index, file_list) for o in self.wrapped_operator]
        return pd.Series(np.max(_r, axis=0))

class TernaryOperator(Operator):
    """三元运算符。类似c语言中的condition_expression ? true_expression : false_expression"""
    def __init__(self, condition_operator:Operator, true_operator:Operator, false_operator:Operator):
        super().__init__()
        self.wrap(condition_operator, true_operator, false_operator)

    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        condition_operator, true_operator, false_operator = self.wrapped_operator
        condition = condition_operator.calculate(day_index, file_list)
        return condition * true_operator.calculate(day_index, file_list) + (1-condition)*false_operator.calculate(day_index, file_list)
    
    def __repr__(self):
        condition_operator, true_operator, false_operator = self.wrapped_operator
        return f"{condition_operator} ? {true_operator} : {false_operator}"
    
class Open(Operator):
    def __init__(self):
        super().__init__()
    
    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        return utils.load_dataframe(path=file_list[day_index])["open"]
    
class Close(Operator):
    def __init__(self):
        super().__init__()
    
    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        return utils.load_dataframe(path=file_list[day_index])["close"]

class High(Operator):
    def __init__(self):
        super().__init__()
    
    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        return utils.load_dataframe(path=file_list[day_index])["high"]
    
class Low(Operator):
    def __init__(self):
        super().__init__()
    
    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        return utils.load_dataframe(path=file_list[day_index])["low"]

class Vol(Operator):
    def __init__(self):
        super().__init__()
    
    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        return utils.load_dataframe(path=file_list[day_index])["vol"]

class Cap(Operator):
    constant:bool=False
    cap:pd.Series=None
    def __init__(self):
        super().__init__()
    
    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        return utils.load_dataframe(path=file_list[day_index])["cap"]
    
    @classmethod
    def enable_constant(cls, cap:pd.Series):
        cls.constant = True
        cls.cap = cap

    @classmethod
    def disable_constant(cls):
        cls.constant = False
        cls.cap = None
    
class IndClass(Operator):
    constant:bool=False
    sector:pd.Series=None
    industry:pd.Series=None
    subindustry:pd.Series=None

    def __init__(self, 
                 level:Literal["sector", "industry", "subindustry"]):
        super().__init__()
        if level == "subindustry":
            level = "industry"
        self.level = level
    
    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        if self.constant:
            if self.level == "sector":
                return self.sector
            elif self.level == "industry":
                return self.industry
            elif self.level == "subindustry":
                if self.subindustry:
                    return self.subindustry
                else:
                    return self.industry
        else:
            return utils.load_dataframe(path=file_list[day_index])[f"indclass.{self.level}"]
    
    @classmethod
    def enable_constant(cls, sector:pd.Series, industry:pd.Series, subindustry:Optional[pd.Series]=None):
        cls.constant = True
        cls.sector = sector
        cls.industry = industry
        cls.subindustry = subindustry

    @classmethod
    def disable_constant(cls):
        cls.constant = False
        cls.sector = None
        cls.industry = None
        cls.subindustry = None

    
    def __repr__(self):
        return f"{self.__class__.__name__}.{self.level}"
    
class Rank(Operator):
    def __init__(self, operator:Operator, ascending=True):
        super().__init__()
        self.ascending=ascending
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        _r = self.wrapped_operator.calculate(day_index, file_list).rank(method="average", na_option="keep", ascending=self.ascending)
        return _r / _r.sum(skipna=True)
    
class Abs(Operator):
    def __init__(self, operator:Operator):
        super().__init__()
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        return self.wrapped_operator.calculate(day_index, file_list).abs()
    
class Log(Operator):
    def __init__(self, operator:Operator):
        super().__init__()
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        return np.log(self.wrapped_operator.calculate(day_index, file_list))
    
class Sign(Operator):
    def __init__(self, operator:Operator):
        super().__init__()
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        return np.sign(self.wrapped_operator.calculate(day_index, file_list))

class Scale(Operator):
    def __init__(self, operator:Operator, a:Number=1):
        super().__init__()
        self.a = a
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str])->pd.Series:
        return self.wrapped_operator.calculate(day_index, file_list) / self.wrapped_operator.calculate(day_index, file_list).sum(skipna=True) * self.a

class TimeSeriesOperator(Operator):
    def __init__(self, d:int=1):
        super().__init__()
        self.d = int(d)
        self.forward_days_required += int(d)

    def __repr__(self):
        if self.wrapped_operator is None:
            return f"{self.__class__.__name__}({self.d})"
        elif isinstance(self.wrapped_operator, Operator):
            wrapped_repr = repr(self.wrapped_operator) + f", d={self.d}"
        else:
            wrapped_repr = ", ".join(repr(operator) for operator in self.wrapped_operator) + f", d={self.d}"
        return f"{self.__class__.__name__}({wrapped_repr})"

class Delay(TimeSeriesOperator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator)

    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        return self.wrapped_operator.calculate(day_index-self.d, file_list)

class Delta(TimeSeriesOperator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator)

    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        return self.wrapped_operator.calculate(day_index, file_list) - Delay(self.wrapped_operator, d=self.d).calculate(day_index, file_list)

class Ts_Corr(TimeSeriesOperator):
    def __init__(self, operator1:Operator, operator2:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator1, operator2)
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        operator1, operator2 = self.wrapped_operator
        series_list1 = [operator1.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        series_list2 = [operator2.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_corr(series_list1, series_list2)

class Ts_Cov(TimeSeriesOperator):
    def __init__(self, operator1:Operator, operator2:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator1, operator2)
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        operator1, operator2 = self.wrapped_operator
        series_list1 = [operator1.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        series_list2 = [operator2.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_cov(series_list1, series_list2)

class Ts_Max(TimeSeriesOperator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_max(series_list)

class Ts_Min(TimeSeriesOperator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_min(series_list)

class Ts_Argmax(TimeSeriesOperator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_argmax(series_list)

class Ts_Argmin(TimeSeriesOperator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_argmin(series_list)

class Ts_Rank(TimeSeriesOperator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_rank(series_list)      

class Ts_Sum(TimeSeriesOperator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_sum(series_list)   

class Ts_Mean(TimeSeriesOperator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_mean(series_list) 

class Ts_Product(TimeSeriesOperator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_prod(series_list) 

class Ts_Stddev(TimeSeriesOperator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator)
    
    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        return ts_std(series_list) 

class IndustryNeutralize(Operator):
    def __init__(self, operator:Operator, level:Union[Literal["sector", "industry", "subindustry"], IndClass]="sector"):
        super().__init__()
        if isinstance(level, str):
            level = IndClass(level)
        self.level:IndClass = level
        self.wrap(operator)

    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        factor_values = self.wrapped_operator.calculate(day_index, file_list)
        industry_classification = self.level.calculate(day_index, file_list)
        return neutralize_factor(factor_values, industry_classification)
    
    def __repr__(self):
        wrapped_repr = repr(self.wrapped_operator) + f", {repr(self.level)}"
        return f"{self.__class__.__name__}({wrapped_repr})"

class Returns(Operator):
    def __init__(self):
        super().__init__()
        self.forward_days_required += 1

    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        return (Delta(Close(), d=1)/Delay(Close(), d=1)).calculate(day_index, file_list)

class ADV(TimeSeriesOperator):
    def __init__(self, d:int=1):
        super().__init__(d)

    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        return Ts_Mean(Vol(), d=self.d).calculate(day_index, file_list)

class DecayLinear(TimeSeriesOperator):
    def __init__(self, operator:Operator, d:int=1):
        super().__init__(d)
        self.wrap(operator)

    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        series_list = [self.wrapped_operator.calculate(day_index-self.d+1+i, file_list) for i in range(self.d)]
        weights = np.arange(self.d, 0, -1)
        return ts_weighted_sum(series_list, weights=weights)
    
class VWAP(TimeSeriesOperator):
    """
    VWAP(Volume-Weighted Average Price)Operator, 成交量加权平均价格算子

        VWAP = sum_{i=1}^{d} TypePrice_i * Vol_i / sum_{i=1}^{d} Vol_i
        TypePrice_i = (High_i + Low_i + Close_i) / 3

    JointQuant使用均价AVG代替。
    这里我们使用典型价格TypePrice代替。
    """

    def __init__(self, d:int=0):
        super().__init__(d)

    def calculate(self, day_index:int, file_list:List[str]) -> pd.Series:
        if self.d == 0:
            return Divide((High()+Low()+Close()), 3).calculate(day_index, file_list)
        else:
            return Divide(Ts_Sum((High()+Low()+Close())/3 * Vol(), d=self.d), Ts_Sum(Vol(), d=self.d)).calculate(day_index, file_list)

def ts_max(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])): 
        max_value = np.max([series_list[j][i] for j in range(d)])
        _r.append(max_value)
    
    return pd.Series(_r)

def ts_min(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])): 
        min_value = np.min([series_list[j][i] for j in range(d)])
        _r.append(min_value)
    
    return pd.Series(_r)

def ts_argmax(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])):  
        argmax_value = np.argmax([series_list[j][i] for j in range(d)])
        _r.append(argmax_value)
    
    return pd.Series(_r)

def ts_argmin(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])):
        argmin_value = np.argmin([series_list[j][i] for j in range(d)])
        _r.append(argmin_value)
    
    return pd.Series(_r)

def ts_rank(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = []
    d = len(series_list)
    
    for i in range(len(series_list[0])): 
        value = pd.Series([series_list[j][i] for j in range(d)]).rank(method="average", na_option="keep", ascending=True)
        rank_value = (value / value.sum(skipna=True))[-1]
        _r.append(rank_value)
    
    return pd.Series(_r)

def ts_sum(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = np.sum(series_list, axis=0)
    return pd.Series(_r)

def ts_mean(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    d = len(series_list)
    _r = np.sum(series_list, axis=0) / d
    return pd.Series(_r)

def ts_weighted_sum(series_list: List[pd.Series], weights):
    weights = weights / weights.sum()
    series_list = align_series_list(series_list)
    _r = np.sum([w * s for w, s in zip(weights, series_list)], axis=0)
    return pd.Series(_r)

def ts_weighted_mean(series_list: List[pd.Series], weights):
    weights = weights / weights.sum()
    series_list = align_series_list(series_list)
    d = len(series_list)
    _r = np.sum([w * s for w, s in zip(weights, series_list)], axis=0) / d
    return pd.Series(_r)

def ts_prod(series_list: List[pd.Series]):
    series_list = align_series_list(series_list)
    _r = np.prod(series_list, axis=0)
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

def neutralize_factor(factor_values, industry_classification):
    """
    剔除行业影响，对因子值进行行业中性化处理。

    参数:
    - factor_values: pd.Series, 因子值，索引为股票代码
    - industry_classification: pd.Series, 行业分类，索引为股票代码

    返回:
    - pd.Series, 剔除行业影响后的因子值（残差）
    """
    # 构建哑变量矩阵
    industry_dummies = pd.get_dummies(industry_classification, prefix='industry')

    # 确保因子值和哑变量矩阵的索引一致
    factor_values = factor_values.reindex(industry_dummies.index)

    # 构建回归模型
    X = industry_dummies.values
    y = factor_values.values.reshape(-1, 1)

    # 线性回归
    model = LinearRegression()
    model.fit(X, y)

    # 提取残差
    residuals = y - model.predict(X)

    # 将残差转换为 Series
    residuals_series = pd.Series(residuals.flatten(), index=factor_values.index)

    return residuals_series

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

if __name__ == "__main__":
    o = DecayLinear(Open()<Close(), d=2)
    a = IndClass("sector")
    b = IndClass("industry")
    print(b.constant)
    a.enable_constant(pd.Series([1,2,3]), pd.Series([4,5,6]))
    print(b.constant)
    print(o.forward_days_required)
    print(o)
    print(o(2, [r"data\test\20130104.csv", r"data\test\20130107.csv", r"data\test\20130108.csv"]))