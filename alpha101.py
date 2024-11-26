import os
import sys
import logging
import argparse
from typing import List, Tuple, Dict, Literal, Union, Optional, Callable, Any

from tqdm import tqdm
import pandas as pd

from operators import *
from preparers import LoggerPreparer
import utils

alpha001 = Rank(Ts_Argmax(Power(TernaryOperator(Returns()<0, Ts_Stddev(Returns(), 5), Close()), 2), d=5)) - 0.5
alpha002 = -1 * Ts_Corr(Rank(Delta(Log(Vol()), d=2)), Rank((Close()-Open())/Open()), d=6)
alpha003 = -1 * Ts_Corr(Rank(Open()), Rank(Vol()), 10)
alpha004 = -1 * Ts_Rank(Rank(Low()), 9)
alpha005 = Rank(Open() - (Ts_Sum(VWAP(), 10) / 10)) * (-1 * Abs(Rank(Close() - VWAP())))
alpha006 = -1 * Ts_Corr(Open(), Vol(), 10)
alpha007 = TernaryOperator((ADV(20) < Vol()) , (-1 * Ts_Rank(Abs(Delta(Close(), 7)), 60) * Sign(Delta(Close(), 7))) , (-1
* 1))
alpha008 = -1 * Rank(((Ts_Sum(Open(), 5) * Ts_Sum(Returns(), 5)) - Delay((Ts_Sum(Open(), 5) * Ts_Sum(Returns(), 5)),10)))
alpha009 = TernaryOperator((0 < Ts_Min(Delta(Close(), 1), 5)) , Delta(Close(), 1) , TernaryOperator((Ts_Max(Delta(Close(), 1), 5) < 0), Delta(Close(), 1) , (-1 * Delta(Close(), 1))))
alpha010 = Rank(TernaryOperator((0 < Ts_Min(Delta(Close(), 1), 4)) , Delta(Close(), 1) , TernaryOperator((Ts_Max(Delta(Close(), 1), 4) < 0), Delta(Close(), 1) , (-1 * Delta(Close(), 1)))))

alpha011 = (Rank(Ts_Max((VWAP() - Close()), 3)) + Rank(Ts_Min((VWAP() - Close()), 3))) * Rank(Delta(Vol(), 3))
alpha012 = Sign(Delta(Vol(), 1)) * (-1 * Delta(Close(), 1))
alpha013 = -1 * Rank(Ts_Cov(Rank(Close()), Rank(Vol()), 5))
alpha014 = (-1 * Rank(Delta(Returns(), 3))) * Ts_Corr(Open(), Vol(), 10)
alpha015 = -1 * Ts_Sum(Rank(Ts_Corr(Rank(High()), Rank(Vol()), 3)), 3)
alpha016 = -1 * Rank(Ts_Cov(Rank(High()), Rank(Vol()), 5))
alpha017 = ((-1 * Rank(Ts_Rank(Close(), 10))) * Rank(Delta(Delta(Close(), 1), 1))) * Rank(Ts_Rank(Vol() / ADV(20), 5))
alpha018 = -1 * Rank(((Ts_Stddev(Abs(Close() - Open()), 5) + (Close() - Open())) + Ts_Corr(Close(), Open(), 10)))
alpha019 = (-1 * Sign(((Close() - Delay(Close(), 7)) + Delta(Close(), 7)))) * (1 + Rank((1 + Ts_Sum(Returns(), 250))))
alpha020 = ((-1 * Rank((Open() - Delay(High(), 1)))) * Rank((Open() - Delay(Close(), 1)))) * Rank((Open() - Delay(Low(), 1)))

alpha021 = TernaryOperator((((Ts_Sum(Close(), 8) / 8) + Ts_Stddev(Close(), 8)) < (Ts_Sum(Close(), 2) / 2)) , (-1 * 1) , TernaryOperator(((Ts_Sum(Close(), 2) / 2) < ((Ts_Sum(Close(), 8) / 8) - Ts_Stddev(Close(), 8))) , 1 , TernaryOperator(((1 < (Vol() / ADV(20))) | ((Vol() / ADV(20)) == 1)) , 1 , (-1 * 1))))
alpha022 = -1 * (Delta(Ts_Corr(High(), Vol(), 5), 5) * Rank(Ts_Stddev(Close(), 20)))
alpha023 = TernaryOperator(((Ts_Sum(High(), 20) / 20) < High()) , (-1 * Delta(High(), 2)) , 0)
alpha024 = TernaryOperator((((Delta((Ts_Sum(Close(), 100) / 100), 100) / Delay(Close(), 100)) < 0.05) | ((Delta((Ts_Sum(Close(), 100) / 100), 100) / Delay(Close(), 100)) == 0.05)) , (-1 * (Close() - Ts_Min(Close(), 100))) , (-1 * Delta(Close(), 3)))
alpha025 = Rank(((((-1 * Returns()) * ADV(20)) * VWAP()) * (High() - Close())))
alpha026 = -1 * Ts_Max(Ts_Corr(Ts_Rank(Vol(), 5), Ts_Rank(High(), 5), 5), 3)
alpha027 = TernaryOperator((0.5 < Rank((Ts_Sum(Ts_Corr(Rank(Vol()), Rank(VWAP()), 6), 2) / 2.0))) , (-1 * 1) , 1)
alpha028 = Scale(((Ts_Corr(ADV(20), Low(), 5) + ((High() + Low()) / 2)) - Close()))
alpha029 = Ts_Min(Ts_Product(Rank(Rank(Scale(Log(Ts_Sum(Ts_Min(Rank(Rank((-1 * Rank(Delta((Close() - 1),
5))))), 2), 1))))), 1), 5) + Ts_Rank(Delay((-1 * Returns()), 6), 5)

alpha030 = ((1.0 - Rank(((Sign((Close() - Delay(Close(), 1))) + Sign((Delay(Close(), 1) - Delay(Close(), 2)))) + Sign((Delay(Close(), 2) - Delay(Close(), 3)))))) * Ts_Sum(Vol(), 5)) / Ts_Sum(Vol(), 20)
alpha031 = ((Rank(Rank(Rank(DecayLinear((-1 * Rank(Rank(Delta(Close(), 10)))), 10)))) + Rank((-1 * Delta(Close(), 3)))) + Sign(Scale(Ts_Corr(ADV(20), Low(), 12))))
alpha032 = Scale(((Ts_Sum(Close(), 7) / 7) - Close())) + (20 * Scale(Ts_Corr(VWAP(), Delay(Close(), 5), 230)))
alpha033 = Rank((-1 * ((1 - (Open() / Close()))**1)))
alpha034 = Rank(((1 - Rank((Ts_Stddev(Returns(), 2) / Ts_Stddev(Returns(), 5)))) + (1 - Rank(Delta(Close(), 1)))))
alpha035 = (Ts_Rank(Vol(), 32) * (1 - Ts_Rank(((Close() + High()) - Low()), 16))) * (1 - Ts_Rank(Returns(), 32))
alpha036 = ((((2.21 * Rank(Ts_Corr((Close() - Open()), Delay(Vol(), 1), 15))) + (0.7 * Rank((Open() - Close())))) + (0.73 * Rank(Ts_Rank(Delay((-1 * Returns()), 6), 5)))) + Rank(Abs(Ts_Corr(VWAP(), ADV(20), 6)))) + (0.6 * Rank((((Ts_Sum(Close(), 200) / 200) - Open()) * (Close() - Open()))))
alpha037 = Rank(Ts_Corr(Delay((Open() - Close()), 1), Close(), 200)) + Rank((Open() - Close()))
alpha038 = (-1 * Rank(Ts_Rank(Close(), 10))) * Rank((Close() / Open()))
alpha039 = (-1 * Rank((Delta(Close(), 7) * (1 - Rank(DecayLinear((Vol() / ADV(20)), 9)))))) * (1 + Rank(Ts_Sum(Returns(), 250)))
alpha040 = (-1 * Rank(Ts_Stddev(High(), 10))) * Ts_Corr(High(), Vol(), 10)

alpha041 = ((High() * Low()) ** 0.5) - VWAP()
alpha042 = Rank((VWAP() - Close())) / Rank((VWAP() + Close()))
alpha043 = Ts_Rank((Vol() / ADV(20)), 20) * Ts_Rank((-1 * Delta(Close(), 7)), 8)
alpha044 = -1 * Ts_Corr(High(), Rank(Vol()), 5)
alpha045 = -1 * ((Rank((Ts_Sum(Delay(Close(), 5), 20) / 20)) * Ts_Corr(Close(), Vol(), 2)) * Rank(Ts_Corr(Ts_Sum(Close(), 5), Ts_Sum(Close(), 20), 2)))
alpha046 = TernaryOperator((0.25 < (((Delay(Close(), 20) - Delay(Close(), 10)) / 10) - ((Delay(Close(), 10) - Close()) / 10))) , (-1 * 1) , TernaryOperator(((((Delay(Close(), 20) - Delay(Close(), 10)) / 10) - ((Delay(Close(), 10) - Close()) / 10)) < 0) , 1 , ((-1 * 1) * (Close() - Delay(Close(), 1)))))
alpha047 = (((Rank((1 / Close())) * Vol()) / ADV(20)) * ((High() * Rank((High() - Close()))) / (Ts_Sum(High(), 5) / 5))) - Rank((VWAP() - Delay(VWAP(), 5)))
alpha048 = IndustryNeutralize(((Ts_Corr(Delta(Close(), 1), Delta(Delay(Close(), 1), 1), 250) * Delta(Close(), 1)) / Close()), IndClass("subindustry")) / Ts_Sum(((Delta(Close(), 1) / Delay(Close(), 1))**2), 250)
alpha049 = TernaryOperator(((((Delay(Close(), 20) - Delay(Close(), 10)) / 10) - ((Delay(Close(), 10) - Close()) / 10)) < (-1 *
0.1)) , 1 , ((-1 * 1) * (Close() - Delay(Close(), 1))))
alpha050 = -1 * Ts_Max(Rank(Ts_Corr(Rank(Vol()), Rank(VWAP()), 5)), 5)

alpha051 = TernaryOperator(((((Delay(Close(), 20) - Delay(Close(), 10)) / 10) - ((Delay(Close(), 10) - Close()) / 10)) < (-1 *
0.05)), 1 , ((-1 * 1) * (Close() - Delay(Close(), 1))))
alpha052 = (((-1 * Ts_Min(Low(), 5)) + Delay(Ts_Min(Low(), 5), 5)) * Rank(((Ts_Sum(Returns(), 240) - Ts_Sum(Returns(), 20)) / 220))) * Ts_Rank(Vol(), 5)
alpha053 = -1 * Delta((((Close() - Low()) - (High() - Close())) / (Close() - Low())), 9)
alpha054 = (-1 * ((Low() - Close()) * (Open() ** 5))) / ((Low() - High()) * (Close() ** 5))
alpha055 = -1 * Ts_Corr(Rank(((Close() - Ts_Min(Low(), 12)) / (Ts_Max(High(), 12) - Ts_Min(Low(), 12)))), Rank(Vol()), 6)
alpha056 = 0 - (1 * (Rank((Ts_Sum(Returns(), 10) / Ts_Sum(Ts_Sum(Returns(), 2), 3))) * Rank((Returns() * Cap()))))
alpha057 = 0 - (1 * ((Close() - VWAP()) / DecayLinear(Rank(Ts_Argmax(Close(), 30)), 2)))
alpha058 = -1 * Ts_Rank(DecayLinear(Ts_Corr(IndustryNeutralize(VWAP(), IndClass("sector")), Vol(), 3.92795), 7.89291), 5.50322)
alpha059 = -1 * Ts_Rank(DecayLinear(Ts_Corr(IndustryNeutralize(((VWAP() * 0.728317) + (VWAP() * (1 - 0.728317))), IndClass("industry")), Vol(), 4.25197), 16.2289), 8.19648)
alpha060 = 0 - (1 * ((2 * Scale(Rank(((((Close() - Low()) - (High() - Close())) / (High() - Low())) * Vol())))) - Scale(Rank(Ts_Argmax(Close(), 10)))))

alpha061 = Rank((VWAP() - Ts_Min(VWAP(), 16.1219))) < Rank(Ts_Corr(VWAP(), ADV(180), 17.9282))
alpha062 = (Rank(Ts_Corr(VWAP(), Ts_Sum(ADV(20), 22.4101), 9.91009)) < Rank(((Rank(Open()) + Rank(Open())) < (Rank(((High() + Low()) / 2)) + Rank(High()))))) * (-1)
alpha063 = (Rank(DecayLinear(Delta(IndustryNeutralize(Close(), IndClass("industry")), 2.25164), 8.22237)) - Rank(DecayLinear(Ts_Corr(((VWAP() * 0.318108) + (Open() * (1 - 0.318108))), Ts_Sum(ADV(180), 37.2467), 13.557), 12.2883))) * (-1)
alpha064 = (Rank(Ts_Corr(Ts_Sum(((Open() * 0.178404) + (Low() * (1 - 0.178404))), 12.7054), Ts_Sum(ADV(120), 12.7054), 16.6208)) < Rank(Delta(((((High() + Low()) / 2) * 0.178404) + (VWAP() * (1 - 0.178404))), 3.69741))) * (-1)
alpha065 = (Rank(Ts_Corr(((Open() * 0.00817205) + (VWAP() * (1 - 0.00817205))), Ts_Sum(ADV(60), 8.6911), 6.40374)) < Rank((Open() - Ts_Min(Open(), 13.635)))) * (-1)
alpha066 = (Rank(DecayLinear(Delta(VWAP(), 3.51013), 7.23052)) + Ts_Rank(DecayLinear(((((Low() * 0.96633) + (Low() * (1 - 0.96633))) - VWAP()) / (Open() - ((High() + Low()) / 2))), 11.4157), 6.72611)) * (-1)
alpha067 = (Rank((High() - Ts_Min(High(), 2.14593))) ** Rank(Ts_Corr(IndustryNeutralize(VWAP(), IndClass("sector")), IndustryNeutralize(ADV(20), IndClass("subindustry")), 6.02936))) * (-1)
alpha068 = (Ts_Rank(Ts_Corr(Rank(High()), Rank(ADV(15)), 8.91644), 13.9333) < Rank(Delta(((Close() * 0.518371) + (Low() * (1 - 0.518371))), 1.06157))) * (-1)
alpha069 = (Rank(Ts_Max(Delta(IndustryNeutralize(VWAP(), IndClass("industry")), 2.72412), 4.79344))**Ts_Rank(Ts_Corr(((Close() * 0.490655) + (VWAP() * (1 - 0.490655))), ADV(20), 4.92416), 9.0615)) * (-1)
alpha070 = (Rank(Delta(VWAP(), 1.29456))**Ts_Rank(Ts_Corr(IndustryNeutralize(Close(), IndClass("industry")), ADV(50), 17.8256), 17.9171)) * (-1)

alpha071 = Max(Ts_Rank(DecayLinear(Ts_Corr(Ts_Rank(Close(), 3.43976), Ts_Rank(ADV(180), 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(DecayLinear((Rank(((Low() + Open()) - (VWAP() + VWAP())))^2), 16.4662), 4.4388))
alpha072 = Rank(DecayLinear(Ts_Corr(((High() + Low()) / 2), ADV(40), 8.93345), 10.1519)) / Rank(DecayLinear(Ts_Corr(Ts_Rank(VWAP(), 3.72469), Ts_Rank(Vol(), 18.5188), 6.86671), 2.95011))
alpha073 = Max(Rank(DecayLinear(Delta(VWAP(), 4.72775), 2.91864)), Ts_Rank(DecayLinear(((Delta(((Open() * 0.147155) + (Low() * (1 - 0.147155))), 2.03608) / ((Open() * 0.147155) + (Low() * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * (-1)
alpha074 = (Rank(Ts_Corr(Close(), Ts_Sum(ADV(30), 37.4843), 15.1365)) < Rank(Ts_Corr(Rank(((High() * 0.0261661) + (VWAP() * (1 - 0.0261661)))), Rank(Vol()), 11.4791))) * (-1)
alpha075 = Rank(Ts_Corr(VWAP(), Vol(), 4.24304)) < Rank(Ts_Corr(Rank(Low()), Rank(ADV(50)), 12.4413))
alpha076 = Max(Rank(DecayLinear(Delta(VWAP(), 1.24383), 11.8259)), Ts_Rank(DecayLinear(Ts_Rank(Ts_Corr(IndustryNeutralize(Low(), IndClass("sector")), ADV(81), 8.14941), 19.569), 17.1543), 19.383)) * (-1)
alpha077 = Min(Rank(DecayLinear(((((High() + Low()) / 2) + High()) - (VWAP() + High())), 20.0451)), Rank(DecayLinear(Ts_Corr(((High() + Low()) / 2), ADV(40), 3.1614), 5.64125)))
alpha078 = Rank(Ts_Corr(Ts_Sum(((Low() * 0.352233) + (VWAP() * (1 - 0.352233))), 19.7428), Ts_Sum(ADV(40), 19.7428), 6.83313)) ** Rank(Ts_Corr(Rank(VWAP()), Rank(Vol()), 5.77492))
alpha079 = Rank(Delta(IndustryNeutralize(((Close() * 0.60733) + (Open() * (1 - 0.60733))), IndClass("sector")), 1.23438)) < Rank(Ts_Corr(Ts_Rank(VWAP(), 3.60973), Ts_Rank(ADV(150), 9.18637), 14.6644))
alpha080 = (Rank(Sign(Delta(IndustryNeutralize(((Open() * 0.868128) + (High() * (1 - 0.868128))), IndClass("industry")), 4.04545)))^Ts_Rank(Ts_Corr(High(), ADV(10), 5.11456), 5.53756)) * (-1)
alpha081 = (Rank(Log(Ts_Product(Rank((Rank(Ts_Corr(VWAP(), Ts_Sum(ADV(10), 49.6054), 8.47743))**4)), 14.9655))) < Rank(Ts_Corr(Rank(VWAP()), Rank(Vol()), 5.07914))) * (-1)
alpha082 = Min(Rank(DecayLinear(Delta(Open(), 1.46063), 14.8717)), Ts_Rank(DecayLinear(Ts_Corr(IndustryNeutralize(Vol(), IndClass("sector")), ((Open() * 0.634196) + (Open() * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * (-1)
alpha083 = (Rank(Delay(((High() - Low()) / (Ts_Sum(Close(), 5) / 5)), 2)) * Rank(Rank(Vol()))) / (((High() - Low()) / (Ts_Sum(Close(), 5) / 5)) / (VWAP() - Close()))
alpha084 = Power(Ts_Rank((VWAP() - Ts_Max(VWAP(), 15.3217)), 20.7127), Delta(Close(), 4.96796))
alpha085 = Rank(Ts_Corr(((High() * 0.876703) + (Close() * (1 - 0.876703))), ADV(30), 9.61331)) ** Rank(Ts_Corr(Ts_Rank(((High() + Low()) / 2), 3.70596), Ts_Rank(Vol(), 10.1595), 7.11408))
alpha086 = (Ts_Rank(Ts_Corr(Close(), Ts_Sum(ADV(20), 14.7444), 6.00049), 20.4195) < Rank(((Open() + Close()) - (VWAP() + Open())))) * (-1)
alpha087 = Max(Rank(DecayLinear(Delta(((Close() * 0.369701) + (VWAP() * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(DecayLinear(Abs(Ts_Corr(IndustryNeutralize(ADV(81), IndClass("industry")), Close(), 13.4132)), 4.89768), 14.4535)) * (-1)
alpha088 = Min(Rank(DecayLinear(((Rank(Open()) + Rank(Low())) - (Rank(High()) + Rank(Close()))), 8.06882)), Ts_Rank(DecayLinear(Ts_Corr(Ts_Rank(Close(), 8.44728), Ts_Rank(ADV(60), 20.6966), 8.01266), 6.65053), 2.61957))
alpha089 = Ts_Rank(DecayLinear(Ts_Corr(((Low() * 0.967285) + (Low() * (1 - 0.967285))), ADV(10), 6.94279), 5.51607), 3.79744) - Ts_Rank(DecayLinear(Delta(IndustryNeutralize(VWAP(), IndClass("industry")), 3.48158), 10.1466), 15.3012)
alpha090 = (Rank((Close() - Ts_Max(Close(), 4.66719))) ** Ts_Rank(Ts_Corr(IndustryNeutralize(ADV(40), IndClass("subindustry")), Low(), 5.38375), 3.21856)) * (-1)

alpha091 = (Ts_Rank(DecayLinear(DecayLinear(Ts_Corr(IndustryNeutralize(Close(), IndClass("industry")), Vol(), 9.74928), 16.398), 3.83219), 4.8667) - Rank(DecayLinear(Ts_Corr(VWAP(), ADV(30), 4.01303), 2.6809))) * (-1)
alpha092 = Min(Ts_Rank(DecayLinear(((((High() + Low()) / 2) + Close()) < (Low() + Open())), 14.7221), 18.8683), Ts_Rank(DecayLinear(Ts_Corr(Rank(Low()), Rank(ADV(30)), 7.58555), 6.94024), 6.80584))
alpha093 = Ts_Rank(DecayLinear(Ts_Corr(IndustryNeutralize(VWAP(), IndClass("industry")), ADV(81), 17.4193), 19.848), 7.54455) / Rank(DecayLinear(Delta(((Close() * 0.524434) + (VWAP() * (1 - 0.524434))), 2.77377), 16.2664))
alpha094 = (Rank((VWAP() - Ts_Min(VWAP(), 11.5783))) ** Ts_Rank(Ts_Corr(Ts_Rank(VWAP(), 19.6462), Ts_Rank(ADV(60), 4.02992), 18.0926), 2.70756)) * (-1)
alpha095 = Rank((Open() - Ts_Min(Open(), 12.4105))) < Ts_Rank((Rank(Ts_Corr(Ts_Sum(((High() + Low()) / 2), 19.1351), Ts_Sum(ADV(40), 19.1351), 12.8742))**5), 11.7584)
alpha096 = max(Ts_Rank(DecayLinear(Ts_Corr(Rank(VWAP()), Rank(Vol()), 3.83878), 4.16783), 8.38151), Ts_Rank(DecayLinear(Ts_Argmax(Ts_Corr(Ts_Rank(Close(), 7.45404), Ts_Rank(ADV(60), 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * (-1)
alpha097 = (Rank(DecayLinear(Delta(IndustryNeutralize(((Low() * 0.721001) + (VWAP() * (1 - 0.721001))), IndClass("industry")), 3.3705), 20.4523)) - Ts_Rank(DecayLinear(Ts_Rank(Ts_Corr(Ts_Rank(Low(), 7.87871), Ts_Rank(ADV(60), 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * (-1)
alpha098 = Rank(DecayLinear(Ts_Corr(VWAP(), Ts_Sum(ADV(5), 26.4719), 4.58418), 7.18088)) - Rank(DecayLinear(Ts_Rank(Ts_Argmin(Ts_Corr(Rank(Open()), Rank(ADV(15)), 20.8187), 8.62571), 6.95668), 8.07206))
alpha099 = (Rank(Ts_Corr(Ts_Sum(((High() + Low()) / 2), 19.8975), Ts_Sum(ADV(60), 19.8975), 8.8136)) < Rank(Ts_Corr(Low(), Vol(), 6.28259))) * -1
alpha100 = (0 - (1 * (((1.5 * Scale(IndustryNeutralize(IndustryNeutralize(Rank(((((Close() - Low()) - (High() - Close())) / (High() - Low())) * Vol())), IndClass("subindustry")), IndClass("subindustry")))) - Scale(IndustryNeutralize((Ts_Corr(Close(), Rank(ADV(20)), 5) - Rank(Ts_Argmin(Close(), 30))), IndClass("subindustry")))) * (Vol() / ADV(20)))))
alpha101 = (Close() - Open()) / ((High() - Low()) + 0.001)

class SingleAlphaProcessor:
    alphas = [alpha001, alpha002, alpha003, alpha004, alpha005, alpha006, alpha007, alpha008, alpha009, alpha010, alpha011, alpha012, alpha013, alpha014, alpha015, alpha016, alpha017, alpha018, alpha019, alpha020, alpha021, alpha022, alpha023, alpha024, alpha025, alpha026, alpha027, alpha028, alpha029, alpha030, alpha031, alpha032, alpha033, alpha034, alpha035, alpha036, alpha037, alpha038, alpha039, alpha040, alpha041, alpha042, alpha043, alpha044, alpha045, alpha046, alpha047, alpha048, alpha049, alpha050, alpha051, alpha052, alpha053, alpha054, alpha055, alpha056, alpha057, alpha058, alpha059, alpha060, alpha061, alpha062, alpha063, alpha064, alpha065, alpha066, alpha067, alpha068, alpha069, alpha070, alpha071, alpha072, alpha073, alpha074, alpha075, alpha076, alpha077, alpha078, alpha079, alpha080, alpha081, alpha082, alpha083, alpha084, alpha085, alpha086, alpha087, alpha088, alpha089, alpha090, alpha091, alpha092, alpha093, alpha094, alpha095, alpha096, alpha097, alpha098, alpha099, alpha100, alpha101]
    def __init__(self, 
                 name:str, 
                 file_list:List[str], 
                 save_folder:str,
                 save_format:Literal["auto", "csv", "pkl", "parquet", "feather"] = "csv"):
        self.alpha_name = name.lower()
        self.operator:Operator = self.alphas[int(self.alpha_name.removeprefix("alpha"))-1]

        self.logger:logging.Logger

        self.file_list = file_list
        self.save_folder = os.path.join(save_folder, self.alpha_name)
        self.save_format:Literal["auto", "csv", "pkl", "parquet", "feather"] = save_format

    def process(self):
        forward_days_required = self.operator.forward_days_required
        for i in tqdm(range(forward_days_required, len(self.file_list)), desc=self.alpha_name):
            date_name = os.path.splitext(os.path.basename(self.file_list[i]))[0]
            daily_alpha = self.operator(i , self.file_list).rename(self.alpha_name, inplace=True)
            utils.save_dataframe(df=daily_alpha, 
                                 path=os.path.join(self.save_folder, date_name), 
                                 format=self.format)
            
class AlphasProcessor:
    def __init__(self, 
                 read_folder:str, 
                 save_folder:str,
                 read_format:Literal["auto", "csv", "pkl", "parquet", "feather"] = "auto",
                 save_format:Literal["auto", "csv", "pkl", "parquet", "feather"] = "csv"):
        
        self.read_folder:str = read_folder
        self.save_folder:str = save_folder

        self.read_format:Literal["csv", "pkl", "parquet", "feather"] = read_format
        self.save_format:Literal["csv", "pkl", "parquet", "feather"] = save_format

        self.file_list:List[str] = [os.path.join(self.read_folder, f) for f in os.listdir(self.read_folder) if f.endswith(self.read_format)]
        self.stock_code:pd.Series

        self.logger:logging.Logger

        self.alpha_names:List[str] = [f"alpha{i:03d}" for i in range(1, 102)]
        self.alpha_processors:List[SingleAlphaProcessor] = [SingleAlphaProcessor(name=n, 
                                                                                 file_list=self.file_list, 
                                                                                 save_folder=self.save_folder,
                                                                                 save_format=self.save_format) for n in self.alpha_names]
        self.alpha_forward_days:List[int] = [processor.operator.forward_days_required for processor in self.alpha_processors]

    def set_loggger(self, logger:logging.Logger):
        self.logger = logger
        for processor in self.alpha_processors:
            processor.logger = logger

    def check_consistency(self, file_list:List[str], col_name:str='ts_code'):
        if not file_list:
            self.logger.warning("No matched file in read folder")
            return False

        reference_ts_code = None
        for file in file_list:
            df = utils.load_dataframe(file, format=self.read_format)
            if col_name not in df.columns:
                self.logger.warning(f"File `{file}` does not have `{col_name}` column")
                return False

            # 获取当前文件的 ts_code 列内容
            current_ts_code = df['ts_code'].tolist()

            # 如果是第一个文件，保存其 ts_code 列作为参考
            if reference_ts_code is None:
                reference_ts_code = current_ts_code
            else:
                # 比较 ts_code 列内容是否一致（考虑顺序）
                if reference_ts_code != current_ts_code:
                    self.logger.warning(f"Inconsistency occurred in file `{file}`")
                    return False

        self.logger.info("Consistency check passed")
        self.stock_code = utils.load_dataframe(self.file_list[0], format=self.read_format)[col_name]
        return True
    
    def load_constant(self, cap_file:str, sector_file:str, industry_file:str, subindustry_file:str):
        cap = utils.load_dataframe(cap_file, format=self.read_format)
        sector = utils.load_dataframe(sector_file, format=self.read_format)
        industry = utils.load_dataframe(industry_file, format=self.read_format)
        subindustry = utils.load_dataframe(subindustry_file, format=self.read_format)
        Cap.enable_constant(cap)
        IndClass.enable_constant(sector, industry, subindustry)

    def process(self):
        if not self.check_consistency(self.file_list):
            raise ValueError(f"Inconsistency in {self.read_folder}")

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        for processor in self.alpha_processors:
            processor.process()
        
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate Alpha101")
    parser.add_argument("--read_folder", type=str, required=True, help="Folder path to read data")
    parser.add_argument("--save_folder", type=str, required=True, help="Folder path to save data")
    parser.add_argument("--read_format", type=str, default="auto", choices=["auto", "csv", "pkl", "parquet", "feather"], help="Read file format")
    parser.add_argument("--save_format", type=str, default="csv", choices=["auto", "csv", "pkl", "parquet", "feather"], help="Save file format")
    parser.add_argument("--cap", type=str, default=None, help="(Optional)Path to cap file. If specified, Cap data will be used as constant")
    parser.add_argument("--sector", type=str, default=None, help="(Optional)Path to sector file. If specified, Sector data will be used as constant")
    parser.add_argument("--industry", type=str, default=None, help="(Optional)Path to industry file. If specified, Industry data will be used as constant")
    parser.add_argument("--subindustry", type=str, default=None, help="(Optional)Path to subindustry file. If specified, Subindustry data will be used as constant")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    logger = LoggerPreparer(name="Alpha101", 
                            file_level=logging.INFO, 
                            log_file=args.log_path).prepare()
    logger.debug(f"Command: {' '.join(sys.argv)}")

    processor = AlphasProcessor(read_folder=args.read_folder, 
                                save_folder=args.save_folder, 
                                read_format=args.read_format, 
                                save_format=args.save_format)
    processor.set_loggger(logger)

    if args.cap and args.sector and args.industry and args.subindustry:
        processor.load_constant(args.cap, args.sector, args.industry, args.subindustry)
    processor.process()

            


