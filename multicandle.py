# %%

import pandas as pd
import os
import numpy as np
import pickle
from math import sqrt, floor
from tqdm import tqdm
import random
from matplotlib import pyplot as plt


def sigmoid(a, x):
    return 1 / (1 + np.exp(-a*x))


class MultiCandleAnalyzer():
    dir = os.path.dirname(os.path.abspath(__file__))
    positionLifetime = 1000*60*60  # ms. 포지션 수명은 1시간 정도로.
    # sample number.
    sn1m = 30
    sn5m = 10
    sn15m = 10
    sn1h = 5
    candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                     "closePrice": "", "volume": "", 'numberOfTrades': ""}]
    paramNum = 6
    pw1mF = np.ones((paramNum, 1)) / sqrt(paramNum)
    pw5mF = np.ones((paramNum, 1)) / sqrt(paramNum)
    pw15mF = np.ones((paramNum, 1)) / sqrt(paramNum)
    pw1hF = np.ones((paramNum, 1)) / sqrt(paramNum)
    tw1mF = np.ones((1, sn1m)) / sqrt(sn1m)
    tw5mF = np.ones((1, sn5m)) / sqrt(sn5m)
    tw15mF = np.ones((1, sn15m)) / sqrt(sn15m)
    tw1hF = np.ones((1, sn1h)) / sqrt(sn1h)

    pw1mR = np.ones((paramNum, 1)) / sqrt(paramNum)
    pw5mR = np.ones((paramNum, 1)) / sqrt(paramNum)
    pw15mR = np.ones((paramNum, 1)) / sqrt(paramNum)
    pw1hR = np.ones((paramNum, 1)) / sqrt(paramNum)
    tw1mR = np.ones((1, sn1m)) / sqrt(sn1m)
    tw5mR = np.ones((1, sn5m)) / sqrt(sn5m)
    tw15mR = np.ones((1, sn15m)) / sqrt(sn15m)
    tw1hR = np.ones((1, sn1h)) / sqrt(sn1h)

    pw1mL = np.ones((paramNum, 1)) / sqrt(paramNum)
    pw5mL = np.ones((paramNum, 1)) / sqrt(paramNum)
    pw15mL = np.ones((paramNum, 1)) / sqrt(paramNum)
    pw1hL = np.ones((paramNum, 1)) / sqrt(paramNum)
    tw1mL = np.ones((1, sn1m)) / sqrt(sn1m)
    tw5mL = np.ones((1, sn5m)) / sqrt(sn5m)
    tw15mL = np.ones((1, sn15m)) / sqrt(sn15m)
    tw1hL = np.ones((1, sn1h)) / sqrt(sn1h)

    def loadWeight(self, filename):
        with open(f"{self.dir}/weights/{filename}.pickle", 'rb') as f:
            return pickle.load(f)

    def loadAllWeights(self, versionName):
        pw1mF = self.loadWeight(f"{versionName}_pw1mF")
        pw5mF = self.loadWeight(f"{versionName}_pw5mF")
        pw15mF = self.loadWeight(f"{versionName}_pw15mF")
        pw1hF = self.loadWeight(f"{versionName}_pw1hF")
        tw1mF = self.loadWeight(f"{versionName}_tw1mF")
        tw5mF = self.loadWeight(f"{versionName}_tw5mF")
        tw15mF = self.loadWeight(f"{versionName}_tw15mF")
        tw1hF = self.loadWeight(f"{versionName}_tw1hF")

        pw1mR = self.loadWeight(f"{versionName}_pw1mR")
        pw5mR = self.loadWeight(f"{versionName}_pw5mR")
        pw15mR = self.loadWeight(f"{versionName}_pw15mR")
        pw1hR = self.loadWeight(f"{versionName}_pw1hR")
        tw1mR = self.loadWeight(f"{versionName}_tw1mR")
        tw5mR = self.loadWeight(f"{versionName}_tw5mR")
        tw15mR = self.loadWeight(f"{versionName}_tw15mR")
        tw1hR = self.loadWeight(f"{versionName}_tw1hR")

        pw1mL = self.loadWeight(f"{versionName}_pw1mL")
        pw5mL = self.loadWeight(f"{versionName}_pw5mL")
        pw15mL = self.loadWeight(f"{versionName}_pw15mL")
        pw1hL = self.loadWeight(f"{versionName}_pw1hL")
        tw1mL = self.loadWeight(f"{versionName}_tw1mL")
        tw5mL = self.loadWeight(f"{versionName}_tw5mL")
        tw15mL = self.loadWeight(f"{versionName}_tw15mL")
        tw1hL = self.loadWeight(f"{versionName}_tw1hL")

        return pw1mF, pw5mF, pw15mF, pw1hF, tw1mF, tw5mF, tw15mF, tw1hF, pw1mR, pw5mR, pw15mR, pw1hR, tw1mR, tw5mR, tw15mR, tw1hR, pw1mL, pw5mL, pw15mL, pw1hL, tw1mL, tw5mL, tw15mL, tw1hL

    def createWeights(self, versionName):
        with open(f"{self.dir}/weights/{versionName}_pw1mF.pickle", "wb") as f:
            pickle.dump(self.pw1mF, f)
        with open(f"{self.dir}/weights/{versionName}_pw5mF.pickle", "wb") as f:
            pickle.dump(self.pw5mF, f)
        with open(f"{self.dir}/weights/{versionName}_pw15mF.pickle", "wb") as f:
            pickle.dump(self.pw15mF, f)
        with open(f"{self.dir}/weights/{versionName}_pw1hF.pickle", "wb") as f:
            pickle.dump(self.pw1hF, f)
        with open(f"{self.dir}/weights/{versionName}_tw1mF.pickle", "wb") as f:
            pickle.dump(self.tw1mF, f)
        with open(f"{self.dir}/weights/{versionName}_tw5mF.pickle", "wb") as f:
            pickle.dump(self.tw5mF, f)
        with open(f"{self.dir}/weights/{versionName}_tw15mF.pickle", "wb") as f:
            pickle.dump(self.tw15mF, f)
        with open(f"{self.dir}/weights/{versionName}_tw1hF.pickle", "wb") as f:
            pickle.dump(self.tw1hF, f)

        with open(f"{self.dir}/weights/{versionName}_pw1mR.pickle", "wb") as f:
            pickle.dump(self.pw1mR, f)
        with open(f"{self.dir}/weights/{versionName}_pw5mR.pickle", "wb") as f:
            pickle.dump(self.pw5mR, f)
        with open(f"{self.dir}/weights/{versionName}_pw15mR.pickle", "wb") as f:
            pickle.dump(self.pw15mR, f)
        with open(f"{self.dir}/weights/{versionName}_pw1hR.pickle", "wb") as f:
            pickle.dump(self.pw1hR, f)
        with open(f"{self.dir}/weights/{versionName}_tw1mR.pickle", "wb") as f:
            pickle.dump(self.tw1mR, f)
        with open(f"{self.dir}/weights/{versionName}_tw5mR.pickle", "wb") as f:
            pickle.dump(self.tw5mR, f)
        with open(f"{self.dir}/weights/{versionName}_tw15mR.pickle", "wb") as f:
            pickle.dump(self.tw15mR, f)
        with open(f"{self.dir}/weights/{versionName}_tw1hR.pickle", "wb") as f:
            pickle.dump(self.tw1hR, f)

        with open(f"{self.dir}/weights/{versionName}_pw1mL.pickle", "wb") as f:
            pickle.dump(self.pw1mL, f)
        with open(f"{self.dir}/weights/{versionName}_pw5mL.pickle", "wb") as f:
            pickle.dump(self.pw5mL, f)
        with open(f"{self.dir}/weights/{versionName}_pw15mL.pickle", "wb") as f:
            pickle.dump(self.pw15mL, f)
        with open(f"{self.dir}/weights/{versionName}_pw1hL.pickle", "wb") as f:
            pickle.dump(self.pw1hL, f)
        with open(f"{self.dir}/weights/{versionName}_tw1mL.pickle", "wb") as f:
            pickle.dump(self.tw1mL, f)
        with open(f"{self.dir}/weights/{versionName}_tw5mL.pickle", "wb") as f:
            pickle.dump(self.tw5mL, f)
        with open(f"{self.dir}/weights/{versionName}_tw15mL.pickle", "wb") as f:
            pickle.dump(self.tw15mL, f)
        with open(f"{self.dir}/weights/{versionName}_tw1hL.pickle", "wb") as f:
            pickle.dump(self.tw1hL, f)

    def initMat(self, mat, df, len):
        for i in range(len):
            mat[i][0] = df.iloc[i]['high_price']
            mat[i][1] = df.iloc[i]['low_price']
            mat[i][2] = df.iloc[i]['open_price']
            mat[i][3] = df.iloc[i]['close_price']
            mat[i][4] = df.iloc[i]['volume']
            mat[i][5] = df.iloc[i]['number_of_trades']
        return mat

    def trainFluid(self, versionName, startday, endday, profitCut, ratio):
        # profitCut 은 1.003 으로 하고, 10배율을 기본값으로 생각하자.
        fee = 0.0004*ratio
        pw1mF, pw5mF, pw15mF, pw1hF, tw1mF, tw5mF, tw15mF, tw1hF, pw1mR, pw5mR, pw15mR, pw1hR, tw1mR, tw5mR, tw15mR, tw1hR, pw1mL, pw5mL, pw15mL, pw1hL, tw1mL, tw5mL, tw15mL, tw1hL = self.loadAllWeights(
            versionName)

        len1m = len(tw1mF[0])  # 30
        len5m = len(tw5mF[0])  # 10
        len15m = len(tw15mF[0])  # 10
        len1h = len(tw1hF[0])  # 5

        mat1m = np.zeros((len1m, self.paramNum))  # 30 x 6
        mat5m = np.zeros((len5m, self.paramNum))  # 10 x 6
        mat15m = np.zeros((len15m, self.paramNum))  # 10 x 6
        mat1h = np.zeros((len1h, self.paramNum))  # 5 x 6

        lastBet = {"betSize": 0, "betTime": 0,
                   "betDir": "", "betPrice": 0, 'candleType': ""}
        df1m = pd.read_csv(
            self.dir+f'/data/future_BTC_1m_{startday}_{endday}.csv')
        df5m = pd.read_csv(
            self.dir+f'/data/future_BTC_5m_{startday}_{endday}.csv')
        df15m = pd.read_csv(
            self.dir+f'/data/future_BTC_15m_{startday}_{endday}.csv')
        df1h = pd.read_csv(
            self.dir+f'/data/future_BTC_1h_{startday}_{endday}.csv')

        mat1m = self.initMat(mat1m, df1m, len1m)
        mat5m = self.initMat(mat5m, df5m, len5m)
        mat15m = self.initMat(mat15m, df15m, len15m)
        mat1h = self.initMat(mat1h, df1h, len1h)

        for i in tqdm(range(len1m, len(df1m)), desc='train fluid...'):
            # 1m 기준으로 계속 보고, 5개 될때마다 5m, 15개 될때마다 15m, 60개 될때마다 1h 업데이트 해주자.

            if lastBet["betTime"] != 0:
                # 지난 배팅 체크하고 배팅하기
                print("g")


# %%
mca = MultiCandleAnalyzer()
# mca.createWeights("220223")
mca.trainFluid("220223")

# %%

# %%
