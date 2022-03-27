# %%

import multiprocessing
import pandas as pd
import os
import numpy as np
import pickle
from math import sqrt, floor
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import json


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
    paramNum = 7
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

    totF = np.ones((1, 4)) / sqrt(4)
    totR = np.ones((1, 4)) / sqrt(4)
    totL = np.ones((1, 4)) / sqrt(4)

    def saveJson(self, obj, filename):
        with open(filename, "w") as f:
            json.dump(obj, f, indent=2)

    def loadJson(self, filename):
        with open(filename, "r") as f:
            return json.load(f)

    def saveWeight(self, weight, filename):
        with open(f"{self.dir}/weights/{filename}.pickle", 'wb') as f:
            pickle.dump(weight, f)

    def loadWeight(self, filename):
        with open(f"{self.dir}/weights/{filename}.pickle", 'rb') as f:
            return pickle.load(f)

    def printAllWeights(self, versionName):
        weights = []
        weights = self.loadAllWeights(versionName)
        for i in range(len(weights)):
            print(weights[i])

    def loadAllWeights(self, versionName):
        pw1mF = self.loadWeight(f"{versionName}_pw1mF")
        pw5mF = self.loadWeight(f"{versionName}_pw5mF")
        pw15mF = self.loadWeight(f"{versionName}_pw15mF")
        pw1hF = self.loadWeight(f"{versionName}_pw1hF")
        tw1mF = self.loadWeight(f"{versionName}_tw1mF")
        tw5mF = self.loadWeight(f"{versionName}_tw5mF")
        tw15mF = self.loadWeight(f"{versionName}_tw15mF")
        tw1hF = self.loadWeight(f"{versionName}_tw1hF")
        totF = self.loadWeight(f"{versionName}_totF")

        pw1mR = self.loadWeight(f"{versionName}_pw1mR")
        pw5mR = self.loadWeight(f"{versionName}_pw5mR")
        pw15mR = self.loadWeight(f"{versionName}_pw15mR")
        pw1hR = self.loadWeight(f"{versionName}_pw1hR")
        tw1mR = self.loadWeight(f"{versionName}_tw1mR")
        tw5mR = self.loadWeight(f"{versionName}_tw5mR")
        tw15mR = self.loadWeight(f"{versionName}_tw15mR")
        tw1hR = self.loadWeight(f"{versionName}_tw1hR")
        totR = self.loadWeight(f"{versionName}_totR")

        pw1mL = self.loadWeight(f"{versionName}_pw1mL")
        pw5mL = self.loadWeight(f"{versionName}_pw5mL")
        pw15mL = self.loadWeight(f"{versionName}_pw15mL")
        pw1hL = self.loadWeight(f"{versionName}_pw1hL")
        tw1mL = self.loadWeight(f"{versionName}_tw1mL")
        tw5mL = self.loadWeight(f"{versionName}_tw5mL")
        tw15mL = self.loadWeight(f"{versionName}_tw15mL")
        tw1hL = self.loadWeight(f"{versionName}_tw1hL")
        totL = self.loadWeight(f"{versionName}_totL")

        return [pw1mF, pw5mF, pw15mF, pw1hF, tw1mF, tw5mF, tw15mF, tw1hF, pw1mR, pw5mR, pw15mR, pw1hR, tw1mR, tw5mR, tw15mR, tw1hR, pw1mL, pw5mL, pw15mL, pw1hL, tw1mL, tw5mL, tw15mL, tw1hL, totF, totR, totL]

    def createCertainWeights(self, versionName, type):
        if type == "F":
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
            with open(f"{self.dir}/weights/{versionName}_totF.pickle", "wb") as f:
                pickle.dump(self.totF, f)

        elif type == "R":
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
            with open(f"{self.dir}/weights/{versionName}_totR.pickle", "wb") as f:
                pickle.dump(self.totR, f)

        elif type == "L":
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
            with open(f"{self.dir}/weights/{versionName}_totL.pickle", "wb") as f:
                pickle.dump(self.totL, f)

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

        with open(f"{self.dir}/weights/{versionName}_totF.pickle", "wb") as f:
            pickle.dump(self.totF, f)
        with open(f"{self.dir}/weights/{versionName}_totL.pickle", "wb") as f:
            pickle.dump(self.totL, f)
        with open(f"{self.dir}/weights/{versionName}_totR.pickle", "wb") as f:
            pickle.dump(self.totR, f)

    def initMat(self, mat, df, len):
        for i in range(len):
            mat[i][0] = df.iloc[i]['high_price']
            mat[i][1] = df.iloc[i]['low_price']
            mat[i][2] = df.iloc[i]['open_price']
            mat[i][3] = df.iloc[i]['close_price']
            mat[i][4] = df.iloc[i]['volume']
            mat[i][5] = df.iloc[i]['number_of_trades']
            mat[i][6] = df.iloc[i]['close_time']
        return mat

    def trainModel(self, versionName, startday, endday, upThr, downThr, profitR, lossR, ratio):
        fee = 0.0004*ratio
        money = 1000
        apiTroubleLoss = fee/ratio
        slipageLoss = fee/ratio

        weightObj = self.loadJson(f"weights/{versionName}.json")
        weights1mP = weightObj['weights1mP']
        weights5mP = weightObj['weights5mP']
        weights15mP = weightObj['weights15mP']
        weights1hP = weightObj['weights1hP']
        weights1mT = weightObj['weights1mT']
        weights5mT = weightObj['weights5mT']
        weights15mT = weightObj['weights15mT']
        weights1hT = weightObj['weights1hT']
        weightsTot = weightObj['weightsTot']

        len1m = len(weights1mT)  # 30
        len5m = len(weights5mT)  # 15
        len15m = len(weights15mT)  # 15
        len1h = len(weights1hT)  # 5
        mat1m = np.zeros((len1m, self.paramNum))  # 30 x 7
        mat5m = np.zeros((len5m, self.paramNum))  # 15 x 7
        mat15m = np.zeros((len15m, self.paramNum))  # 15 x 7
        mat1h = np.zeros((len1h, self.paramNum))  # 5 x 7

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
        prob = 0
        moneyHistory = []
        positionLifetime = 1000*60*15  # ms
        for ii in tqdm(range(len1m, len(df1m)), desc='training...'):
            # 1m 기준으로 계속 보고, 5개 될때마다 5m, 15개 될때마다 15m, 60개 될때마다 1h 업데이트 해주자.
            candle1m = df1m.iloc[ii]
            candle5m = df5m.iloc[floor(ii/5)]
            candle15m = df15m.iloc[floor(ii/15)]
            candle1h = df1h.iloc[floor(ii/60)]
            for jj in range(len1m-1):
                mat1m[jj] = mat1m[jj+1]
            mat1m[len1m-1][0] = candle1m['high_price'] + 0.0005 * \
                (2*random.random()-1)*candle1m['high_price']
            mat1m[len1m-1][1] = candle1m['low_price'] + 0.0005 * \
                (2*random.random()-1)*candle1m['low_price']
            mat1m[len1m-1][2] = candle1m['open_price'] + 0.0005 * \
                (2*random.random()-1)*candle1m['open_price']
            mat1m[len1m-1][3] = candle1m['close_price'] + 0.0005 * \
                (2*random.random()-1)*candle1m['close_price']
            mat1m[len1m-1][4] = candle1m['volume'] + 0.0005 * \
                (2*random.random()-1)*candle1m['volume']
            mat1m[len1m -
                  1][5] = candle1m['number_of_trades'] + 0.0005*(2*random.random()-1)*candle1m['number_of_trades']
            mat1m[len1m -
                  1][6] = candle1m['close_time']

            if candle1m['close_time'] == candle5m['close_time']:
                # 5분봉 업데이트
                for jj in range(len5m-1):
                    mat5m[jj] = mat5m[jj+1]
                mat5m[len5m-1][0] = candle5m['high_price'] + 0.0005 * \
                    (2*random.random()-1)*candle5m['high_price']
                mat5m[len5m-1][1] = candle5m['low_price'] + 0.0005 * \
                    (2*random.random()-1)*candle5m['low_price']
                mat5m[len5m-1][2] = candle5m['open_price'] + 0.0005 * \
                    (2*random.random()-1)*candle5m['open_price']
                mat5m[len5m-1][3] = candle5m['close_price'] + 0.0005 * \
                    (2*random.random()-1)*candle5m['close_price']
                mat5m[len5m-1][4] = candle5m['volume'] + 0.0005 * \
                    (2*random.random()-1)*candle5m['high_price']
                mat5m[len5m -
                      1][5] = candle5m['number_of_trades'] + 0.0005*(2*random.random()-1)*candle5m['number_of_trades']
                mat5m[len5m -
                      1][6] = candle5m['close_time']
            if candle1m['close_time'] == candle15m['close_time']:
                # 15분봉 업데이트
                for jj in range(len15m-1):
                    mat15m[jj] = mat15m[jj+1]
                mat15m[len15m-1][0] = candle15m['high_price'] + \
                    0.0005*(2*random.random()-1)*candle15m['high_price']
                mat15m[len15m-1][1] = candle15m['low_price'] + \
                    0.0005*(2*random.random()-1)*candle15m['low_price']
                mat15m[len15m-1][2] = candle15m['open_price'] + \
                    0.0005*(2*random.random()-1)*candle15m['open_price']
                mat15m[len15m-1][3] = candle15m['close_price'] + \
                    0.0005*(2*random.random()-1)*candle15m['close_price']
                mat15m[len15m-1][4] = candle15m['volume'] + 0.0005 * \
                    (2*random.random()-1)*candle15m['high_price']
                mat15m[len15m -
                       1][5] = candle15m['number_of_trades'] + 0.0005*(2*random.random()-1)*candle15m['number_of_trades']
                mat15m[len15m -
                       1][6] = candle15m['close_time']
            if candle1m['close_time'] == candle1h['close_time']:
                # 1시간 봉 업데이트
                for jj in range(len1h-1):
                    mat1h[jj] = mat1h[jj+1]
                mat1h[len1h-1][0] = candle1h['high_price'] + 0.0005 * \
                    (2*random.random()-1)*candle1h['high_price']
                mat1h[len1h-1][1] = candle1h['low_price'] + 0.0005 * \
                    (2*random.random()-1)*candle1h['low_price']
                mat1h[len1h-1][2] = candle1h['open_price'] + 0.0005 * \
                    (2*random.random()-1)*candle1h['open_price']
                mat1h[len1h-1][3] = candle1h['close_price'] + 0.0005 * \
                    (2*random.random()-1)*candle1h['close_price']
                mat1h[len1h-1][4] = candle1h['volume'] + 0.0005 * \
                    (2*random.random()-1)*candle1h['high_price']
                mat1h[len1h -
                      1][5] = candle1h['number_of_trades'] + 0.0005*(2*random.random()-1)*candle1h['number_of_trades']
                mat1h[len1h -
                      1][6] = candle1h['close_time']
            # 수익 체크.
            if lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime > candle1m['close_time']:
                if lastBet['betDir'] == "up":
                    if candle1m['high_price'] > lastBet['betPrice']*profitR:
                        # 상승배팅 수익 실현
                        money = (1-apiTroubleLoss)*(1-slipageLoss) * \
                            (1-fee)*money*(1+(profitR-1)*ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        self.saveJson({"weights1mP": weights1mP,
                                       "weights1mT": weights1mT,
                                       "weights5mP": weights5mP,
                                       "weights5mT": weights5mT,
                                       "weights15mP": weights15mP,
                                       "weights15mT": weights15mT,
                                       "weights1hP": weights1hP,
                                       "weights1hT": weights1hT,
                                       "weightsTot": weightsTot
                                       }, f"weights/{versionName}.json")
                        #print(f"상승배팅 수익실현: {money}")
                        moneyHistory.append(money)
                    elif candle1m['low_price'] < lastBet['betPrice']*lossR:
                        # 상승배팅 손해
                        money = (1-apiTroubleLoss)*(1-slipageLoss) * \
                            (1-fee)*money*(1+(lossR-1)*ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        #print(f"상승배팅 손해: {money}")
                        moneyHistory.append(money)
                elif lastBet['betDir'] == "down":
                    if candle1m['high_price'] > lastBet['betPrice']*profitR:
                        # 하락배팅 손해
                        money = (1-apiTroubleLoss)*(1-slipageLoss) * \
                            (1-fee)*money * (1-(profitR-1)*ratio)

                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        #print(f"하락배팅 손해: {money}")
                        moneyHistory.append(money)
                    elif candle1m['low_price'] < lastBet['betPrice']*lossR:
                        # 하락배팅 수익실현
                        money = (1-apiTroubleLoss)*(1-slipageLoss) * \
                            (1-fee)*money*(1-(lossR-1)*ratio)

                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        self.saveJson({"weights1mP": weights1mP,
                                       "weights1mT": weights1mT,
                                       "weights5mP": weights5mP,
                                       "weights5mT": weights5mT,
                                       "weights15mP": weights15mP,
                                       "weights15mT": weights15mT,
                                       "weights1hP": weights1hP,
                                       "weights1hT": weights1hT,
                                       "weightsTot": weightsTot
                                       }, f"weights/{versionName}.json")
                        #print(f"하락배팅 수익실현: {money}")
                        moneyHistory.append(money)
            elif lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime < candle1m['close_time']:
                # 시간초과로 인한 포지션 종료.
                if lastBet['betDir'] == "up":
                    if candle1m['close_price'] > lastBet['betPrice']:
                        # 상승배팅 수익 실현
                        money = (1-apiTroubleLoss)*(1-slipageLoss)*(1-fee)*money * \
                            (1+((candle1m['close_price'] /
                             lastBet['betPrice'])-1) * ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        self.saveJson({"weights1mP": weights1mP,
                                       "weights1mT": weights1mT,
                                       "weights5mP": weights5mP,
                                       "weights5mT": weights5mT,
                                       "weights15mP": weights15mP,
                                       "weights15mT": weights15mT,
                                       "weights1hP": weights1hP,
                                       "weights1hT": weights1hT,
                                       "weightsTot": weightsTot
                                       }, f"weights/{versionName}.json")
                        #print(f"타임아웃 상승배팅 수익실현: {money}")
                        moneyHistory.append(money)
                    elif candle1m['close_price'] < lastBet['betPrice']:
                        # 상승배팅 손해
                        money = (1-apiTroubleLoss)*(1-slipageLoss)*(1-fee)*money * \
                            (1-(1-(candle1m['close_price'] /
                             lastBet['betPrice'])) * ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        #print(f"타임아웃 상승배팅 손해: {money}")
                        moneyHistory.append(money)
                elif lastBet['betDir'] == "down":
                    if candle1m['close_price'] > lastBet['betPrice']:
                        # 다운배팅 손해
                        money = (1-apiTroubleLoss)*(1-slipageLoss)*(1-fee)*money * \
                            (1-(candle1m['close_price'] /
                             lastBet['betPrice'] - 1)*ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        #print(f"타임아웃 다운배팅 손해: {money}")
                        moneyHistory.append(money)
                    elif candle1m['close_price'] < lastBet['betPrice']:
                        # 다운배팅 수익 실현
                        money = (1-apiTroubleLoss)*(1-slipageLoss)*(1-fee)*money * \
                            (1-((candle1m['close_price'] /
                             lastBet['betPrice'] - 1)*ratio))
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        self.saveJson({"weights1mP": weights1mP,
                                       "weights1mT": weights1mT,
                                       "weights5mP": weights5mP,
                                       "weights5mT": weights5mT,
                                       "weights15mP": weights15mP,
                                       "weights15mT": weights15mT,
                                       "weights1hP": weights1hP,
                                       "weights1hT": weights1hT,
                                       "weightsTot": weightsTot
                                       }, f"weights/{versionName}.json")
                        #print(f"타임아웃 다운배팅 수익실현: {money}")
                        moneyHistory.append(money)
                #print("걍 이도저도 아닌 타임아웃")
                lastBet['betTime'] = 0
                lastBet['betPrice'] = 0
                lastBet['betDir'] = ""
            if lastBet['betTime'] == 0 and lastBet['betPrice'] == 0:
                for i in range(len(weights1mP)):
                    weights1mP[i] = weights1mP[i] + \
                        weights1mP[i] * 0.05*(2*random.random()-1)
                    weights5mP[i] = weights5mP[i] + \
                        weights5mP[i] * 0.05*(2*random.random()-1)
                    weights15mP[i] = weights15mP[i] + \
                        weights15mP[i] * 0.05*(2*random.random()-1)
                    weights1hP[i] = weights1hP[i] + \
                        weights1hP[i] * 0.05*(2*random.random()-1)
                for kk in range(len1m):
                    weights1mT[kk] = weights1mT[kk] + weights1mT[kk] * \
                        0.05*(2*random.random()-1)
                for kk in range(len5m):
                    weights5mT[kk] = weights5mT[kk] + weights5mT[kk] * \
                        0.05*(2*random.random()-1)
                for kk in range(len15m):
                    weights15mT[kk] = weights15mT[kk] + weights15mT[kk] * \
                        0.05*(2*random.random()-1)
                for kk in range(len1h):
                    weights1hT[kk] = weights1hT[kk] + weights1hT[kk] * \
                        0.05*(2*random.random()-1)
                plst = []
                x1m = np.matmul(mat1m, np.transpose(
                    np.array([weights1mP])))
                p1m = np.matmul((np.array([weights1mT])), x1m)
                plst.append(p1m[0][0])
                x5m = np.matmul(mat5m, np.transpose(
                    np.array([weights5mP])))
                p5m = np.matmul(np.array([weights5mT]), x5m)
                plst.append(p5m[0][0])
                x15m = np.matmul(mat15m, np.transpose(
                    np.array([weights15mP])))
                p15m = np.matmul(np.array([weights15mT]), x15m)
                plst.append(p15m[0][0])
                x1h = np.matmul(mat1h, np.transpose(
                    np.array([weights1hP])))
                p1h = np.matmul(np.array([weights1hT]), x1h)
                plst.append(p1h[0][0])
                for kk in range(len(weightsTot)):
                    weightsTot[kk] = weightsTot[kk]+weightsTot[kk] * \
                        0.05*(2*random.random()-1)
                for kk in range(len(weightsTot)):
                    prob += weightsTot[kk]*plst[kk]
                prob = np.tanh(prob)
                prob = max(0.1*prob, prob)
                if prob > upThr:
                    lastBet['betTime'] = candle1m['close_time']
                    lastBet['betPrice'] = candle1m['close_price']
                    lastBet['betDir'] = "up"
                    money = money * (1-fee)
                    #print(f"상승배팅 시작: {money}")
                elif prob < downThr:
                    lastBet['betTime'] = candle1m['close_time']
                    lastBet['betPrice'] = candle1m['close_price']
                    lastBet['betDir'] = "down"
                    money = money * (1-fee)
                    #print(f"하락배팅 시작: {money}")
                else:
                    pass
                    #print(f"배팅없음: {money}")

        xline = np.arange(len(moneyHistory))
        #print(f"종료. 금액: {money}")
        plt.plot(xline, moneyHistory)
        plt.show()

        return

    def testModel(self, versionName, startday, endday, thr, profitR, lossR, ratio):
        fee = 0.0004*ratio
        money = 1000
        apiTroubleLoss = fee/ratio
        slipageLoss = fee/ratio

        weightObj = self.loadJson(f"weights/{versionName}.json")
        weights1mP = weightObj['weights1mP']
        weights5mP = weightObj['weights5mP']
        weights15mP = weightObj['weights15mP']
        weights1hP = weightObj['weights1hP']
        weights1mT = weightObj['weights1mT']
        weights5mT = weightObj['weights5mT']
        weights15mT = weightObj['weights15mT']
        weights1hT = weightObj['weights1hT']
        weightsTot = weightObj['weightsTot']

        len1m = len(weights1mT)  # 30
        len5m = len(weights5mT)  # 15
        len15m = len(weights15mT)  # 15
        len1h = len(weights1hT)  # 5
        mat1m = np.zeros((len1m, self.paramNum))  # 30 x 7
        mat5m = np.zeros((len5m, self.paramNum))  # 15 x 7
        mat15m = np.zeros((len15m, self.paramNum))  # 15 x 7
        mat1h = np.zeros((len1h, self.paramNum))  # 5 x 7

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
        prob = 0
        moneyHistory = []
        positionLifetime = 1000*60*15  # ms
        for i in tqdm(range(len1m, len(df1m)), desc='train fluid...'):
            # 1m 기준으로 계속 보고, 5개 될때마다 5m, 15개 될때마다 15m, 60개 될때마다 1h 업데이트 해주자.
            candle1m = df1m.iloc[i]
            candle5m = df5m.iloc[floor(i/5)]
            candle15m = df15m.iloc[floor(i/15)]
            candle1h = df1h.iloc[floor(i/60)]
            for jj in range(len1m-1):
                mat1m[jj] = mat1m[jj+1]
            mat1m[len1m-1][0] = candle1m['high_price'] + 0.0005 * \
                (2*random.random()-1)*candle1m['high_price']
            mat1m[len1m-1][1] = candle1m['low_price'] + 0.0005 * \
                (2*random.random()-1)*candle1m['low_price']
            mat1m[len1m-1][2] = candle1m['open_price'] + 0.0005 * \
                (2*random.random()-1)*candle1m['open_price']
            mat1m[len1m-1][3] = candle1m['close_price'] + 0.0005 * \
                (2*random.random()-1)*candle1m['close_price']
            mat1m[len1m-1][4] = candle1m['volume'] + 0.0005 * \
                (2*random.random()-1)*candle1m['volume']
            mat1m[len1m -
                  1][5] = candle1m['number_of_trades'] + 0.0005*(2*random.random()-1)*candle1m['number_of_trades']
            mat1m[len1m -
                  1][6] = candle1m['close_time']

            if candle1m['close_time'] == candle5m['close_time']:
                # 5분봉 업데이트
                for jj in range(len5m-1):
                    mat5m[jj] = mat5m[jj+1]
                mat5m[len5m-1][0] = candle5m['high_price'] + 0.0005 * \
                    (2*random.random()-1)*candle5m['high_price']
                mat5m[len5m-1][1] = candle5m['low_price'] + 0.0005 * \
                    (2*random.random()-1)*candle5m['low_price']
                mat5m[len5m-1][2] = candle5m['open_price'] + 0.0005 * \
                    (2*random.random()-1)*candle5m['open_price']
                mat5m[len5m-1][3] = candle5m['close_price'] + 0.0005 * \
                    (2*random.random()-1)*candle5m['close_price']
                mat5m[len5m-1][4] = candle5m['volume'] + 0.0005 * \
                    (2*random.random()-1)*candle5m['high_price']
                mat5m[len5m -
                      1][5] = candle5m['number_of_trades'] + 0.0005*(2*random.random()-1)*candle5m['number_of_trades']
                mat5m[len5m -
                      1][6] = candle5m['close_time']
            if candle1m['close_time'] == candle15m['close_time']:
                # 15분봉 업데이트
                for jj in range(len15m-1):
                    mat15m[jj] = mat15m[jj+1]
                mat15m[len15m-1][0] = candle15m['high_price'] + \
                    0.0005*(2*random.random()-1)*candle15m['high_price']
                mat15m[len15m-1][1] = candle15m['low_price'] + \
                    0.0005*(2*random.random()-1)*candle15m['low_price']
                mat15m[len15m-1][2] = candle15m['open_price'] + \
                    0.0005*(2*random.random()-1)*candle15m['open_price']
                mat15m[len15m-1][3] = candle15m['close_price'] + \
                    0.0005*(2*random.random()-1)*candle15m['close_price']
                mat15m[len15m-1][4] = candle15m['volume'] + 0.0005 * \
                    (2*random.random()-1)*candle15m['high_price']
                mat15m[len15m -
                       1][5] = candle15m['number_of_trades'] + 0.0005*(2*random.random()-1)*candle15m['number_of_trades']
                mat15m[len15m -
                       1][6] = candle15m['close_time']
            if candle1m['close_time'] == candle1h['close_time']:
                # 1시간 봉 업데이트
                for jj in range(len1h-1):
                    mat1h[jj] = mat1h[jj+1]
                mat1h[len1h-1][0] = candle1h['high_price'] + 0.0005 * \
                    (2*random.random()-1)*candle1h['high_price']
                mat1h[len1h-1][1] = candle1h['low_price'] + 0.0005 * \
                    (2*random.random()-1)*candle1h['low_price']
                mat1h[len1h-1][2] = candle1h['open_price'] + 0.0005 * \
                    (2*random.random()-1)*candle1h['open_price']
                mat1h[len1h-1][3] = candle1h['close_price'] + 0.0005 * \
                    (2*random.random()-1)*candle1h['close_price']
                mat1h[len1h-1][4] = candle1h['volume'] + 0.0005 * \
                    (2*random.random()-1)*candle1h['high_price']
                mat1h[len1h -
                      1][5] = candle1h['number_of_trades'] + 0.0005*(2*random.random()-1)*candle1h['number_of_trades']
                mat1h[len1h -
                      1][6] = candle1h['close_time']
            # 수익 체크.
            if lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime > candle1m['close_time']:
                if lastBet['betDir'] == "up":
                    if candle1m['high_price'] > lastBet['betPrice']*profitR:
                        # 상승배팅 수익 실현
                        money = (1-apiTroubleLoss)*(1-slipageLoss) * \
                            (1-fee)*money*(1+(profitR-1)*ratio)
                        # money = (1-fee)*money * \
                        #    (1+((candle['high_price'] /
                        #     lastBet['betPrice'])-1) * ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        #print(f"상승배팅 수익실현: {money}")
                        moneyHistory.append(money)
                    elif candle1m['low_price'] < lastBet['betPrice']*lossR:
                        # 상승배팅 손해
                        money = (1-apiTroubleLoss)*(1-slipageLoss) * \
                            (1-fee)*money*(1+(lossR-1)*ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        #print(f"상승배팅 손해: {money}")
                        moneyHistory.append(money)
                elif lastBet['betDir'] == "down":
                    if candle1m['high_price'] > lastBet['betPrice']*profitR:
                        # 하락배팅 손해
                        money = (1-apiTroubleLoss)*(1-slipageLoss) * \
                            (1-fee)*money*(2-(profitR-1)*ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        #print(f"하락배팅 손해: {money}")
                        moneyHistory.append(money)
                    elif candle1m['low_price'] < lastBet['betPrice']*lossR:
                        # 하락배팅 수익실현
                        money = (1-apiTroubleLoss)*(1-slipageLoss) * \
                            (1-fee)*money*(1-(lossR-1)*ratio)

                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        #print(f"하락배팅 수익실현: {money}")
                        moneyHistory.append(money)
            elif lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime < candle1m['close_time']:
                # 시간초과로 인한 포지션 종료.
                if lastBet['betDir'] == "up":
                    if candle1m['close_price'] > lastBet['betPrice']:

                        # 상승배팅 수익 실현
                        money = (1-apiTroubleLoss)*(1-slipageLoss)*(1-fee)*money * \
                            (1+((candle1m['close_price'] /
                                 lastBet['betPrice'])-1) * ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        #print(f"타임아웃 상승배팅 수익실현: {money}")
                        moneyHistory.append(money)
                    elif candle1m['close_price'] < lastBet['betPrice']:
                        # 상승배팅 손해
                        money = (1-apiTroubleLoss)*(1-slipageLoss)*(1-fee)*money * \
                            (1-(1-(candle1m['close_price'] /
                             lastBet['betPrice'])) * ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        #print(f"타임아웃 상승배팅 손해: {money}")
                        moneyHistory.append(money)
                elif lastBet['betDir'] == "down":
                    if candle1m['close_price'] > lastBet['betPrice']:
                        # 다운배팅 손해
                        money = (1-apiTroubleLoss)*(1-slipageLoss)*(1-fee)*money * \
                            (1-(candle1m['close_price'] /
                             lastBet['betPrice'] - 1)*ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        #print(f"타임아웃 다운배팅 손해: {money}")
                        moneyHistory.append(money)
                    elif candle1m['close_price'] < lastBet['betPrice']:
                        # 다운배팅 수익 실현
                        money = (1-apiTroubleLoss)*(1-slipageLoss)*(1-fee)*money * \
                            (1-((candle1m['close_price'] /
                             lastBet['betPrice'] - 1)*ratio))
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        #print(f"타임아웃 다운배팅 수익실현: {money}")
                        moneyHistory.append(money)
            for i in range(len(weights1mP)):
                weights1mP[i] = weights1mP[i] + \
                    weights1mP[i] * 0.05*(2*random.random()-1)
                weights5mP[i] = weights5mP[i] + \
                    weights5mP[i] * 0.05*(2*random.random()-1)
                weights15mP[i] = weights15mP[i] + \
                    weights15mP[i] * 0.05*(2*random.random()-1)
                weights1hP[i] = weights1hP[i] + \
                    weights1hP[i] * 0.05*(2*random.random()-1)
                for i in range(len1m):
                    weights1mT[i] = weights1mT[i] + weights1mT[i] * \
                        0.05*(2*random.random()-1)
                for i in range(len5m):
                    weights5mT[i] = weights5mT[i] + weights5mT[i] * \
                        0.05*(2*random.random()-1)
                for i in range(len15m):
                    weights15mT[i] = weights15mT[i] + weights15mT[i] * \
                        0.05*(2*random.random()-1)
                for i in range(len1h):
                    weights1hT[i] = weights1hT[i] + weights1hT[i] * \
                        0.05*(2*random.random()-1)
                plst = []
                x1m = np.matmul(mat1m, np.transpose(np.array([weights1mP])))
                p1m = np.matmul((np.array([weights1mT])), x1m)
                plst.append(p1m[0][0])
                x5m = np.matmul(mat5m, np.transpose(np.array([weights5mP])))
                p5m = np.matmul(np.array([weights5mT]), x5m)
                plst.append(p5m[0][0])
                x15m = np.matmul(mat15m, np.transpose(np.array([weights15mP])))
                p15m = np.matmul(np.array([weights15mT]), x15m)
                plst.append(p15m[0][0])
                x1h = np.matmul(mat1h, np.transpose(np.array([weights1hP])))
                p1h = np.matmul(np.array([weights1hT]), x1h)
                plst.append(p1h[0][0])
                for i in range(len(weightsTot)):
                    weightsTot[i] = weightsTot[i]+weightsTot[i] * \
                        0.05*(2*random.random()-1)
                for i in range(len(weightsTot)):
                    prob += weightsTot[i]*plst[i]
                prob = np.tanh(prob)
                prob = max(0.1*prob, prob)
                if prob > thr:
                    lastBet['betTime'] = candle1m['close_time']
                    lastBet['betPrice'] = candle1m['close_price']
                    money = money * (1-fee)
        xline = np.arange(len(moneyHistory))
        plt.plot(xline, moneyHistory)
        plt.show()

        return


# %%
if __name__ == '__main__':
    mca = MultiCandleAnalyzer()
    procs = []
    # mca.trainModel(versionName="220320_1", startday="2021-01-01",
    #               endday="2021-12-31", upThr=0.8, downThr=0.2, profitR=1.003, lossR=0.997, ratio=10)
    versionName = ["220320_1", "220327_1", "220327_2", "220327_3"]
    startday = ["2021-01-01", "2021-01-01", "2021-01-01", "2021-01-01"]
    endday = ["2021-12-31", "2021-12-31", "2021-12-31", "2021-12-31"]
    upThr = [0.8, 0.8, 0.8, 0.8]
    downThr = [0.2, 0.2, 0.2, 0.2]
    profitR = [1.003, 1.003, 1.003, 1.003]
    lossR = [0.997, 0.997, 0.997, 0.997]
    ratio = [10, 10, 10, 10]
    for i in range(3):
        for i, v, s, e, ut, dt, p, l, r in zip(range(len(versionName)), versionName, startday, endday, upThr, downThr, profitR, lossR, ratio):
            p = multiprocessing.Process(target=mca.trainModel, args=(
                v, s, e, ut, dt, p, l, r))
            procs.append(p)
            p.start()
        for proc in procs:
            proc.join()
    pool = multiprocessing.Pool(processes=4)
    pool.map(mca.trainModel, versionName, startday,
             endday, upThr, downThr, profitR, lossR, ratio)


#    procs = []
#    versionName = ["220307_1", "220307_2", "220307_3", "220307_4"]
#    startday = ["2021-01-01", "2021-01-01", "2021-01-01", "2021-01-01"]
#    endday = ["2021-12-31", "2021-12-31", "2021-12-31", "2021-12-31"]
#    thr = [0.8, 0.8, 0.8, 0.8]
#    profitR = [1.003, 1.003, 1.003, 1.003]
#    lossR = [0.997, 0.997, 0.997, 0.997]
#    ratio = [10, 10, 10, 10]
#    for i in range(3):
#        for i, v, s, e, t, p, l, r in zip(range(len(versionName)), versionName, startday, endday, thr, profitR, lossR, ratio):
#            p = multiprocessing.Process(target=mca.trainRaise, args=(
#                v, s, e, t, p, l, r))
#            procs.append(p)
#            p.start()
#        for proc in procs:
#            proc.join()
    #pool = multiprocessing.Pool(processes=4)
    #pool.map(mca.trainRaise, ["220307_1", "220307_2", "220307_3", "220307_4"],["2021-01-01", "2021-01-01", "2021-01-01", "2021-01-01"],["2021-12-31", "2021-12-31", "2021-12-31", "2021-12-31"],[0.8,0.8,0.8,0.8],[1.003,1.003,1.003,1.003],[0.997,0.997,0.997,0.997],[10,10,10,10])


# mca.createWeights("220223")
# for i in range(1):
# mca.trainFluid("220223", startday="2021-01-01", endday="2021-12-31",
#               thr=0.7, profitR=1.003, lossR=0.997, ratio=10)


# mca.trainRaise("220223", startday="2021-01-01", endday="2021-12-31",
#               thr=0.8, profitR=1.003, lossR=0.997, ratio=10)

# %%
