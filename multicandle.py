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

    totF = np.ones((1, 4)) / sqrt(4)
    totR = np.ones((1, 4)) / sqrt(4)
    totL = np.ones((1, 4)) / sqrt(4)

    def saveWeight(self, weight, filename):
        with open(f"{self.dir}/weights/{filename}.pickle", 'wb') as f:
            pickle.dump(weight, f)

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

        return pw1mF, pw5mF, pw15mF, pw1hF, tw1mF, tw5mF, tw15mF, tw1hF, pw1mR, pw5mR, pw15mR, pw1hR, tw1mR, tw5mR, tw15mR, tw1hR, pw1mL, pw5mL, pw15mL, pw1hL, tw1mL, tw5mL, tw15mL, tw1hL, totF, totR, totL

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

        with open(f"{self.dir}/weights/{versionName}_totL.pickle", "wb") as f:
            pickle.dump(self.totL, f)

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

    def trainFluid(self, versionName, startday, endday, thr, profitR, lossR, ratio):
        # profitCut 은 1.003 으로 하고, 10배율을 기본값으로 생각하자.
        fee = 0.0004*ratio
        pw1mF, pw5mF, pw15mF, pw1hF, tw1mF, tw5mF, tw15mF, tw1hF, pw1mR, pw5mR, pw15mR, pw1hR, tw1mR, tw5mR, tw15mR, tw1hR, pw1mL, pw5mL, pw15mL, pw1hL, tw1mL, tw5mL, tw15mL, tw1hL, totF, totR, totL = self.loadAllWeights(
            versionName)

        len1m = len(tw1mF[0])  # 30
        len5m = len(tw5mF[0])  # 10
        len15m = len(tw15mF[0])  # 10
        len1h = len(tw1hF[0])  # 5

        mat1m = np.zeros((len1m, self.paramNum))  # 30 x 7
        mat5m = np.zeros((len5m, self.paramNum))  # 10 x 7
        mat15m = np.zeros((len15m, self.paramNum))  # 10 x 7
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
        for i in tqdm(range(len1m, len(df1m)), desc='train fluid...'):
            # 1m 기준으로 계속 보고, 5개 될때마다 5m, 15개 될때마다 15m, 60개 될때마다 1h 업데이트 해주자.
            candle1m = df1m.iloc[i]
            candle5m = df5m.iloc[floor(i/5)]
            candle15m = df15m.iloc[floor(i/15)]
            candle1h = df1h.iloc[floor(i/60)]

            # 5번 - 1번. 10번 - 2번.
            # 15번 - 1번. 30번 - 2번.
            # 60번 - 1번.
            # 1분봉 데이터 업데이트
            for jj in range(len1m-1):
                mat1m[jj] = mat1m[jj+1]
            mat1m[len1m-1][0] = candle1m['high_price']
            mat1m[len1m-1][1] = candle1m['low_price']
            mat1m[len1m-1][2] = candle1m['open_price']
            mat1m[len1m-1][3] = candle1m['close_price']
            mat1m[len1m-1][4] = candle1m['volume']
            mat1m[len1m -
                  1][5] = candle1m['number_of_trades']
            mat1m[len1m -
                  1][6] = candle1m['close_time']

            if candle1m['close_time'] == candle5m['close_time']:
                # 5분봉 업데이트
                for jj in range(len5m-1):
                    mat5m[jj] = mat5m[jj+1]
                mat5m[len1m-1][0] = candle5m['high_price']
                mat5m[len1m-1][1] = candle5m['low_price']
                mat5m[len1m-1][2] = candle5m['open_price']
                mat5m[len1m-1][3] = candle5m['close_price']
                mat5m[len1m-1][4] = candle5m['volume']
                mat5m[len1m -
                      1][5] = candle5m['number_of_trades']
                mat5m[len1m -
                      1][6] = candle5m['close_time']
            if candle1m['close_time'] == candle15m['close_time']:
                # 15분봉 업데이트
                for jj in range(len15m-1):
                    mat15m[jj] = mat15m[jj+1]
                mat15m[len1m-1][0] = candle15m['high_price']
                mat15m[len1m-1][1] = candle15m['low_price']
                mat15m[len1m-1][2] = candle15m['open_price']
                mat15m[len1m-1][3] = candle15m['close_price']
                mat15m[len1m-1][4] = candle15m['volume']
                mat15m[len1m -
                       1][5] = candle15m['number_of_trades']
                mat15m[len1m -
                       1][6] = candle15m['close_time']
            if candle1m['close_time'] == candle1h['close_time']:
                # 1시간 봉 업데이트
                for jj in range(len1h-1):
                    mat1h[jj] = mat1h[jj+1]
                mat1h[len1m-1][0] = candle1h['high_price']
                mat1h[len1m-1][1] = candle1h['low_price']
                mat1h[len1m-1][2] = candle1h['open_price']
                mat1h[len1m-1][3] = candle1h['close_price']
                mat1h[len1m-1][4] = candle1h['volume']
                mat1h[len1m -
                      1][5] = candle1h['number_of_trades']
                mat1h[len1m -
                      1][6] = candle1h['close_time']

            if lastBet["betTime"] != 0:
                # 지난 배팅 체크하고 배팅하기
                if lastBet["betTime"]+self.positionLifetime > candle1m['close_time']:
                    if lastBet['betPrice']*profitR < candle1m['high_price'] or lastBet['betPrice']*(2-profitR) > candle1m['low_price']:
                        # 변동성이 있었으니 정답 맞는지 확인
                        if prob > thr:
                            # 정답 맞았으니 지금것 세이브.
                            self.saveWeight(pw1mF, "{versionName}_pw1mF".format(
                                versionName=versionName))
                            self.saveWeight(pw5mF, "{versionName}_pw5mF".format(
                                versionName=versionName))
                            self.saveWeight(pw15mF, "{versionName}_pw15mF".format(
                                versionName=versionName))
                            self.saveWeight(pw1hF, "{versionName}_pw1hF".format(
                                versionName=versionName))
                            self.saveWeight(tw1mF, "{versionName}_tw1mF".format(
                                versionName=versionName))
                            self.saveWeight(tw5mF, "{versionName}_tw5mF".format(
                                versionName=versionName))
                            self.saveWeight(tw15mF, "{versionName}_tw15mF".format(
                                versionName=versionName))
                            self.saveWeight(tw1hF, "{versionName}_tw1hF".format(
                                versionName=versionName))
                            self.saveWeight(totF, f"{versionName}_totF")

                        else:
                            continue
                        lastBet['betTime'] = 0

            if lastBet['betTime'] == 0:
                # 배팅하기
                pw1mF, pw5mF, pw15mF, pw1hF, tw1mF, tw5mF, tw15mF, tw1hF, pw1mR, pw5mR, pw15mR, pw1hR, tw1mR, tw5mR, tw15mR, tw1hR, pw1mL, pw5mL, pw15mL, pw1hL, tw1mL, tw5mL, tw15mL, tw1hL = self.loadAllWeights(
                    versionName)
                totF = self.loadWeight(f"{versionName}_totF")
                for i in range(len(pw1mF)):
                    pw1mF[i][0] = pw1mF[i][0] + pw1mF[i][0] * \
                        0.05*(2*random.random()-1)
                    pw5mF[i][0] = pw5mF[i][0] + pw5mF[i][0] * \
                        0.05*(2*random.random()-1)
                    pw15mF[i][0] = pw15mF[i][0] + \
                        pw15mF[i][0]*0.05*(2*random.random()-1)
                    pw1hF[i][0] = pw1hF[i][0] + pw1hF[i][0] * \
                        0.05*(2*random.random()-1)
                for i in range(len1m):
                    tw1mF[0][i] = tw1mF[0][i] + tw1mF[0][i] * \
                        0.05*(2*random.random()-1)
                plst = []
                x1m = np.matmul(mat1m, pw1mF)
                p1m = np.matmul(tw1mF, x1m)
                plst.append(p1m[0][0])
                x5m = np.matmul(mat5m, pw5mF)
                p5m = np.matmul(tw5mF, x5m)
                plst.append(p5m[0][0])
                x15m = np.matmul(mat15m, pw15mF)
                p15m = np.matmul(tw15mF, x15m)
                plst.append(p15m[0][0])
                x1h = np.matmul(mat1h, pw1hF)
                p1h = np.matmul(tw1hF, x1h)
                plst.append(p1h[0][0])
                for i in range(len(totF[0])):
                    totF[0][i] = totF[0][i]+totF[0][i] * \
                        0.05*(2*random.random()-1)
                for i in range(len(totF[0])):
                    prob += totF[0][i]*plst[i]
                prob = np.tanh(1, prob)
                prob = max(0.1*prob, prob)
                print(f"prob: {prob}")
                lastBet['betTime'] = candle1m['close_time']
                lastBet['betPrice'] = candle1m['close_price']
                print("new betting")


# %%
mca = MultiCandleAnalyzer()
# mca.createWeights("220223")
mca.trainFluid("220223")

# %%

# %%
