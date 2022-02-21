# %%

import pandas as pd
import os
import numpy as np
import pickle
from math import sqrt, floor
from tqdm import tqdm
import random
from matplotlib import pyplot as plt

# %%


# 캔들 패턴 여러가지를 상정하고, 그에 해당하는 패턴이 됐을때 상승 / 하락 랜덤하게 배팅하고 승률을 기억해두자.
# 1분봉은 휩쏘가 많다고 하니까, 캔들패턴은 15분봉을 보자.
# 기억해둔것에 따라 확률을 점점 조정하는식으로 랜덤 프로세스.
#
# 캔들패턴으로 방향만 알아내는 것을 목표로.
# 배팅 사이즈는 거래량을 기준으로 봐야할까.

class CandleAnalyzer():
    def getData(self):
        # web sockets 를 통해 데이터를 가져오기. 콜백함수에서 뭔가를 하겠지.
        print("g")
    dir = os.path.dirname(os.path.abspath(__file__))
    filename = "/BTC_kline_1d_210101_211231.csv"
    df = pd.DataFrame()
    oneMinuteBar = pd.DataFrame()
    volumeBar = pd.DataFrame()
    dollarBar = pd.DataFrame()
    lastBet = {"betSize": 0, "betTime": 0,
               "betDir": "", "betPrice": 0, 'candleType': ""}
    positionLifetime = 1000*60*15  # ms
    # positionLifetime = 4 # 4개의 sample 시간동안 포지션을 살려둔다.
    benefit = 1.02
    sampleLength = 10  # 다섯개의 캔들 데이터를 기반으로 판단할 것이다.
    candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                     "closePrice": "", "volume": "", 'numberOfTrades': ""}]
    paramNum = 6
    candleMatrix = np.zeros((sampleLength, paramNum))
    paramWeight = np.ones((paramNum, 1)) / sqrt(paramNum)
    timeWeight = np.ones((1, sampleLength)) / sqrt(sampleLength)
    raisingParamWeight = np.ones((paramNum, 1)) / sqrt(paramNum)
    raisingTimeWeight = np.ones((1, sampleLength)) / sqrt(sampleLength)
    loweringParamWeight = np.ones((paramNum, 1))/sqrt(paramNum)
    loweringTimeWeight = np.ones((1, sampleLength))/sqrt(sampleLength)

    def createWeights(self, date, interval):
        pWeight = np.ones((self.paramNum, 1)) / sqrt(self.paramNum)
        tWeight = np.ones((1, self.sampleLength)) / sqrt(self.sampleLength)
        with open(f"{date}_{interval}_fluid_p.pickle", "wb") as f:
            pickle.dump(pWeight, f)
        with open(f"{date}_{interval}_fluid_t.pickle", "wb") as f:
            pickle.dump(tWeight, f)
        with open(f"{date}_{interval}_up_p.pickle", "wb") as f:
            pickle.dump(pWeight, f)
        with open(f"{date}_{interval}_up_t.pickle", "wb") as f:
            pickle.dump(tWeight, f)
        with open(f"{date}_{interval}_down_p.pickle", "wb") as f:
            pickle.dump(pWeight, f)
        with open(f"{date}_{interval}_down_t.pickle", "wb") as f:
            pickle.dump(tWeight, f)

    def printWeights(self, date, interval):
        with open(f"{date}_{interval}_paramWeight.pickle", "rb") as f:
            print(f"param: {pickle.load(f)}")
        with open(f"{date}_{interval}_timeWeight.pickle", "rb") as f:
            print(f"time: {pickle.load(f)}")
        with open(f"{date}_{interval}_raisingParamWeight.pickle", "rb") as f:
            print(f"raising param: {pickle.load(f)}")
        with open(f"{date}_{interval}_raisingTimeWeight.pickle", "rb") as f:
            print(f"raising time: {pickle.load(f)}")
        with open(f"{date}_{interval}_loweringParamWeight.pickle", "rb") as f:
            print(f"lowering param: {pickle.load(f)}")
        with open(f"{date}_{interval}_loweringTimeWeight.pickle", "rb") as f:
            print(f"lowering time: {pickle.load(f)}")

    def saveWeight(self, weight, filename):
        with open(f"{filename}.pickle", "wb") as f:
            pickle.dump(weight, f)

    def loadWeight(self, filename):
        with open(f"{filename}.pickle", "rb") as f:
            weight = pickle.load(f)
        return weight

    def saveParamWeight(self, weight, filename):
        with open(f"{filename}.pickle", "wb") as f:
            pickle.dump(weight, f)

    def loadParamWeight(self, filename):
        with open(f"{filename}.pickle", "rb") as f:
            weight = pickle.load(f)
            self.paramWeight = weight
        return weight

    def saveTimeWeight(self, weight, date, interval):
        with open(f"{date}_{interval}_timeWeight.pickle", "wb") as f:
            pickle.dump(weight, f)

    def loadTimeWeight(self, date, interval):
        with open(f"{date}_{interval}_timeWeight.pickle", "rb") as f:
            weight = pickle.load(f)
            self.timeWeight = weight
        return weight

    def saveRaisingParamWeight(self, weight, date, interval):
        with open(f"{date}_{interval}_raisingParamWeight.pickle", "wb") as f:
            pickle.dump(weight, f)

    def loadRaisingParamWeight(self, date, interval):
        with open(f"{date}_{interval}_raisingParamWeight.pickle", "rb") as f:
            self.raisingParamWeight = pickle.load(f)
        return self.raisingParamWeight

    def saveRaisingTimeWeight(self, date, interval):
        with open(f"{date}_{interval}_raisingTimeWeight.pickle", "wb") as f:
            pickle.dump(self.raisingTimeWeight, f)

    def loadRaisingTimeWeight(self, date, interval):
        with open(f"{date}_{interval}_raisingTimeWeight.pickle", "rb") as f:
            self.raisingTimeWeight = pickle.load(f)
        return self.raisingTimeWeight

    def saveLoweringParamWeight(self, date, interval):
        with open(f"{date}_{interval}_loweringParamWeight.pickle", "wb") as f:
            pickle.dump(self.loweringParamWeight, f)

    def loadLoweringParamWeight(self, date, interval):
        with open(f"{date}_{interval}_loweringParamWeight.pickle", "rb") as f:
            self.loweringParamWeight = pickle.load(f)
        return self.loweringParamWeight

    def saveLoweringTimeWeight(self, date, interval):
        with open(f"{date}_{interval}_loweringTimeWeight.pickle", "wb") as f:
            pickle.dump(self.loweringTimeWeight, f)

    def loadLoweringTimeWeight(self, date, interval):
        with open(f"{date}_{interval}_loweringTimeWeight.pickle", "rb") as f:
            self.loweringTimeWeight = pickle.load(f)
        return self.loweringTimeWeight

    def trainFluid(self, dataName, weightName, probCut=0.6, profitR=1.01, lifetime=1000*60*5*5):
        # initialize
        sampleLength = 10  # N개의 캔들 데이터를 기반으로 판단할 것이다.
        candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                         "closePrice": "", "volume": "", 'numberOfTrades': ""}]
        lossR = 2-profitR
        paramNum = 6
        candleMatrix = np.zeros((sampleLength, paramNum))
        pWeight = self.loadWeight(filename=f"{weightName}_fluid_p")
        tWeight = self.loadWeight(filename=f"{weightName}_fluid_t")
        prob = 0
        lastBet = {"betSize": 0, "betTime": 0,
                   "betDir": "", "betPrice": 0, 'candleType': ""}
        df = pd.read_csv(
            self.dir+'/data/'+dataName)
        for i in range(sampleLength):
            candleMatrix[i][0] = df.iloc[i]['high_price']
            candleMatrix[i][1] = df.iloc[i]['low_price']
            candleMatrix[i][2] = df.iloc[i]['open_price']
            candleMatrix[i][3] = df.iloc[i]['close_price']
            candleMatrix[i][4] = df.iloc[i]['volume']
            candleMatrix[i][5] = df.iloc[i]['number_of_trades']

        # training
        for i in tqdm(range(sampleLength, len(df)), desc='train fluid...'):
            candle = df.iloc[i]
            for jj in range(sampleLength-1):
                candleMatrix[jj] = candleMatrix[jj+1]

            candleMatrix[sampleLength-1][0] = candle['high_price']
            candleMatrix[sampleLength-1][1] = candle['low_price']
            candleMatrix[sampleLength-1][2] = candle['open_price']
            candleMatrix[sampleLength-1][3] = candle['close_price']
            candleMatrix[sampleLength-1][4] = candle['volume']
            candleMatrix[sampleLength -
                         1][5] = candle['number_of_trades']

            # 기존 주문이 있는지 확인하고
            # 있다면 타임아웃인지 아닌지 확인하고
            # 타임아웃 아니라면 변동성 있었는지 체크하고
            # 타임아웃이었다면 버리고.
            if lastBet['betTime'] != 0:  # 기존 주문이 있다.
                if lastBet['betTime']+lifetime > candle['close_time']:  # 아직 타임아웃이 아니다
                    if candle['high_price'] > lastBet['betPrice']*profitR or candle['low_price'] < lastBet['betPrice']*lossR:
                        # 지금 변동성이 있네?
                        # 그럼 얘가 정답을 맞췄는지 체크해 볼까?
                        if prob > probCut:
                            self.saveWeight(pWeight, f"{weightName}_fluid_p")
                            self.saveWeight(tWeight, f"{weightName}_fluid_t")
                        lastBet['betTime'] = 0
                        lastBet['betSize'] = 0
                        lastBet['betPrice'] = 0
                elif lastBet['betTime']+lifetime < candle['close_time']:  # 타임아웃 케이스
                    lastBet['betTime'] = 0
                    lastBet['betSize'] = 0
                    lastBet['betPrice'] = 0
            if lastBet['betTime'] == 0:
                pWeight = self.loadWeight(f"{weightName}_fluid_p")
                for i in range(paramNum):
                    pWeight[i][0] = pWeight[i][0] + random.random()*2-1
                tWeight = self.loadWeight(f"{weightName}_fluid_t")
                for i in range(sampleLength):
                    tWeight[0][i] = tWeight[0][i] + random.random()*2-1
                x = np.matmul(candleMatrix, pWeight)
                prob = np.matmul(
                    tWeight / np.linalg.norm(tWeight), x / np.linalg.norm(x))
                prob = (prob[0][0])
                lastBet['betTime'] = candle['close_time']
                lastBet['betPrice'] = candle['close_price']

    def trainUp(self, dataName, weightName, probCut=0.8, profitR=1.01, lifetime=1000*60*5*5):
        # initialize
        sampleLength = 10  # N개의 캔들 데이터를 기반으로 판단할 것이다.
        candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                         "closePrice": "", "volume": "", 'numberOfTrades': ""}]
        lossR = 2-profitR
        paramNum = 6
        candleMatrix = np.zeros((sampleLength, paramNum))
        pWeight = self.loadWeight(filename=f"{weightName}_up_p")
        tWeight = self.loadWeight(filename=f"{weightName}_up_t")
        prob = 0
        lastBet = {"betSize": 0, "betTime": 0,
                   "betDir": "", "betPrice": 0, 'candleType': ""}
        df = pd.read_csv(
            self.dir+'/data/'+dataName)

        for i in range(sampleLength):
            candleMatrix[i][0] = df.iloc[i]['high_price']
            candleMatrix[i][1] = df.iloc[i]['low_price']
            candleMatrix[i][2] = df.iloc[i]['open_price']
            candleMatrix[i][3] = df.iloc[i]['close_price']
            candleMatrix[i][4] = df.iloc[i]['volume']
            candleMatrix[i][5] = df.iloc[i]['number_of_trades']
        for i in tqdm(range(sampleLength, len(df)), desc='train up...'):
            candle = df.iloc[i]
            # 캔들행렬 업데이트
            for jj in range(sampleLength-1):
                candleMatrix[jj] = candleMatrix[jj+1]

            candleMatrix[sampleLength-1][0] = candle['high_price']
            candleMatrix[sampleLength-1][1] = candle['low_price']
            candleMatrix[sampleLength-1][2] = candle['open_price']
            candleMatrix[sampleLength-1][3] = candle['close_price']
            candleMatrix[sampleLength-1][4] = candle['volume']
            candleMatrix[sampleLength -
                         1][5] = candle['number_of_trades']

            # 기존 주문이 있는지 확인하고
            # 있다면 타임아웃인지 아닌지 확인하고
            # 타임아웃 아니라면 변동성 있었는지 체크하고
            # 타임아웃이었다면 버리고.
            if lastBet['betTime'] != 0:  # 기존 주문이 있다.
                if lastBet['betTime']+lifetime > candle['close_time']:  # 아직 타임아웃이 아니다
                    if candle['high_price'] > lastBet['betPrice']*profitR:  # 가격이 상승했다
                        # 지금 변동성이 있네?
                        # 그럼 얘가 정답을 맞췄는지 체크해 볼까?
                        if prob > probCut:
                            self.saveWeight(pWeight, f"{weightName}_up_p")
                            self.saveWeight(tWeight, f"{weightName}_up_t")
                        lastBet['betTime'] = 0
                        lastBet['betSize'] = 0
                        lastBet['betPrice'] = 0
                    elif candle['low_price'] < lastBet['betPrice']*lossR:  # price down. incorrect
                        lastBet['betTime'] = 0
                        lastBet['betSize'] = 0
                        lastBet['betPrice'] = 0
                elif lastBet['betTime']+lifetime < candle['close_time']:  # 타임아웃 케이스
                    lastBet['betTime'] = 0
                    lastBet['betSize'] = 0
                    lastBet['betPrice'] = 0
            if lastBet['betTime'] == 0:
                pWeight = self.loadWeight(f"{weightName}_up_p")
                for i in range(paramNum):
                    pWeight[i][0] = pWeight[i][0] + random.random()*2-1
                tWeight = self.loadWeight(f"{weightName}_up_t")
                for i in range(sampleLength):
                    tWeight[0][i] = tWeight[0][i] + random.random()*2-1
                x = np.matmul(candleMatrix, pWeight)
                prob = np.matmul(
                    tWeight / np.linalg.norm(tWeight), x / np.linalg.norm(x))
                prob = (prob[0][0])
                lastBet['betTime'] = candle['close_time']
                lastBet['betPrice'] = candle['close_price']

    def trainDown(self, dataName, weightName, probCut=0.8, profitR=1.01, lifetime=1000*60*5*5):
        # initialize
        sampleLength = 10  # N개의 캔들 데이터를 기반으로 판단할 것이다.
        candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                         "closePrice": "", "volume": "", 'numberOfTrades': ""}]
        lossR = 2-profitR
        paramNum = 6
        candleMatrix = np.zeros((sampleLength, paramNum))
        pWeight = self.loadWeight(filename=f"{weightName}_down_p")
        tWeight = self.loadWeight(filename=f"{weightName}_down_t")
        prob = 0
        lastBet = {"betSize": 0, "betTime": 0,
                   "betDir": "", "betPrice": 0, 'candleType': ""}
        df = pd.read_csv(
            self.dir+'/data/'+dataName)

        for i in range(sampleLength):
            candleMatrix[i][0] = df.iloc[i]['high_price']
            candleMatrix[i][1] = df.iloc[i]['low_price']
            candleMatrix[i][2] = df.iloc[i]['open_price']
            candleMatrix[i][3] = df.iloc[i]['close_price']
            candleMatrix[i][4] = df.iloc[i]['volume']
            candleMatrix[i][5] = df.iloc[i]['number_of_trades']
        for i in tqdm(range(sampleLength, len(df)), desc='train down...'):
            candle = df.iloc[i]
            # 캔들행렬 업데이트
            for jj in range(sampleLength-1):
                candleMatrix[jj] = candleMatrix[jj+1]

            candleMatrix[sampleLength-1][0] = candle['high_price']
            candleMatrix[sampleLength-1][1] = candle['low_price']
            candleMatrix[sampleLength-1][2] = candle['open_price']
            candleMatrix[sampleLength-1][3] = candle['close_price']
            candleMatrix[sampleLength-1][4] = candle['volume']
            candleMatrix[sampleLength -
                         1][5] = candle['number_of_trades']

            # 기존 주문이 있는지 확인하고
            # 있다면 타임아웃인지 아닌지 확인하고
            # 타임아웃 아니라면 변동성 있었는지 체크하고
            # 타임아웃이었다면 버리고.
            if lastBet['betTime'] != 0:  # 기존 주문이 있다.
                if lastBet['betTime']+lifetime > candle['close_time']:  # 아직 타임아웃이 아니다
                    if candle['low_price'] < lastBet['betPrice']*lossR:  # 가격이 상승했다
                        # 지금 변동성이 있네?
                        # 그럼 얘가 정답을 맞췄는지 체크해 볼까?
                        if prob > probCut:
                            self.saveWeight(pWeight, f"{weightName}_down_p")
                            self.saveWeight(tWeight, f"{weightName}_down_t")
                        lastBet['betTime'] = 0
                        lastBet['betSize'] = 0
                        lastBet['betPrice'] = 0
                    # price down. incorrect
                    elif candle['high_price'] > lastBet['betPrice']*profitR:
                        lastBet['betTime'] = 0
                        lastBet['betSize'] = 0
                        lastBet['betPrice'] = 0

                elif lastBet['betTime']+lifetime < candle['close_time']:  # 타임아웃 케이스
                    lastBet['betTime'] = 0
                    lastBet['betSize'] = 0
                    lastBet['betPrice'] = 0
            if lastBet['betTime'] == 0:  # 기존 주문이 없다.
                pWeight = self.loadWeight(f"{weightName}_down_p")
                for i in range(paramNum):
                    pWeight[i][0] = pWeight[i][0] + random.random()*2-1
                tWeight = self.loadWeight(f"{weightName}_down_t")
                for i in range(sampleLength):
                    tWeight[0][i] = tWeight[0][i] + random.random()*2-1
                x = np.matmul(candleMatrix, pWeight)
                prob = np.matmul(
                    tWeight / np.linalg.norm(tWeight), x / np.linalg.norm(x))
                prob = (prob[0][0])
                lastBet['betTime'] = candle['close_time']
                lastBet['betPrice'] = candle['close_price']

    def testFluid(self, dataName, weightName, probCut=0.6, profitR=1.01, lifetime=1000*60*25):
        # initialize
        sampleLength = 10  # N개의 캔들 데이터를 기반으로 판단할 것이다.
        candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                         "closePrice": "", "volume": "", 'numberOfTrades': ""}]
        lossR = 2-profitR
        paramNum = 6
        correct = 0
        incorrect = 0
        largeN = 0
        candleMatrix = np.zeros((sampleLength, paramNum))
        pWeight = self.loadWeight(filename=f"{weightName}_fluid_p")
        tWeight = self.loadWeight(filename=f"{weightName}_fluid_t")
        prob = 0
        lastBet = {"betSize": 0, "betTime": 0,
                   "betDir": "", "betPrice": 0, 'candleType': ""}
        df = pd.read_csv(
            self.dir+'/data/'+dataName)

        for i in range(sampleLength):
            candleMatrix[i][0] = df.iloc[i]['high_price']
            candleMatrix[i][1] = df.iloc[i]['low_price']
            candleMatrix[i][2] = df.iloc[i]['open_price']
            candleMatrix[i][3] = df.iloc[i]['close_price']
            candleMatrix[i][4] = df.iloc[i]['volume']
            candleMatrix[i][5] = df.iloc[i]['number_of_trades']
        for i in tqdm(range(sampleLength, len(df)), desc='test fluid...'):
            candle = df.iloc[i]
            # 캔들행렬 업데이트
            for jj in range(sampleLength-1):
                candleMatrix[jj] = candleMatrix[jj+1]

            candleMatrix[sampleLength-1][0] = candle['high_price']
            candleMatrix[sampleLength-1][1] = candle['low_price']
            candleMatrix[sampleLength-1][2] = candle['open_price']
            candleMatrix[sampleLength-1][3] = candle['close_price']
            candleMatrix[sampleLength-1][4] = candle['volume']
            candleMatrix[sampleLength -
                         1][5] = candle['number_of_trades']

            # 기존 주문이 있는지 확인하고
            # 있다면 타임아웃인지 아닌지 확인하고
            # 타임아웃 아니라면 변동성 있었는지 체크하고
            # 타임아웃이었다면 버리고.
            if lastBet['betTime'] != 0:  # 기존 주문이 있다.
                if lastBet['betTime']+lifetime > candle['close_time']:  # 아직 타임아웃이 아니다
                    if candle['high_price'] > lastBet['betPrice']*profitR or candle['low_price'] < lastBet['betPrice']*lossR:
                        # 지금 변동성이 있네?
                        # 그럼 얘가 정답을 맞췄는지 체크해 볼까?
                        if prob > probCut:
                            correct += 1
                        # 변동성은 있었는데 얘가 제대로 맞추질 못했네
                        largeN += 1
                        lastBet['betTime'] = 0
                        lastBet['betSize'] = 0
                        lastBet['betPrice'] = 0

                elif lastBet['betTime']+lifetime < candle['close_time']:  # 타임아웃 케이스
                    lastBet['betTime'] = 0
                    lastBet['betSize'] = 0
                    lastBet['betPrice'] = 0
                    incorrect += 1
                    largeN += 1
            if lastBet['betTime'] == 0:  # 기존 주문이 없다.
                x = np.matmul(candleMatrix, pWeight)
                prob = np.matmul(
                    tWeight / np.linalg.norm(tWeight), x / np.linalg.norm(x))
                prob = (prob[0][0])
                lastBet['betTime'] = candle['close_time']
                lastBet['betPrice'] = candle['close_price']
        print(
            f"정답: {correct}, 타임아웃: {incorrect}, 오답: {largeN-correct}, 정답률: {100*correct/largeN}%, 타임아웃률: {100*incorrect/largeN}%")

    def testUp(self, dataName, weightName, probCut=0.8, profitR=1.01, lifetime=1000*60*25):
        # initialize
        sampleLength = 10  # N개의 캔들 데이터를 기반으로 판단할 것이다.
        candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                         "closePrice": "", "volume": "", 'numberOfTrades': ""}]
        lossR = 2-profitR
        paramNum = 6
        correct = 0
        incorrect = 0
        largeN = 0
        candleMatrix = np.zeros((sampleLength, paramNum))
        pWeight = self.loadWeight(filename=f"{weightName}_up_p")
        tWeight = self.loadWeight(filename=f"{weightName}_up_t")
        prob = 0
        lastBet = {"betSize": 0, "betTime": 0,
                   "betDir": "", "betPrice": 0, 'candleType': ""}
        df = pd.read_csv(
            self.dir+'/data/'+dataName)

        for i in range(sampleLength):
            candleMatrix[i][0] = df.iloc[i]['high_price']
            candleMatrix[i][1] = df.iloc[i]['low_price']
            candleMatrix[i][2] = df.iloc[i]['open_price']
            candleMatrix[i][3] = df.iloc[i]['close_price']
            candleMatrix[i][4] = df.iloc[i]['volume']
            candleMatrix[i][5] = df.iloc[i]['number_of_trades']
        for i in tqdm(range(sampleLength, len(df)), desc='test up...'):
            candle = df.iloc[i]
            # 캔들행렬 업데이트
            for jj in range(sampleLength-1):
                candleMatrix[jj] = candleMatrix[jj+1]

            candleMatrix[sampleLength-1][0] = candle['high_price']
            candleMatrix[sampleLength-1][1] = candle['low_price']
            candleMatrix[sampleLength-1][2] = candle['open_price']
            candleMatrix[sampleLength-1][3] = candle['close_price']
            candleMatrix[sampleLength-1][4] = candle['volume']
            candleMatrix[sampleLength -
                         1][5] = candle['number_of_trades']

            # 기존 주문이 있는지 확인하고
            # 있다면 타임아웃인지 아닌지 확인하고
            # 타임아웃 아니라면 변동성 있었는지 체크하고
            # 타임아웃이었다면 버리고.
            if lastBet['betTime'] != 0:  # 기존 주문이 있다.
                if lastBet['betTime']+lifetime > candle['close_time']:  # 아직 타임아웃이 아니다
                    if candle['high_price'] > lastBet['betPrice']*profitR:
                        # 지금 변동성이 있네?
                        # 그럼 얘가 정답을 맞췄는지 체크해 볼까?
                        if prob > probCut:
                            correct += 1
                            lastBet['betTime'] = 0
                            lastBet['betSize'] = 0
                            lastBet['betPrice'] = 0
                        # 변동성은 있었는데 얘가 제대로 맞추질 못했네
                        largeN += 1
                    elif candle['low_price'] < lastBet['betPrice']*lossR:
                        incorrect += 1
                        largeN += 1
                        lastBet['betTime'] = 0
                        lastBet['betSize'] = 0
                        lastBet['betPrice'] = 0

                elif lastBet['betTime']+lifetime < candle['close_time']:  # 타임아웃 케이스
                    lastBet['betTime'] = 0
                    lastBet['betSize'] = 0
                    lastBet['betPrice'] = 0
                    largeN += 1
            if lastBet['betTime'] == 0:  # 기존 주문이 있다.
                x = np.matmul(candleMatrix, pWeight)
                prob = np.matmul(
                    tWeight / np.linalg.norm(tWeight), x / np.linalg.norm(x))
                prob = (prob[0][0])
                lastBet['betTime'] = candle['close_time']
                lastBet['betPrice'] = candle['close_price']
        print(
            f"정답: {correct}, 타임아웃: {incorrect}, 오답: {largeN-correct}, 정답률: {100*correct/largeN}%, 타임아웃률: {100*incorrect/largeN}%")

    def testDown(self, dataName, weightName, probCut=0.8, profitR=1.01, lifetime=1000*60*25):
        # initialize
        sampleLength = 10  # N개의 캔들 데이터를 기반으로 판단할 것이다.
        candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                         "closePrice": "", "volume": "", 'numberOfTrades': ""}]
        lossR = 2-profitR
        paramNum = 6
        correct = 0
        incorrect = 0
        largeN = 0
        candleMatrix = np.zeros((sampleLength, paramNum))
        pWeight = self.loadWeight(filename=f"{weightName}_down_p")
        tWeight = self.loadWeight(filename=f"{weightName}_down_t")
        prob = 0
        lastBet = {"betSize": 0, "betTime": 0,
                   "betDir": "", "betPrice": 0, 'candleType': ""}
        df = pd.read_csv(
            self.dir+'/data/'+dataName)

        for i in range(sampleLength):
            candleMatrix[i][0] = df.iloc[i]['high_price']
            candleMatrix[i][1] = df.iloc[i]['low_price']
            candleMatrix[i][2] = df.iloc[i]['open_price']
            candleMatrix[i][3] = df.iloc[i]['close_price']
            candleMatrix[i][4] = df.iloc[i]['volume']
            candleMatrix[i][5] = df.iloc[i]['number_of_trades']
        for i in tqdm(range(sampleLength, len(df)), desc='test down...'):
            candle = df.iloc[i]
            # 캔들행렬 업데이트
            for jj in range(sampleLength-1):
                candleMatrix[jj] = candleMatrix[jj+1]

            candleMatrix[sampleLength-1][0] = candle['high_price']
            candleMatrix[sampleLength-1][1] = candle['low_price']
            candleMatrix[sampleLength-1][2] = candle['open_price']
            candleMatrix[sampleLength-1][3] = candle['close_price']
            candleMatrix[sampleLength-1][4] = candle['volume']
            candleMatrix[sampleLength -
                         1][5] = candle['number_of_trades']

            # 기존 주문이 있는지 확인하고
            # 있다면 타임아웃인지 아닌지 확인하고
            # 타임아웃 아니라면 변동성 있었는지 체크하고
            # 타임아웃이었다면 버리고.
            if lastBet['betTime'] != 0:  # 기존 주문이 있다.
                if lastBet['betTime']+lifetime > candle['close_time']:  # 아직 타임아웃이 아니다
                    if candle['low_price'] < lastBet['betPrice']*lossR:
                        # 지금 변동성이 있네?
                        # 그럼 얘가 정답을 맞췄는지 체크해 볼까?
                        if prob > probCut:
                            correct += 1
                            lastBet['betTime'] = 0
                            lastBet['betSize'] = 0
                            lastBet['betPrice'] = 0
                        # 변동성은 있었는데 얘가 제대로 맞추질 못했네
                        largeN += 1
                    elif candle['high_price'] > lastBet['betPrice']*profitR:
                        incorrect += 1
                        largeN += 1
                        lastBet['betTime'] = 0
                        lastBet['betSize'] = 0
                        lastBet['betPrice'] = 0

                elif lastBet['betTime']+lifetime < candle['close_time']:  # 타임아웃 케이스
                    lastBet['betTime'] = 0
                    lastBet['betSize'] = 0
                    lastBet['betPrice'] = 0
                    largeN += 1
            if lastBet['betTime'] == 0:  # 기존 주문이 있다.
                x = np.matmul(candleMatrix, pWeight)
                prob = np.matmul(
                    tWeight / np.linalg.norm(tWeight), x / np.linalg.norm(x))
                prob = (prob[0][0])
                lastBet['betTime'] = candle['close_time']
                lastBet['betPrice'] = candle['close_price']
        print(
            f"정답: {correct}, 타임아웃: {incorrect}, 오답: {largeN-correct}, 정답률: {100*correct/largeN}%, 타임아웃률: {100*incorrect/largeN}%")

    def trainUpSignal(self):
        return

    def trainDownSignal(self):
        return

    def testSignalCatch(self):
        self.loadParamWeight()
        self.loadTimeWeight()
        signalStart = []
        signalGoal = []
        signalFail = []
        df = pd.read_csv(
            self.dir+'/data/future_BTC_15m_2020-01-01_2020-12-31.csv')

        for i in range(self.sampleLength):
            self.candleMatrix[i][0] = df.iloc[i]['high_price']
            self.candleMatrix[i][1] = df.iloc[i]['low_price']
            self.candleMatrix[i][2] = df.iloc[i]['open_price']
            self.candleMatrix[i][3] = df.iloc[i]['close_price']
            self.candleMatrix[i][4] = df.iloc[i]['volume']
            self.candleMatrix[i][5] = df.iloc[i]['number_of_trades']

        for i in tqdm(range(self.sampleLength, len(df))):
            candle = df.iloc[i]

            for jj in range(self.sampleLength-1):
                self.candleMatrix[jj] = self.candleMatrix[jj+1]

            self.candleMatrix[self.sampleLength-1][0] = candle['high_price']
            self.candleMatrix[self.sampleLength-1][1] = candle['low_price']
            self.candleMatrix[self.sampleLength-1][2] = candle['open_price']
            self.candleMatrix[self.sampleLength-1][3] = candle['close_price']
            self.candleMatrix[self.sampleLength-1][4] = candle['volume']
            self.candleMatrix[self.sampleLength -
                              1][5] = candle['number_of_trades']

            if self.lastBet['betTime'] != 0 and self.lastBet['betTime']+self.positionLifetime < candle['close_time']:
                if candle['high_price'] > self.lastBet['betPrice']*self.benefit or candle['low_price'] < self.lastBet['betPrice']*(2-self.benefit):
                    signalGoal.append(candle['close_time'])
                    self.lastBet['betTime'] = 0
                    self.lastBet['betPrice'] = 0
            elif self.lastBet['betTime'] != 0 and self.lastBet['betTime']+self.positionLifetime > candle['close_time']:
                self.lastBet['betTime'] = 0
                self.lastBet['betPrice'] = 0
                signalFail.append(candle['close_time'])
            else:
                signalFail.append(candle['close_time'])

            x = np.matmul(
                self.candleMatrix, self.paramWeight)

            prob = np.matmul(
                self.timeWeight / np.linalg.norm(self.timeWeight), x / np.linalg.norm(x))
            self.signalProb = prob[0][0]
            print(self.signalProb)
            if self.signalProb > 0.5 and self.lastBet['betTime'] == 0:
                self.lastBet['betTime'] = candle['close_time']
                self.lastBet['betPrice'] = candle['close_price']

                signalStart.append(self.lastBet['betTime'])

        plt.figure(figsize=(20, 10))
        plt.plot(df['close_time'], df['close_price'], label='close_price')
        plt.show()
        print(len(signalStart))
        print(len(signalGoal))
        print(len(signalFail))
        print("done")

        return

    def trainSignalCatch(self, filename, date, interval):
        positionLifetime = 1000*60*15*3  # ms

        benefit = 1.02
        sampleLength = 5  # 다섯개의 캔들 데이터를 기반으로 판단할 것이다.
        candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                         "closePrice": "", "volume": "", 'numberOfTrades': ""}]
        paramNum = 6
        candleMatrix = np.zeros((sampleLength, paramNum))
        df = pd.read_csv(
            self.dir+'/data/'+filename)
        signalProb = 0
        # 캔들행렬 초기화
        prob = 0
        lastBet = {"betSize": 0, "betTime": 0,
                   "betDir": "", "betPrice": 0, 'candleType': ""}
        df = pd.read_csv(
            self.dir+'/data/'+filename)
# '/data/BTC_kline_15m_2021-01-01_2022-01-07.csv'
        # 캔들행렬 초기화
        for i in range(self.sampleLength):
            candleMatrix[i][0] = df.iloc[i]['high_price']
            candleMatrix[i][1] = df.iloc[i]['low_price']
            candleMatrix[i][2] = df.iloc[i]['open_price']
            candleMatrix[i][3] = df.iloc[i]['close_price']
            candleMatrix[i][4] = df.iloc[i]['volume']
            candleMatrix[i][5] = df.iloc[i]['number_of_trades']

        # 본격적으로 모델 돌아감
        for i in tqdm(range(sampleLength, len(df)), desc='trainSignalCatch...'):
            candle = df.iloc[i]
            # 캔들행렬 업데이트
            for jj in range(sampleLength-1):
                candleMatrix[jj] = candleMatrix[jj+1]

            candleMatrix[sampleLength-1][0] = candle['high_price']
            candleMatrix[sampleLength-1][1] = candle['low_price']
            candleMatrix[sampleLength-1][2] = candle['open_price']
            candleMatrix[sampleLength-1][3] = candle['close_price']
            candleMatrix[sampleLength-1][4] = candle['volume']
            candleMatrix[sampleLength -
                         1][5] = candle['number_of_trades']

            # lastBet 이 있다면 수익 체크.
            # 일단은 lastBet 이 가격변동있는지부터 체크하자.

            #print(f"lastBet: {lastBet}")
            #print(f"lastBet['betTime']: {lastBet['betTime']}")
            #print(f"candle close_time: {candle['close_time']}")
            if float(lastBet['betTime']) != 0 and float(lastBet['betTime'])+positionLifetime > float(candle['close_time']):
                if float(candle['high_price']) > lastBet['betPrice']*benefit or float(candle['low_price']) < lastBet['betPrice']*(2-benefit):
                    # 시그널이 포착됐을때 signalProb 값을 체크. 이 값이 0.5 보다 컸다?
                    # 그럼 시그널 포착 잘한거니까 가중치를 저장한다음 가중치에 살짝 변화주기.
                    # 가중치에 변화주는건 맞췄든 틀렸든 할거니까. 그대신 맞춘것만 저장됨으로써 생존하는거지.
                    lastBet['betTime'] = 0
                    lastBet['betPrice'] = 0
                    if signalProb > 0.5:
                        # 정답을 맞췄다면 저장하자
                        self.saveParamWeight(date, interval)
                        self.saveTimeWeight(date, interval)

            # timeout case
            elif lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime < candle['close_time']:
                lastBet['betTime'] = 0
                lastBet['betPrice'] = 0

            # 여기서 param 과 time weight 에 약간 랜덤 변동을 줘야한다.
            self.loadParamWeight(date, interval)
            for i in range(paramNum):
                self.paramWeight[i] = self.paramWeight[i] + random.random()*2-1

            self.loadTimeWeight(date, interval)
            for i in range(sampleLength):
                self.timeWeight[0][i] = self.timeWeight[0][i] + \
                    random.random()*2-1

            # prob 는 사건이 발생할 확률. 여기서 뭔가가 발생한다면, 그때 이게 상승인지 하락인지 체크하는 모델로 넘어감.
            # 일단은 prob 만 최적화 하기 위해 트레이닝하자.

            x = np.matmul(
                candleMatrix, self.paramWeight)

            prob = np.matmul(
                self.timeWeight / np.linalg.norm(self.timeWeight), x / np.linalg.norm(x))
            signalProb = abs(prob[0][0])

            if lastBet['betTime'] == 0:
                lastBet['betTime'] = candle['close_time']
                lastBet['betPrice'] = candle['close_price']

        print("done")
        return

    def trainUpSignalCatch(self, filename, date, interval):
        df = pd.read_csv(
            self.dir+'/data/'+filename)
        # 캔들행렬 초기화
        positionLifetime = self.positionLifetime
        prob = 0
        lastBet = {"betSize": 0, "betTime": 0,
                   "betDir": "", "betPrice": 0, 'candleType': ""}
        for i in range(self.sampleLength):
            self.candleMatrix[i][0] = df.iloc[i]['high_price']
            self.candleMatrix[i][1] = df.iloc[i]['low_price']
            self.candleMatrix[i][2] = df.iloc[i]['open_price']
            self.candleMatrix[i][3] = df.iloc[i]['close_price']
            self.candleMatrix[i][4] = df.iloc[i]['volume']
            self.candleMatrix[i][5] = df.iloc[i]['number_of_trades']

        # 본격적으로 모델 돌아감
        for i in tqdm(range(self.sampleLength, len(df)), desc='trainUpSignalCatch...'):
            candle = df.iloc[i]
            # 캔들행렬 업데이트
            for jj in range(self.sampleLength-1):
                self.candleMatrix[jj] = self.candleMatrix[jj+1]

            self.candleMatrix[self.sampleLength-1][0] = candle['high_price']
            self.candleMatrix[self.sampleLength-1][1] = candle['low_price']
            self.candleMatrix[self.sampleLength-1][2] = candle['open_price']
            self.candleMatrix[self.sampleLength-1][3] = candle['close_price']
            self.candleMatrix[self.sampleLength-1][4] = candle['volume']
            self.candleMatrix[self.sampleLength -
                              1][5] = candle['number_of_trades']

            # lastBet 이 있다면 수익 체크.
            # 일단은 lastBet 이 가격변동있는지부터 체크하자.
            if lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime > candle['close_time']:
                if candle['high_price'] > lastBet['betPrice']*self.benefit:
                    lastBet['betTime'] = 0
                    lastBet['betPrice'] = 0
                    if prob > 0.7:
                        # 정답을 맞췄다면 저장하자
                        self.saveRaisingParamWeight(date, interval)
                        self.saveRaisingTimeWeight(date, interval)

            # timeout case
            elif lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime < candle['close_time']:
                lastBet['betTime'] = 0
                lastBet['betPrice'] = 0

            # 여기서 param 과 time weight 에 약간 랜덤 변동을 줘야한다.
            self.loadRaisingParamWeight(date, interval)
            for i in range(self.paramNum):
                self.raisingParamWeight[i] = self.raisingParamWeight[i] + \
                    random.random()*2-1

            self.loadRaisingTimeWeight(date, interval)
            for i in range(self.sampleLength):
                self.raisingTimeWeight[0][i] = self.raisingTimeWeight[0][i] + \
                    random.random()*2-1

            x = np.matmul(
                self.candleMatrix, self.paramWeight)

            prob = np.matmul(
                self.raisingTimeWeight / np.linalg.norm(self.raisingTimeWeight), x / np.linalg.norm(x))

            prob = abs(prob[0][0])
            if lastBet['betTime'] == 0:
                lastBet['betTime'] = candle['close_time']
                lastBet['betPrice'] = candle['close_price']
        print("done")
        return

    def trainDownSignalCatch(self, filename, date, interval):
        positionLifetime = 1000*60*15  # ms

        benefit = 1.02
        sampleLength = 5  # 다섯개의 캔들 데이터를 기반으로 판단할 것이다.
        candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                         "closePrice": "", "volume": "", 'numberOfTrades': ""}]
        paramNum = 6
        candleMatrix = np.zeros((sampleLength, paramNum))
        df = pd.read_csv(
            self.dir+'/data/'+filename)

        # 캔들행렬 초기화
        prob = 0
        lastBet = {"betSize": 0, "betTime": 0,
                   "betDir": "", "betPrice": 0, 'candleType': ""}
        for i in range(sampleLength):
            candleMatrix[i][0] = df.iloc[i]['high_price']
            candleMatrix[i][1] = df.iloc[i]['low_price']
            candleMatrix[i][2] = df.iloc[i]['open_price']
            candleMatrix[i][3] = df.iloc[i]['close_price']
            candleMatrix[i][4] = df.iloc[i]['volume']
            candleMatrix[i][5] = df.iloc[i]['number_of_trades']

        # 본격적으로 모델 돌아감
        for i in tqdm(range(sampleLength, len(df)), desc='trainSignalCatch...'):
            candle = df.iloc[i]
            # 캔들행렬 업데이트
            for jj in range(sampleLength-1):
                candleMatrix[jj] = candleMatrix[jj+1]

            candleMatrix[sampleLength-1][0] = candle['high_price']
            candleMatrix[sampleLength-1][1] = candle['low_price']
            candleMatrix[sampleLength-1][2] = candle['open_price']
            candleMatrix[sampleLength-1][3] = candle['close_price']
            candleMatrix[sampleLength-1][4] = candle['volume']
            candleMatrix[sampleLength -
                         1][5] = candle['number_of_trades']

            # lastBet 이 있다면 수익 체크.
            # 일단은 lastBet 이 가격변동있는지부터 체크하자.
            if lastBet['betTime'] != 0 and lastBet['betTime']+self.positionLifetime > candle['close_time']:
                if candle['high_price'] > lastBet['betPrice']*self.benefit:
                    # 시그널이 포착됐을때 signalProb 값을 체크. 이 값이 0.5 보다 컸다?
                    # 그럼 시그널 포착 잘한거니까 가중치를 저장한다음 가중치에 살짝 변화주기.
                    # 가중치에 변화주는건 맞췄든 틀렸든 할거니까. 그대신 맞춘것만 저장됨으로써 생존하는거지.
                    lastBet['betTime'] = 0
                    lastBet['betPrice'] = 0
                    if prob > 0.7:
                        # 정답을 맞췄다면 저장하자
                        self.saveLoweringParamWeight(date, interval)
                        self.saveLoweringTimeWeight(date, interval)

            # timeout case
            elif lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime < candle['close_time']:
                lastBet['betTime'] = 0
                lastBet['betPrice'] = 0

            # 여기서 param 과 time weight 에 약간 랜덤 변동을 줘야한다.
            self.loadLoweringParamWeight(date, interval)
            for i in range(self.paramNum):
                self.loweringParamWeight[i] = self.loweringParamWeight[i] + \
                    random.random()*2-1

            self.loadLoweringTimeWeight(date, interval)
            for i in range(self.sampleLength):
                self.loweringTimeWeight[0][i] = self.loweringTimeWeight[0][i] + \
                    random.random()*2-1

            # prob 는 사건이 발생할 확률. 여기서 뭔가가 발생한다면, 그때 이게 상승인지 하락인지 체크하는 모델로 넘어감.
            # 일단은 prob 만 최적화 하기 위해 트레이닝하자.

            x = np.matmul(
                candleMatrix, self.loweringParamWeight)

            prob = np.matmul(
                self.loweringTimeWeight / np.linalg.norm(self.loweringTimeWeight), x / np.linalg.norm(x))

            prob = abs(prob[0][0])

            if lastBet['betTime'] == 0:
                lastBet['betTime'] = candle['close_time']
                lastBet['betPrice'] = candle['close_price']
        print("done")
        return

    def trainBettingSize(self):
        return

    def testWholeModel(self, testDataDir, date, interval):
        money = 1000
        # 바이낸스 선물 시장가 주문의 경우 수수료 0.04%. 0.0004.
        # 배율이 올라가면 수수료도 올라감.
        ratio = 5
        fee = 0.0004*ratio
        apiTroubleLoss = 0.0004  # *ratio
        slipageLoss = 0.0004  # *ratio
        df = pd.read_csv(
            self.dir+testDataDir)
        lastBet = {"betSize": 0, "betTime": 0,
                   "betDir": "", "betPrice": 0, 'candleType': ""}
        positionLifetime = 1000*60*15  # ms
        benefit = 1.02
        benefitCut = 1+(benefit-1)/ratio
        losscut = 2-benefit
        sampleLength = 5  # 다섯개의 캔들 데이터를 기반으로 판단할 것이다.
        candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                         "closePrice": "", "volume": "", 'numberOfTrades': ""}]
        paramNum = 6
        candleMatrix = np.zeros((sampleLength, paramNum))
        paramWeight = np.ones((paramNum, 1)) / sqrt(paramNum)
        timeWeight = np.ones((1, sampleLength)) / sqrt(sampleLength)
        raisingParamWeight = np.ones((paramNum, 1)) / sqrt(paramNum)
        raisingTimeWeight = np.ones((1, sampleLength)) / sqrt(sampleLength)
        loweringParamWeight = np.ones((paramNum, 1))/sqrt(paramNum)
        loweringTimeWeight = np.ones((1, sampleLength))/sqrt(sampleLength)
        timeWeight = self.loadTimeWeight(date, interval)
        paramWeight = self.loadParamWeight(date, interval)
        raisingTimeWeight = self.loadRaisingTimeWeight(date, interval)
        raisingParamWeight = self.loadRaisingParamWeight(date, interval)
        loweringTimeWeight = self.loadLoweringTimeWeight(date, interval)
        loweringParamWeight = self.loadLoweringParamWeight(date, interval)
        print(raisingTimeWeight)
        moneyHistory = []

        for i in range(self.sampleLength):
            candleMatrix[i][0] = df.iloc[i]['high_price']
            candleMatrix[i][1] = df.iloc[i]['low_price']
            candleMatrix[i][2] = df.iloc[i]['open_price']
            candleMatrix[i][3] = df.iloc[i]['close_price']
            candleMatrix[i][4] = df.iloc[i]['volume']
            candleMatrix[i][5] = df.iloc[i]['number_of_trades']

        for i in tqdm(range(sampleLength, floor(len(df)/300))):
            candle = df.iloc[i]

            for jj in range(sampleLength-1):
                candleMatrix[jj] = candleMatrix[jj+1]

            candleMatrix[sampleLength-1][0] = candle['high_price']
            candleMatrix[sampleLength-1][1] = candle['low_price']
            candleMatrix[sampleLength-1][2] = candle['open_price']
            candleMatrix[sampleLength-1][3] = candle['close_price']
            candleMatrix[sampleLength-1][4] = candle['volume']
            candleMatrix[sampleLength -
                         1][5] = candle['number_of_trades']

            # 수익 체크.
            if lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime > candle['close_time']:
                if lastBet['betDir'] == "up":
                    if candle['high_price'] > lastBet['betPrice']*benefitCut:
                        # 상승배팅 수익 실현
                        money = (1-apiTroubleLoss)*(1-slipageLoss) * \
                            (1-fee)*money*benefit
                        # money = (1-fee)*money * \
                        #    (1+((candle['high_price'] /
                        #     lastBet['betPrice'])-1) * ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"상승배팅 수익실현: {money}")
                        moneyHistory.append(money)
                    elif candle['low_price'] < lastBet['betPrice']*losscut:
                        # 상승배팅 손해
                        money = (1-apiTroubleLoss)*(1-slipageLoss) * \
                            (1-fee)*money*losscut
                        # money = (1-fee)*money * \
                        #    (1-(1-(candle['low_price'] /
                        #     lastBet['betPrice'])) * ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"상승배팅 손해: {money}")
                        moneyHistory.append(money)
                elif lastBet['betDir'] == "down":
                    if candle['high_price'] > lastBet['betPrice']*benefitCut:
                        # 하락배팅 손해
                        money = (1-apiTroubleLoss)*(1-slipageLoss) * \
                            (1-fee)*money * (2-benefit)
                        # money = (1-fee)*money * \
                        #    (1-(candle['high_price'] /
                        #     lastBet['betPrice'] - 1)*ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"하락배팅 손해: {money}")
                        moneyHistory.append(money)
                    elif candle['low_price'] < lastBet['betPrice']*losscut:
                        # 하락배팅 수익실현
                        money = (1-apiTroubleLoss)*(1-slipageLoss) * \
                            (1-fee)*money*benefit
                        # money = (1-fee)*money * \
                        #    (1-((candle['low_price'] /
                        #     lastBet['betPrice'] - 1)*ratio))
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"하락배팅 수익실현: {money}")
                        moneyHistory.append(money)
            elif lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime < candle['close_time']:
                # 시간초과로 인한 포지션 종료.
                if lastBet['betDir'] == "up":
                    if candle['close_price'] > lastBet['betPrice']:

                        # 상승배팅 수익 실현
                        money = (1-apiTroubleLoss)*(1-slipageLoss)*(1-fee)*money * \
                            (1+((candle['close_price'] /
                             lastBet['betPrice'])-1) * ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"타임아웃 상승배팅 수익실현: {money}")
                        moneyHistory.append(money)
                    elif candle['close_price'] < lastBet['betPrice']:
                        # 상승배팅 손해
                        money = (1-apiTroubleLoss)*(1-slipageLoss)*(1-fee)*money * \
                            (1-(1-(candle['close_price'] /
                             lastBet['betPrice'])) * ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"타임아웃 상승배팅 손해: {money}")
                        moneyHistory.append(money)
                elif lastBet['betDir'] == "down":
                    if candle['close_price'] > lastBet['betPrice']:
                        # 다운배팅 손해
                        money = (1-apiTroubleLoss)*(1-slipageLoss)*(1-fee)*money * \
                            (1-(candle['close_price'] /
                             lastBet['betPrice'] - 1)*ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"타임아웃 다운배팅 손해: {money}")
                        moneyHistory.append(money)
                    elif candle['close_price'] < lastBet['betPrice']:
                        # 다운배팅 수익 실현
                        money = (1-apiTroubleLoss)*(1-slipageLoss)*(1-fee)*money * \
                            (1-((candle['close_price'] /
                             lastBet['betPrice'] - 1)*ratio))
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"타임아웃 다운배팅 수익실현: {money}")
                        moneyHistory.append(money)
            fluidProb = 0
            upProb = 0
            downProb = 0
            x = np.matmul(
                candleMatrix, paramWeight)

            fluidProb = np.matmul(
                timeWeight / np.linalg.norm(timeWeight), x / np.linalg.norm(x))
            fluidProb = abs(fluidProb[0][0])
            # print(fluidProb)
            fluidThr = 0.5
            bettingThr = 0.8

            if fluidProb > fluidThr and lastBet['betTime'] == 0:
                xup = np.matmul(
                    candleMatrix, raisingParamWeight)
                upProb = np.matmul(
                    raisingTimeWeight / np.linalg.norm(raisingTimeWeight), xup / np.linalg.norm(xup))
                upProb = abs(upProb[0][0])

                xdown = np.matmul(
                    candleMatrix, loweringParamWeight)
                downProb = np.matmul(
                    loweringTimeWeight / np.linalg.norm(loweringTimeWeight), xdown / np.linalg.norm(xdown))
                downProb = abs(downProb[0][0])

                #print(f"상승 확률: {upProb}")
                #print(f"하락 확률: {downProb}")

                if upProb > downProb and upProb > bettingThr:
                    print(f"상승배팅 시도: {upProb}")
                    lastBet['betDir'] = 'up'
                    lastBet['betTime'] = candle['close_time']
                    lastBet['betPrice'] = candle['close_price']
                    money = money * (1-fee)
                elif downProb > upProb and downProb > bettingThr:
                    print(f"하락배팅 시도: {downProb}")
                    lastBet['betDir'] = 'down'
                    lastBet['betTime'] = candle['close_time']
                    lastBet['betPrice'] = candle['close_price']
                    money = money * (1-fee)

        xline = np.arange(len(moneyHistory))
        print(f"종료. 금액: {money}")
        plt.plot(xline, moneyHistory)
        plt.show()

        return

        def trainWholeModel(self):
            # 만약 타임아웃 된다면 시그널 캐치 파라미터를 바꾸고
            # 만약 상승배팅인데 틀렸다면 상승배팅 파라미터를 바꾸고 메모리에 올리고
            # 만약 하강배팅인데 틀렸다면 하락배팅 파라미터를 바꾸고 메모리에 올리고
            # 저장을 하는 것은 수익을 올렸을때만 저장을 하자.
            return


# %%
ca = CandleAnalyzer()
#ca.createWeights("220220", "5m")
for i in range(300):
    ca.trainFluid("future_BTC_5m_2022-01-01_2022-01-31.csv", "220220_5m")
    ca.trainUp("future_BTC_5m_2022-01-01_2022-01-31.csv", "220220_5m")
    ca.trainDown("future_BTC_5m_2022-01-01_2022-01-31.csv", "220220_5m")
    #ca.testFluid("future_BTC_5m_2021-01-01_2021-12-31.csv", "220220_5m")
    #ca.testUp("future_BTC_5m_2021-01-01_2021-12-31.csv", "220220_5m")
    #ca.testDown("future_BTC_5m_2021-01-01_2021-12-31.csv", "220220_5m")
ca.testFluid("future_BTC_5m_2022-01-01_2022-01-31.csv", "220220_5m")
ca.testUp("future_BTC_5m_2022-01-01_2022-01-31.csv", "220220_5m")
ca.testDown("future_BTC_5m_2022-01-01_2022-01-31.csv", "220220_5m")


#filename = "future_BTC_5m_2021-01-01_2021-12-31.csv"
filename = "BTC_kline_15m_2021-01-01_2022-01-07.csv"
#ca.createWeights("202220", "15m")
#ca.printWeights("202220", "5m")
# for i in range(10):
#    ca.trainSignalCatch(filename, "202220", "15m")
#    ca.trainUpSignalCatch(filename, "202220", "15m")
#    ca.trainDownSignalCatch(filename, "202220", "15m")
#ca.printWeights("202220", "15m")
#testDataDir = '/data/future_BTC_5m_2022-01-01_2022-01-31.csv'
#testDataDir = '/data/future_BTC_15m_2022-01-01_2022-01-31.csv'
testDataDir = "/data/BTC_kline_15m_2021-01-01_2022-01-07.csv"
#ca.testWholeModel(testDataDir, "202220", "15m")
#testDataDir = '/data/BTC_kline_1h_200101_201231.csv'
#testDataDir = '/data/BTC_kline_1h_210101_211231.csv'
#testDataDir = '/data/future_BTC_15m_2022-01-01_2022-01-31.csv'

#testDataDir = '/data/future_BTC_15m_2020-01-01_2020-12-31.csv'
#ca.testWholeModel(testDataDir, "202220", "5m")
# ca.testSignalCatch()


# %%
