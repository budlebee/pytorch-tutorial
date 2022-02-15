# %%

import pandas as pd
import os
import numpy as np
import pickle
from math import sqrt
from torch import sign
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

    def saveParamWeight(self):
        with open("paramWeight.pickle", "wb") as f:
            pickle.dump(self.paramWeight, f)
            # pickle.dump(
            #    [[1], [1], [1], [1], [1], [1]], f)

    def loadParamWeight(self):
        with open("paramWeight.pickle", "rb") as f:
            self.paramWeight = pickle.load(f)
            # print(self.paramWeight)
        return self.paramWeight

    def saveTimeWeight(self):
        with open("timeWeight.pickle", "wb") as f:
            pickle.dump(self.timeWeight, f)

    def loadTimeWeight(self):
        with open("timeWeight.pickle", "rb") as f:
            self.timeWeight = pickle.load(f)
        return self.timeWeight

    def saveRaisingParamWeight(self):
        with open("raisingParamWeight.pickle", "wb") as f:
            pickle.dump(self.raisingParamWeight, f)

    def loadRaisingParamWeight(self):
        with open("raisingParamWeight.pickle", "rb") as f:
            self.raisingParamWeight = pickle.load(f)
        return self.raisingParamWeight

    def saveRaisingTimeWeight(self):
        with open("raisingTimeWeight.pickle", "wb") as f:
            pickle.dump(self.raisingTimeWeight, f)

    def loadRaisingTimeWeight(self):
        with open("raisingTimeWeight.pickle", "rb") as f:
            self.raisingTimeWeight = pickle.load(f)
        return self.raisingTimeWeight

    def saveLoweringParamWeight(self):
        with open("loweringParamWeight.pickle", "wb") as f:
            pickle.dump(self.loweringParamWeight, f)

    def loadLoweringParamWeight(self):
        with open("loweringParamWeight.pickle", "rb") as f:
            self.loweringParamWeight = pickle.load(f)
        return self.loweringParamWeight

    def saveLoweringTimeWeight(self):
        with open("loweringTimeWeight.pickle", "wb") as f:
            pickle.dump(self.loweringTimeWeight, f)

    def loadLoweringTimeWeight(self):
        with open("loweringTimeWeight.pickle", "rb") as f:
            self.loweringTimeWeight = pickle.load(f)
        return self.loweringTimeWeight

    # 해당 캔들 데이터를 통해 바의 길이와 테일의 길이를 재구성할 수 있으니 하나의 다변수 방정식을 만들어낼 수 있을것.
    # 방정식의 파라미터를 교정하는 식으로.
    # 행렬로 만들자. 다섯개의 샘플이니 행이 5개인 행렬. 각 요소별로 칼럼이 되겠네
    # 그럼 웨이트 벡터를 곱해서 나온건 벡터일텐데, 이제 시간가중치 듀얼벡터를 곱해서 스칼라 값 하나가 나오게끔 하자.

    def handleData(self, msg):
        if self.lastBet["betDir"] == "up":
            if msg['k']['h'] > self.lastBet["betPrice"]*self.benefit:
                # betting win. add weight to last betting strategy.
                # get lastBet['candleType'] and tuning weight.
                # 캔들타입을 굳이 명시해야되나? 어차피 다변수 방정식이니 캔들타입이 아니라 방정식의 계수들이 중요한거잖아.
                self.saveParamWeight()
                self.saveTimeWeight()
                print('betting win')
            elif msg['k']['l'] < self.lastBet['betPrice']/self.benefit:
                # betting lose. lose weight to last betting strategy.
                print('betting lose')
        elif self.lastBet["betDir"] == "down":
            if msg['k']['h'] > self.lastBet["betPrice"]/self.benefit:
                # betting win. add weight to last betting strategy
                print('betting win')
            elif msg['k']['l'] < self.lastBet['betPrice']*self.benefit:
                # betting lose lose weight to last betting strategy
                print('betting lose')
        if msg['E']-self.lastBet["betTime"] > self.positionLifetime:
            print("position close")

            # 해당 포지션을 정리해야됨.
            # msg 가 massge 임. 해당 메시지를 처리하는 콜백함수.
            # msg 를 일정 임계값에 따라 볼륨바와 달러바로 구성한다.
            # 볼륨바와 달러바는 고정 길이로 해야되나 변동 길이로 해야되나? ㄱ
            # 분류기가 한번 동작하고 나서 볼륨바를 초기화 해버리는건 데이터의 독립성은 보장하겠지만
            # 형편없는 전략이라는 지적이 있다.
            # 고정길이로 해서 checkSignal 함수에 집어넣고, 체크시그널안에서는 볼륨바에 가중치를 줘서
            # 의존성 문제를 해결하는식으로 하자.

            # open-close = bodyLength. 양수면 상승 캔들, 음수면 하락 캔들.
            # high-max(open,close) = upperTale
            # low-min(open,close) = lowerTale

            # 몇가지 시나리오 : 바디렝쓰가 짧고, 위아래로 캔들이 김 : 이후로 횡보
            # 하락장이었는데(연속해서 바디렝쓰가 음수였음) 캔들이 바디가 짧고 로워테일이 김 : 반등
            # 상승장이었는데 이후 나타난 캔들이 바디가 길고 어퍼 테일이 김: 하강
            # 두개의 다른방향 빅캔들이 붙어서 나타남: 하락캔들이후 바로 상승캔들인데 둘이 위차가 서로 비슷함: 그럼 서서히 상승.

            # 분봉을 보고, 한두시간마다 사고판다라. 캔들을 위주로 보고, 거래량은 보조정도라.

        self.checkRaisingSignal()
        self.checkLoweringSignal()
        return

    # 상승을 주도하는 패턴과 하락을 주도하는 패턴은 서로 다르다고 하니, 여러 분류기를 동시에 돌려야되나?

    def checkRaisingSignal(self, df):
        chance = False
        if chance == True:
            self.decideBetSize(df)
        return

    def checkLoweringSignal(self, df):
        chance = False
        if chance == True:
            self.decideBetSize(df)
        return

    def decideBetSize(self, df):
        # 여기서 베팅사이즈를 결정하고 주문을 넣은뒤, 주문을 기억하고 있어야함.
        return

    test_df = pd.DataFrame()

    """
    candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                     "closePrice": "", "volume": "", 'numberOfTrades': ""}]
    paramNum = 6
    candleMatrix = np.ones((sampleLength, paramNum))
    paramWeight = np.ones((paramNum, 1))
    timeWeight = np.ones((1, sampleLength))

    """
    signalProb = 0

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

    def trainSignalCatch(self):
        df = pd.read_csv(
            self.dir+'/data/BTC_kline_15m_2021-01-01_2022-01-07.csv')
# '/data/BTC_kline_15m_2021-01-01_2022-01-07.csv'
        # 캔들행렬 초기화
        for i in range(self.sampleLength):
            self.candleMatrix[i][0] = df.iloc[i]['high_price']
            self.candleMatrix[i][1] = df.iloc[i]['low_price']
            self.candleMatrix[i][2] = df.iloc[i]['open_price']
            self.candleMatrix[i][3] = df.iloc[i]['close_price']
            self.candleMatrix[i][4] = df.iloc[i]['volume']
            self.candleMatrix[i][5] = df.iloc[i]['number_of_trades']

        # 본격적으로 모델 돌아감
        for i in tqdm(range(self.sampleLength, len(df)), desc='trainSignalCatch...'):
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
            if self.lastBet['betTime'] != 0 and self.lastBet['betTime']+self.positionLifetime < candle['close_time']:
                if candle['high_price'] > self.lastBet['betPrice']*self.benefit or candle['low_price'] < self.lastBet['betPrice']*(2-self.benefit):
                    # 시그널이 포착됐을때 self.signalProb 값을 체크. 이 값이 0.5 보다 컸다?
                    # 그럼 시그널 포착 잘한거니까 가중치를 저장한다음 가중치에 살짝 변화주기.
                    # 가중치에 변화주는건 맞췄든 틀렸든 할거니까. 그대신 맞춘것만 저장됨으로써 생존하는거지.
                    self.lastBet['betTime'] = 0
                    self.lastBet['betPrice'] = 0
                    if self.signalProb > 0.5:
                        # 정답을 맞췄다면 저장하자
                        self.saveParamWeight()
                        self.saveTimeWeight()

            # timeout case
            elif self.lastBet['betTime'] != 0 and self.lastBet['betTime']+self.positionLifetime > candle['close_time']:
                self.lastBet['betTime'] = 0
                self.lastBet['betPrice'] = 0

            # 여기서 param 과 time weight 에 약간 랜덤 변동을 줘야한다.
            self.loadParamWeight()
            for i in range(self.paramNum):
                self.paramWeight[i] = self.paramWeight[i] + random.random()*2-1

            self.loadTimeWeight()
            for i in range(self.sampleLength):
                self.timeWeight[0][i] = self.timeWeight[0][i] + \
                    random.random()*2-1

            # prob 는 사건이 발생할 확률. 여기서 뭔가가 발생한다면, 그때 이게 상승인지 하락인지 체크하는 모델로 넘어감.
            # 일단은 prob 만 최적화 하기 위해 트레이닝하자.

            x = np.matmul(
                self.candleMatrix, self.paramWeight)

            prob = np.matmul(
                self.timeWeight / np.linalg.norm(self.timeWeight), x / np.linalg.norm(x))
            self.signalProb = prob[0][0]
            print(f"prob: {self.signalProb}")

            if self.lastBet['betTime'] == 0:
                self.lastBet['betTime'] = candle['close_time']
                self.lastBet['betPrice'] = candle['close_price']

            # if prob > 0.5:
            #    raisingProb = self.checkRaisingSignal(df)
            #    loweringProb = self.checkLoweringSignal(df)
            #    if raisingProb > loweringProb:
            #        self.decideBetSize(df)
            #    else:
            #        self.decideBetSize(df)

        # init candleMatrix

        # 15분봉 데이터를 받아서 처리하는 함수.
        print("done")
        return

    def trainUpSignalCatch(self):
        df = pd.read_csv(
            self.dir+'/data/BTC_kline_15m_2021-01-01_2022-01-07.csv')
# '/data/BTC_kline_15m_2021-01-01_2022-01-07.csv'
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
        for i in tqdm(range(self.sampleLength, len(df)), desc='trainSignalCatch...'):
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
            if lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime < candle['close_time']:
                if candle['high_price'] > lastBet['betPrice']*self.benefit:
                    # 시그널이 포착됐을때 signalProb 값을 체크. 이 값이 0.5 보다 컸다?
                    # 그럼 시그널 포착 잘한거니까 가중치를 저장한다음 가중치에 살짝 변화주기.
                    # 가중치에 변화주는건 맞췄든 틀렸든 할거니까. 그대신 맞춘것만 저장됨으로써 생존하는거지.
                    lastBet['betTime'] = 0
                    lastBet['betPrice'] = 0
                    if prob > 0.5:
                        # 정답을 맞췄다면 저장하자
                        self.saveRaisingParamWeight()
                        self.saveRaisingTimeWeight()

            # timeout case
            elif lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime > candle['close_time']:
                lastBet['betTime'] = 0
                lastBet['betPrice'] = 0

            # 여기서 param 과 time weight 에 약간 랜덤 변동을 줘야한다.
            self.loadRaisingParamWeight()
            for i in range(self.paramNum):
                self.raisingParamWeight[i] = self.raisingParamWeight[i] + \
                    random.random()*2-1

            self.loadRaisingTimeWeight()
            for i in range(self.sampleLength):
                self.raisingTimeWeight[0][i] = self.raisingTimeWeight[0][i] + \
                    random.random()*2-1

            # prob 는 사건이 발생할 확률. 여기서 뭔가가 발생한다면, 그때 이게 상승인지 하락인지 체크하는 모델로 넘어감.
            # 일단은 prob 만 최적화 하기 위해 트레이닝하자.

            x = np.matmul(
                self.candleMatrix, self.paramWeight)

            prob = np.matmul(
                self.raisingTimeWeight / np.linalg.norm(self.raisingTimeWeight), x / np.linalg.norm(x))

            if lastBet['betTime'] == 0:
                lastBet['betTime'] = candle['close_time']
                lastBet['betPrice'] = candle['close_price']
        print("done")
        return

    def trainDownSignalCatch(self):
        positionLifetime = 1000*60*15  # ms

        benefit = 1.02
        sampleLength = 5  # 다섯개의 캔들 데이터를 기반으로 판단할 것이다.
        candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                         "closePrice": "", "volume": "", 'numberOfTrades': ""}]
        paramNum = 6
        candleMatrix = np.zeros((sampleLength, paramNum))
        df = pd.read_csv(
            self.dir+'/data/BTC_kline_15m_2021-01-01_2022-01-07.csv')
# '/data/BTC_kline_15m_2021-01-01_2022-01-07.csv'
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
            if lastBet['betTime'] != 0 and lastBet['betTime']+self.positionLifetime < candle['close_time']:
                if candle['high_price'] > lastBet['betPrice']*self.benefit:
                    # 시그널이 포착됐을때 signalProb 값을 체크. 이 값이 0.5 보다 컸다?
                    # 그럼 시그널 포착 잘한거니까 가중치를 저장한다음 가중치에 살짝 변화주기.
                    # 가중치에 변화주는건 맞췄든 틀렸든 할거니까. 그대신 맞춘것만 저장됨으로써 생존하는거지.
                    lastBet['betTime'] = 0
                    lastBet['betPrice'] = 0
                    if prob > 0.5:
                        # 정답을 맞췄다면 저장하자
                        self.saveLoweringParamWeight()
                        self.saveLoweringTimeWeight()

            # timeout case
            elif lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime > candle['close_time']:
                lastBet['betTime'] = 0
                lastBet['betPrice'] = 0

            # 여기서 param 과 time weight 에 약간 랜덤 변동을 줘야한다.
            self.loadLoweringParamWeight()
            for i in range(self.paramNum):
                self.loweringParamWeight[i] = self.loweringParamWeight[i] + \
                    random.random()*2-1

            self.loadLoweringTimeWeight()
            for i in range(self.sampleLength):
                self.loweringTimeWeight[0][i] = self.loweringTimeWeight[0][i] + \
                    random.random()*2-1

            # prob 는 사건이 발생할 확률. 여기서 뭔가가 발생한다면, 그때 이게 상승인지 하락인지 체크하는 모델로 넘어감.
            # 일단은 prob 만 최적화 하기 위해 트레이닝하자.

            x = np.matmul(
                candleMatrix, self.loweringParamWeight)

            prob = np.matmul(
                self.loweringTimeWeight / np.linalg.norm(self.loweringTimeWeight), x / np.linalg.norm(x))

            if lastBet['betTime'] == 0:
                lastBet['betTime'] = candle['close_time']
                lastBet['betPrice'] = candle['close_price']
        print("done")
        return

    def trainBettingSize(self):
        return

    def testWholeModel(self, testDataDir):
        money = 1000
        # 바이낸스 선물 시장가 주문의 경우 수수료 0.04%
        # 배율이 올라가면 수수료도 올라감.
        ratio = 5
        fee = 0.0004*ratio
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
        timeWeight = self.loadTimeWeight()
        paramWeight = self.loadParamWeight()
        raisingTimeWeight = self.loadRaisingTimeWeight()
        raisingParamWeight = self.loadRaisingParamWeight()
        loweringTimeWeight = self.loadLoweringTimeWeight()
        loweringParamWeight = self.loadLoweringParamWeight()
        moneyHistory = []

        for i in range(self.sampleLength):
            candleMatrix[i][0] = df.iloc[i]['high_price']
            candleMatrix[i][1] = df.iloc[i]['low_price']
            candleMatrix[i][2] = df.iloc[i]['open_price']
            candleMatrix[i][3] = df.iloc[i]['close_price']
            candleMatrix[i][4] = df.iloc[i]['volume']
            candleMatrix[i][5] = df.iloc[i]['number_of_trades']

        for i in tqdm(range(sampleLength, len(df))):
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
            if lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime < candle['close_time']:
                if lastBet['betDir'] == "up":
                    if candle['high_price'] > lastBet['betPrice']*benefitCut:
                        # 상승배팅 수익 실현
                        money = (1-fee)*money*benefit
                        # money = (1-fee)*money * \
                        #    (1+((candle['high_price'] /
                        #     lastBet['betPrice'])-1) * ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"상승배팅 수익실현: {money}")
                        moneyHistory.append(money)
                    elif candle['low_price'] < lastBet['betPrice']*losscut:
                        # 상승배팅 손해
                        money = (1-fee)*money*losscut
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
                        money = (1-fee)*money * (2-benefit)
                        # money = (1-fee)*money * \
                        #    (1-(candle['high_price'] /
                        #     lastBet['betPrice'] - 1)*ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"하락배팅 손해: {money}")
                        moneyHistory.append(money)
                    elif candle['low_price'] < lastBet['betPrice']*losscut:
                        # 하락배팅 수익실현
                        money = (1-fee)*money*benefit
                        # money = (1-fee)*money * \
                        #    (1-((candle['low_price'] /
                        #     lastBet['betPrice'] - 1)*ratio))
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"하락배팅 수익실현: {money}")
                        moneyHistory.append(money)
            elif lastBet['betTime'] != 0 and lastBet['betTime']+positionLifetime > candle['close_time']:
                # 시간초과로 인한 포지션 종료.
                if lastBet['betDir'] == "up":
                    if candle['close_price'] > lastBet['betPrice']:

                        # 상승배팅 수익 실현
                        money = (1-fee)*money * \
                            (1+((candle['close_price'] /
                             lastBet['betPrice'])-1) * ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"타임아웃 상승배팅 수익실현: {money}")
                        moneyHistory.append(money)
                    elif candle['close_price'] < lastBet['betPrice']:
                        # 상승배팅 손해
                        money = (1-fee)*money * \
                            (1-(1-(candle['close_price'] /
                             lastBet['betPrice'])) * ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"타임아웃 상승배팅 손해: {money}")
                        moneyHistory.append(money)
                elif lastBet['betDir'] == "down":
                    if candle['close_price'] > lastBet['betPrice']:
                        # 다운배팅 손해
                        money = (1-fee)*money * \
                            (1-(candle['close_price'] /
                             lastBet['betPrice'] - 1)*ratio)
                        lastBet['betTime'] = 0
                        lastBet['betPrice'] = 0
                        print(f"타임아웃 다운배팅 손해: {money}")
                        moneyHistory.append(money)
                    elif candle['close_price'] < lastBet['betPrice']:
                        # 다운배팅 수익 실현
                        money = (1-fee)*money * \
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
            fluidProb = fluidProb[0][0]

            if fluidProb > 0.5 and lastBet['betTime'] == 0:
                xup = np.matmul(
                    candleMatrix, raisingParamWeight)
                upProb = np.matmul(
                    raisingTimeWeight / np.linalg.norm(raisingTimeWeight), xup / np.linalg.norm(xup))
                upProb = upProb[0][0]

                xdown = np.matmul(
                    candleMatrix, loweringParamWeight)
                downProb = np.matmul(
                    loweringTimeWeight / np.linalg.norm(loweringTimeWeight), xdown / np.linalg.norm(xdown))
                downProb = downProb[0][0]

                if upProb > downProb and upProb > 0.8:
                    lastBet['betDir'] = 'up'
                    lastBet['betTime'] = candle['close_time']
                    lastBet['betPrice'] = candle['close_price']
                    money = money * (1-fee)
                elif downProb > upProb and downProb > 0.8:
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
# for i in range(20):
#    ca.trainDownSignalCatch()

#testDataDir = '/data/BTC_kline_1h_210101_211231.csv'
testDataDir = '/data/future_BTC_15m_2020-01-01_2020-12-31.csv'
ca.testWholeModel(testDataDir)
# ca.testSignalCatch()


# %%
