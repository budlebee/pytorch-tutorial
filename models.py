# %%
from tkinter import HORIZONTAL
import pandas as pd
import os
import numpy as np
import pickle

directory_of_python_script = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(directory_of_python_script +
                       "/data/"+"BTC"+"_kline_"+"1minute"+"_210101_211231.csv")

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
    df = pd.DataFrame()
    oneMinuteBar = pd.DataFrame()
    volumeBar = pd.DataFrame()
    dollarBar = pd.DataFrame()
    lastBet = {"betSize": 0, "betTime": 0,
               "betDwertir": "", "betPrice": 0, 'candleType': ""}
    horizontalTime = 1000  # ms
    benefit = 1.02
    sampleLength = 5  # 다섯개의 캔들 데이터를 기반으로 판단할 것이다.
    candleSample = [{"highPrice": "", "lowPrice": "", "openPrice": "",
                     "closePrice": "", "volume": "", 'numberOfTrades': ""}]
    paramNum = 6
    candleMatrix = np.zeros((sampleLength, paramNum))
    paramWeight = np.ones((paramNum, 1))
    timeWeight = np.ones((1, sampleLength))
    raisingParamWeight = np.ones((paramNum, 1))
    raisingTimeWeight = np.ones((1, sampleLength))
    loweringParamWeight = np.ones((paramNum, 1))
    loweringTimeWeight = np.ones((1, sampleLength))

    def saveParamWeight(self):
        with open("paramWeight.pickle", "wb") as f:
            pickle.dump(self.paramWeight*3, f)

    def loadParamWeight(self):
        with open("paramWeight.pickle", "rb") as f:
            self.paramWeight = pickle.load(f)
            print(self.paramWeight)

    def saveTimeWeight(self):
        with open("timeWeight.pickle", "wb") as f:
            pickle.dump(self.timeWeight, f)

    def loadTimeWeight(self):
        with open("timeWeight.pickle", "rb") as f:
            self.timeWeight = pickle.load(f)

    def saveRaisingParamWeight(self):
        with open("raisingParamWeight.pickle", "wb") as f:
            pickle.dump(self.raisingParamWeight*3, f)

    def loadRaisingParamWeight(self):
        with open("raisingParamWeight.pickle", "rb") as f:
            self.raisingParamWeight = pickle.load(f)

    def saveRaisingTimeWeight(self):
        with open("raisingTimeWeight.pickle", "wb") as f:
            pickle.dump(self.raisingTimeWeight, f)

    def loadRaisingTimeWeight(self):
        with open("raisingTimeWeight.pickle", "rb") as f:
            self.raisingTimeWeight = pickle.load(f)

    def saveLoweringParamWeight(self):
        with open("loweringParamWeight.pickle", "wb") as f:
            pickle.dump(self.loweringParamWeight*3, f)

    def loadLoweringParamWeight(self):
        with open("loweringParamWeight.pickle", "rb") as f:
            self.loweringParamWeight = pickle.load(f)

    def saveLoweringTimeWeight(self):
        with open("loweringTimeWeight.pickle", "wb") as f:
            pickle.dump(self.loweringTimeWeight, f)

    def loadLoweringTimeWeight(self):
        with open("loweringTimeWeight.pickle", "rb") as f:
            self.loweringTimeWeight = pickle.load(f)

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
        if msg['E']-self.lastBet["betTime"] > self.horizontalTime:
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

    def trainModel(self):
        df = pd.read_csv(
            self.dir+'/data/BTC_kline_15m_2021-01-01_2022-01-07.csv')

        # 캔들행렬 초기화
        for i in range(self.sampleLength):
            self.candleMatrix[i, 0] = df.iloc[i]['high_price']
            self.candleMatrix[i, 1] = df.iloc[i]['low_price']
            self.candleMatrix[i, 2] = df.iloc[i]['open_price']
            self.candleMatrix[i, 3] = df.iloc[i]['close_price']
            self.candleMatrix[i, 4] = df.iloc[i]['volume']
            self.candleMatrix[i, 5] = df.iloc[i]['number_of_trades']

        # 본격적으로 모델 돌아감
        for i in range(6, len(df)):
            # lastBet 이 있다면 수익 체크.

            # 캔들행렬 업데이트
            self.candleMatrix[0] = self.candleMatrix[1]
            self.candleMatrix[1] = self.candleMatrix[2]
            self.candleMatrix[2] = self.candleMatrix[3]
            self.candleMatrix[3] = self.candleMatrix[4]
            self.candleMatrix[4, 0] = df.iloc[i]['high_price']
            self.candleMatrix[4, 1] = df.iloc[i]['low_price']
            self.candleMatrix[4, 2] = df.iloc[i]['open_price']
            self.candleMatrix[4, 3] = df.iloc[i]['close_price']
            self.candleMatrix[4, 4] = df.iloc[i]['volume']
            self.candleMatrix[4, 5] = df.iloc[i]['number_of_trades']

            # prob 는 사건이 발생할 확률. 여기서 뭔가가 발생한다면, 그때 이게 상승인지 하락인지 체크하는 모델로 넘어감.
            prob = np.matmul(self.timeWeight, np.matmul(
                self.candleMatrix, self.paramWeight))
            if prob > 0.5:
                raisingProb = self.checkRaisingSignal(df)
                loweringProb = self.checkLoweringSignal(df)
                if raisingProb > loweringProb:
                    self.decideBetSize(df)
                else:
                    self.decideBetSize(df)

        # init candleMatrix

        # 15분봉 데이터를 받아서 처리하는 함수.

        return


# %%
ca = CandleAnalyzer()

ca.trainModel()


# %%
