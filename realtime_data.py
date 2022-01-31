from binance import Client, ThreadedWebsocketManager
from decouple import config
import pandas as pd
import os
import websockets
import asyncio

binance_key = config('binance_key')
binance_secret = config('binance_secret')
client = Client(binance_key, binance_secret)
dir = os.path.dirname(os.path.abspath(__file__))


class Handler():
    symbol = 'BTCUSDT'
    numOfTrBefore = 1
    numOfTrNow = 1
    speedBefore = 1
    speedNow = 1
    momentum = 0

    def measure_trade_num_growth(before, after):
        return after - before

    def set_triple_barrier(cprice, timestamp_now):
        # cprice 를 기반으로 익절컷과 손절컷을 설정.
        print("g")

    event_time = []
    open_price = []
    close_price = []
    high_price = []
    low_price = []
    base_asset_volume = []
    number_of_trades = []
    kline_closed = []
    quote_asset_volume = []
    taker_buy_base_asset_volume = []
    taker_buy_quote_asset_volume = []
    df = pd.DataFrame()

    def collect_data(self, msg):
        print("run...")
        self.event_time.append(msg['E'])
        self.open_price.append(msg['k']['o'])
        self.close_price.append(msg['k']['c'])
        self.high_price.append(msg['k']['h'])
        self.low_price.append(msg['k']['l'])
        self.base_asset_volume.append(msg['k']['v'])
        self.number_of_trades.append(msg['k']['n'])
        self.kline_closed.append(msg['k']['x'])
        self.quote_asset_volume.append(msg['k']['q'])
        self.taker_buy_base_asset_volume.append(msg['k']['V'])
        self.taker_buy_quote_asset_volume.append(msg['k']['Q'])
        if len(self.event_time) > 5000:
            self.df['event_time'] = self.event_time
            self.df['open_price'] = self.open_price
            self.df['close_price'] = self.close_price
            self.df['high_price'] = self.high_price
            self.df['low_price'] = self.low_price
            self.df['base_asset_volume'] = self.base_asset_volume
            self.df['number_of_trades'] = self.number_of_trades
            self.df['kline_closed'] = self.kline_closed
            self.df['quote_asset_volume'] = self.quote_asset_volume
            self.df['taker_buy_base_asset_volume'] = self.taker_buy_base_asset_volume
            self.df['taker_buy_quote_asset_volume'] = self.taker_buy_quote_asset_volume
            self.df.to_csv(f"{dir}/realtime_data_{msg['E']}.csv")
            print("data saved!")
            self.event_time = []
            self.open_price = []
            self.close_price = []
            self.high_price = []
            self.low_price = []
            self.base_asset_volume = []
            self.number_of_trades = []
            self.kline_closed = []
            self.quote_asset_volume = []
            self.taker_buy_base_asset_volume = []
            self.taker_buy_quote_asset_volume = []

    def handle_socket_message(self, msg):
        # 여기서 모델이 msg 를 받고, 임계값이 넘었다 싶으면 주문을 전송.

        endOfCandle = msg['k']['x']

        numOfTrNow = msg['k']['n']
        speedNow = numOfTrNow-self.numOfTrBefore
        print(f"event time: {msg['E']}")
        print(f"speed change: {(speedNow/self.speedBefore -1)*100}%")
        if speedNow > self.speedBefore:
            self.momentum += 1
        else:
            self.momentum = 0

        if endOfCandle == True:
            self.numOfTrBefore = 0
        else:
            self.numOfTrBefore = numOfTrNow
        self.speedBefore = speedNow

        print("momentum: ", self.momentum)
        print("close price: ", msg['k']['c'])

    def ws_start(self):
        twm = ThreadedWebsocketManager(
            api_key=binance_key, api_secret=binance_secret)
        # start is required to initialise its internal loop
        twm.start()
        twm.start_kline_socket(
            callback=self.collect_data, symbol=self.symbol)
        twm.join()

    def startUpbitSocket(self):
        websockets.connect('wss://api.upbit.com/websocket/v1')


def main():
    print("start!")
    handler = Handler()
    handler.ws_start()


if __name__ == "__main__":
    main()
