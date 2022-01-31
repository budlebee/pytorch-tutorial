from models import CandleAnalyzer
import pandas as pd


def main():
    directory_of_python_script = os.path.dirname(os.path.abspath(__file__))
    train_df = pd.read_csv(directory_of_python_script +
                           "/data/"+"BTC"+"_kline_"+"1minute"+"_210101_211231.csv")
    cc = CandleAnalyzer()
    cc.handleData(train_df)
    # 1분봉 데이터를 집어넣고, 순차적으로 넣은뒤
    # 수익률을 알아내기.
    cc = CandleAnalyzer()
    cc.getData()
    return


if __name__ == "__main__":
    main()
