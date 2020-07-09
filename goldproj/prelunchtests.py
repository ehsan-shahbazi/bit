import pandas as pd
from collections import OrderedDict
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import sys
import matplotlib.pyplot as plt


class Predictor:

    @staticmethod
    def derivate(df):
        ind = df.index[0:-1]
        t = df.diff().iloc[1:]
        t.index = ind

        tt = df.iloc[0:-1]

        ttt = t.div(tt) * 100
        # print('what the hell 4:', ttt)
        return ttt

    def __init__(self):
        model_dir = 'polls/trained/chp-1H-c-1LSTM-L-25-12e-Bi.h5'
        i_scale_dir = 'polls/trained/I_scaler.gz'
        o_scale_dir = 'polls/trained/O_scaler.gz'
        self.net = load_model(model_dir)
        self.i_scale = joblib.load(i_scale_dir)
        self.o_scale = joblib.load(o_scale_dir)

    def predict(self, df=''):
        """
        Inter your code
        :param: df: the df should be appropriated for prediction in size and time framing
        :return:
        """
        ul = 0.05
        ll = ul

        df = pd.to_numeric(df['Close']['mean'], downcast='float')
        arr = self.derivate(df).values.reshape(-1, 1)
        scaled_input = self.i_scale.transform(arr)
        the_input = np.reshape(scaled_input, (arr.shape[1], arr.shape[0], -1))

        prediction = self.net.predict(the_input)
        scaled_prediction = self.o_scale.inverse_transform(prediction)
        if scaled_prediction > ul:
            return 'B'
        elif scaled_prediction < ll:
            return 'S'
        else:
            return '?'


tot_df = pd.read_csv('data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
print(len(tot_df) - 72000)
tot_df = tot_df.iloc[4291457:]
print(len(tot_df))


def re_sample(the_df):
    period = len(the_df)
    ids = pd.period_range('2014-12-01 00:00:00', freq='T', periods=period)
    # print(the_df['Close'].head())
    the_df[['Open', 'High', 'Close', 'Low']] = the_df[['Open', 'High', 'Close', 'Low']].astype(float)
    df_res = the_df.set_index(ids).resample('1H').agg(OrderedDict([('Open', 'first'),
                                                                   ('High', 'max'),
                                                                   ('Low', 'min'),
                                                                   ('Close', ['mean', 'last'])]))
    # print(the_df)
    return df_res


input_size = 12
predictor = Predictor()
actions = []
prices = []
budget = 1000.0
asset = 0.0
tot_asset = []
all_asset = 0
total = len(tot_df) - (input_size + 2) * 60
for i in range(len(tot_df) - (input_size + 2) * 60):
    sys.stdout.flush()
    print(i * 100 / total, '%')
    sample_df = re_sample(tot_df.iloc[i: i + (input_size + 1) * 60])
    last = sample_df.tail(1)
    close = last['Close']['last']
    answer = predictor.predict(sample_df)
    actions.append(answer)
    prices.append(close.item())
    tot_asset.append(all_asset)
    print(answer, asset, budget, close.item())
    if (answer == 'S') and (asset != 0.0):
        budget = asset * 0.999 * close.item()
        asset = 0.0
        all_asset = budget
    elif (answer == 'B') and (budget != 0.0):
        asset = (budget * 0.999) / close.item()
        budget = 0.0
        all_asset = asset * close.item()
    print('number of sells:', actions.count('S'))
    print('number of buys:', actions.count('B'))
    print('number of don\'t know:', actions.count('?'))
    if len(tot_asset) > 1:
        print('hi', tot_asset[-1])

plt.plot(prices, label='price')
plt.plot(tot_asset, label='asset')
plt.legend()
plt.show()
