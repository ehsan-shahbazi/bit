import pandas as pd
from collections import OrderedDict
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import sys
import ta
import matplotlib.pyplot as plt
import sklearn.tree as tree
import time
num_of_trees = 15


class Predictor:

    @staticmethod
    def derivate(df):
        ind = df.index[0:-1]
        t = df.diff().iloc[1:]
        t.index = ind

        tt = df.iloc[0:-1]

        ttt = t.div(tt) * 100
        labels = []
        """
        plt.hist(list(ttt), bins=100)
        plt.show()
        """
        for dif in ttt:
            if (dif < 10000) & (dif > 1):
                labels.append([1, 0, 0, 0, 0, 0])
            elif (dif < 1) & (dif > 0.2):
                labels.append([0, 1, 0, 0, 0, 0])
            elif (dif < 0.2) & (dif > 0):
                labels.append([0, 0, 1, 0, 0, 0])
            elif (dif < 0) & (dif > -0.2):
                labels.append([0, 0, 0, 1, 0, 0])
            elif (dif < -0.2) & (dif > -1):
                labels.append([0, 0, 0, 0, 1, 0])
            else:
                labels.append([0, 0, 0, 0, 0, 1])
        # print('what the hell 4:', ttt)
        return labels

    @staticmethod
    def make_inputs(df, test=False):
        df.drop(['Open', 'Close', 'High', 'Low'], axis=1)
        inputs = []
        for index, data in df.iterrows():
            # print(list(data))
            inputs.append(list(data)[1:])
        if test:
            return inputs
        else:
            return inputs[0:-1]

    def learn_tree(self, df, name='first_tree'):
        the_tree = tree.DecisionTreeClassifier(max_depth=10)
        the_tree = the_tree.fit(self.make_inputs(df), self.derivate(df['Close']))
        from joblib import dump, load
        dump(the_tree, 'data/' + name + '.joblib')
        return the_tree

    def predict(self, df):
        """
        Inter your code
        :param: df: the df should be appropriated for prediction in size and time framing
        :return:
        """

        return '?'


def re_sample(the_df, method='1H'):
    """
    :param the_df:
    :param method: It should be 1H or 1D or 1Min or ...
    :return:
    """
    period = len(the_df)
    ids = pd.period_range('2000-12-01 00:00:00', freq='T', periods=period)
    the_df[['Open', 'High', 'Close', 'Low', "Volume_(Currency)"]] = \
        the_df[['Open', 'High', 'Close', 'Low', "Volume_(Currency)"]].astype(float)
    df_res = the_df.set_index(ids).resample(method).agg(OrderedDict([('Open', 'first'),
                                                                     ('High', 'max'),
                                                                     ('Low', 'min'),
                                                                     ('Close', 'last'),
                                                                     ('Volume_(Currency)', 'sum')]))
    return df_res


tot_df = pd.read_csv('data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
print(len(tot_df))
print(len(tot_df) - 720000)
test_df = tot_df.iloc[len(tot_df) - 720000:]
train_df = tot_df.iloc[720000: len(tot_df) - 720000]
print(train_df.head(), '\n re_sampling test df')
test_df = re_sample(test_df, method='1H')
print(test_df.head())
print('done.\n making features.')
print(time.time())
test_df = ta.add_all_ta_features(test_df, open="Open", high="High", low="Low", close="Close",
                                 volume="Volume_(Currency)", fillna=True)
print(time.time())
test_df.to_csv('data/featured_data_test')
train_df = re_sample(train_df, method='1H')
print('making test csv is done')
print(time.time())
train_df = ta.add_all_ta_features(train_df, open="Open", high="High", low="Low", close="Close",
                                  volume="Volume_(Currency)", fillna=True)
print(time.time())
test_df.to_csv('data/featured_data_train')
print(test_df.head())



"""
train part
"""


def train():
    trained_df = pd.read_csv('data/featured_data_train')
    print(trained_df.head())
    new_predictor = Predictor()
    for i in range(num_of_trees):
        print(i, '% completed')
        new_predictor.learn_tree(trained_df, name=str(i))

train()
"""
test part
"""


def test():
    new_predictor = Predictor()
    new_test_data = pd.read_csv('data/featured_data_test')
    new_test_outputs = new_predictor.derivate(new_test_data['Close'])
    new_test_inputs = new_predictor.make_inputs(new_test_data, test=True)
    new_test_data = pd.read_csv('data/featured_data_test')
    accuracy = []
    for i in range(num_of_trees):
        new_tree = joblib.load('data/' + str(i) + '.joblib')
        new_predictions = new_tree.predict(new_test_inputs)
        true = 0
        for j in range(len(new_predictions) - 1):
            if (list(new_test_outputs[j]) in [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0]]) & (
                    list(new_predictions[j]) in [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]):
                true += 1
            if (list(new_test_outputs[j]) not in [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0, 0]]) & (
                    list(new_predictions[j]) not in [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]):
                true += 1
        accuracy.append(true / len(new_predictions))
        print('accuracy is: ', true / len(new_predictions))
    plt.hist(accuracy, bins=10)
    plt.show()

test()

"""
simulations part
"""
predictor = Predictor()
test_data = pd.read_csv('data/featured_data_test')
# test_outputs = predictor.derivate(test_data['Close'])
test_inputs = predictor.make_inputs(test_data, test=True)
test_data = pd.read_csv('data/featured_data_test')
tree1 = joblib.load('data/' + str(0) + '.joblib')
predictions = tree1.predict(test_inputs)
prices = []
asset = []
buys = []
sells = []
budget = 1000
mat = 0
fee = 0.999
print(len(predictions), len(test_data))
tot = len(test_data)
for index, data in test_data.iterrows():
    if index == len(predictions):
        continue
    sys.stdout.flush()
    sys.stdout.write('\r ' + str(index / tot) + '%')
    price = data['Close']
    if (list(predictions[index]) == [1, 0, 0, 0, 0, 0]) | (list(predictions[index]) == [0, 1, 0, 0, 0, 0]) |\
            (list(predictions[index]) == [0, 0, 1, 0, 0, 0]):
        if budget > 0:
            mat = budget * fee / price
            budget = 0
            buys.append(index)

    if (list(predictions[index]) == [0, 0, 0, 1, 0, 0]) | (list(predictions[index]) == [0, 0, 0, 0, 1, 0]) |\
            (list(predictions[index]) == [0, 0, 0, 0, 0, 1]):
        if mat > 0:
            budget = fee * price * mat
            mat = 0
            sells.append(index)

    prices.append(price)
    asset.append(price * mat + budget)

plt.plot(prices, label='price')
plt.plot(asset, label='asset')
for xc in buys:
    plt.axvline(x=xc, color='green', linestyle='--', linewidth=0.5)
for xc in sells:
    plt.axvline(x=xc, color='red', linestyle='--', linewidth=0.5)

plt.legend()
plt.show()

"""
test part
"""
'''
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
'''