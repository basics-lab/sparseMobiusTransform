from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
pd.options.mode.chained_assignment = None

class XGBoostModule:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        # construct subsets
        a = np.arange(2 ** 16, dtype=int)[np.newaxis, :]
        b = np.arange(16, dtype=int)[::-1, np.newaxis]
        self.subsets = np.array(a & 2 ** b > 0, dtype=int)

    def train_model(self, indices):
        accuracies = np.zeros(len(indices))
        for i, ind in enumerate(indices):
            subs = self.subsets[:,ind]

            # grab subset from features
            x_train_subs = self.x_train.loc[:, map(bool, subs)]
            x_test_subs = self.x_test.loc[:, map(bool, subs)]

            # train model on subset
            if np.sum(subs) == 0:
                accuracies[i] = 0.5
            else:
                clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                                        subsample=0.8, nthread=10, learning_rate=0.1, tree_method='hist',
                                        enable_categorical=True)
                clf.fit(x_train_subs, self.y_train)
                accuracies[i] = clf.score(x_test_subs, self.y_test)
        return accuracies

if __name__ == '__main__':
    print('Fetching dataset...')
    # fetch dataset
    bank_marketing = fetch_ucirepo(id=222)

    print('Preprocessing dataset...')
    # data (as pandas dataframes)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets.eq('yes').mul(1)

    # label categorical
    X['job'] = X['job'].astype("category")
    X['marital'] = X['marital'].astype("category")
    X['education'] = X['education'].astype("category")
    X['default'] = X['default'].astype("category")
    X['housing'] = X['housing'].astype("category")
    X['loan'] = X['loan'].astype("category")
    X['contact'] = X['contact'].astype("category")
    X['month'] = X['month'].astype("category")
    X['poutcome'] = X['poutcome'].astype("category")

    # scale numerical data
    min_max_scaler = MinMaxScaler()
    X[['age', 'balance', 'duration', 'campaign', 'day_of_week', 'pdays', 'previous']] = min_max_scaler.fit_transform(
        X[['age', 'balance', 'duration', 'campaign', 'day_of_week', 'pdays', 'previous']])

    # split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # create XGBoost Module
    XGBm = XGBoostModule(x_train, x_test, y_train, y_test)


    batch_size = 200

    pool = Pool()

    # train model for each subset
    accuracies = np.zeros(2 ** 16)

    query_batches = np.array_split(range(2**16), 2**16 // batch_size)
    counter = 0
    print('Running training of XGBoost for all models...')
    pbar = tqdm(desc='Model Training Batches', total=len(query_batches))
    for new_res in pool.imap(XGBm.train_model, query_batches):
        accuracies[counter: counter + len(new_res)] = new_res
        counter += len(new_res)
        pbar.update(1)

    # save results
    print(accuracies)
    with open('feature_selection_accs_two.npy', 'wb') as f:
        np.save(f, accuracies)

    pool.close()
