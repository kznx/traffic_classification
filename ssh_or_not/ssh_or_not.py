
# coding: utf-8

from scipy.io import arff
import numpy as np
from sklearn import cross_validation

##  データセットを読み込む
dataset, meta = arff.loadarff("DARPA99Week3-0.arff")

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

## 配列の型をtensorflow/kerasが扱えるnumpyのarrayに変換
ds=np.asarray(dataset.tolist(), dtype=np.float32)
target=np.asarray(ds[:,22].tolist(), dtype=np.int8)
train=ds[:, :21]

## 正規化
mms = MinMaxScaler()
x_norm = mms.fit_transform(train)

## 訓練データとテストデータに分離
train_x, test_x, train_y, test_y = cross_validation.train_test_split(
    x_norm, target, test_size=0.2
)

# convert class vectors to 1-of-K format
y_train = np_utils.to_categorical(train_y, 2)
y_test = np_utils.to_categorical(test_y, 2)

print('x for train: ', train_x.shape)
print('x for test: ', test_x.shape)
print('y for train: ', y_train.shape)
print('y for test: ', y_test.shape)


# モデルの定義
model = Sequential()

# ネットワークの定義
model.add(Dense(input_dim=21, output_dim=20))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(input_dim=20, output_dim=20))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(input_dim=20, output_dim=10))
model.add(Activation('relu'))
model.add(Dropout(0.2))

## OUTPUT  SSH or NOTSSH
model.add(Dense(output_dim=2))
model.add(Activation('softmax'))

# ネットワークのコンパイル
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'sgd',
              metrics = ['accuracy'])

# 学習処理
hist = model.fit(train_x, y_train, nb_epoch = 10, batch_size = 100, verbose=1)

# 学習結果の評価
score = model.evaluate(test_x, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




