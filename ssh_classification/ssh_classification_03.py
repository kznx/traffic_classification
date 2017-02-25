
# coding: utf-8

# In[1]:

from scipy.io import arff
import numpy as np
from sklearn import cross_validation

##  データセットを読み込む
dataset, meta = arff.loadarff("NIMS2000.arff")
# dataset[0]


# In[2]:

from keras.models import Sequential
import keras.backend.tensorflow_backend as KTF
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

## 配列の型をtensorflow/kerasが扱えるnumpyのarrayに変換
ds=np.asarray(dataset.tolist(), dtype=np.float32)
target=np.asarray(ds[:,22].tolist(), dtype=np.int8)
train=ds[:, :22]

## 正規化
mms = MinMaxScaler()
x_norm = mms.fit_transform(train)

## 訓練データとテストデータに分離
train_x, test_x, train_y, test_y = cross_validation.train_test_split(
    x_norm, target, test_size=0.2
)

# convert class vectors to 1-of-K format
y_train = np_utils.to_categorical(train_y, 13)
y_test = np_utils.to_categorical(test_y, 13)

print('x for train: ', train_x.shape)
print('x for test: ', test_x.shape)
print('y for train: ', y_train.shape)
print('y for test: ', y_test.shape)


# In[3]:

# Backendの設定

# モデルの定義
model = Sequential()

# ネットワークの定義
model.add(Dense(input_dim=22, output_dim=40))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(input_dim=40, output_dim=30))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(input_dim=30, output_dim=30))
model.add(Activation('relu'))

model.add(Dense(input_dim=30, output_dim=20))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(input_dim=20, output_dim=10))
model.add(Activation('relu'))

model.add(Dense(input_dim=10, output_dim=20))
model.add(Activation('relu'))

## OUTPUT  SSH or NOTSSH
model.add(Dense(output_dim=13))
model.add(Activation('softmax'))

rms = RMSprop()
# ネットワークのコンパイル
model.compile(loss = 'categorical_crossentropy',
##              optimizer = 'sgd',
              optimizer = rms,
              metrics = ['accuracy'])

## コールバック
## es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
tb_cb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

# 学習処理
hist = model.fit(train_x, y_train, nb_epoch = 1000, batch_size = 200, verbose=1, 
                callbacks=[tb_cb])

# 学習結果の評価
score = model.evaluate(test_x, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




