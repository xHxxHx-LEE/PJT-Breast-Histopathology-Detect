<h1>Breast Histopathology Detect

## 목차

**[1. 프로젝트 소개](#1-프로젝트-소개)**

**[2. 코드 리뷰](#2-코드-리뷰)**

**[3. 시연](#3-시연)**

**[4. 느낀 점](#4-느낀-점)**

---

## 1. 프로젝트 소개



> **개요**: 유방암은 전 세계 여성 암의 25.2%를 차지할 만큼 발생 빈도가 높은 질병이다.(출처: 2018 유방암 백서) 미국 등 여러 선진국에서 자주 발생하는 선진국형 질병이나,  최근 서구화된 식습관과 고령화로 인해 국내 유방암 환자도 크게 증가하고 있다. 한국유방암학회의 자료에 따르면 여성 인구 10만명당 전체 유방암 환자 수는 2000년 26.3명에서 2015년 88.1명으로 지속적으로 증가했다. 특히 우리나라는 폐경 전 유방암 환자 비중이 높은 편에 속한다. 또한 통계청 자료에 따르면 유방암으로 인한 사망률은 2010년 여성 인구 10만명 당 4.8명에서 2015년 약 2배 늘어난 9.2명을 기록했다. 전조증상이 거의 없다는 점, 폐 전이와 뼈 전이가 흔히 나타난다는 점, 치료 과정에서 유방 절제가 필요하다는 점과 재발률이 높다는 점 등의 이유 때문에 많은 여성이 두려워 하는 질병이다. 유방암 중 침윤성 유관암은 전체 유방암의 약 75-85%를 차지하는 대표적인 유방암이다. 따라서 삼성 멀티캠퍼스에서 배웠던 이미지 분류를 통해 프로젝트를 수행하고자 한다.

> **기간**: 2020.03.17-2020.03.23

> **구성원 및 역할**: 이찬호(팀장, DL), 이정철(Back End(Flask)), 정소현(DL), 정용주(Front End), 황지민(DL, Back End(Flask))

> **스택**: Python

> **툴**: Jupyter Notebook

> **모델 예측 성능**: 85% (Test 데이터 예측시)

---



## 2. 코드 리뷰

1)  **Import 부분**

```python
# import 부분
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="1"

tf.debugging.set_log_device_placement(True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
```

---

2) **이미지 불러오기 및 전처리**

이미지 갯수가 달라 이미지 갯수를 맞추었고 train(80%) test(20%) validation(20%) 3개로 분류

```python
folders = [folder + x for x in sorted(os.listdir(folder))]
folders_0 = list(pd.Series(folders)+'/0/')
folders_1 = list(pd.Series(folders)+'/1/')

images_path = []
images_0=[]
images_1=[]

for i in range(0,len(folders_0)):
    images_0.extend([folders_0[i] + x for x in sorted(os.listdir(folders_0[i]))])
    images_1.extend([folders_1[i] + x for x in sorted(os.listdir(folders_1[i]))])
    
print(len(images_0)) # 198738
print(len(images_1)) # 78786

images_path = np.hstack([images_1,np.random.choice(images_0, len(images_1))])

id_num = []
path = []
name = []
result = []

for i in images_path:
    split=i.split("/")
    id_num.append(split[-3])
    path.append(i)
    name.append(split[-1])
    result.append(split[-2])
    
data = pd.DataFrame({"id_num":id_num, "path":path, "name":name, "result":result})

data["result"].value_counts()
# 0    78786
# 1    78786

X=data.iloc[:, :3]
y=data.iloc[:, 3].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=5)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state=5)

train_data = X_train
train_data['label'] = y_train

test_data = X_test
test_data['label'] = y_test

val_data = X_val
val_data['label'] = y_val

IM_WIDTH = 50
IM_HEIGHT = 50

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=180, width_shift_range=0.2,
                                   height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)

train_generator = train_datagen.flow_from_dataframe(train_data, x_col="path", y_col="label", shuffle=True,
                                                  batch_size=50, class_mode="binary", target_size=(IM_WIDTH, IM_HEIGHT))

train_generator.n # 141814

test_datagen = ImageDataGenerator(rescale=1./255, rotation_range=180, width_shift_range=0.2,
                                height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)

test_generator = test_datagen.flow_from_dataframe(test_data, x_col="path", y_col="label", shuffle=True,
                                                  batch_size=50, class_mode="binary", target_size=(IM_WIDTH, IM_HEIGHT))

test_generator.n # 15758

val_datagen=ImageDataGenerator(rescale=1./255, rotation_range=180, width_shift_range=0.2,
                                height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)

val_generator=val_datagen.flow_from_dataframe(val_data, x_col="path", y_col="label", shuffle=True,
                                                  batch_size=50, class_mode="binary", target_size=(IM_WIDTH, IM_HEIGHT))

val_generator.n # 15758

img, label = train_generator.next()

label = label.reshape(-1,1)

enc = OneHotEncoder()
enc.fit(label)

label_onehot = enc.transform(label).toarray()
label_onehot = np.array(label_onehot, dtype="float32")
```

---

3) **모델 학습**

Keras에서 지원하는 MobileNetV2를 이용하여 학습

```python
mobV2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IM_WIDTH, IM_HEIGHT, 3))

mobV2.summary()

model = Sequential()
model.add(mobV2)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00001), metrics=['acc'])

model_path ="./model/somacV4-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=7)

history = model.fit_generator(train_generator, epochs=30, validation_data = val_generator, callbacks=[checkpoint, early_stopping])

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])
model_path ="./model/somacV5-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=7)
```

---

4) **모델 히스토리**

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'], loc='upper left')
plt.show()
```

![모델 히스토리](C:\Users\cksgh\Desktop\다운로드.png)

---

5) **테스트 데이터 예측**

```python
model.evaluate(test_generator)
```

---



## 3. 시연

1) **메인 페이지**

파란색 글씨를 클릭하면 이미지 업로드 창으로 이동

![메인 페이지](C:\Users\cksgh\AppData\Roaming\Typora\typora-user-images\image-20210203141348144.png)

---



2) **이미지 업로드 페이지**

분홍색 사각형을 클릭하면 파일 업로드 가능

![이미지 업로드](C:\Users\cksgh\AppData\Roaming\Typora\typora-user-images\image-20210203141520525.png)

---



3) **결과 페이지**

이미지를 업로드 하면 서버에 내장된 모델로 예측하여 양성일 확률을 도출해내어 사용자에게 알려주는 페이지

![결과 페이지](C:\Users\cksgh\AppData\Roaming\Typora\typora-user-images\image-20210203141714966.png)

---



## 4. 느낀 점

유방암 데이터가 캐글에서 제공하는 이미지 데이터이고 프로젝트 기간이 매우 짧아 빠르지만 성능이 좋은 Keras 이미지 어플리케이션 중 MobilenetV2 를 이용해 이미지를 이진 분류로 모델을 만들게 되었다. 프로젝트를 진행하면서 세상의 모든 데이터는 캐글에서 제공하는 데이터 만큼 깔끔하지 않고 가공이 되어있지 않은 데이터라는 것을 인지하게 되었으며 그에 따라 전처리 잘하는 것도 실력이라고 느끼게 되었다. 세미 프로젝트 주제인 유방암에 대해 무지했지만 이번 프로젝트를 진행하면서 유방암의 종류에 대해 알게 되었으며 유방암이 여성만 걸리는 것이 아니라 남성도 걸릴 수 있다는 것에 대해 알게 되었다.

시간 관계상 Keras의 모델을 사용하였지만 시간이 충분히 있었으면 모델을 직접 만들어 봤으면 하는 아쉬움이 있다.