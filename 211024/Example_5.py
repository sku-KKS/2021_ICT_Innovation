import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('diabetes.csv')

# outlier 처리(0 값을 NaN으로, 임신은 0번도 가능해서 바꾸지 않음)
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)

# missing value 처리(NaN을 평균 값으로)
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

"""
for col in df.columns:
    missing_rows = df.loc[df[col] == 0].shape[0]
    print(col + ": " + str(missing_rows))
"""

# 표준화
from sklearn.preprocessing import StandardScaler

std_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

scaler = StandardScaler()

df_std = scaler.fit_transform(df[std_cols])

print(type(df_std))

df_std = pd.DataFrame(df_std, columns=std_cols)

df_std['Outcome'] = df['Outcome'].values

pd.set_option('display.max_columns', None)
print(df_std.describe())

df_scaled = df_std # 원본 DataFrame 보존
# print(df_scaled.describe())

# feature column, label column 추출 후 DataFrame 생성
feature_df = df_scaled[df_scaled.columns.difference(['Outcome'])]
label_df = df_scaled['Outcome']
print(feature_df.shape, label_df.shape)

# pandas <=> numpy 테스트
feature_np = feature_df.to_numpy()
label_np = label_df.to_numpy()
print(feature_np.shape, label_np.shape)

# train / test data 분리
split = 0.15

test_num = int(split*len(label_np))

x_test = feature_np[0:test_num]
y_test = label_np[0:test_num]

x_train = feature_np[test_num:]
y_train = label_np[test_num:]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(8, )))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy', metrics=['accuracy'])

from datetime import datetime

start_time = datetime.now()
hist = model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test))
end_time = datetime.now()
print('elapsed time => ', end_time-start_time)

model.evaluate(x_test, y_test)

plt.title('loss trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')

plt.show()

plt.title('accuracy trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')
plt.legend(loc='best')

plt.show()