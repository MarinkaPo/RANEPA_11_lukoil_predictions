# from functools import cache
import streamlit as st
import pandas as pd #Пандас
# import matplotlib
import matplotlib.pyplot as plt #Отрисовка графиков
import numpy as np #Numpy
#import pickle
from PIL import Image
#from tqdm import tqdm
#import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils #Для to_categorical
from tensorflow.keras.optimizers import Adam #Оптимизатор
from tensorflow.keras.models import Sequential, Model #Два варианты моделей
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, Conv2D, LSTM #Стандартные слои
from sklearn.preprocessing import StandardScaler, MinMaxScaler #Нормировщики
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # для генерации выборки временных рядов
# %matplotlib inline # Рисовать графики сразу же
# from lukoil_functions.py import *
# matplotlib.use('TkAgg')

st.title('Lukoil stock price prediction')
img = Image.open('Lukoil.jpg')
st.image(img, use_column_width='auto') #width=400

st.write("""
Приложение показывает, как работает нейронная сеть для предсказания цен акций.

Предсказвает цену на акции Лукойл, исходя из данных предыдущих периодов.

Данные подготовил Николай Лисин.
""")
#-------------------------О проекте-------------------------
expander_bar = st.expander("Перед тем, как начать:")
expander_bar.markdown(
    """
\n**Регрессия** - относится к классу задач обучения с учителем, когда по заданному набору признаков наблюдаемого объекта необходимо спрогнозировать некоторую целевую переменную.
Таким образом можно прогнозировать цену недвижимости, капитализацию компании или стоимость акций. 
\nВ этом приложении вы узнаете, как разрабатывать и оценивать модели нейронных сетей с использованием библиотеки глубокого обучения Keras для решения проблемы регрессии.
\n**Используемые библиотеки:** [tensorflow (keras)](https://keras.io/guides/sequential_model/), [streamlit](https://docs.streamlit.io/library/get-started), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html), [numpy](https://numpy.org/doc/stable/reference/index.html).
\n**Полезно почитать:** [Ссылка 1](https://www.machinelearningmastery.ru/regression-tutorial-keras-deep-learning-library-python), 
[Ссылка 2](https://habr.com/ru/company/vk/blog/513842/), [Ссылка 3](https://www.bizkit.ru/2019/11/05/14921/).

"""
)
#-------------------------Боковая панель-------------------------
# st.sidebar.header('Загрузить датафрейм - для варианта 1:')

# uploaded_file = st.sidebar.file_uploader("Выбрать CSV-файл", type=["csv"])
# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file, index_col=0)
# else:
#     st.sidebar.header('Установить фичи самостоятельно - для варианта 2:')
#     def user_input_features():
#         open_price = st.sidebar.slider('Цена открытия', 2041.0,5995.5,3000.9)
#         max_price = st.sidebar.slider('Максимальная цена', 2046.9,5996.0,3300.9)
#         min_price = st.sidebar.slider('Минимальная цена', 2040.1,5993.0,3400.9)
#         close_price = st.sidebar.slider('Цена закрытия', 2041.1,5996.0,3450.9)
#         v_volume = st.sidebar.slider('Объем продаж', 0,4296341,1845000)

#         data = {'open_price': open_price,
#                 'max_price': max_price,
#                 'min_price': min_price,
#                 'close_price': close_price,
#                 'v_volume': v_volume}
#         features = pd.DataFrame(data, index=[0])
#         return features
#     input_df = user_input_features()

# Соединяем наши фичи с имеющимся датасетом
# This will be useful for the encoding phase
# concat_data = pd.read_csv('concat_data.csv')
# concat_data.drop(['Unnamed: 0'], axis=1)
# # penguins = concat_data.drop(columns=['species'])
# # data = concat_data
# data = pd.concat([input_df,concat_data],axis=0, ignore_index=True)

input_df = pd.read_csv('concat_data.csv', index_col=0)
data = input_df
# data.head(8)
# -------------------------Смотрим датафрейм-------------------------
st.header('Блок 1: работа с данными')
st.dataframe(data.head(8)) # 
# st.write(sinput_df.head(10))
st.write("Весь размер таблицы: строк:", data.shape[0], "столбцов: ", data.shape[1])

expander_bar = st.expander('Информация о датасете')
expander_bar.warning('''Как мы видим, у нас есть данные о 481872 наблюдениях за ценами акций Лукоил. Данные распределены по 5ти колонкам:
\nOPEN - цена на открытие торгов, 
\nMAX - максимальная цена, 
\nMIN - минимальная цена, 
\nCLOSE - цена закрытия торгов, 
\nVOLUME - это количество “проторгованных” акций до текущего момента.
'''
)

data = np.array(input_df)

#-------------------------Визуализируем данные на графиках-------------------------
# vizualize = st.button('Визуализируем данные на графиках')
flag_viz = False
if st.button('Визуализируем данные на графиках'):
    flag_viz = True
else:
    pass
    # data = input_df
if flag_viz == True: 
    st.caption('''Перед вами 5 графиков, по каждой из колонок датафрейма:''')    
    #Отображаем исходные от точки start и длинной step
    start = 0            #С какой точки начинаем
    step = data.shape[0] #Сколько точек отрисуем
    #Заполняем текстовые названия каналов данных
    chanelNames = ['Open', 'Max', 'Min', 'Close', 'Volume']

    # st.line_chart(data[start:start+step,0], use_container_width=True) # ДОЛГО!!!

    # create a plot figure
    fig1 = plt.figure(figsize=(22,12), tight_layout=True)   #
    # create the first of two panels and set current axis 
    plt.subplot(2, 2, 1) # подграфик 1, деление 2х2, ячейка 1
    plt.plot(data[start:start+step,0], label = 'Цена открытия', color='red')
    plt.title('Цена открытия') # title of the subplot
    plt.legend()
    plt.ylabel('Цена.руб')
    
    plt.subplot(2, 2, 2) # подграфик 2
    plt.plot(data[start:start+step,1], label = 'Максимальная цена', )
    plt.title('Максимальная цена')
    plt.legend()
    plt.ylabel('Цена.руб')

    plt.subplot(2, 2, 3)  # подграфик 3
    plt.plot(data[start:start+step,2], label = 'Минимальная цена', c = 'green')
    plt.title('Минимальная цена')
    plt.legend()
    plt.ylabel('Цена.руб')

    plt.subplot(2, 2, 4) #  подграфик 4
    plt.plot(data[start:start+step,3], label = 'Цена закрытия', c = '#FFA500')
    plt.title('Цена закрытия')
    plt.ylabel('Цена.руб')
    plt.legend()
    # plt.show() # открывает в ОТДЕЛЬНОМ ОКНЕ
    # plt.tight_layout()
    st.pyplot(fig1)
     
    # #Канал volume
    fig2 = plt.figure(figsize=(22,12), tight_layout=True)
    plt.plot(data[start:start+step,4], label="Объем продаж")
    plt.title('Объем продаж')
    plt.ylabel('Цена.руб')
    plt.legend()
    st.pyplot(fig2)

#-------------------------Создаём генератор данных!!!-------------------------
# if data.shape[0] == 0:
#     pass
# else:
xLen = 300                      #Анализируем по 300 прошедшим точкам 
valLen = 30000                  #Используем 30.000 записей для проверки
#Формируем параметры загрузки данных

trainLen = data.shape[0]-valLen # Размер тренировочной выборки

#Делим данные на тренировочную и тестовую выборки 
xTrain,xTest = data[:trainLen], data[trainLen+xLen+2:]

#Масштабируем данные (отдельно для X и Y), чтобы их легче было скормить сетке
xScaler = MinMaxScaler()
if xTrain.shape[0]==0:
    pass
else: 
    xScaler.fit(xTrain)
    xTrain = xScaler.transform(xTrain)
    xTest = xScaler.transform(xTest)

    #Делаем reshape,т.к. у нас только один столбец по одному значению
    yTrain,yTest = np.reshape(data[:trainLen,3],(-1,1)), np.reshape(data[trainLen+xLen+2:,3],(-1,1)) 
    yScaler = MinMaxScaler()
    yScaler.fit(yTrain)
    yTrain = yScaler.transform(yTrain)
    yTest = yScaler.transform(yTest)

    #Создаем генератор для обучения
    trainDataGen = TimeseriesGenerator(xTrain, yTrain,           #В качестве параметров наши выборки
                                length=xLen, stride=1,        #Для каждой точки (из промежутка длины xLen)
                                batch_size=20)                #Размер batch, который будем скармливать модели

    #Создаем аналогичный генератор для валидации при обучении
    testDataGen = TimeseriesGenerator(xTest, yTest,
                                length=xLen, stride=1,
                                batch_size=20)

    #Создадим генератор проверочной выборки, из которой потом вытащим xVal, yVal для проверки                           
    DataGen = TimeseriesGenerator(xTest, yTest,
                                length=300, stride=1,
                                batch_size=len(xTest)) #размер batch будет равен длине нашей выборки
    xVal = []
    yVal = []
    for i in DataGen:
        xVal.append(i[0])
        yVal.append(i[1])

    xVal = np.array(xVal)
    yVal = np.array(yVal)

if st.button('Создаём генератор данных'):
    st.caption('''Сейчас мы: 
    \n1. разделили данные на тренировочную (Train) и тестовую (Test) выборки (обязательно БЕЗ shuffle, т.к. данные имеют временную упорядоченность), 
    \n2. отмасштабировали данные с помощью MinMaxScaler() [документация](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), 
    \n3. создали генератор для Train и для Test [документация](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/TimeseriesGenerator).''')
    st.write(f'''
    **Таким образом получились np.array:** 
    \n xTrain: 
    \n {xTrain}, \n размером {xTrain.shape}, 
    \n yTrain: 
    \n {yTrain}, \n размером {yTrain.shape}
    \n xTest: 
    \n {xTest}, \n размером {xTest.shape}
    \n yTest: 
    \n {yTest} \n размером {yTest.shape} 
    \n **...которые мы будем использовать для обучения модели (Train) и для проверки качества обучения (Test).**  
        ''')

#--------------------Функции для визуализации--------------------
# Функция рассчитываем результаты прогнозирования сети
# В аргументы принимает сеть (currModel) и проверочную выборку
# Выдаёт результаты предсказания predVal
# И правильные ответы в исходной размерности yValUnscaled (какими они были до нормирования)
def getPred(currModel, xVal, yVal, yScaler):
  # Предсказываем ответ сети по проверочной выборке
  # И возвращаем исходны масштаб данных, до нормализации
  predVal = yScaler.inverse_transform(currModel.predict(xVal))
  yValUnscaled = yScaler.inverse_transform(yVal)
  
  return (predVal, yValUnscaled)

# Функция визуализирует графики, что предсказала сеть и какие были правильные ответы
# start - точка с которой начинаем отрисовку графика
# step - длина графика, которую отрисовываем
# channel - какой канал отрисовываем
def showPredict(start, step, channel, predVal, yValUnscaled):
  fig4 = plt.figure(figsize=(22,12), tight_layout=True)
  plt.plot(predVal[start:start+step, 0],
           label='Прогноз')
  plt.plot(yValUnscaled[start:start+step, channel], 
           label='Базовый ряд')
  plt.xlabel('Время')
  plt.ylabel('Значение Close')
  plt.legend()
  st.pyplot(fig4) 
  
# Функция расёта корреляции дух одномерных векторов
def correlate(a, b):
  # Рассчитываем основные показатели
  ma = a.mean() # Среднее значение первого вектора
  mb = b.mean() # Среднее значение второго вектора
  mab = (a*b).mean() # Среднее значение произведения векторов
  sa = a.std() # Среднеквадратичное отклонение первого вектора
  sb = b.std() # Среднеквадратичное отклонение второго вектора
  
  #Рассчитываем корреляцию
  val = 1
  if ((sa>0) & (sb>0)):
    val = (mab-ma*mb)/(sa*sb)
  return val

# Функция рисуем корреляцию прогнозированного сигнала с правильным
# Смещая на различное количество шагов назад
# Для проверки появления эффекта автокорреляции
# channels - по каким каналам отображать корреляцию
# corrSteps - на какое количество шагов смещать сигнал назад для рассчёта корреляции
def showCorr(channels, corrSteps, predVal, yValUnscaled):
  # Проходим по всем каналам
  for ch in channels:
    corr = [] # Создаём пустой лист, в нём будут корреляции при смезении на i рагов обратно
    yLen = yValUnscaled.shape[0] # Запоминаем размер проверочной выборки

      # Постепенно увеличикаем шаг, насколько смещаем сигнал для проверки автокорреляции
    for i in range(corrSteps):
      # Получаем сигнал, смещённый на i шагов назад
      # predVal[i:, ch]
      # Сравниваем его с верными ответами, без смещения назад
      # yValUnscaled[:yLen-i,ch]
      # Рассчитываем их корреляцию и добавляем в лист
      corr.append(correlate(yValUnscaled[:yLen-i,ch], predVal[i:, 0]))

    own_corr = [] # Создаём пустой лист, в нём будут корреляции при смезении на i рагов обратно

      # Постепенно увеличикаем шаг, насколько смещаем сигнал для проверки автокорреляции
    for i in range(corrSteps):
      # Получаем сигнал, смещённый на i шагов назад
      # predVal[i:, ch]
      # Сравниваем его с верными ответами, без смещения назад
      # yValUnscaled[:yLen-i,ch]
      # Рассчитываем их корреляцию и добавляем в лист
      own_corr.append(correlate(yValUnscaled[:yLen-i,ch], yValUnscaled[i:, ch]))

    # Отображаем график коррелций для данного шага
    plt.plot(corr, label='Предсказание на ' + str(ch+1) + ' шаг')
    plt.plot(own_corr, label='Эталон')
  
  fig5 = plt.figure(figsize=(22,12), tight_layout=True)  
  plt.xlabel('Время')
  plt.ylabel('Значение')
  plt.legend()
  st.pyplot(fig5) 

#-------------------------Загрузить уже обученную модель-------------------------
st.header('Блок 2: загрузить уже обученную модель')
expander_bar = st.expander("Какую модель загружаем и почему?")
expander_bar.success('''
В связи с тем, что обучение моделей - долгий процесс, 
в этом блоке мы будем **не обучать новую модель**, а **загружать уже обученную модель** *model_20_ep.h5*. 
\nДля её обучения был взят исходный датасет и генератор данных из Блока 1.
\nОбучение длилось 20 эпох.
\nПолный процесс создания нейросети "с нуля" разобран в Блоке 3.
'''
)
model_upload = keras.models.load_model('model_20_ep.h5')


#-------------------------Выводим результаты-------------------------
if st.button('Выводим результаты загруженной модели'):
    for i in range(10):
        y1 = yScaler.inverse_transform(yVal[0][i].reshape(-1,1))
        y2 = yScaler.inverse_transform(model_upload.predict(xVal[0][i].reshape(1,300,5)))
        st.write('Реальное: ', y1[0][0],'     ', 'Предсказанное', y2[0][0])

#-------------------------Прогнозируем данные загруженной сетью-------------------------
if st.button('Прогноз загруженной моделью'):
    st.caption('''Перед вами график реальных цен акций и цен, предсказанных моделью:
    ''')
    currModel = model_upload #Выбираем загруженную модель
    (predVal, yValUnscaled) = getPred(currModel, xVal[0], yVal[0], yScaler) #Прогнозируем данные

    #Отображаем графики
    showPredict(0, 160, 0, predVal, yValUnscaled)

#-------------------------Создадим полносвязанную нейронную сеть-------------------------
st.header('Блок 3: создать нейронную сеть "с нуля"')
expander_bar = st.expander('Немного о создании модели')
expander_bar.info('''Для создания модели воспользуемся фреймворком **keras**.
\nМодель будем задавать через класс Sequential(), добавляя слои [документация](https://keras.io/api/models/sequential/). 
\nДалее модель компилируем, указывая loss-функцию (mse) и optimizer (Adam) [документация](https://keras.io/api/losses/#:~:text=categorical_hinge%20function-,Usage%20of%20losses%20with%20compile()%20%26%20fit(),-A%20loss%20function).
''')
modelD = Sequential()
modelD.add(Dense(150,input_shape = (xLen,5), activation="linear" )) # 5 - количество каналов
modelD.add(Flatten())
modelD.add(Dense(1, activation="linear"))
#Компилируем
modelD.compile(loss="mse", optimizer=Adam(lr=1e-4))
modelD.summary()

if st.button('Создадим полносвязанную нейронную сеть'):
  st.code('''
  # создаём сеть через класс Sequential:
  modelD = Sequential()
  modelD.add(Dense(150,input_shape = (xLen,5), activation="linear" )) # 5 - количество каналов
  modelD.add(Flatten())
  modelD.add(Dense(1, activation="linear"))
  # компилируем:
  modelD.compile(loss="mse", optimizer=Adam(lr=1e-4))
  ''')
  
  #tf.keras.utils.plot_model(modelD, to_file='modelD.png', show_shapes=True)
  st.image('modelD.png', caption='Архитектура нашей нейронной сети', 
          width=None, use_column_width=None, clamp=False, 
          channels="RGB", output_format="auto")

# if st.button('прогресс бар'):
#   my_bar = st.progress(0)
  # for percent_complete in range(100):
  #     time.sleep(1)
  #     my_bar.progress(percent_complete + 1)

#--------------------Запускаем обучение и визуализацию--------------------
#--------------------Непосредственно обучение
epchs = st.selectbox('Выберете количество эпох обучения:', (1,2,5,10,20))
if st.button('Запускаем обучение и прогноз'):
    with st.echo():
      history = modelD.fit(trainDataGen, 
                          epochs=int(epchs), 
                          verbose=1,
                          validation_data = testDataGen)
    # for epch in tqdm(range(0,epchs)):
    #   progress_bar = st.progress(0)
    #   for percent_complete in range(100):
    # time.sleep(0.1)
    # progress_bar.progress(percent_complete + 1)
    # progress_bar.progress(1.0)
    #Выводим графики обучения
    fig3 = plt.figure(figsize=(22,12), tight_layout=True)
    plt.plot(history.history['loss'], 
            label='Средняя абсолютная ошибка на обучающем наборе')
    plt.plot(history.history['val_loss'], 
            label='Средняя абсолютная ошибка на проверочном наборе')
    plt.ylabel('Средняя ошибка')
    plt.legend()
    st.pyplot(fig3) 


    #--------------------Выводим результаты обученной модели на Val:
    for i in range(10):
        y1 = yScaler.inverse_transform(yVal[0][i].reshape(-1,1))
        y2 = yScaler.inverse_transform(modelD.predict(xVal[0][i].reshape(1,300,5)))
        st.write('Реальное: ', y1[0][0],'     ', 'Предсказанное', y2[0][0])


    #if st.button('Прогноз обученной моделью'):
    currModel = modelD #Выбираем текущую модель
    (predVal, yValUnscaled) = getPred(currModel, xVal[0], yVal[0], yScaler) #Прогнозируем данные
    #Отображаем графики
    showPredict(0, 160, 0, predVal, yValUnscaled)




