import streamlit as st
import pandas as pd #Пандас
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt #Отрисовка графиков
from tensorflow.keras import utils #Для to_categorical
import numpy as np #Numpy
from tensorflow.keras.optimizers import Adam #Оптимизатор
from tensorflow.keras.models import Sequential, Model #Два варианты моделей
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, Conv2D, LSTM #Стандартные слои
from sklearn.preprocessing import StandardScaler, MinMaxScaler #Нормировщики
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # для генерации выборки временных рядов
# %matplotlib inline # Рисовать графики сразу же
import pickle
from PIL import Image

st.title('Lukoil stock price prediction')
img = Image.open('Lukoil.jpg')
st.image(img, use_column_width='auto') #width=400

st.write("""
Приложение показывает, как работает нейронная сеть с временными рядами.

Предсказвает цену на акции Лукойл, исходя из данных предыдущих периодов.

Данные подготовил Николай Лисин.
""")
#-------------------------О проекте-------------------------
expander_bar = st.expander("Перед тем, как начать - немного теории:")
expander_bar.markdown(
    """
**Временной ряд** — это упорядоченная последовательность значений какого-либо показателя за несколько периодов времени. 
Основная характеристика, которая отличает временной ряд от простой выборки данных, — указанное время измерения или номер изменения по порядку.
**Временные ряды** используются для аналитики и прогнозирования, когда важно определить, что будет происходить с показателями в ближайший час/день/месяц/год: 
например, сколько пользователей скачают за день мобильное приложение. Показатели для составления временных рядов могут быть не только техническими, 
но и экономическими, социальными и даже природными. 

**Используемые библиотеки:** tensorflow (keras), sklearn, streamlit, pandas, matplotlib, numpy.

**Полезно почитать:** [Общее](http://www.machinelearning.ru/wiki/index.php?title=%D0%92%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D0%BE%D0%B9_%D1%80%D1%8F%D0%B4), 
[Хабр](https://habr.com/ru/post/553658/).

"""
)

st.sidebar.header('Выбор действия:')

# df = pd.read_csv("concat_data.csv")
# @st.cache
# def convert_df(df):
#    return df.to_csv().encode('utf-8')
# csv = convert_df(df)

# ДЛЯ СОХРАНЕНИЯ!!!
# st.download_button(label ='Нажмите, чтобы сохранить файл',
#    data = csv,
#    file_name='concat_data.csv',
#    mime='text/csv'
#    key='download-csv'
# )

# ДЛЯ ЗАГРУЗКИ С РЕСУРСА!!!
# st.sidebar.markdown("""
# [Загрузить данные Лукойла](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
# """)

# -------------------------Собираем датафрейм-------------------------
uploaded_file = st.sidebar.file_uploader("Загрузить CSV-файл", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file, index_col=0)
else:
    def user_input_features():
        open_price = st.sidebar.slider('Цена открытия', 2041.0,5995.5,3000.9)
        max_price = st.sidebar.slider('Максимальная цена', 2046.9,5996.0,3300.9)
        min_price = st.sidebar.slider('Минимальная цена', 2040.1,5993.0,3400.9)
        close_price = st.sidebar.slider('Цена закрытия', 2041.1,5996.0,3450.9)
        v_volume = st.sidebar.slider('Объем продаж', 0,4296341,1845000)

        data = {'open_price': open_price,
                'max_price': max_price,
                'min_price': min_price,
                'close_price': close_price,
                'v_volume': v_volume}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Соединяем наши фичи с имеющимся датасетом
# This will be useful for the encoding phase
# concat_data = pd.read_csv('concat_data.csv')
# concat_data.drop(['Unnamed: 0'], axis=1)
# # penguins = concat_data.drop(columns=['species'])
# # data = concat_data
# data = pd.concat([input_df,concat_data],axis=0, ignore_index=True)

data = input_df
# -------------------------Смотрим датафрейм-------------------------
if st.button('Посмотрим загруженный dataframe:'):
    st.dataframe(data.head(10)) 
    # st.write(input_df.head(10))
    st.write("Весь размер таблицы:", data.shape[0], "строк, ", data.shape[1], "столбцов." )

data = np.array(input_df)

#-------------------------Визуализируем!!!-------------------------
# vizualize = st.button('Визуализируем данные на графиках')
if st.button('Визуализируем данные на графиках'):
    # data = input_df
    
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
if data.shape[0] == 0:
    pass
else:
    xLen = 300                      #Анализируем по 300 прошедшим точкам 
    valLen = 30000                  #Используем 30.000 записей для проверки
    #Формируем параметры загрузки данных

    trainLen = data.shape[0]-valLen # Размер тренировочной выборки

    #Делим данные на тренировочную и тестовую выборки 
    xTrain,xTest = data[:trainLen], data[trainLen+xLen+2:]

    #Масштабируем данные (отдельно для X и Y), чтобы их легче было скормить сетке
    xScaler = MinMaxScaler()
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
    st.write(f'''
    Таким образом получились np.array: 
    \n xTrain: 
    \n {xTrain}, \n размером {xTrain.shape}, 
    \n yTrain: 
    \n {yTrain}, \n размером {yTrain.shape}
    \n xTest: 
    \n {xTest}, \n размером {xTest.shape}
    \n yTest: 
    \n {yTest} \n размером {yTest.shape}   
        ''')

#-------------------------Создадим полносвязанную нейронную сеть-------------------------
#Создаём нейронку
modelD = Sequential()
modelD.add(Dense(150,input_shape = (xLen,5), activation="linear" )) # 5 - количество каналов
modelD.add(Flatten())
modelD.add(Dense(1, activation="linear"))
#Компилируем
modelD.compile(loss="mse", optimizer=Adam(lr=1e-4))
modelD.summary()

if st.button('Создадим полносвязанную нейронную сеть'):
    st.write('!!! Необходимо вставить сюда modelD.summary() !!!')

    
#-------------------------Запускаем обучение-------------------------
epchs = st.selectbox('Выберете количество эпох обучения:', (1,2,5,10,20))
if st.button('Запускаем обучение'):
    st.write('!!! Необходимо вставить сюда визуализацию эпох !!!')
    history = modelD.fit(
                        trainDataGen, 
                        epochs=int(epchs), 
                        verbose=1,
                        validation_data = testDataGen 
                        )

    #Выводим графики
    fig3 = plt.figure(figsize=(22,12), tight_layout=True)
    plt.plot(history.history['loss'], 
            label='Средняя абсолютная ошибка на обучающем наборе')
    plt.plot(history.history['val_loss'], 
            label='Средняя абсолютная ошибка на проверочном наборе')
    plt.ylabel('Средняя ошибка')
    plt.legend()
    st.pyplot(fig3) 

    #Выводим результаты
    for i in range(10):
        y1 = yScaler.inverse_transform(yVal[0][i].reshape(-1,1))
        y2 = yScaler.inverse_transform(modelD.predict(xVal[0][i].reshape(1,300,5)))
        st.write('Реальное: ', y1[0][0],'     ', 'Предсказанное', y2[0][0])


#-------------------------Визуализация результатов-------------------------   
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



#-------------------------Прогнозируем данные текущей сетью-------------------------
if st.button('Прогнозируем данные'):
    currModel = modelD #Выбираем текущую модель
    (predVal, yValUnscaled) = getPred(currModel, xVal[0], yVal[0], yScaler) #Прогнозируем данные

    #Отображаем графики
    showPredict(0, 160, 0, predVal, yValUnscaled)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# encode = ['sex','island']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
# df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
# st.subheader('Выбранные характеристики пингвина:')

# if uploaded_file is not None:
#     st.write(df)
# else:
#     st.write('Подождите, пока CSV-файл загрузится. Текущие выбранные параметры представлены ниже.')
#     st.write(df)

# # Reads in saved classification model
# load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# # Apply model to make predictions
# prediction = load_clf.predict(df)
# prediction_proba = load_clf.predict_proba(df)


# st.subheader('Предсказание вида пингвина:')
# penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
# st.write(penguins_species[prediction])

# st.subheader('Вероятность предсказания по каждому виду:')
# st.write(prediction_proba)
