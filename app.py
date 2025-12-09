import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression

model = LinearRegression()

df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')


st.write('1. Показать основные информативные графики/гистограммы в рамках EDA\n')


#График №1
st.write("Попарные распределения числовых признаков для `test`")
numeric_cols = ['year', 'selling_price', 'km_driven', 'engine', 'seats', 'max_power', 'mileage']
pairplot = sns.pairplot(df_train[numeric_cols],
                        diag_kind='hist',
                        plot_kws={'alpha': 0.6, 's': 20},
                        )
plt.tight_layout()
plt.show()
fig = plt.gcf()
st.pyplot(fig)

st.write("Попарные распределения числовых признаков для `train`")
pairplot = sns.pairplot(df_test[numeric_cols],
                        diag_kind='hist',
                        plot_kws={'alpha': 0.6, 's': 20},
                        )
plt.tight_layout()
fig = plt.gcf()
st.pyplot(fig)

#График №2
st.write("Тепловая карта корреляции")
corr_matrix = df_train.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True)
plt.title('Корреляции признаков')
plt.tight_layout()
fig = plt.gcf()
st.pyplot(fig)

#График №2
st.write("Зависимость цены от возраста автомобиля")
df_train_age = df_train.copy()
df_train_age['age'] = 2020 - df_train_age['year']
plt.subplots(figsize=(12, 6))
sns.scatterplot(x='age', y='selling_price', data=df_train_age, alpha=0.5)
plt.title('Зависимость цены от возраста автомобиля')
plt.xlabel('Возраст автомобиля')
plt.ylabel('Цена')
plt.tight_layout()
fig = plt.gcf()
st.pyplot(fig)

#График №4
st.write("Зависимость стоимости от года продажи")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_train, x='year', y='selling_price')
plt.title('Зависимость стоимости от года продажи')
plt.xlabel('Год выпуска')
plt.ylabel('Цена')
fig = plt.gcf()
st.pyplot(fig)

#-----------------------------------------------------------------------------------------------------------
st.write("2. На вход запрашивать csv-файл с признаками объектов или запрашивать признаки объекта в окошках для ввода, и применять на поступивших объектах модель")

def load_model():
    try:
        with open('to_gitHub/lasso_model.pkl', 'rb') as f:
            model_pickle = pickle.load(f)
        return model_pickle
    except FileNotFoundError:
        st.error("Файл не найден")
        return None

model = load_model()
input_method = st.radio("Есть два пути, выбирай:", ["Csv-файл с признаками объектов будешь загружать", "В окошки ввода будешь вводить"])

if input_method == "Csv-файл с признаками объектов будешь загружать":
    uploaded = st.file_uploader("Сюда клади CSV файл", type=['csv'])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Данные:")
        st.write(df.head())

        required = model.feature_names
        X = df[required].values
        predictions = model.predict(X)

        df['predicted'] = predictions
        st.write("Результаты:")
        st.write(df[['predicted'] + required])

else:
    st.write("Введите признаки в окошки:")

    year = st.number_input("Год выпуска автомобиля", min_value=1990, max_value=2020, value=2000)
    km_driven = st.number_input("Пробег автомобиля", min_value=0, value=50000)
    mileage = st.number_input("Расход топлива автомобиля", min_value=0.0, value=15.0)
    engine = st.number_input("Объем двигателя", min_value=0.0, value=1.5)
    max_power = st.number_input("Мощность двигателя", min_value=0.0, value=200.0)
    seats = st.number_input("Количество мест", min_value=2, max_value=10, value=5)

    if st.button("Предсказание:"):
        features = np.array([[year, km_driven, mileage, engine, max_power, seats]])
        price = model.predict(features)[0]
        st.success(f"Предсказанная цена: {price}")
        st.write("Влияние признаков:")

    feature_names = ['year', 'km_driven', 'engine', 'max_power', 'mileage', 'seats']
    df_weight = pd.DataFrame({
        'feature': feature_names,
        'weight': model.coef_
    })

    st.write("3. Визуализация весов обученной модели")
    plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='feature', y='weight', data=df_weight, alpha=0.5)
    plt.title('Веса обученной модели')
    plt.xlabel('Признак')
    plt.ylabel('Вес')
    plt.tight_layout()
    fig = plt.gcf()
    st.pyplot(fig)



