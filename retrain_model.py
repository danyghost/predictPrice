import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Загрузка данных
data = pd.read_csv("all_v2.csv")

def preprocess_data(df):
    df = df.copy()

    # Обработка пропущенных значений
    df['building_type'] = df['building_type'].fillna(-1)
    df['object_type'] = df['object_type'].fillna(-1)
    df['rooms'] = df['rooms'].fillna(-1)
    df['area'] = df['area'].fillna(df['area'].median())
    df['kitchen_area'] = df['kitchen_area'].fillna(df['kitchen_area'].median())
    df['level'] = df['level'].fillna(df['level'].median())
    df['levels'] = df['levels'].fillna(df['levels'].median())

    # Обработка выбросов
    df = df[df['area'] <= 200]
    df = df[df['price'] <= 20000000]
    df = df[df['price'] >= 1000000]
    df = df[df['level'] <= df['levels']]

    # Защита от деления на ноль
    df['levels'] = df['levels'].replace(0, 1)

    # Создание новых признаков
    df['room_size'] = df['area'] / df['rooms'].replace(-1, 0.5).clip(lower=0.5)
    df['floor_ratio'] = df['level'] / df['levels']
    df['kitchen_ratio'] = df['kitchen_area'] / df['area']

    # Убираем бесконечные значения
    for col in ['room_size', 'floor_ratio', 'kitchen_ratio']:
        df[col] = df[col].replace([np.inf, -np.inf], df[col].median())

    return df


def train_model():
    """Обучение модели с использованием и региона, и города"""
    # Загрузка и предобработка данных
    data_clean = preprocess_data(data)

    # Определение признаков - используем и город, и регион!
    features = ['region', 'city', 'building_type', 'object_type', 'level', 'levels',
                'rooms', 'area', 'kitchen_area', 'room_size', 'floor_ratio', 'kitchen_ratio']
    X = data_clean[features]
    y = data_clean['price']  # общая цена

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Определение типов признаков
    num_cols = ['level', 'levels', 'area', 'kitchen_area', 'room_size', 'floor_ratio', 'kitchen_ratio']
    cat_cols = ['region', 'city', 'building_type', 'object_type', 'rooms']

    # Пайплайн
    model = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scaler', StandardScaler())]), num_cols),
            ('cat', SimpleImputer(strategy='most_frequent'), cat_cols)
        ])),
        ('model', LGBMRegressor(
            random_state=0,
            verbose=-1,
            objective='regression',
            metric='mae',
            n_estimators=1000,
            learning_rate=0.05
        ))
    ])

    # Обучение
    print("Обучение модели с использованием региона и города...")
    model.fit(X_train, y_train)

    # Оценка
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    print(f'MAE: {mae:,.0f} руб.')
    print(f'MAPE: {mape:.2f}%')
    print(f'Средняя цена: {y_test.mean():,.0f} руб.')

    # Сохранение
    joblib.dump(model, 'model_with_city.joblib')
    print("Модель сохранена как 'model_with_city.joblib'")

    return model


if __name__ == "__main__":
    train_model()