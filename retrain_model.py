import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
import warnings

warnings.filterwarnings('ignore')


# Загрузка данных
def load_data():
    data = pd.read_csv("real_estate.csv")
    print(f"Загружено данных: {len(data)}")
    print(f"Колонки: {data.columns.tolist()}")
    return data


def preprocess_data(df):
    df = df.copy()
    print("Начальная предобработка")

    # Обработка пропущенных значений
    df['building_type'] = df['building_type'].fillna(-1)
    df['object_type'] = df['object_type'].fillna(-1)
    df['rooms'] = df['rooms'].fillna(-1)
    df['area'] = df['area'].fillna(df['area'].median())
    df['kitchen_area'] = df['kitchen_area'].fillna(df['kitchen_area'].median())
    df['level'] = df['level'].fillna(df['level'].median())
    df['levels'] = df['levels'].fillna(df['levels'].median())

    # Обработка выбросов
    initial_size = len(df)
    df = df[df['area'] <= 200]
    df = df[df['price'] <= 20000000]
    df = df[df['price'] >= 1000000]
    df = df[df['level'] <= df['levels']]
    print(f"Удалено выбросов: {initial_size - len(df)}")

    # Защита от деления на ноль
    df['levels'] = df['levels'].replace(0, 1)

    # Создание новых признаков
    df['room_size'] = df['area'] / df['rooms'].replace(-1, 0.5).clip(lower=0.5)
    df['floor_ratio'] = df['level'] / df['levels']

    # Убираем бесконечные значения
    for col in ['room_size', 'floor_ratio']:
        df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
        df[col] = df[col].fillna(df[col].median())

    # Оптимизация типов данных для экономии памяти
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    print(f"После предобработки: {len(df)} записей")
    return df


def train_model_optimized():
    """Оптимизированная версия с экономией памяти"""
    data = load_data()
    data_clean = preprocess_data(data)

    features = ['region_code', 'building_type', 'object_type', 'level', 'levels',
                'rooms', 'area', 'kitchen_area', 'room_size', 'floor_ratio']

    X = data_clean[features]
    y = data_clean['price']

    print(f"Размерность X: {X.shape}")
    print(f"Уникальные значения категориальных признаков:")
    for col in ['region_code', 'building_type', 'object_type', 'rooms']:
        print(f"  {col}: {X[col].nunique()} уникальных значений")

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Применяем TargetEncoder к категориальным признакам ДО пайплайна
    cat_cols = ['region_code', 'building_type', 'object_type', 'rooms']

    print("Применяем TargetEncoder...")

    # Создаем отдельные энкодеры для каждой колонки
    encoders = {}
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    for col in cat_cols:
        print(f"  Кодируем {col}...")
        encoder = TargetEncoder()

        # Кодируем тренировочные данные
        X_train_encoded[col] = encoder.fit_transform(
            X_train[[col]], y_train
        )
        # Кодируем тестовые данные
        X_test_encoded[col] = encoder.transform(
            X_test[[col]]
        )
        encoders[col] = encoder

    # Теперь все признаки числовые
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    model = Pipeline(steps=[
        ('preprocessor', numeric_transformer),
        ('model', LGBMRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='regression',
            metric='mae',
            verbose=-1,
            n_jobs=-1
        ))
    ])

    print("Обучение модели")
    model.fit(X_train_encoded, y_train)

    # Оценка
    preds = model.predict(X_test_encoded)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    print(f'MAE: {mae:,.0f} руб.')
    print(f'MAPE: {mape:.2f}%')
    print(f'Средняя цена: {y_test.mean():,.0f} руб.')
    print(f'Отношение MAE к средней цене: {mae / y_test.mean():.3f}')

    # Сохранение модели и энкодеров
    joblib.dump({'model': model, 'encoders': encoders}, 'model/sale_model.joblib')
    print("Модель и энкодеры сохранены")

    return model, encoders


if __name__ == "__main__":
    try:
        model_optimized, encoders = train_model_optimized()
    except Exception as e:
        print(f"Ошибка в версии с category_encoders: {e}")
        import traceback


        traceback.print_exc()
