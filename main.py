import joblib
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Загрузка данных
data = pd.read_csv("all_v2.csv")


# Предобработка
def preprocess_data(df):
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

    # Создание новых признаков
    df['price_per_sqm'] = df['price'] / df['area']
    # df['price_per_sqm_group'] = df.groupby(['region', 'building_type', 'rooms'])['price'].transform('median') / df['area']
    df['room_size'] = df['area'] / (df['rooms'] + 1)  # +1 для студий
    df['floor_ratio'] = df['level'] / df['levels']

    return df


# Применяем предобработку
data = preprocess_data(data)

# Выбор признаков и метки
features = ['region', 'building_type', 'object_type', 'level', 'levels',
            'rooms', 'area', 'kitchen_area', 'price_per_sqm', 'floor_ratio']
X = data[features]
y = data['price']

# Разделение данных
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Определение числовых и категориальных признаков
num_cols = ['level', 'levels', 'rooms', 'area', 'kitchen_area', 'price_per_sqm', 'floor_ratio']
cat_cols = ['region', 'building_type', 'object_type']

# Предобработка
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])


# ----- Правила -----
# Чем меньше learning_rate, тем больше нужно n_estimators (и наоборот).
# n_estimators ≈ 50 / learning_rate
# num_leaves ≤ 2^max_depth


# Пайплайн
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=5,
        num_leaves=31,
        min_child_samples=25,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        objective='regression',
        metric='mae',
    ))
])

# Обучение модели
preprocessor.fit(X_train)
X_val_processed = preprocessor.transform(X_val)

# Затем обучаем модель с правильными параметрами
model.named_steps['model'].fit(
    preprocessor.transform(X_train), y_train,
    eval_set=[(X_val_processed, y_val)],
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(period=50)
    ],
)

# Предсказание и оценка
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f'Mean Absolute Error: {mae}')


# ------ Визуализация важности признаков ------
# feature_importances = model.named_steps['model'].feature_importances_
#
# Получаем имена признаков после OneHotEncoder
# onehot_columns = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps[
#     'onehot'].get_feature_names_out(cat_cols)
# all_features = num_cols + list(onehot_columns)
#
# plt.figure(figsize=(12, 8))
# plt.barh(all_features, feature_importances)
# plt.xlabel('Важность признака')
# plt.title('Важность признаков в модели')
# plt.tight_layout()
# plt.show()


# Сохраняем пайплайн
# joblib.dump(model, 'model/model.joblib', compress=3)