import joblib
import pandas as pd
from typing import Dict, Any
from parser import get_cian_analogs
import cianparser

# Загружаем модель один раз при импорте модуля
pipeline = joblib.load('model/model.joblib')

# Загружаем mapping: название региона -> код
try:
    region_mapping_df = pd.read_csv('region_mapping.csv')
    REGION_NAME_TO_CODE = dict(zip(region_mapping_df['region_name'], region_mapping_df['region_code']))
except Exception:
    REGION_NAME_TO_CODE = {}

REQUIRED_FEATURES = ['region', 'building_type', 'object_type', 'level', 'levels', 'rooms', 'area', 'kitchen_area', 'price_per_sqm', 'floor_ratio']

def predict_price(input_data: Dict[str, Any]) -> float:
    """
    Принимает словарь параметров, возвращает прогноз цены.
    Обрабатывает отсутствующие значения и преобразует типы.
    """
    # Значения по умолчанию для отсутствующих полей
    defaults = {
        'price_per_sqm': input_data.get('area', 50) * 100000,
        'floor_ratio': input_data.get('level', 1) / input_data.get('levels', 5),
        'building_type': input_data.get('building_type', 'кирпичный'),
        'object_type': input_data.get('object_type', 'вторичка'),
        'kitchen_area': input_data.get('kitchen_area', input_data.get('area', 50) * 0.2)
    }
    # Исключаем deal_type из признаков для модели
    data = {**defaults, **{k: v for k, v in input_data.items() if k != 'deal_type'}}
    # Оставляем только нужные признаки
    features = {k: data.get(k) for k in REQUIRED_FEATURES}
    df = pd.DataFrame([features])
    price = pipeline.predict(df)[0]
    return float(price)


def predict_price_with_analogs(input_data):
    # Строгая валидация обязательных параметров
    required_fields = ['region_name', 'rooms', 'area']
    for field in required_fields:
        if field not in input_data or input_data[field] in (None, ''):
            raise ValueError(f"Не указан обязательный параметр: {field}")
    # deal_type обязателен только для поиска аналогов
    if 'deal_type' not in input_data or input_data['deal_type'] in (None, ''):
        raise ValueError("Не указан обязательный параметр: deal_type")
    # Получаем внутренний код региона для ML
    region_name = input_data['region_name']
    region_code = REGION_NAME_TO_CODE.get(region_name)
    if not region_code:
        raise ValueError(f"Регион '{region_name}' не поддерживается. Выберите другой регион.")
    input_data['region'] = region_code  # для ML
    # Проверка региона на валидность для cianparser
    valid_locations = cianparser.list_locations()
    if region_name not in valid_locations:
        raise ValueError(f"Регион '{region_name}' не поддерживается парсером. Выберите другой регион.")

    price_ml = predict_price(input_data)
    analogs = get_cian_analogs(
        location=region_name,
        deal_type=input_data['deal_type'],
        rooms=int(input_data['rooms']),
        area=float(input_data['area']),
        start_page=1, end_page=1
    )
    if analogs:
        prices = [float(flat['price']) for flat in analogs if flat.get('price') and str(flat['price']).replace('.','',1).isdigit()]
        if prices:
            price_cian = sum(prices) / len(prices)
            price_final = (price_ml + price_cian) / 2
        else:
            price_final = price_ml
    else:
        price_final = price_ml
    return price_final, price_ml, analogs

if __name__ == "__main__":
    print("Тест функции")
    # Примеры для теста
    example = {
        'region': 'Москва',
        'building_type': 'панельный',
        'object_type': 'вторичка',
        'level': 5,
        'levels': 12,
        'rooms': 2,
        'area': 55,
        'kitchen_area': 10,
        'deal_type': 'sale'
    }
    print(f"Прогноз: {predict_price(example):,.2f} руб.") 