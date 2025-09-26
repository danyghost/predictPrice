import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from parser import get_cian_analogs
import cianparser

# Загружаем модель
try:
    model_data = joblib.load('model/model_optimized.joblib')
    model = model_data['model']  # Извлекаем саму модель
    encoders = model_data['encoders']  # Извлекаем энкодеры
    print("Модель и энкодеры успешно загружены")
except FileNotFoundError:
    print("Модель не найдена")

# Загружаем mapping: название региона -> код
try:
    region_mapping_df = pd.read_csv('region_mapping.csv')
    REGION_NAME_TO_CODE = dict(zip(region_mapping_df['region_name'], region_mapping_df['region_code']))
except Exception as e:
    print(f"Ошибка загрузки region_mapping.csv: {e}")
    REGION_NAME_TO_CODE = {}

REQUIRED_FEATURES = ['region_code', 'building_type', 'object_type', 'level', 'levels',
                     'rooms', 'area', 'kitchen_area', 'room_size', 'floor_ratio']


def filter_outliers(prices: List[float]) -> List[float]:

    # Удаляет выбросы из цен аналогов используя межквартильный размах (IQR)
    if len(prices) < 3:
        return prices

    prices_array = np.array(prices)
    Q1 = np.percentile(prices_array, 25)
    Q3 = np.percentile(prices_array, 75)
    IQR = Q3 - Q1

    # Фильтруем выбросы (outside 1.5*IQR)
    filtered = prices_array[
        (prices_array >= Q1 - 1.5 * IQR) &
        (prices_array <= Q3 + 1.5 * IQR)
        ]
    return filtered.tolist()


def prepare_input_data(input_data: Dict[str, Any]) -> pd.DataFrame:

    # Подготавливает входные данные для модели
    if model is None:
        raise ValueError("Модель не загружена")

    processed_data = input_data.copy()

    # Преобразуем название региона в код
    region_name = processed_data.get('region', 'Москва')
    if region_name in REGION_NAME_TO_CODE:
        processed_data['region_code'] = REGION_NAME_TO_CODE[region_name]
    else:
        processed_data['region_code'] = '77'  # Москва по умолчанию

    # Создаем engineered features
    rooms_val = processed_data.get('rooms', 1)
    area_val = processed_data.get('area', 50)
    level_val = processed_data.get('level', 1)
    levels_val = max(processed_data.get('levels', 5), 1)

    processed_data['room_size'] = area_val / (0.5 if rooms_val == 0 else max(rooms_val, 0.5))
    processed_data['floor_ratio'] = level_val / levels_val

    # Применяем target encoding
    cat_cols = ['region_code', 'building_type', 'object_type', 'rooms']

    for col in cat_cols:
        if col in processed_data and col in encoders:
            try:
                # Энкодеры ожидают DataFrame
                col_df = pd.DataFrame({col: [str(processed_data[col])]})
                encoded_array = encoders[col].transform(col_df)
                encoded_value = float(encoded_array.iloc[0])
                processed_data[col] = encoded_value
            except Exception as e:
                print(f"Ошибка кодирования {col}: {e}")
                processed_data[col] = 0

    # Создаем DataFrame с нужными признаками
    features = {}
    for k in REQUIRED_FEATURES:
        features[k] = processed_data.get(k, 0)

    return pd.DataFrame([features])


def predict_price(input_data: Dict[str, Any]) -> float:

    # Основная функция прогнозирования
    features_df = prepare_input_data(input_data)
    price = float(model.predict(features_df)[0])
    return price


def predict_price_with_analogs(input_data: Dict[str, Any]) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Прогнозирует цену с учетом аналогов с CIAN
    Возвращает: (финальная_цена, ml_прогноз, аналоги)
    """
    # Строгая валидация обязательных параметров
    required_fields = ['region', 'city', 'rooms', 'area']
    for field in required_fields:
        if field not in input_data or input_data[field] in (None, ''):
            raise ValueError(f"Не указан обязательный параметр: {field}")

    # deal_type обязателен только для поиска аналогов
    if 'deal_type' not in input_data or input_data['deal_type'] in (None, ''):
        raise ValueError("Не указан обязательный параметр: deal_type")

    # Проверка города на валидность для cianparser
    city_name = input_data['city']
    valid_locations = cianparser.list_locations()

    # Извлекаем названия городов из структуры [['Москва', '1'], ['Санкт-Петербург', '2'], ...]
    available_cities = []
    if isinstance(valid_locations, list):
        for loc in valid_locations:
            if isinstance(loc, list) and len(loc) >= 1:
                available_cities.append(loc[0])  # Берем первый элемент (название города)

    print(f"Доступные города: {available_cities[:10]}...")  # Покажем первые 10

    # Ищем город в списке поддерживаемых (регистронезависимо)
    city_lower = city_name.lower()
    supported_city = next((city for city in available_cities if city.lower() == city_lower), None)

    if not supported_city:
        # Пробуем найти похожий город
        similar_cities = [city for city in available_cities if city_lower in city.lower()]
        if similar_cities:
            raise ValueError(
                f"Город '{city_name}' не найден. Возможно вы имели в виду: {', '.join(similar_cities[:3])}")
        else:
            raise ValueError(
                f"Город '{city_name}' не поддерживается парсером. Доступные города: {', '.join(available_cities[:10])}...")

    print(f"Найден город в списке: {supported_city}")

    # Прогноз ML
    price_ml = predict_price(input_data)
    print(f"ML прогноз: {price_ml:,.0f} руб.")

    # Поиск аналогов - используем город!
    analogs = get_cian_analogs(
        location=supported_city,
        deal_type=input_data['deal_type'],
        rooms=int(input_data['rooms']),
        area=float(input_data['area']),
        start_page=1,
        end_page=1
    )

    print(f"Найдено аналогов: {len(analogs)}")

    # Расчет финальной цены с умным взвешиванием
    if analogs:
        prices = []
        for flat in analogs:
            if flat.get('price'):
                try:
                    # Очищаем цену от пробелов и символов
                    price_val = float(flat['price'])
                    prices.append(price_val)
                except (ValueError, TypeError):
                    continue

        if prices:
            print(f"Цены аналогов до фильтрации: {[f'{p:,.0f}' for p in prices]}")

            # Фильтруем выбросы
            filtered_prices = filter_outliers(prices)
            print(f"Цены аналогов после фильтрации: {[f'{p:,.0f}' for p in filtered_prices]}")

            if filtered_prices:
                # Используем МЕДИАНУ вместо среднего - она устойчивее к выбросам
                price_cian = np.median(filtered_prices)
                print(f"Медианная цена аналогов: {price_cian:,.0f} руб.")

                # Простое и надежное взвешивание
                ml_weight = 0.3  # Всегда 30% веса модели
                cian_weight = 0.7  # 70% веса аналогам

                print(f"Веса: ML={ml_weight:.2f}, CIAN={cian_weight:.2f}")

                price_final = (price_ml * ml_weight + price_cian * cian_weight)
            else:
                print("Все аналоги были отфильтрованы как выбросы, используем ML прогноз")
                price_final = price_ml
        else:
            print("Не удалось извлечь цены из аналогов")
            price_final = price_ml
    else:
        print("Аналоги не найдены, используем ML прогноз")
        price_final = price_ml

    print(f"Финальный прогноз: {price_final:,.0f} руб.")
    return price_final, price_ml, analogs


if __name__ == "__main__":
    print("Тест функции predict_price")

    # Пример для теста
    example = {
        'region': 'Ростовская область',
        'city': 'Ростов-на-Дону',
        'building_type': '2',
        'object_type': '1',
        'level': 5,
        'levels': 12,
        'rooms': 2,
        'area': 55.0,
        'kitchen_area': 10.0,
        'deal_type': 'sale'
    }

    try:
        result = predict_price(example)
        print(f"Прогноз: {result:,.2f} руб.")

        # Тест с аналогами
        final_price, ml_price, analogs = predict_price_with_analogs(example)
        print(f"ML прогноз: {ml_price:,.2f} руб.")
        print(f"Финальный прогноз (с аналогами): {final_price:,.2f} руб.")
        print(f"Найдено аналогов: {len(analogs)}")

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback

        traceback.print_exc()