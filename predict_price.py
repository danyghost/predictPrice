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
    region_name_to_code = dict(zip(region_mapping_df['region_name'], region_mapping_df['region_code']))
except Exception as e:
    print(f"Ошибка загрузки region_mapping.csv: {e}")
    region_name_to_code = {}

sale_required_features = [
    'region_code', 'building_type', 'object_type', 'level', 'levels',
    'rooms', 'area', 'kitchen_area', 'room_size', 'floor_ratio'
]

rent_required_features = [
    'type', 'gas', 'area', 'rooms', 'kitchen_area', 'build_year', 'material',
    'build_series_category', 'level', 'levels','rubbish_chute', 'build_overlap',
    'build_walls', 'heating', 'city', 'floor_ratio', 'is_new_building'
]


def filter_outliers(prices: List[float]) -> List[float]:
    """Удаляет выбросы из цен аналогов используя межквартильный размах (IQR)"""
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


def prepare_sale_input_data(input_data: Dict[str, Any], encoders: dict = None) -> pd.DataFrame:
    """Подготавливает входные данные для модели продажи"""

    if encoders is None:
        encoders = {}

    processed_data = input_data.copy()

    # Преобразуем название региона в код
    region_name = processed_data.get('region')
    if region_name in region_name_to_code:
        processed_data['region_code'] = region_name_to_code[region_name]

    # Создаем engineered features
    rooms_val = processed_data.get('rooms', 1)
    area_val = processed_data.get('area', 50)
    level_val = processed_data.get('level', 1)
    levels_val = max(processed_data.get('levels', 5), 1)

    processed_data['room_size'] = area_val / (0.5 if rooms_val == 0 else max(rooms_val, 0.5))
    processed_data['floor_ratio'] = level_val / levels_val

    # Применяем target encoding если энкодеры доступны
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
    for k in sale_required_features:
        features[k] = processed_data.get(k, 0)

    return pd.DataFrame([features])


def prepare_rent_input_data(input_data: Dict[str, Any]) -> pd.DataFrame:
    """Подготавливает входные данные для модели аренды"""
    processed_data = input_data.copy()

    # Маппинг типов для модели аренды
    building_type_mapping = {
        '0': 'unknown', '1': 'panel', '2': 'monolithic',
        '3': 'brick', '4': 'block', '5': 'wood'
    }

    object_type_mapping = {
        '1': 'secondary', '2': 'new'
    }

    # Преобразуем типы зданий
    building_type = str(processed_data.get('building_type', '0'))
    processed_data['material'] = building_type_mapping.get(building_type, 'unknown')

    # Преобразуем типы объектов
    object_type = str(processed_data.get('object_type', '1'))
    processed_data['type'] = object_type_mapping.get(object_type, 'secondary')

    # Заполняем обязательные поля для модели аренды
    processed_data['gas'] = 'unknown'
    processed_data['build_year'] = 2000 if object_type == '2' else 1990  # Новостройка или вторичка
    processed_data['build_series_category'] = 'unknown'
    processed_data['rubbish_chute'] = 'unknown'
    processed_data['build_overlap'] = 'unknown'
    processed_data['build_walls'] = 'unknown'
    processed_data['heating'] = 'unknown'
    processed_data['city'] = processed_data.get('city', 'unknown')

    # Создаем engineered features
    level_val = processed_data.get('level', 1)
    levels_val = max(processed_data.get('levels', 5), 1)
    processed_data['floor_ratio'] = level_val / levels_val
    processed_data['is_new_building'] = object_type == '2'

    # Обработка бесконечных значений
    processed_data['floor_ratio'] = processed_data['floor_ratio'] if not np.isinf(
        processed_data['floor_ratio']) else 0.5

    # Создаем DataFrame с нужными признаками
    features = {}
    for k in rent_required_features:
        features[k] = processed_data.get(k, 'unknown') if k in ['type', 'gas', 'material', 'build_series_category',
                                                                'rubbish_chute', 'build_overlap', 'build_walls',
                                                                'heating', 'city'] else processed_data.get(k, 0)

    return pd.DataFrame([features])


def predict_sale_price(input_data: Dict[str, Any], model, encoders: dict = None) -> float:
    """Прогнозирование цены для продажи"""
    features_df = prepare_sale_input_data(input_data, encoders)
    price = float(model.predict(features_df)[0])
    return price

def predict_rent_price(input_data: Dict[str, Any], model) -> float:
    """Прогнозирование цены для аренды"""
    features_df = prepare_rent_input_data(input_data)
    price_log = float(model.predict(features_df)[0])
    price = np.expm1(price_log)
    return price

def predict_price_with_analogs(input_data: Dict[str, Any], model, encoders: dict = None) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Прогнозирует цену с учетом аналогов с CIAN
    Возвращает: (финальная_цена, ml_прогноз, аналоги)
    """
    # Строгая валидация обязательных параметров
    required_fields = ['region', 'city', 'rooms', 'area']
    for field in required_fields:
        if field not in input_data or input_data[field] in (None, ''):
            raise ValueError(f"Не указан обязательный параметр: {field}")

    # Проверка города на валидность для cianparser
    city_name = input_data['city']
    valid_locations = cianparser.list_locations()

    # Извлекаем названия городов из структуры
    available_cities = []
    if isinstance(valid_locations, list):
        for loc in valid_locations:
            if isinstance(loc, list) and len(loc) >= 1:
                available_cities.append(loc[0])  # Берем первый элемент (название города)

    print(f"Доступные города: {available_cities[:10]}...")

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

    # Прогноз ML в зависимости от типа сделки
    deal_type = input_data['deal_type']
    if deal_type == 'sale':
        price_ml = predict_sale_price(input_data, model, encoders)
    else:  # rent
        price_ml = predict_rent_price(input_data, model)

    print(f"ML прогноз ({deal_type}): {price_ml:,.0f} {'руб./мес' if deal_type == 'rent' else 'руб.'}")

    # Поиск аналогов (используем город)
    analogs = get_cian_analogs(
        location=supported_city,
        deal_type=deal_type,
        rooms=int(input_data['rooms']),
        area=float(input_data['area']),
        start_page=1,
        end_page=1
    )

    print(f"Найдено аналогов для {input_data['deal_type']}: {len(analogs)}")

    # Расчет финальной цены с учетом типа сделки
    if analogs:
        prices = []
        for flat in analogs:
            if flat.get('price'):
                try:
                    price_val = float(flat['price'])
                    # Для аренды проверяем адекватность цены
                    if deal_type == 'rent':
                        # Аренда обычно от 5к до 500к руб/мес
                        if 8000 <= price_val <= 500000:
                            prices.append(price_val)
                    else:
                        # Продажа обычно от 1 млн до 50 млн
                        if 1000000 <= price_val <= 50000000:
                            prices.append(price_val)
                except (ValueError, TypeError):
                    continue

        if prices:
            filtered_prices = filter_outliers(prices)

            if filtered_prices:
                price_cian = np.median(filtered_prices)

                # Разные веса для аренды и продажи
                if deal_type == 'rent':
                    ml_weight = 0.4
                    cian_weight = 0.6
                else:
                    ml_weight = 0.3
                    cian_weight = 0.7

                price_final = (price_ml * ml_weight + price_cian * cian_weight)
            else:
                price_final = price_ml
        else:
            price_final = price_ml
    else:
        price_final = price_ml

    # Форматирование вывода
    price_type = "руб./мес" if input_data['deal_type'] == "rent" else "руб."
    print(f"Финальный прогноз: {price_final:,.0f} {price_type}")

    return price_final, price_ml, analogs

# Сохраняем обратную совместимость со старой версией
def predict_price(input_data: Dict[str, Any]) -> float:
    """Совместимость со старой версией - только для продажи"""
    try:
        sale_model_data = joblib.load('model/model_optimized.joblib')
        model = sale_model_data['model']
        encoders = sale_model_data.get('encoders', {})

        features_df = prepare_sale_input_data(input_data, encoders)
        price = float(model.predict(features_df)[0])
        return price
    except Exception as e:
        print(f"Ошибка в predict_price: {e}")
        return 0

if __name__ == "__main__":
    print("Тест функции predict_price")

    # Тест прогноза продажи
    test_sale = {
        'region': 'Ростовская область',
        'city': 'Ростов-на-Дону',
        'building_type': '2',
        'object_type': '1',
        'level': 5,
        'levels': 9,
        'rooms': 2,
        'area': 55.0,
        'kitchen_area': 10.0,
        'deal_type': 'sale'
    }

    # Тест прогноза аренды
    test_rent = {
        'region': 'Ростовская область',
        'city': 'Ростов-на-Дону',
        'building_type': '2',
        'object_type': '1',
        'level': 5,
        'levels': 9,
        'rooms': 2,
        'area': 55.0,
        'kitchen_area': 10.0,
        'deal_type': 'rent'
    }

    try:
        # Загружаем модели для теста
        sale_model_data = joblib.load('model/model_optimized.joblib')
        sale_model = sale_model_data['model']
        sale_encoders = sale_model_data.get('encoders', {})

        rent_model_data = joblib.load('model/rent_model.joblib')
        rent_model = rent_model_data['model']

        # Тест продажи
        print("\nТест прогноза продажи")
        final_price_sale, ml_price_sale, analogs_sale = predict_price_with_analogs(
            test_sale,
            model = sale_model,
            encoders = sale_encoders
        )
        print(f"ML прогноз продажи: {ml_price_sale:,.2f} руб.")
        print(f"Финальный прогноз продажи: {final_price_sale:,.2f} руб.")
        print(f"Найдено аналогов продажи: {len(analogs_sale)}")

        # Тест аренды
        print("\nТест прогноза аренды")
        final_price_rent, ml_price_rent, analogs_rent = predict_price_with_analogs(
            test_rent,
            model = rent_model,
            encoders = {}
        )
        print(f"ML прогноз аренды: {ml_price_rent:,.2f} руб./мес")
        print(f"Финальный прогноз аренды: {final_price_rent:,.2f} руб./мес")
        print(f"Найдено аналогов аренды: {len(analogs_rent)}")

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback

        traceback.print_exc()