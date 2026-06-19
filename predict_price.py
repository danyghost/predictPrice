import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from parser import get_cian_analogs
import cianparser


MARKET_GROWTH_INDEX = {
    # КЛАСТЕР 1: СВЕРХВЫСОКИЙ РОСТ (Абсолютные лидеры, курорты и флагманы)
    # Точечные города-исключения с аномальным ростом
    "Сочи": {"sale": 3.90, "rent": 2.30},
    "Геленджик": {"sale": 3.40, "rent": 1.95},
    "Анапа": {"sale": 3.20, "rent": 1.85},
    "Казань": {"sale": 3.10, "rent": 1.80},

    # Ростовская агломерация
    "Ростов-на-Дону": {"sale": 2.3, "rent": 1.55}, # Главный город высокий спрос на аренду (вузы, логистика)
    "Батайск": {"sale": 1.9, "rent": 1.45},
    "Аксай": {"sale": 1.8, "rent": 1.45},
    "Азов": {"sale": 1.6, "rent": 1.35}, # Исторический спутник, более удален, рост стабильно-умеренный
    "Шахты": {"sale": 1.2, "rent": 1.35},
    "Таганрог": {"sale": 1.8, "rent": 1.45},
    "Новочеркасск": {"sale": 1.9, "rent": 1.4},
    "Волгодонск": {"sale": 1.5, "rent": 1.4},

    # Ближнее Подмосковье (города-спутники первого эшелона)
    "Красногорск": {"sale": 2.75, "rent": 1.75},
    "Реутов": {"sale": 2.75, "rent": 1.75},
    "Химки": {"sale": 2.70, "rent": 1.70},
    "Одинцово": {"sale": 2.70, "rent": 1.70},
    "Мытищи": {"sale": 2.65, "rent": 1.70},
    "Люберцы": {"sale": 2.65, "rent": 1.65},

    # Региональные базовые коэффициенты кластера
    "Краснодарский край": {"sale": 2.85, "rent": 1.65},  # Базовый для Краснодара/Новороссийска
    "Ростовская область": {"sale": 2.20, "rent": 1.38},
    "Калининградская область": {"sale": 2.90, "rent": 1.70},
    "Республика Татарстан": {"sale": 2.50, "rent": 1.45},
    "Санкт-Петербург": {"sale": 2.70, "rent": 1.75},
    "Республика Крым": {"sale": 2.65, "rent": 1.60},
    "Севастополь": {"sale": 2.65, "rent": 1.60},
    "Республика Дагестан": {"sale": 2.60, "rent": 1.55},
    "Москва": {"sale": 2.55, "rent": 1.85},
    "Московская область": {"sale": 2.40, "rent": 1.50},  # Среднеобластной показатель
    "Ленинградская область": {"sale": 2.45, "rent": 1.50},

    # КЛАСТЕР 2: КРУПНЫЕ ИНДУСТРИАЛЬНЫЕ ЦЕНТРЫ И МИЛЛИОННИКИ
    "Владивосток": {"sale": 2.55, "rent": 1.60},
    "Сургут": {"sale": 2.40, "rent": 1.60},

    "Новосибирская область": {"sale": 2.35, "rent": 1.50},
    "Свердловская область": {"sale": 2.35, "rent": 1.50},
    "Нижегородская область": {"sale": 2.35, "rent": 1.45},
    "Тюменская область": {"sale": 2.40, "rent": 1.55},
    "Приморский край": {"sale": 2.30, "rent": 1.45},
    "Самарская область": {"sale": 2.25, "rent": 1.40},
    "Красноярский край": {"sale": 2.30, "rent": 1.45},
    "Челябинская область": {"sale": 2.25, "rent": 1.40},
    "Воронежская область": {"sale": 2.30, "rent": 1.45},
    "Иркутская область": {"sale": 2.30, "rent": 1.45},
    "Ханты-Мансийский автономный округ — Югра": {"sale": 2.30, "rent": 1.50},
    "Ямало-Ненецкий автономный округ": {"sale": 2.35, "rent": 1.55},

    # КЛАСТЕР 3: СТАБИЛЬНЫЙ УМЕРЕННЫЙ РОСТ И ГОРОДА СРЕДНЕЙ ПОЛОСЫ
    # Аутсайдеры Московской области (Дальний пояс, требующий занижения)
    "Шатура": {"sale": 1.95, "rent": 1.30},
    "Рошаль": {"sale": 1.90, "rent": 1.25},
    "Ликино-Дулёво": {"sale": 1.95, "rent": 1.30},

    "Республика Адыгея": {"sale": 2.15, "rent": 1.38},
    "Республика Башкортостан": {"sale": 2.20, "rent": 1.42},
    "Республика Бурятия": {"sale": 2.15, "rent": 1.38},
    "Кабардино-Балкарская Республика": {"sale": 2.15, "rent": 1.35},
    "Карачаево-Черкесская Republic": {"sale": 2.15, "rent": 1.35},
    "Республика Карелия": {"sale": 2.15, "rent": 1.40},
    "Республика Коми": {"sale": 2.10, "rent": 1.35},
    "Республика Марий Эл": {"sale": 2.10, "rent": 1.35},
    "Республика Мордовия": {"sale": 2.10, "rent": 1.35},
    "Республика Саха (Якутия)": {"sale": 2.25, "rent": 1.45},
    "Республика Северная Осетия — Алания": {"sale": 2.15, "rent": 1.38},
    "Удмуртская Республика": {"sale": 2.15, "rent": 1.38},
    "Республика Хакасия": {"sale": 2.15, "rent": 1.38},
    "Чувашская Республика": {"sale": 2.15, "rent": 1.40},
    "Алтайский край": {"sale": 2.15, "rent": 1.38},
    "Ставропольский край": {"sale": 2.20, "rent": 1.42},
    "Хабаровский край": {"sale": 2.20, "rent": 1.42},
    "Амурская область": {"sale": 2.15, "rent": 1.38},
    "Архангельская область": {"sale": 2.15, "rent": 1.40},
    "Астраханская область": {"sale": 2.15, "rent": 1.38},
    "Белгородская область": {"sale": 2.00, "rent": 1.30},
    "Брянская область": {"sale": 2.10, "rent": 1.35},
    "Владимирская область": {"sale": 2.15, "rent": 1.38},
    "Волгоградская область": {"sale": 2.15, "rent": 1.38},
    "Вологодская область": {"sale": 2.15, "rent": 1.38},
    "Ивановская область": {"sale": 2.10, "rent": 1.35},
    "Калужская область": {"sale": 2.20, "rent": 1.40},
    "Камчатский край": {"sale": 2.20, "rent": 1.45},
    "Кемеровская область": {"sale": 2.15, "rent": 1.38},
    "Кировская область": {"sale": 2.10, "rent": 1.35},
    "Костромская область": {"sale": 2.10, "rent": 1.35},
    "Курская область": {"sale": 2.00, "rent": 1.30},
    "Липецкая область": {"sale": 2.15, "rent": 1.38},
    "Мурманская область": {"sale": 2.15, "rent": 1.40},
    "Новгородская область": {"sale": 2.15, "rent": 1.38},
    "Омская область": {"sale": 2.15, "rent": 1.38},
    "Оренбургская область": {"sale": 2.10, "rent": 1.35},
    "Орловская область": {"sale": 2.10, "rent": 1.35},
    "Пензенская область": {"sale": 2.15, "rent": 1.38},
    "Пермский край": {"sale": 2.20, "rent": 1.40},
    "Псковская область": {"sale": 2.10, "rent": 1.35},
    "Рязанская область": {"sale": 2.15, "rent": 1.38},
    "Саратовская область": {"sale": 2.15, "rent": 1.38},
    "Сахалинская область": {"sale": 2.25, "rent": 1.45},
    "Смоленская область": {"sale": 2.15, "rent": 1.38},
    "Тамбовская область": {"sale": 2.10, "rent": 1.35},
    "Тверская область": {"sale": 2.15, "rent": 1.38},
    "Томская область": {"sale": 2.20, "rent": 1.42},
    "Тульская область": {"sale": 2.20, "rent": 1.40},
    "Ульяновская область": {"sale": 2.10, "rent": 1.35},
    "Забайкальский край": {"sale": 2.15, "rent": 1.38},
    "Ярославская область": {"sale": 2.15, "rent": 1.40},

    # КЛАСТЕР 4: НИЗКООБОРОТНЫЕ И СПЕЦИФИЧЕСКИЕ РЫНКИ
    "Республика Алтай": {"sale": 2.10, "rent": 1.35},
    "Республика Ингушетия": {"sale": 1.90, "rent": 1.25},
    "Республика Калмыкия": {"sale": 1.90, "rent": 1.25},
    "Республика Тыва": {"sale": 1.90, "rent": 1.25},
    "Чеченская Республика": {"sale": 2.00, "rent": 1.30},
    "Курганская область": {"sale": 1.95, "rent": 1.28},
    "Магаданская область": {"sale": 1.95, "rent": 1.30},
    "Еврейская автономная область": {"sale": 1.90, "rent": 1.25},
    "Ненецкий автономный округ": {"sale": 2.00, "rent": 1.35},
    "Чукотский автономный округ": {"sale": 1.95, "rent": 1.30},

    "DEFAULT": {"sale": 2.15, "rent": 1.38}
}


# Загружаем модель
try:
    model_data = joblib.load('model/sale_model.joblib')
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
    'build_series_category', 'level', 'levels', 'rubbish_chute', 'build_overlap',
    'build_walls', 'heating', 'city', 'floor_ratio', 'is_new_building'
]


def apply_inflation_markup(base_price: float, region_name: str, deal_type: str) -> float:
    """Применяет коэффициент актуализации цены 2018 года к реалиям 2026 года"""
    if not region_name:
        return base_price * MARKET_GROWTH_INDEX["DEFAULT"].get(deal_type, 1.0)

    clean_region = str(region_name).strip()
    coefficients = MARKET_GROWTH_INDEX.get(clean_region, MARKET_GROWTH_INDEX["DEFAULT"])
    factor = coefficients.get(deal_type, 1.0)
    return base_price * factor


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
                val = processed_data[col]
                if isinstance(val, str):
                    if val.isdigit():
                        val = int(val)
                    else:
                        try:
                            val = float(val)
                        except ValueError:
                            pass

                # Передаем чистое значение без str()
                col_df = pd.DataFrame({col: [val]})
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


def predict_price_with_analogs(input_data: Dict[str, Any], model, encoders: dict = None) -> Tuple[
    float, float, List[Dict[str, Any]]]:
    """
    Прогнозирует цену с учетом аналогов с CIAN
    Возвращает: (финальная_цена, ml_прогноз, аналоги)
    """
    # Строгая валидация обязательных параметров
    required_fields = ['region', 'city', 'rooms', 'area', 'deal_type']
    for field in required_fields:
        if field not in input_data or input_data[field] in (None, ''):
            raise ValueError(f"Не указан обязательный параметр: {field}")

    city_name = input_data['city']
    region_name = input_data.get('region', 'DEFAULT')
    deal_type = input_data['deal_type']

    valid_locations = cianparser.list_locations()
    available_cities = []
    if isinstance(valid_locations, list):
        for loc in valid_locations:
            if isinstance(loc, list) and len(loc) >= 1:
                available_cities.append(loc[0])

    city_lower = city_name.lower()
    supported_city = next((city for city in available_cities if city.lower() == city_lower), None)

    if not supported_city:
        print(f"⚠️ Город '{city_name}' отсутствует в cianparser. Парсинг аналогов будет пропущен.")
    else:
        print(f"✅ Найдено соответствие в cianparser: {supported_city}")

    # Базовый прогноз ML в зависимости от типа сделки
    if deal_type == 'sale':
        price_ml = predict_sale_price(input_data, model, encoders)
    else:  # rent
        price_ml = predict_rent_price(input_data, model)

    print(
        f"Базовый ML прогноз ({deal_type}, база 2018 года): {price_ml:,.0f} {'руб./мес' if deal_type == 'rent' else 'руб.'}")

    growth_data = MARKET_GROWTH_INDEX.get(city_name) or MARKET_GROWTH_INDEX.get(region_name) or MARKET_GROWTH_INDEX[
        "DEFAULT"]
    coefficient = growth_data[deal_type]

    # Применяем точечный коэффициент к базовой цене ML
    price_ml = price_ml * coefficient
    print(f"📈 Коэффициент для [{city_name} / {region_name}]: {coefficient}")
    print(
        f"Актуализированный ML прогноз под текущий рынок: {price_ml:,.0f} {'руб./мес' if deal_type == 'rent' else 'руб.'}")

    # Поиск аналогов (используем город)
    analogs = []
    if supported_city:
        try:
            analogs = get_cian_analogs(
                location=supported_city,
                deal_type=deal_type,
                rooms=int(input_data['rooms']),
                area=float(input_data['area']),
                start_page=1,
                end_page=1
            )
            print(f"Найдено аналогов для {deal_type}: {len(analogs)}")
        except Exception as e:
            print(f"Не удалось собрать аналоги через парсер ({e}). Используем резервную актуализированную ML-модель.")
    else:
        print("Поиск аналогов пропущен: расчет ведется по кастомизированной ML-модели.")

    # Расчет финальной цены с учетом типа сделки
    if analogs:
        prices = []
        for flat in analogs:
            if flat.get('price'):
                try:
                    price_val = float(flat['price'])
                    # Для аренды проверяем адекватность цены
                    if deal_type == 'rent':
                        if 8000 <= price_val <= 500000:
                            prices.append(price_val)
                    else:
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
                    ml_weight = 0.3
                    cian_weight = 0.7
                else:
                    ml_weight = 0.2
                    cian_weight = 0.8

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


def predict_price(input_data: Dict[str, Any]) -> float:
    """Совместимость со старой версией - только для продажи"""
    try:
        sale_model_data = joblib.load('model/sale_model.joblib')
        model = sale_model_data['model']
        encoders = sale_model_data.get('encoders', {})

        features_df = prepare_sale_input_data(input_data, encoders)
        price = float(model.predict(features_df)[0])

        region_name = input_data.get('region', 'DEFAULT')
        price = apply_inflation_markup(price, region_name, 'sale')
        return price
    except Exception as e:
        print(f"Ошибка в predict_price: {e}")
        return 0