import cianparser
from typing import List, Dict, Any
import json


# Загружаем маппинг город->регион
def load_region_mapping(mapping_path='region_to_cities.json'):
    """Загружает маппинг регионов и городов"""
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Файл {mapping_path} не найден")
        return {}
    except json.JSONDecodeError:
        print(f"Ошибка формата JSON в файле {mapping_path}")
        return {}


# Создаем обратный маппинг город->регион
def create_city_to_region_mapping(region_data):
    """Создает маппинг город -> регион"""
    city_to_region = {}
    for region, cities in region_data.items():
        for city in cities:
            city_to_region[city.lower()] = region
    return city_to_region


# Загружаем данные при импорте
region_data = load_region_mapping()
city_to_region_mapping = create_city_to_region_mapping(region_data)


def get_region_from_city(city_name):
    """Определяет регион по названию города"""
    if not city_name:
        return "Неизвестный регион"
    return city_to_region_mapping.get(city_name.lower(), "Неизвестный регион")


def get_cian_analogs(
    location: str,
    deal_type: str,
    rooms: int,
    area: float,
    start_page: int = 1,
    end_page: int = 1
) -> List[Dict[str, Any]]:

    parser = cianparser.CianParser(location=location)

    # Парсим с запасом по комнатам (±1)
    rooms_range = tuple({max(1, rooms - 1), rooms, rooms + 1})
    data = parser.get_flats(
        deal_type=deal_type,
        rooms=rooms_range,
        with_saving_csv=False,
        additional_settings={"start_page": start_page, "end_page": end_page}
    )

    # ДОБАВЛЯЕМ ГОРОД И РЕГИОН К КАЖДОМУ ОБЪЯВЛЕНИЮ
    for flat in data:
        flat['city'] = location  # сохраняем город поиска
        flat['region'] = get_region_from_city(location)  # определяем регион

    # Гибкие диапазоны
    area_tol = int(max(10, area * 0.2))  # ±20% или минимум 10 м²

    def is_analog(flat, area_tol):
        try:
            flat_area = float(flat.get("area_total", 0))
            return abs(flat_area - area) <= area_tol
        except Exception:
            return False

    # Первый проход: стандартный диапазон
    analogs = [flat for flat in data if is_analog(flat, area_tol)]

    # Если мало аналогов — расширяем диапазон
    if len(analogs) < 3:
        area_tol = int(max(20, area * 0.20))  # ±20% или минимум 20 м²
        analogs = [flat for flat in data if is_analog(flat, area_tol)]

    # Сортировка по похожести (по площади и комнатам)
    def similarity(flat):
        flat_area = float(flat.get("area_total", 0))
        flat_rooms = int(flat.get("rooms", 0))
        return abs(flat_area - area) + 5 * abs(flat_rooms - rooms)

    analogs = sorted(analogs, key=similarity)

    # Возвращаем до 7 наиболее похожих
    return analogs[:7]


def save_analogs_to_csv(analogs: List[Dict[str, Any]], filename: str = 'cian_analogs.csv'):
    """Сохраняет аналоги в CSV файл"""
    import pandas as pd
    if analogs:
        df = pd.DataFrame(analogs)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Данные сохранены в {filename}")
    else:
        print("Нет данных для сохранения")


if __name__ == "__main__":
    # Тестируем с сохранением в CSV
    analogs = get_cian_analogs(location="Москва", deal_type="sale", rooms=2, area=55, start_page=1, end_page=1)

    # Сохраняем в CSV для проверки
    save_analogs_to_csv(analogs, 'test_analogs.csv')

    # Выводим в консоль
    for flat in analogs:
        print(
            f"{flat.get('city', 'N/A')} | {flat.get('region', 'N/A')} | {flat.get('address', '')} | {flat.get('price', '')} | {flat.get('url', '')}")