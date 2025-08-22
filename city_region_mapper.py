import json
import pandas as pd


class CityRegionMapper:
    def __init__(self, mapping_path='region_to_cities.json'):
        self.region_to_cities = self._load_mapping(mapping_path)
        self.city_to_region = self._create_reverse_mapping()

    def _load_mapping(self, path):
        """Загружает маппинг из JSON"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Файл {path} не найден")
            return {}
        except json.JSONDecodeError:
            print(f"Ошибка формата JSON в файле {path}")
            return {}

    def _create_reverse_mapping(self):
        """Создает обратный маппинг город -> регион"""
        reverse_mapping = {}
        for region, cities in self.region_to_cities.items():
            for city in cities:
                reverse_mapping[city.lower()] = region
        return reverse_mapping

    def get_region_from_city(self, city_name):
        """Конвертирует город в регион"""
        if not city_name or pd.isna(city_name):
            return "Неизвестный регион"

        city_name = str(city_name).strip().lower()
        return self.city_to_region.get(city_name, "Неизвестный регион")

    def get_cities_by_region(self, region_name):
        """Получает список городов по региону"""
        if not region_name or pd.isna(region_name):
            return []

        region_name = str(region_name).strip()
        return self.region_to_cities.get(region_name, [])

    def validate_city_in_region(self, city_name, region_name):
        """Проверяет, принадлежит ли город региону"""
        if not city_name or not region_name:
            return False

        city_name = str(city_name).strip().lower()
        region_name = str(region_name).strip()

        cities_in_region = self.get_cities_by_region(region_name)
        return any(city.lower() == city_name for city in cities_in_region)