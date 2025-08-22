import pandas as pd
import json
import re


# Загружаем мэппинг кодов регионов
def load_region_mapping():
    try:
        mapping_df = pd.read_csv('region_mapping.csv')
        region_mapping = {}
        for _, row in mapping_df.iterrows():
            region_mapping[float(row['region_code'])] = row['region_name']
        return region_mapping
    except FileNotFoundError:
        print("Файл region_mapping.csv не найден")
        return {}


# Загружаем мэппинг регионов к городам
def load_region_to_cities():
    try:
        with open('region_to_cities.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Файл region_to_cities.json не найден")
        return {}


# Создаем обратный мэппинг: город -> регион
def create_city_to_region_mapping(region_data):
    city_to_region = {}
    for region, cities in region_data.items():
        for city in cities:
            city_to_region[city.lower()] = region
    return city_to_region


# Функция для извлечения города из адреса
def extract_city_from_address(address, city_to_region):
    """Извлекает название города из адреса"""
    if pd.isna(address):
        return None

    address_str = str(address).lower()

    # Ищем точное совпадение с городами из нашего mapping
    for city in city_to_region.keys():
        if city.lower() in address_str:
            # Возвращаем оригинальное написание города
            for original_city in region_data.values():
                if city.lower() in [c.lower() for c in original_city]:
                    return next(c for c in original_city if c.lower() == city.lower())

    # Пытаемся извлечь город по паттернам
    patterns = [
        r'г\.\s*([^,]+)',  # г. Москва
        r'город\s+([^,]+)',  # город Москва
        r'([а-яё]+)\s*\(',  # Москва (район)
        r',\s*([а-яё\s-]+),\s*[а-я]',  # , Москва, ул.
    ]

    for pattern in patterns:
        match = re.search(pattern, address_str, re.IGNORECASE)
        if match:
            potential_city = match.group(1).strip()
            # Проверяем, есть ли этот город в нашем списке
            for city in city_to_region.keys():
                if city.lower() == potential_city.lower():
                    # Возвращаем оригинальное написание города
                    for original_city in region_data.values():
                        if city.lower() in [c.lower() for c in original_city]:
                            return next(c for c in original_city if c.lower() == city.lower())

    return None


def update_dataset_with_city():
    try:
        # Загружаем мэппинги
        region_mapping = load_region_mapping()
        region_data = load_region_to_cities()

        if not region_mapping or not region_data:
            print("Не удалось загрузить мэппинги")
            return None

        # Создаем обратный мэппинг город->регион
        city_to_region = create_city_to_region_mapping(region_data)

        # Загружаем текущий датасет
        print("Загрузка датасета...")
        data = pd.read_csv("all_v2.csv")
        print(f"Размер датасета: {data.shape}")

        if 'city' not in data.columns:
            print("Добавляем столбец 'city'...")

            # Функция для определения города
            def find_city(row):
                address = row.get('address', '')
                region_code = row.get('region', '')

                # 1. Пытаемся извлечь город из адреса
                city_from_address = extract_city_from_address(address, city_to_region)
                if city_from_address:
                    return city_from_address

                # 2. Если не нашли в адресе, используем код региона
                try:
                    region_code_float = float(region_code)
                    if region_code_float in region_mapping:
                        region_name = region_mapping[region_code_float]

                        # Если регион есть в нашем мэппинге, берем его административный центр
                        if region_name in region_data and region_data[region_name]:
                            return region_data[region_name][0]  # Первый город - административный центр

                        return region_name  # Возвращаем название региона
                except (ValueError, TypeError):
                    pass

                # 3. Если ничего не нашли, возвращаем код региона
                return str(region_code)

            # Применяем функцию к каждой строке
            print("Обработка данных...")
            data['city'] = data.apply(find_city, axis=1)

            # Сохраняем обновленный датасет
            data.to_csv("main_dataset.csv", index=False)
            print("Датасет обновлен: main_dataset.csv")

            # Статистика по городам
            city_stats = data['city'].value_counts()
            print("\nТоп-20 городов по количеству объявлений:")
            print(city_stats.head(20))

            # Информация о пропущенных значениях
            missing_cities = data['city'].isna().sum()
            print(f"\nПропущенных значений города: {missing_cities}")

            # Сохраняем статистику по городам
            city_stats.to_csv("city_statistics.csv")
            print("Статистика по городам сохранена в city_statistics.csv")

            return data
        else:
            print("Столбец 'city' уже существует")
            return data

    except FileNotFoundError:
        print("Файл all_v2.csv не найден")
        return None
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_all_cities_from_mapping():
    """Получаем все города из маппинга"""
    region_data = load_region_to_cities()
    all_cities = []
    for cities in region_data.values():
        all_cities.extend(cities)
    return sorted(list(set(all_cities)))


def get_main_cities():
    """Получает основные города (административные центры регионов)"""
    region_data = load_region_to_cities()
    main_cities = []
    for region, cities in region_data.items():
        if cities:
            main_cities.append(cities[0])
    return sorted(main_cities)


# Для обратной совместимости с parser.py
region_data = load_region_to_cities()

if __name__ == "__main__":
    # Обновляем существующий датасет
    updated_data = update_dataset_with_city()