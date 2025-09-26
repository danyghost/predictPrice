import requests
from typing import List, Dict, Any
import json
import re
import time
from bs4 import BeautifulSoup


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


def get_city_id(city_name, valid_locations):
    """Получает ID города из списка cianparser"""
    for loc in valid_locations:
        if isinstance(loc, list) and len(loc) >= 2 and loc[0].lower() == city_name.lower():
            return loc[1]  # Возвращаем ID
    return None


def parse_cian_page(url, params, headers):
    """Парсит одну страницу CIAN"""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        analogs = []

        # Ищем карточки объявлений
        cards = soup.find_all('article', {'data-name': 'CardComponent'})

        for card in cards:
            try:
                # Извлекаем цену
                price_elem = card.find('span', {'data-mark': 'MainPrice'})
                if not price_elem:
                    continue

                price_text = price_elem.get_text()
                price = float(re.sub(r'[^\d]', '', price_text))

                # Извлекаем площадь
                area_elem = card.find('div', string=re.compile(r'м²'))
                area_val = None
                if area_elem:
                    area_text = area_elem.get_text()
                    area_match = re.search(r'(\d+[,.]?\d*)\s*м²', area_text)
                    if area_match:
                        area_val = float(area_match.group(1).replace(',', '.'))

                # Извлекаем количество комнат
                rooms_elem = card.find('div', string=re.compile(r'-комн|комнат|комн'))
                rooms_val = None
                if rooms_elem:
                    rooms_text = rooms_elem.get_text()
                    rooms_match = re.search(r'(\d+)\s*[-]?комн', rooms_text)
                    if rooms_match:
                        rooms_val = int(rooms_match.group(1))

                # Извлекаем адрес
                address_elem = card.find('div', {'data-name': 'AddressContainer'})
                address = address_elem.get_text().strip() if address_elem else ''

                # Извлекаем ссылку
                link_elem = card.find('a', {'data-name': 'Link'})
                url = link_elem['href'] if link_elem else ''

                # Извлекаем этаж
                floor_elem = card.find('div', string=re.compile(r'этаж'))
                floor_info = ''
                if floor_elem:
                    floor_info = floor_elem.get_text().strip()

                analogs.append({
                    'price': price,
                    'area_total': area_val,
                    'rooms': rooms_val,
                    'address': address,
                    'url': url,
                    'floor_info': floor_info
                })

            except Exception as e:
                print(f"Ошибка парсинга карточки: {e}")
                continue

        return analogs

    except Exception as e:
        print(f"Ошибка парсинга страницы: {e}")
        return []


def get_cian_analogs(
        location: str,
        deal_type: str,
        rooms: int,
        area: float,
        start_page: int = 1,
        end_page: int = 1
) -> List[Dict[str, Any]]:
    """
    Парсит аналоги с CIAN используя прямой HTTP запрос
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    # Получаем ID города через cianparser (только для получения ID)
    import cianparser
    valid_locations = cianparser.list_locations()
    city_id = get_city_id(location, valid_locations)

    if not city_id:
        print(f"Не найден ID для города {location}")
        return []

    print(f"Парсим город {location} (ID: {city_id})")

    all_analogs = []
    base_url = "https://www.cian.ru/cat.php"

    for page in range(start_page, end_page + 1):
        print(f"Парсим страницу {page}...")

        params = {
            'deal_type': deal_type,
            'engine_version': 2,
            'offer_type': 'flat',
            'region': city_id,
            'p': page,
            'room1': rooms,
            'room2': rooms,
            'room3': rooms
        }

        analogs = parse_cian_page(base_url, params, headers)
        all_analogs.extend(analogs)

        # Задержка между запросами
        time.sleep(2)

        # Если на странице мало объявлений, возможно это последняя страница
        if len(analogs) < 10 and page > start_page:
            print(f"На странице {page} мало объявлений, прекращаем парсинг")
            break

    print(f"Всего найдено объявлений: {len(all_analogs)}")

    # Фильтруем по площади (±10%)
    area_tol = area * 0.1
    filtered_analogs = [
        flat for flat in all_analogs
        if flat['area_total'] and abs(flat['area_total'] - area) <= area_tol
    ]

    # Сортировка по похожести
    def similarity(flat):
        flat_area = flat.get('area_total', 0)
        flat_rooms = flat.get('rooms', 0)
        area_diff = abs(flat_area - area) if flat_area else 100
        rooms_diff = abs(flat_rooms - rooms) * 10 if flat_rooms else 100
        return area_diff + rooms_diff

    filtered_analogs.sort(key=similarity)

    # Добавляем информацию о городе и регионе
    for flat in filtered_analogs:
        flat['city'] = location
        flat['region'] = get_region_from_city(location)

    # Возвращаем до 7 наиболее похожих
    return filtered_analogs[:7]


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
            f"{flat.get('city', 'N/A')} | {flat.get('price', 'N/A')} руб. | {flat.get('area_total', 'N/A')} м² | {flat.get('rooms', 'N/A')} комн. | {flat.get('address', '')}")
from typing import List, Dict, Any
import json
import re
import time
from bs4 import BeautifulSoup

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

def get_city_id(city_name, valid_locations):
    """Получает ID города из списка cianparser"""
    for loc in valid_locations:
        if isinstance(loc, list) and len(loc) >= 2 and loc[0].lower() == city_name.lower():
            return loc[1]  # Возвращаем ID
    return None

def parse_cian_page(url, params, headers):
    """Парсит одну страницу CIAN"""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        analogs = []

        # Ищем карточки объявлений
        cards = soup.find_all('article', {'data-name': 'CardComponent'})

        for card in cards:
            try:
                # Извлекаем цену
                price_elem = card.find('span', {'data-mark': 'MainPrice'})
                if not price_elem:
                    continue

                price_text = price_elem.get_text()
                price = float(re.sub(r'[^\d]', '', price_text))

                # Извлекаем площадь
                area_elem = card.find('div', string=re.compile(r'м²'))
                area_val = None
                if area_elem:
                    area_text = area_elem.get_text()
                    area_match = re.search(r'(\d+[,.]?\d*)\s*м²', area_text)
                    if area_match:
                        area_val = float(area_match.group(1).replace(',', '.'))

                # Извлекаем количество комнат
                rooms_elem = card.find('div', string=re.compile(r'-комн|комнат|комн'))
                rooms_val = None
                if rooms_elem:
                    rooms_text = rooms_elem.get_text()
                    rooms_match = re.search(r'(\d+)\s*[-]?комн', rooms_text)
                    if rooms_match:
                        rooms_val = int(rooms_match.group(1))

                # Извлекаем адрес
                address_elem = card.find('div', {'data-name': 'AddressContainer'})
                address = address_elem.get_text().strip() if address_elem else ''

                # Извлекаем ссылку
                link_elem = card.find('a', {'data-name': 'Link'})
                url = link_elem['href'] if link_elem else ''

                # Извлекаем этаж
                floor_elem = card.find('div', string=re.compile(r'этаж'))
                floor_info = ''
                if floor_elem:
                    floor_info = floor_elem.get_text().strip()

                analogs.append({
                    'price': price,
                    'area_total': area_val,
                    'rooms': rooms_val,
                    'address': address,
                    'url': url,
                    'floor_info': floor_info
                })

            except Exception as e:
                print(f"Ошибка парсинга карточки: {e}")
                continue

        return analogs

    except Exception as e:
        print(f"Ошибка парсинга страницы: {e}")
        return []

def get_cian_analogs(
    location: str,
    deal_type: str,
    rooms: int,
    area: float,
    start_page: int = 1,
    end_page: int = 1
) -> List[Dict[str, Any]]:
    """
    Парсит аналоги с CIAN используя прямой HTTP запрос
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    # Получаем ID города через cianparser (только для получения ID)
    import cianparser
    valid_locations = cianparser.list_locations()
    city_id = get_city_id(location, valid_locations)

    if not city_id:
        print(f"Не найден ID для города {location}")
        return []

    print(f"Парсим город {location} (ID: {city_id})")

    all_analogs = []
    base_url = "https://www.cian.ru/cat.php"

    for page in range(start_page, end_page + 1):
        print(f"Парсим страницу {page}...")

        params = {
            'deal_type': deal_type,
            'engine_version': 2,
            'offer_type': 'flat',
            'region': city_id,
            'p': page,
            'room1': rooms,
            'room2': rooms,
            'room3': rooms
        }

        analogs = parse_cian_page(base_url, params, headers)
        all_analogs.extend(analogs)

        # Задержка между запросами
        time.sleep(2)

        # Если на странице мало объявлений, возможно это последняя страница
        if len(analogs) < 10 and page > start_page:
            print(f"На странице {page} мало объявлений, прекращаем парсинг")
            break

    print(f"Всего найдено объявлений: {len(all_analogs)}")

    # Фильтруем по площади (±10%)
    area_tol = area * 0.1
    filtered_analogs = [
        flat for flat in all_analogs
        if flat['area_total'] and abs(flat['area_total'] - area) <= area_tol
    ]

    # Сортировка по похожести
    def similarity(flat):
        flat_area = flat.get('area_total', 0)
        flat_rooms = flat.get('rooms', 0)
        area_diff = abs(flat_area - area) if flat_area else 100
        rooms_diff = abs(flat_rooms - rooms) * 10 if flat_rooms else 100
        return area_diff + rooms_diff

    filtered_analogs.sort(key=similarity)

    # Добавляем информацию о городе и регионе
    for flat in filtered_analogs:
        flat['city'] = location
        flat['region'] = get_region_from_city(location)

    # Возвращаем до 7 наиболее похожих
    return filtered_analogs[:7]

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
        print(f"{flat.get('city', 'N/A')} | {flat.get('price', 'N/A')} руб. | {flat.get('area_total', 'N/A')} м² | {flat.get('rooms', 'N/A')} комн. | {flat.get('address', '')}")