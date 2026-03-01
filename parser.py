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


def parse_cian_page(url, params, headers, deal_type):
    """Парсит одну страницу CIAN с учетом типа сделки"""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        analogs = []

        # Ищем карточки объявлений - разные селекторы для аренды и продажи
        cards = soup.find_all('article', {'data-name': 'CardComponent'})

        # Альтернативные селекторы для аренды
        if not cards and deal_type == 'rent':
            cards = soup.find_all('div', {'data-name': 'OfferCard'})

        if not cards:
            print(f"Не найдено карточек объявлений на странице для {deal_type}")
            return []

        for card in cards:
            try:
                # Извлекаем цену - разные селекторы для аренды и продажи
                price_elem = None

                if deal_type == 'sale':
                    # Для продажи
                    price_elem = (card.find('span', {'data-mark': 'MainPrice'}) or
                                  card.find('span', class_=re.compile(r'price', re.I)))
                else:
                    # Для аренды
                    price_elem = (card.find('span', {'data-mark': 'MainPrice'}) or
                                  card.find('span', class_=re.compile(r'price|rent', re.I)) or
                                  card.find('p', class_=re.compile(r'price', re.I)))

                if not price_elem:
                    continue

                price_text = price_elem.get_text()
                # Очищаем цену и учитываем, что аренда может быть в руб/мес
                price_text_clean = re.sub(r'[^\d]', '', price_text.split('/')[0])  # Берем часть до "/"
                price = float(price_text_clean) if price_text_clean else 0

                if price == 0:
                    continue

                # Извлекаем площадь
                area_elem = card.find('div', string=re.compile(r'м²'))
                if not area_elem:
                    # Альтернативные поиски площади
                    area_elems = card.find_all('div', string=re.compile(r'(\d+[,.]?\d*)\s*м²'))
                    if area_elems:
                        area_elem = area_elems[0]

                area_val = None
                if area_elem:
                    area_text = area_elem.get_text()
                    area_match = re.search(r'(\d+[,.]?\d*)\s*м²', area_text)
                    if area_match:
                        area_val = float(area_match.group(1).replace(',', '.'))

                # Извлекаем количество комнат
                rooms_elem = card.find('div', string=re.compile(r'-комн|комнат|комн|Студия'))
                rooms_val = None
                if rooms_elem:
                    rooms_text = rooms_elem.get_text()
                    # Обработка студий
                    if 'студия' in rooms_text.lower() or 'studio' in rooms_text.lower():
                        rooms_val = 0
                    else:
                        rooms_match = re.search(r'(\d+)\s*[-]?комн', rooms_text)
                        if rooms_match:
                            rooms_val = int(rooms_match.group(1))

                # Извлекаем адрес
                address_elem = card.find('div', {'data-name': 'AddressContainer'})
                if not address_elem:
                    address_elem = card.find('a', {'data-name': 'Link'})  # Иногда адрес в ссылке

                address = address_elem.get_text().strip() if address_elem else ''

                # Извлекаем ссылку
                link_elem = card.find('a', {'data-name': 'Link'})
                if not link_elem:
                    link_elem = card.find('a', href=re.compile(r'cian.ru/rent/flat|cian.ru/sale/flat'))

                url = link_elem['href'] if link_elem and link_elem.has_attr('href') else ''
                if url and not url.startswith('http'):
                    url = 'https://www.cian.ru' + url

                # Извлекаем этаж
                floor_elem = card.find('div', string=re.compile(r'этаж'))
                floor_info = ''
                if floor_elem:
                    floor_info = floor_elem.get_text().strip()

                analog_data = {
                    'price': price,
                    'area_total': area_val,
                    'rooms': rooms_val,
                    'address': address,
                    'url': url,
                    'floor_info': floor_info,
                    'deal_type': deal_type
                }

                # Валидация минимальных требований
                if price > 0 and area_val and area_val > 10:  # Минимальная площадь 10 м²
                    analogs.append(analog_data)

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
    Парсит аналоги с CIAN для аренды или продажи
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
    }

    # Получаем ID города через cianparser
    import cianparser
    valid_locations = cianparser.list_locations()
    city_id = get_city_id(location, valid_locations)

    if not city_id:
        print(f"Не найден ID для города {location}")
        return []

    print(f"Парсим город {location} (ID: {city_id}) для {deal_type}")

    all_analogs = []
    base_url = "https://www.cian.ru/cat.php"

    # Настройки парсинга в зависимости от типа сделки
    max_pages = 2 if deal_type == 'rent' else 3  # Меньше страниц для аренды
    end_page = min(end_page, max_pages)

    for page in range(start_page, end_page + 1):
        print(f"Парсим страницу {page} для {deal_type}...")

        # Базовые параметры
        params = {
            'deal_type': deal_type,
            'engine_version': 2,
            'offer_type': 'flat',
            'region': city_id,
            'p': page,
        }

        # Параметры комнат в зависимости от типа
        if rooms == 0:  # Студия
            params['room9'] = 1  # Студии
        elif rooms >= 1 and rooms <= 3:
            params[f'room{rooms}'] = 1
        else:  # 4+ комнаты
            params['room4'] = 1

        # Дополнительные параметры для уточнения поиска
        params['mintarea'] = max(area * 0.7, 20)  # Минимальная площадь -70%
        params['maxtarea'] = area * 1.3  # Максимальная площадь +30%

        try:
            analogs = parse_cian_page(base_url, params, headers, deal_type)
            all_analogs.extend(analogs)
            print(f"Найдено на странице {page}: {len(analogs)} объявлений")

            # Задержка между запросами (больше для аренды)
            delay = 3 if deal_type == 'rent' else 2
            time.sleep(delay)

            # Критерии остановки парсинга
            if len(analogs) < 5 and page > start_page:
                print(f"На странице {page} мало объявлений, прекращаем парсинг")
                break

        except Exception as e:
            print(f"Ошибка при парсинге страницы {page}: {e}")
            continue

    print(f"Всего найдено объявлений для {deal_type}: {len(all_analogs)}")

    # Фильтрация и сортировка результатов
    if not all_analogs:
        return []

    # Фильтруем по площади (±30%)
    area_tol = area * 0.3
    filtered_analogs = [
        flat for flat in all_analogs
        if flat['area_total'] and abs(flat['area_total'] - area) <= area_tol
    ]

    print(f"После фильтрации по площади: {len(filtered_analogs)} объявлений")

    # Сортировка по похожести (более строгая для аренды)
    def similarity(flat):
        flat_area = flat.get('area_total', 0)
        flat_rooms = flat.get('rooms', -1)

        # Весовые коэффициенты в зависимости от типа сделки
        if deal_type == 'rent':
            area_weight = 2.0  # Больший вес площади для аренды
            rooms_weight = 1.5
        else:
            area_weight = 1.0
            rooms_weight = 1.0

        area_diff = abs(flat_area - area) * area_weight if flat_area else 1000
        rooms_diff = abs(flat_rooms - rooms) * 10 * rooms_weight if flat_rooms != -1 else 500

        return area_diff + rooms_diff

    filtered_analogs.sort(key=similarity)

    # Добавляем информацию о городе и регионе
    for flat in filtered_analogs:
        flat['city'] = location
        flat['region'] = get_region_from_city(location)
        # Добавляем форматированную цену для отображения
        flat['price_formatted'] = f"{flat['price']:,.0f}".replace(',', ' ')

    # Возвращаем наиболее похожие аналоги (больше для аренды)
    max_results = 10 if deal_type == 'rent' else 7
    return filtered_analogs[:max_results]


def test_parsing():
    """Тестирование парсера для обоих типов сделок"""
    test_cases = [
        {
            'location': 'Москва',
            'deal_type': 'sale',
            'rooms': 2,
            'area': 55.0,
            'description': 'Продажа 2-комнатной в Москве'
        },
        {
            'location': 'Москва',
            'deal_type': 'rent',
            'rooms': 2,
            'area': 55.0,
            'description': 'Аренда 2-комнатной в Москве'
        },
        {
            'location': 'Санкт-Петербург',
            'deal_type': 'rent',
            'rooms': 1,
            'area': 40.0,
            'description': 'Аренда 1-комнатной в СПб'
        }
    ]

    for test_case in test_cases:
        print(f"\n=== Тест: {test_case['description']} ===")
        try:
            analogs = get_cian_analogs(
                location=test_case['location'],
                deal_type=test_case['deal_type'],
                rooms=test_case['rooms'],
                area=test_case['area'],
                start_page=1,
                end_page=1
            )

            print(f"Найдено: {len(analogs)} объявлений")
            for i, flat in enumerate(analogs[:3]):  # Покажем первые 3
                price_type = "руб./мес" if test_case['deal_type'] == 'rent' else "руб."
                print(f"  {i + 1}. {flat.get('price_formatted', 'N/A')} {price_type} | "
                      f"{flat.get('area_total', 'N/A')} м² | "
                      f"{flat.get('rooms', 'N/A')} комн. | {flat.get('address', '')[:50]}...")

        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    # Тестируем парсинг для обоих типов сделок
    test_parsing()

    # Дополнительный тест с сохранением в CSV
    print("\n=== Детальный тест аренды ===")
    analogs = get_cian_analogs(
        location="Москва",
        deal_type="rent",
        rooms=2,
        area=55,
        start_page=1,
        end_page=1
    )

    # Выводим в консоль
    for flat in analogs:
        price_type = "руб./мес" if flat.get('deal_type') == 'rent' else "руб."
        print(f"{flat.get('city', 'N/A')} | {flat.get('price_formatted', 'N/A')} {price_type} | "
              f"{flat.get('area_total', 'N/A')} м² | {flat.get('rooms', 'N/A')} комн. | "
              f"{flat.get('address', '')}")