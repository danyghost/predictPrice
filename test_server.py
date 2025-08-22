import requests
import json

def test_server():
    base_url = "http://localhost:5000"
    
    try:
        # Тест health check
        print("1. Тестируем health check...")
        response = requests.get(f"{base_url}/api/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Тест получения регионов
        print("\n2. Тестируем получение регионов...")
        response = requests.get(f"{base_url}/api/regions")
        print(f"Status: {response.status_code}")
        regions = response.json()
        print(f"Количество регионов: {len(regions)}")
        print(f"Первые 5 регионов: {regions[:5]}")
        
        # Тест поиска регионов
        print("\n3. Тестируем поиск регионов (рост)...")
        response = requests.get(f"{base_url}/api/regions?q=рост")
        print(f"Status: {response.status_code}")
        filtered_regions = response.json()
        print(f"Найдено регионов с 'рост': {filtered_regions}")
        
        # Тест получения городов
        if regions:
            test_region = regions[0]
            print(f"\n4. Тестируем получение городов для региона '{test_region}'...")
            response = requests.get(f"{base_url}/api/cities?region={test_region}")
            print(f"Status: {response.status_code}")
            cities = response.json()
            print(f"Количество городов: {len(cities)}")
            print(f"Города: {cities[:5]}")
        
        # Тест предсказания
        print("\n5. Тестируем предсказание...")
        test_data = {
            "region": "Московская область",
            "city": "Москва",
            "area": 50,
            "rooms": 2
        }
        response = requests.post(f"{base_url}/api/predict", json=test_data)
        print(f"Status: {response.status_code}")
        prediction = response.json()
        print(f"Результат: {prediction}")
        
    except requests.exceptions.ConnectionError:
        print("Ошибка: Сервер не запущен. Запустите 'python app.py'")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    test_server()
