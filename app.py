from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json
from city_region_mapper import CityRegionMapper

app = Flask(__name__)
CORS(app)

# Загружаем модель и маппер
try:
    model = joblib.load('model_with_city.joblib')
    print("Модель успешно загружена")
except FileNotFoundError:
    print("Ошибка: модель 'model_with_city.joblib' не найдена")
    model = None

city_mapper = CityRegionMapper('region_to_cities.json')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/regions', methods=['GET'])
def get_regions():
    """Получить список всех регионов"""
    try:
        regions = list(city_mapper.region_to_cities.keys())
        return jsonify(regions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Получить города для выбранного региона"""
    try:
        region = request.args.get('region', '')
        if not region:
            return jsonify({'error': 'Не указан регион'}), 400

        cities = city_mapper.get_cities_by_region(region)
        return jsonify(cities)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_price():
    """Предсказание цены с обязательными регионом и городом"""
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Модель не загружена'}), 500

        data = request.get_json()

        # Получаем оба ОБЯЗАТЕЛЬНЫХ параметра
        city = data.get('city', '').strip()
        region = data.get('region', '').strip()

        # Валидация: оба параметра обязательны
        if not city:
            return jsonify({'success': False, 'error': 'Укажите город'}), 400
        if not region:
            return jsonify({'success': False, 'error': 'Укажите регион'}), 400

        # Проверяем, что город принадлежит региону
        if not city_mapper.validate_city_in_region(city, region):
            expected_region = city_mapper.get_region_from_city(city)
            return jsonify({
                'success': False,
                'error': f'Город "{city}" не принадлежит региону "{region}". Правильный регион: "{expected_region}"'
            }), 400

        # Получаем остальные параметры
        building_type = int(data.get('building_type', 1))
        object_type = int(data.get('object_type', 1))
        level = int(data.get('level', 1))
        levels = int(data.get('levels', 5))
        rooms = int(data.get('rooms', 1))
        area = float(data.get('area', 50))
        kitchen_area = float(data.get('kitchen_area', area * 0.2))

        # Создаем engineered features
        room_size = area / (0.5 if rooms == 0 else rooms)
        floor_ratio = level / max(levels, 1)
        kitchen_ratio = kitchen_area / area

        # Подготовка данных для модели
        features = pd.DataFrame([{
            'region': region,
            'city': city,
            'building_type': building_type,
            'object_type': object_type,
            'level': level,
            'levels': levels,
            'rooms': rooms,
            'area': area,
            'kitchen_area': kitchen_area,
            'room_size': room_size,
            'floor_ratio': floor_ratio,
            'kitchen_ratio': kitchen_ratio
        }])

        # Прогноз
        price = model.predict(features)[0]

        return jsonify({
            'success': True,
            'price': float(price),
            'price_formatted': f"{float(price):,.0f}",
            'region': region,
            'city': city,
            'area': area,
            'rooms': rooms,
            'message': 'Прогноз выполнен с использованием региона и города'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка работоспособности сервера"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'regions_count': len(city_mapper.region_to_cities),
        'message': 'Сервер работает'
    })


if __name__ == '__main__':
    print(f"Загружено регионов: {len(city_mapper.region_to_cities)}")
    app.run(debug=True, host='0.0.0.0', port=5000)