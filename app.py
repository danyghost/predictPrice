from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
from city_region_mapper import CityRegionMapper
from predict_price import predict_price_with_analogs

app = Flask(__name__)
CORS(app)

# Загружаем модели
try:
    sale_model_data = joblib.load('model/sale_model.joblib')
    sale_model = sale_model_data['model']
    sale_encoders = sale_model_data.get('encoders', {})
    print("Модель продажи успешно загружена")
except FileNotFoundError:
    print("Ошибка! Модель продажи не найдена")
    sale_model = None
    sale_encoders = {}

try:
    rent_model_data = joblib.load('model/rent_model.joblib')
    rent_model = rent_model_data['model']
    print("Модель аренды успешно загружена")
except FileNotFoundError:
    print("Ошибка! Модель аренды не найдена")
    rent_model = None

# Загружаем мэппер
city_mapper = CityRegionMapper('region_to_cities.json')


models = {
    'sale': sale_model,
    'rent': rent_model,
}

encoders = {
    'sale': sale_encoders
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Получение всех локаций с группировкой по регионам"""
    try:
        locations = []
        for region, cities in city_mapper.region_to_cities.items():
            locations.append({
                'type': 'region',
                'name': region,
                'value': region
            })
            for city in cities:
                locations.append({
                    'type': 'city',
                    'name': city,
                    'value': city,
                    'region': region,
                    'parent': region
                })
        return jsonify(locations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search-locations', methods=['GET'])
def search_locations():
    """Поиск локаций по названию"""
    try:
        query = request.args.get('q', '').lower().strip()
        if not query:
            return jsonify([])

        results = []
        for region, cities in city_mapper.region_to_cities.items():
            # Проверяем регион
            if query in region.lower():
                results.append({
                    'type': 'region',
                    'name': region,
                    'value': region
                })

            # Проверяем города
            for city in cities:
                if query in city.lower():
                    results.append({
                        'type': 'city',
                        'name': city,
                        'value': city,
                        'region': region,
                        'parent': region
                    })

        # Сортируем: сначала точные совпадения, потом частичные
        def sort_key(item):
            name = item['name'].lower()
            if name.startswith(query):
                return (0, name)
            return (1, name)

        results.sort(key=sort_key)
        return jsonify(results[:20])  # Ограничиваем результаты

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_price_endpoint():
    """Предсказание цены с учетом аналогов"""
    try:
        data = request.get_json()

        location = data.get('location', '').strip()
        deal_type = data.get('deal_type', 'sale')

        # Валидация: обязательные параметры
        if not location:
            return jsonify({'success': False, 'error': 'Укажите город или регион'}), 400
        if not deal_type:
            return jsonify({'success': False, 'error': 'Укажите тип сделки'}), 400

        # Проверяем наличие модели для выбранного типа сделки
        if deal_type not in models or models[deal_type] is None:
            return jsonify({
                'success': False,
                'error': f'Модель для типа сделки "{deal_type}" не загружена'
            }), 400

        # Определяем, что выбрал пользователь: город или регион
        if location in city_mapper.region_to_cities:
            # Пользователь выбрал регион - берем первый город из этого региона
            region = location
            cities = city_mapper.get_cities_by_region(region)
            if not cities:
                return jsonify({'success': False, 'error': f'В регионе "{region}" нет городов'}), 400
            city = cities[0]  # берем первый город региона
        else:
            # Пользователь выбрал город - определяем регион
            city = location
            region = city_mapper.get_region_from_city(city)
            if region == "Неизвестный регион":
                return jsonify({
                    'success': False,
                    'error': f'Город "{city}" не найден в базе. Проверьте правильность написания.'
                }), 400

        # Получаем остальные параметры
        building_type = data.get('building_type', '1')
        object_type = data.get('object_type', '1')
        level = int(data.get('level', 1))
        levels = int(data.get('levels', 5))
        rooms = int(data.get('rooms', 1))
        area = float(data.get('area', 50))
        kitchen_area = float(data.get('kitchen_area', area * 0.2))

        # Подготовка данных для модели
        input_data = {
            'region': region,
            'city': city,
            'building_type': building_type,
            'object_type': object_type,
            'level': level,
            'levels': levels,
            'rooms': rooms,
            'area': area,
            'kitchen_area': kitchen_area,
            'deal_type': deal_type
        }

        # Прогноз с аналогами - передаем модель и энкодеры
        final_price, ml_price, analogs = predict_price_with_analogs(
            input_data,
            model=models[deal_type],
            encoders=encoders.get(deal_type, {})
        )

        # Форматируем ответ в зависимости от типа сделки
        is_rent = deal_type == 'rent'
        price_suffix = "руб./мес" if is_rent else "руб."

        # Форматируем аналоги
        analogs_formatted = []
        for analog in analogs:
            analog_price = analog.get('price', 0)
            analogs_formatted.append({
                'price': analog_price,
                'price_formatted': f"{analog_price:,.0f} {price_suffix}",
                'area': analog.get('area_total'),
                'rooms': analog.get('rooms'),
                'address': analog.get('address', ''),
                'url': analog.get('url', ''),
                'floor_info': analog.get('floor_info', '')
            })

        return jsonify({
            'success': True,
            'price': float(final_price),
            'price_formatted': f"{float(final_price):,.0f} {price_suffix}",
            'ml_price': float(ml_price),
            'ml_price_formatted': f"{float(ml_price):,.0f} {price_suffix}",
            'is_rent': is_rent,
            'price_suffix': price_suffix,
            'region': region,
            'city': city,
            'area': area,
            'rooms': rooms,
            'analogs_count': len(analogs),
            'analogs': analogs_formatted,
            'message': 'Прогноз выполнен успешно'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка работоспособности сервера"""
    return jsonify({
        'status': 'ok',
        'sale_model_loaded': sale_model is not None,
        'rent_model_loaded': rent_model is not None,
        'regions_count': len(city_mapper.region_to_cities),
        'message': 'Сервер работает'
    })


if __name__ == '__main__':
    print(f"Загружено регионов: {len(city_mapper.region_to_cities)}")
    print(f"Модель продажи: {'загружена' if sale_model else 'не загружена'}")
    print(f"Модель аренды: {'загружена' if rent_model else 'не загружена'}")
    app.run(debug=True, host='0.0.0.0', port=5000)