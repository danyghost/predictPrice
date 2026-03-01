// ---------- Конфигурация (скопирована из твоего main.js) ----------
const CONFIG = {
    buildingTypes: [
        {label: "Иное", value: 0},
        {label: "Панельный", value: 1},
        {label: "Монолитный", value: 2},
        {label: "Кирпичный", value: 3},
        {label: "Блочный", value: 4},
        {label: "Деревянный", value: 5}
    ],
    objectTypes: [
        {label: "Вторичный рынок", value: 1},
        {label: "Новостройки", value: 2}
    ],
    rooms: [
        {label: "Студия", value: -1},
        {label: "1", value: 1},
        {label: "2", value: 2},
        {label: "3", value: 3},
        {label: "4", value: 4},
        {label: "5+", value: 5}
    ],
    dealTypes: [
        {label: "Покупка/Продажа", value: "sale"},
        {label: "Аренда", value: "rent"}
    ]
};

// ---------- Глобальные переменные ----------
let selectedLocation = { type: '', name: '', region: '' };
let selectedChips = {
    'deal-type-list': 'sale',
    'building-type-list': '0',
    'object-type-list': '1',
    'rooms-list': '1'
};

// ---------- DOM элементы ----------
const elements = {
    locationSearch: document.getElementById('location-search'),
    searchResults: document.getElementById('search-results'),
    selectedLocationInfo: document.getElementById('selected-location-info'),
    selectedLocationText: document.getElementById('selected-location-text'),
    resultBlock: document.getElementById('result'),
    predictButton: document.getElementById('predict-button'),
    form: document.getElementById('predict-form')
};

// ---------- Утилиты ----------
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// ---------- Работа с чипами ----------
function renderChips(list, containerId, single = true) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    list.forEach(item => {
        const chip = document.createElement('div');
        chip.className = 'chip';
        chip.textContent = item.label;
        chip.dataset.value = item.value;

        chip.addEventListener('click', () => {
            if (single) {
                container.querySelectorAll('.chip').forEach(c => c.classList.remove('selected'));
                chip.classList.add('selected');
                selectedChips[containerId] = chip.dataset.value;
            } else {
                chip.classList.toggle('selected');
            }
        });

        if (selectedChips[containerId] == chip.dataset.value) {
            chip.classList.add('selected');
        }

        container.appendChild(chip);
    });
}

function getSelected(containerId) {
    const sel = document.querySelector(`#${containerId} .chip.selected`);
    return sel ? sel.dataset.value : selectedChips[containerId] || null;
}

// ---------- Поиск локаций (использует /api/search-locations) ----------
async function searchLocations(query) {
    if (query.length < 2) {
        elements.searchResults.style.display = 'none';
        return;
    }

    try {
        const response = await fetch(`/api/search-locations?q=${encodeURIComponent(query)}`);
        if (!response.ok) throw new Error('Ошибка поиска');
        const locations = await response.json();
        renderSearchResults(locations);
    } catch (error) {
        console.error('Ошибка поиска:', error);
        elements.searchResults.innerHTML = '<div class="no-results">Ошибка загрузки</div>';
        elements.searchResults.style.display = 'block';
    }
}

function renderSearchResults(locations) {
    if (!locations || locations.length === 0) {
        elements.searchResults.innerHTML = '<div class="no-results">Ничего не найдено</div>';
        elements.searchResults.style.display = 'block';
        return;
    }

    // Группировка по регионам (как в твоём main.js)
    const grouped = {};
    locations.forEach(loc => {
        if (loc.type === 'region') {
            if (!grouped[loc.value]) grouped[loc.value] = { region: loc, cities: [] };
        } else {
            if (!grouped[loc.region]) grouped[loc.region] = { region: { name: loc.region, value: loc.region }, cities: [] };
            grouped[loc.region].cities.push(loc);
        }
    });

    const sortedRegions = Object.keys(grouped).sort((a, b) => a.localeCompare(b, 'ru'));

    let html = '';
    sortedRegions.forEach(regionName => {
        const group = grouped[regionName];
        const sortedCities = group.cities.sort((a, b) => a.name.localeCompare(b.name, 'ru'));
        html += `
            <div class="region-group">
                <div class="region-name">${group.region.name || regionName}</div>
                ${sortedCities.map(city => `
                    <div class="city-item" data-type="city" data-name="${city.name}" data-region="${city.region}">
                        ${city.name}
                    </div>
                `).join('')}
            </div>
        `;
    });

    elements.searchResults.innerHTML = html;
    elements.searchResults.style.display = 'block';

    // Обработчики для городов
    document.querySelectorAll('.city-item').forEach(item => {
        item.addEventListener('click', () => {
            const name = item.dataset.name;
            const region = item.dataset.region;
            selectedLocation = { type: 'city', name, region };
            elements.locationSearch.value = name;
            elements.selectedLocationText.textContent = `${name} (${region})`;
            elements.selectedLocationInfo.style.display = 'block';
            elements.searchResults.style.display = 'none';
        });
    });
}

// ---------- Валидация формы ----------
function clearFieldErrors() {
    document.querySelectorAll('.field-error').forEach(e => e.textContent = '');
    document.querySelectorAll('.input-error').forEach(e => e.classList.remove('input-error'));
    document.querySelectorAll('.chip-list').forEach(e => e.classList.remove('chip-error'));
}

function showFieldError(errorId, message) {
    document.getElementById(errorId).textContent = message;
}

function showChipError(errorId, message, chipListId) {
    document.getElementById(errorId).textContent = message;
    document.getElementById(chipListId).classList.add('chip-error');
}

function validateForm() {
    let hasError = false;
    clearFieldErrors();

    // Локация
    if (!selectedLocation.name || !selectedLocation.region) {
        showFieldError('error-location', 'Выберите город из списка');
        document.getElementById('location-search').classList.add('input-error');
        hasError = true;
    }

    // Чипы
    const dealType = getSelected('deal-type-list');
    const buildingType = getSelected('building-type-list');
    const objectType = getSelected('object-type-list');
    const rooms = getSelected('rooms-list');

    if (!dealType) {
        showChipError('error-deal-type', 'Выберите тип сделки', 'deal-type-list');
        hasError = true;
    }
    if (!buildingType && buildingType !== 0) {
        showChipError('error-building-type', 'Выберите тип здания', 'building-type-list');
        hasError = true;
    }
    if (!objectType) {
        showChipError('error-object-type', 'Выберите тип объекта', 'object-type-list');
        hasError = true;
    }
    if (!rooms && rooms !== 0) {
        showChipError('error-rooms', 'Выберите количество комнат', 'rooms-list');
        hasError = true;
    }

    // Числовые поля
    const level = document.getElementById('level').value;
    const levels = document.getElementById('levels').value;
    const area = document.getElementById('area').value;
    const kitchen = document.getElementById('kitchen_area').value;

    if (!level) {
        showFieldError('error-level', 'Укажите этаж');
        document.getElementById('level').classList.add('input-error');
        hasError = true;
    }
    if (!levels) {
        showFieldError('error-levels', 'Укажите этажность');
        document.getElementById('levels').classList.add('input-error');
        hasError = true;
    }
    if (!area) {
        showFieldError('error-area', 'Укажите общую площадь');
        document.getElementById('area').classList.add('input-error');
        hasError = true;
    }
    if (!kitchen) {
        showFieldError('error-kitchen_area', 'Укажите площадь кухни');
        document.getElementById('kitchen_area').classList.add('input-error');
        hasError = true;
    }

    return !hasError;
}

// ---------- Отправка формы (POST /api/predict) ----------
async function handleSubmit(e) {
    e.preventDefault();
    if (!validateForm()) return;

    // Скрываем старый результат и показываем загрузку
    elements.resultBlock.style.display = 'none';
    elements.resultBlock.innerHTML = '<div class="result-block" style="text-align:center;">Загрузка...</div>';
    elements.resultBlock.style.display = 'block';

    const dealType = getSelected('deal-type-list');
    const buildingType = getSelected('building-type-list');
    const objectType = getSelected('object-type-list');
    const rooms = getSelected('rooms-list');
    const level = document.getElementById('level').value;
    const levels = document.getElementById('levels').value;
    const area = document.getElementById('area').value;
    const kitchen = document.getElementById('kitchen_area').value;

    const payload = {
        location: selectedLocation.name,
        deal_type: dealType,
        building_type: buildingType,
        object_type: objectType,
        rooms: Number(rooms),
        level: Number(level),
        levels: Number(levels),
        area: Number(area),
        kitchen_area: Number(kitchen)
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();

        if (response.ok && data.success) {
            renderResult(data);
        } else {
            showError(data.error || 'Ошибка при расчёте');
        }
    } catch (err) {
        console.error(err);
        showError('Ошибка соединения с сервером');
    }
}

// ---------- Отображение результата (с учётом аналогов) ----------
function renderResult(data) {
    const isRent = data.is_rent || false;
    const priceSuffix = isRent ? '₽/мес' : '₽';

    let analogsHtml = '';
    if (data.analogs && data.analogs.length > 0) {
        analogsHtml = '<div class="analogs-title">Похожие объявления</div>';
        data.analogs.forEach(analog => {
            analogsHtml += `
                <div class="analog-card">
                    <div class="analog-price">${analog.price_formatted || analog.price.toLocaleString('ru-RU') + ' ' + priceSuffix}</div>
                    <div class="analog-address">${analog.address || 'Адрес не указан'}</div>
                    ${analog.url ? `<a href="${analog.url}" target="_blank" class="analog-link">Перейти к объявлению →</a>` : ''}
                </div>
            `;
        });
    }

    const detailsHtml = `
        <div><strong>Регион:</strong> ${data.region}</div>
        <div><strong>Город:</strong> ${data.city}</div>
        <div><strong>Площадь:</strong> ${data.area} м²</div>
        <div><strong>Комнат:</strong> ${data.rooms}</div>
        ${!isRent ? `<div><strong>Цена за м²:</strong> ${Math.round(data.price / data.area).toLocaleString('ru-RU')} ₽</div>` : ''}
    `;

    elements.resultBlock.innerHTML = `
        <div class="result-block">
            <div class="result-price">${data.price_formatted}</div>
            <div class="result-desc">${isRent ? 'Ориентировочная стоимость аренды' : 'Ориентировочная стоимость'}</div>
            <div class="result-details">${detailsHtml}</div>
            ${analogsHtml}
        </div>
    `;
    elements.resultBlock.style.display = 'block';
    elements.resultBlock.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function showError(message) {
    elements.resultBlock.innerHTML = `<div class="result-block" style="text-align:center; color:#e74c3c;">${message}</div>`;
    elements.resultBlock.style.display = 'block';
}

// ---------- Инициализация ----------
function init() {
    // Рендерим все чипы
    renderChips(CONFIG.buildingTypes, 'building-type-list');
    renderChips(CONFIG.objectTypes, 'object-type-list');
    renderChips(CONFIG.rooms, 'rooms-list');
    renderChips(CONFIG.dealTypes, 'deal-type-list');

    // Поиск локаций
    elements.locationSearch.addEventListener('input', debounce((e) => {
        searchLocations(e.target.value.trim());
    }, 300));

    elements.locationSearch.addEventListener('focus', () => {
        if (elements.locationSearch.value.trim().length >= 2) {
            searchLocations(elements.locationSearch.value.trim());
        }
    });

    // Закрытие подсказок при клике вне
    document.addEventListener('click', (e) => {
        if (!elements.locationSearch.contains(e.target) && !elements.searchResults.contains(e.target)) {
            elements.searchResults.style.display = 'none';
        }
    });

    // Сабмит формы
    elements.form.addEventListener('submit', handleSubmit);
}

document.addEventListener('DOMContentLoaded', init);