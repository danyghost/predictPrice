// ---------- Конфигурация (скопирована из твоего main.js) ----------
const CONFIG = {
    buildingTypes: [
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
    'building-type-list': '1',
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

                if (containerId === 'rooms-list') {
                    updateKitchenLabel(chip.dataset.value);
                }
            } else {
                chip.classList.toggle('selected');
            }
        });

        if (selectedChips[containerId] == chip.dataset.value) {
            chip.classList.add('selected');
        }

        container.appendChild(chip);
    });

    if (containerId === 'rooms-list') {
        updateKitchenLabel(selectedChips[containerId]);
    }
}

function getSelected(containerId) {
    const sel = document.querySelector(`#${containerId} .chip.selected`);
    return sel ? sel.dataset.value : selectedChips[containerId] || null;
}

// Замена названия и управление валидацией, если выбрана Студия
function updateKitchenLabel(roomsValue) {
    const kitchenInput = document.getElementById('kitchen_area');
    const kitchenBlock = document.getElementById('kitchen-block');
    if (!kitchenInput || !kitchenBlock) return;

    console.log("Вызвана updateKitchenLabel. Значение комнат:", roomsValue);

    if (Number(roomsValue) === -1 || roomsValue === '-1' || roomsValue === 'Студия') {
        kitchenBlock.style.display = 'none';
        kitchenInput.value = '0';

        kitchenInput.removeAttribute('min');
        kitchenInput.removeAttribute('max');
        kitchenInput.removeAttribute('required');
    } else {
        kitchenBlock.style.display = 'block';

        if (kitchenInput.value === '0' || kitchenInput.value === 0) {
            kitchenInput.value = '';
        }

        kitchenInput.setAttribute('min', '3');
        kitchenInput.setAttribute('max', '50');
        kitchenInput.setAttribute('required', 'true');

        const kitchenLabel = kitchenBlock.querySelector('.filter-label');
        if (kitchenLabel) kitchenLabel.textContent = 'Кухня (м²)';
        kitchenInput.placeholder = 'Площадь кухни';
    }
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

        let locations = await response.json();

        // --- СОРТИРОВКА ПО РЕЛЕВАНТНОСТИ ---
        const q = query.toLowerCase();
        locations.sort((a, b) => {
            const nameA = a.name.toLowerCase();
            const nameB = b.name.toLowerCase();

            // Точное совпадение
            if (nameA === q && nameB !== q) return -1;
            if (nameB === q && nameA !== q) return 1;

            // Название начинается с искомого текста
            const startsA = nameA.startsWith(q);
            const startsB = nameB.startsWith(q);
            if (startsA && !startsB) return -1;
            if (startsB && !startsA) return 1;

            // Позиция подстроки (чем ближе совпадение к началу слова, тем выше)
            const indexA = nameA.indexOf(q);
            const indexB = nameB.indexOf(q);
            if (indexA !== indexB) return indexA - indexB;

            // Длина строки (более короткие и точные названия городов идут выше)
            if (nameA.length !== nameB.length) return nameA.length - nameB.length;

            // Алфавитный порядок (если все предыдущие параметры совпали)
            return nameA.localeCompare(nameB);
        });
        // ------------------------------------

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

    // Группировка по регионам с сохранением исходного порядка
    const grouped = {};
    const regionOrder = [];

    locations.forEach(loc => {
        if (loc.type === 'region') {
            if (!grouped[loc.value]) {
                grouped[loc.value] = { region: loc, cities: [] };
                regionOrder.push(loc.value);
            }
        } else {
            if (!grouped[loc.region]) {
                grouped[loc.region] = { region: { name: loc.region, value: loc.region }, cities: [] };
                regionOrder.push(loc.region);
            }
            grouped[loc.region].cities.push(loc);
        }
    });

    let html = '';

    regionOrder.forEach(regionName => {
        const group = grouped[regionName];

        if (!group.cities || group.cities.length === 0) {
            return;
        }

        html += `
            <div class="region-group">
                <div class="region-name">${group.region.name || regionName}</div>
                ${group.cities.map(city => `
                    <div class="city-item" data-type="city" data-name="${city.name}" data-region="${city.region}">
                        ${city.name}
                    </div>
                `).join('')}
            </div>
        `;
    });

    if (html === '') {
        elements.searchResults.innerHTML = '<div class="no-results">Ничего не найдено</div>';
        elements.searchResults.style.display = 'block';
        return;
    }

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

    // 1. Локация
    if (!selectedLocation.name || !selectedLocation.region) {
        showFieldError('error-location', 'Выберите город из списка');
        document.getElementById('location-search').classList.add('input-error');
        hasError = true;
    }

    // 2. Чипы
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

    // 3. Числовые поля (получаем сырые строковые значения)
    const levelRaw = document.getElementById('level').value.trim();
    const levelsRaw = document.getElementById('levels').value.trim();
    const areaRaw = document.getElementById('area').value.trim();
    const kitchenRaw = document.getElementById('kitchen_area').value.trim();

    // Проверка на заполненность
    if (!levelRaw) {
        showFieldError('error-level', 'Укажите этаж');
        document.getElementById('level').classList.add('input-error');
        hasError = true;
    }
    if (!levelsRaw) {
        showFieldError('error-levels', 'Укажите этажность');
        document.getElementById('levels').classList.add('input-error');
        hasError = true;
    }
    if (!areaRaw) {
        showFieldError('error-area', 'Укажите общую площадь');
        document.getElementById('area').classList.add('input-error');
        hasError = true;
    }
    if (!kitchenRaw) {
        showFieldError('error-kitchen_area', 'Укажите площадь кухни');
        document.getElementById('kitchen_area').classList.add('input-error');
        hasError = true;
    }

    // 4. Глубокая валидация чисел (если все поля заполнены)
    if (!hasError) {
        const level = parseInt(levelRaw, 10);
        const levels = parseInt(levelsRaw, 10);
        const area = parseFloat(areaRaw);
        const kitchen = parseFloat(kitchenRaw);

        // Проверка на корректность ввода чисел и положительные значения
        if (isNaN(level) || level <= 0) {
            showFieldError('error-level', 'Этаж должен быть положительным числом');
            document.getElementById('level').classList.add('input-error');
            hasError = true;
        }
        if (isNaN(levels) || levels <= 0) {
            showFieldError('error-levels', 'Этажность должна быть положительным числом');
            document.getElementById('levels').classList.add('input-error');
            hasError = true;
        }
        if (isNaN(area) || area <= 0) {
            showFieldError('error-area', 'Площадь должна быть больше 0');
            document.getElementById('area').classList.add('input-error');
            hasError = true;
        }

        const isStudio = Number(selectedChips['rooms-list']) === -1;

        if (!isStudio && (isNaN(kitchen) || kitchen <= 0)) {
            showFieldError('error-kitchen_area', 'Площадь кухни должна быть больше 0');
            document.getElementById('kitchen_area').classList.add('input-error');
            hasError = true;
        }

        // 5. Логические правила
        if (!hasError) {
            // Кухня физически не может быть больше или равна всей квартире
            if (kitchen >= area) {
                showFieldError('error-kitchen_area', 'Площадь кухни не может быть больше или равна общей площади квартиры');
                document.getElementById('kitchen_area').classList.add('input-error');
                hasError = true;
            }
            // Кухня не может занимать более 35% от общей площади квартиры
            else if (kitchen > area * 0.35) {
                showFieldError('error-kitchen_area', 'Площадь кухни не может превышать 35% от общей площади');
                document.getElementById('kitchen_area').classList.add('input-error');
                hasError = true;
            }
            else if (area - kitchen < 16) {
                showFieldError('error-kitchen_area', 'Остаточная жилая площадь слишком мала для полноценной квартиры');
                document.getElementById('kitchen_area').classList.add('input-error');
                hasError = true;
            }
        }
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

// ---------- Отображение результата ----------
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

    const roomsText = Number(data.rooms) === -1 ? 'Студия' : data.rooms;

    const detailsHtml = `
        <div><strong>Регион:</strong> ${data.region}</div>
        <div><strong>Город:</strong> ${data.city}</div>
        <div><strong>Площадь:</strong> ${data.area} м²</div>
        <div><strong>Комнат:</strong> ${roomsText}</div>
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