import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="NumPy", layout="wide")

menu = st.sidebar.radio("***",
    (
        "Общая характеристика",
        "Основные возможности",
        "Примеры",
        "Numpy и Matplotlib: связь технологий",
    )
)


# ---------- helper: компактный вывод массивов ----------
def format_array_compact(arr, precision=3, max_items=8):
    """Возвращает компактную строку представления массива:
    - ограничивает количество показанных элементов до max_items и добавляет '...'
    - задаёт точность вывода
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return "[]"

    # Если одномерный — показываем начало и конец при большом количестве элементов
    if arr.ndim == 1:
        if arr.size <= max_items:
            return np.array2string(arr, precision=precision, separator=', ')
        else:
            head = np.array2string(arr[: max_items // 2], precision=precision, separator=', ')
            tail = np.array2string(arr[-(max_items // 2) :], precision=precision, separator=', ')
            # убираем внешние скобки у head/tail
            head = head.strip('[]')
            tail = tail.strip('[]')
            return f"[{head}, ..., {tail}]"

    # Для многомерных массивов используем краткое представление
    return np.array2string(arr, precision=precision, max_line_width=80, threshold=6)


# -------------------- ОБЩАЯ ХАРАКТЕРИСТИКА --------------------
if menu == "Общая характеристика":
    st.markdown("""
    #### 🔢 Общая характеристика

    **NumPy (Numerical Python)** — это фундаментальная библиотека Python для **численных вычислений** и **работы с многомерными массивами**.

    Основные цели библиотеки:
    * Быстрая обработка больших объёмов числовых данных.
    * Удобная векторная и матричная арифметика.
    * Поддержка широкого спектра **математических функций**: от базовых до статистических и линейной алгебры.
    * Основа для большинства библиотек анализа данных: **Pandas, SciPy, scikit-learn, Matplotlib** и др.

    """)

    st.info("NumPy — это сердце научных вычислений в Python 💡")

    st.markdown("""
    **Главная структура данных** — `ndarray` (n-мерный массив).  
    Он похож на список, но работает **в десятки раз быстрее** благодаря реализации на C.
    """)

# -------------------- ОСНОВНЫЕ ВОЗМОЖНОСТИ --------------------
if menu == "Основные возможности":
    st.markdown("""
    #### ⚙️ Основные возможности NumPy

    **NumPy** предоставляет широкий набор инструментов для математических и статистических операций.
    """)

    st.markdown("""
    **Ключевые возможности:"
    """)

    st.markdown("""
    * Создание массивов (`np.array`, `np.arange`, `np.linspace`)
    * Математические операции над массивами без циклов (векторизация)
    * Линейная алгебра (`np.dot`, `np.linalg.inv`, `np.linalg.eig`)
    * Работа со случайными числами (`np.random`)
    * Удобные средства для **индексации, срезов и трансформации форм**
    * Эффективная интеграция с библиотеками **C, C++ и Fortran**
    """)

    st.divider()
    st.markdown("#### 🧮 Пример векторных вычислений")

    x = np.linspace(0, 10, 6)
    y = x ** 2 + 2 * x + 1

    # компактный вывод вместо развёрнутой таблицы
    st.markdown(f"**x:** `{format_array_compact(x)}`")
    st.markdown(f"**y = x² + 2x + 1:** `{format_array_compact(y)}`")

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', linewidth=1)
    ax.set_title("Парабола: y = x² + 2x + 1")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# -------------------- ПРИМЕРЫ --------------------
if menu == "Примеры":
    st.markdown("""
    #### 📊 Примеры использования NumPy

    NumPy делает работу с данными **компактной и выразительной**.
    Ниже — несколько типичных примеров.
    """)

    st.markdown("##### ▶️ Генерация случайных данных")
    code_random = """import numpy as np

data = np.random.randn(1000)  # 1000 случайных чисел из нормального распределения
mean = np.mean(data)
std = np.std(data)

print("Среднее:", mean)
print("Стандартное отклонение:", std)"""
    st.code(code_random, language="python")

    # Визуализация случайных данных
    data = np.random.randn(1000)
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, alpha=0.7)
    ax.set_title("Гистограмма случайных данных")
    ax.set_xlabel("Значения")
    ax.set_ylabel("Частота")
    st.pyplot(fig)

    st.markdown("##### ▶️ Линейная алгебра")
    code_linalg = """A = np.array([[2, 1], [1, 3]])
b = np.array([8, 18])
x = np.linalg.solve(A, b)
print("Решение системы:", x)"""
    st.code(code_linalg, language="python")

    # Реальный вывод результата и проверка
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([8.0, 18.0])
    x = np.linalg.solve(A, b)
    st.markdown("**Решение системы A x = b:**")
    st.markdown(f"`x = {format_array_compact(x, precision=6)}`")
    st.markdown("**Проверка (A @ x):**")
    st.markdown(f"`A @ x = {format_array_compact(A @ x, precision=6)}`  (должно совпадать с `b`)" )

# -------------------- СВЯЗЬ NUMPY И MATPLOTLIB --------------------
if menu == "Numpy и Matplotlib: связь технологий":
    st.markdown("""
    #### 🔗 NumPy и Matplotlib: связь технологий

    **Matplotlib** тесно интегрирован с **NumPy**, поскольку большинство графиков
    строится на основе массивов `numpy.ndarray` — неважно, создаются ли эти значения
    как `np.sin(x)`, как результат арифметики над массивами или как эмпирические данные.

    Пример:
    - NumPy создаёт данные: `x = np.linspace(0, 10, 100)`.
    - Вычисляемые с помощью NumPy значения (например, `y = np.sin(x)` или любая другая функция).
    - Matplotlib строит график по массивам `plt.plot(x, y)` — то есть Matplotlib отображает
      *массивы чисел*, а не "функции".

    Вместе они образуют **фундамент визуального анализа данных** в Python.
    """)

    st.divider()
    st.markdown("#### 📈 Интерактивная демонстрация: синусоида с затухающей амплитудой")

    # --- Колонки: настройки и график ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("##### ⚙️ Параметры графика")
        amplitude = st.slider("Амплитуда", 0.1, 2.0, 1.0, 0.1)
        frequency = st.slider("Частота", 0.5, 5.0, 1.0, 0.1)
        decay = st.slider("Затухание", 0.0, 1.0, 0.3, 0.05)
        st.markdown("Измени параметры, чтобы увидеть, как меняется форма сигнала 👉")

    with col2:
        # --- Данные ---
        x = np.linspace(0, 10, 400)
        y = amplitude * np.sin(frequency * x) * np.exp(-decay * x)

        # --- Построение графика ---
        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=2)
        ax.set_title("Синусоида с затухающей амплитудой")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle='--', alpha=0.5)

        # --- Фиксация диапазонов осей ---
        ax.set_xlim(0, 10)
        ax.set_ylim(-2, 2)  # диапазон амплитуды фиксирован для наглядности

        st.pyplot(fig)

    st.success("Оси зафиксированы, чтобы форма сигнала менялась наглядно 🔍")
