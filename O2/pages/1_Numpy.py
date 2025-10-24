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
    **Ключевые возможности:**
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
    st.write("**x:**", x)
    st.write("**y = x² + 2x + 1:**", y)

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', color='royalblue')
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
    ax.hist(data, bins=30, color='mediumseagreen', alpha=0.7)
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

# -------------------- СВЯЗЬ NUMPY И MATPLOTLIB --------------------
if menu == "Numpy и Matplotlib: связь технологий":
    st.markdown("""
    #### 🔗 NumPy и Matplotlib: связь технологий

    **Matplotlib** тесно интегрирован с **NumPy**, поскольку большинство графиков  
    строится на основе массивов `numpy.ndarray`.

    Например:
    - NumPy создаёт данные: `x = np.linspace(0, 10, 100)`
    - Matplotlib строит визуализацию: `plt.plot(x, np.sin(x))`

    Вместе они образуют **фундамент визуального анализа данных** в Python.
    """)

    st.divider()
    st.markdown("#### 📈 Пример: синусоида с затухающей амплитудой")

    x = np.linspace(0, 10, 400)
    y = np.sin(x) * np.exp(-x / 3)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='crimson', linewidth=2)
    ax.set_title("Пример взаимодействия NumPy и Matplotlib")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    st.success("NumPy формирует данные, а Matplotlib превращает их в наглядные образы 🔍")
