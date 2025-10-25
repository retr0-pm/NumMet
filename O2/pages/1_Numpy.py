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
    """Возвращает компактное представление массива."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return "[]"

    if arr.ndim == 1:
        if arr.size <= max_items:
            return np.array2string(arr, precision=precision, separator=', ')
        else:
            head = np.array2string(arr[: max_items // 2], precision=precision, separator=', ')
            tail = np.array2string(arr[-(max_items // 2) :], precision=precision, separator=', ')
            head = head.strip('[]')
            tail = tail.strip('[]')
            return f"[{head}, ..., {tail}]"
    return np.array2string(arr, precision=precision, max_line_width=80, threshold=6)


# -------------------- ОБЩАЯ ХАРАКТЕРИСТИКА --------------------
if menu == "Общая характеристика":
    st.markdown("""
    #### 🔢 Общая характеристика

    **NumPy (Numerical Python)** — фундаментальная библиотека Python для **численных вычислений** и работы с **многомерными массивами**.

    Основные цели:
    * Быстрая обработка больших массивов чисел
    * Векторные и матричные операции
    * Поддержка широкого спектра **математических функций**
    * Основа для **Pandas, SciPy, Matplotlib** и др.
    """)
    st.info("NumPy — сердце научных вычислений в Python 💡")
    st.markdown("**Главная структура данных:** `ndarray` — n-мерный массив, работающий **в десятки раз быстрее**, чем списки Python.")


# -------------------- ОСНОВНЫЕ ВОЗМОЖНОСТИ --------------------
if menu == "Основные возможности":
    st.markdown("""
    #### ⚙️ Основные возможности NumPy

    NumPy предоставляет набор инструментов для математических и статистических операций:
    * Создание массивов (`np.array`, `np.arange`, `np.linspace`)
    * Векторизация математических операций
    * Линейная алгебра (`np.dot`, `np.linalg.inv`, `np.linalg.eig`)
    * Работа со случайными числами (`np.random`)
    * Индексация, срезы, трансформация форм
    * Интеграция с библиотеками **C, C++ и Fortran**
    """)

    st.divider()
    st.markdown("#### 🧮 Пример векторных вычислений")

    x = np.linspace(0, 10, 6)
    y = x ** 2 + 2 * x + 1
    st.markdown(f"**x:** `{format_array_compact(x)}`")
    st.markdown(f"**y = x² + 2x + 1:** `{format_array_compact(y)}`")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, y, marker='o', linewidth=1)
    ax.set_title("Парабола: $y = x^2 + 2x + 1$", pad=12)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig, use_container_width=False)


# -------------------- ПРИМЕРЫ --------------------
if menu == "Примеры":
    st.markdown("""
    #### 📊 Примеры использования NumPy
    """)

    st.markdown("##### ▶️ Генерация случайных данных")
    code_random = """import numpy as np
data = np.random.randn(1000)
mean = np.mean(data)
std = np.std(data)
print("Среднее:", mean)
print("Стандартное отклонение:", std)"""
    st.code(code_random, language="python")

    data = np.random.randn(1000)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(data, bins=30, alpha=0.7)
    ax.set_title("Гистограмма случайных данных", pad=12)
    ax.set_xlabel("Значения")
    ax.set_ylabel("Частота")
    st.pyplot(fig, use_container_width=False)

    st.markdown("##### ▶️ Линейная алгебра")
    code_linalg = """A = np.array([[2, 1], [1, 3]])
b = np.array([8, 18])
x = np.linalg.solve(A, b)
print("Решение системы:", x)"""
    st.code(code_linalg, language="python")

    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([8.0, 18.0])
    x = np.linalg.solve(A, b)
    st.markdown("**Решение системы A x = b:**")
    st.markdown(f"`x = {format_array_compact(x, precision=6)}`")
    st.markdown("**Проверка (A @ x):**")
    st.markdown(f"`A @ x = {format_array_compact(A @ x, precision=6)}`")


# -------------------- СВЯЗЬ NUMPY И MATPLOTLIB --------------------
if menu == "Numpy и Matplotlib: связь технологий":
    st.markdown("""
    #### 🔗 NumPy и Matplotlib

    Matplotlib тесно интегрирован с NumPy. Большинство графиков строится по массивам `numpy.ndarray`.
    """)

    st.divider()
    st.markdown("#### 📈 Демонстрация: синусоида с затухающей амплитудой")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("##### ⚙️ Параметры графика")
        amplitude = st.slider("Амплитуда", 0.1, 2.0, 1.0, 0.1)
        frequency = st.slider("Частота", 0.5, 5.0, 1.0, 0.1)
        decay = st.slider("Затухание", 0.0, 1.0, 0.3, 0.05)
        st.markdown("Измени параметры, чтобы увидеть, как меняется форма сигнала 👉")

    with col2:
        x = np.linspace(0, 10, 400)
        y = amplitude * np.sin(frequency * x) * np.exp(-decay * x)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, y, linewidth=2)
        ax.set_title("Синусоида с затухающей амплитудой", pad=12)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid(True, linestyle='--', alpha=0.5)

        ax.set_xlim(0, 10)
        ax.set_ylim(-2, 2)

        st.pyplot(fig, use_container_width=False)

    st.success("Оси зафиксированы, чтобы форма сигнала менялась наглядно 🔍")
