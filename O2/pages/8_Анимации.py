import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import base64
import tempfile
import os

st.set_page_config(page_title="Анимация в Matplotlib", layout="centered")

menu = st.sidebar.radio("***",
    (
        "Что такое анимация",
        "Подключение и создание анимации",
        "Пример живой анимации функции",
        "Анимация случайных данных",
    )
)

# -------------------- ЧТО ТАКОЕ АНИМАЦИЯ --------------------
if menu == "Что такое анимация":
    st.markdown(r"""
    ##### 🎬 Что такое анимация в Matplotlib

    **Анимация** — это последовательное изменение данных или параметров графика во времени.  
    В `Matplotlib` за неё отвечает модуль `matplotlib.animation`.

    💡 Основные типы применения:
    * Визуализация изменения **функций во времени** — например, колебания, волны, осцилляции;
    * Отображение **движения точек** или **динамики процессов**;
    * Создание **презентаций и обучающих демонстраций**.

    С помощью анимации можно оживить графики, показывая динамику данных.
    """)

# -------------------- ПОДКЛЮЧЕНИЕ И СОЗДАНИЕ --------------------
if menu == "Подключение и создание анимации":
    st.markdown(r"""
    #### ⚙️ Подключение модуля и создание анимации

    1. Импортируем модуль анимации:
    ```python
    import matplotlib.animation as animation
    ```

    2. Создаём фигуру и линию для обновления:
    ```python
    fig, ax = plt.subplots()
    line, = ax.plot(x, y, color='royalblue', lw=2)
    ```

    3. Пишем функцию обновления данных для каждого кадра:
    ```python
    def update(frame):
        line.set_ydata(np.sin(x + frame / 5))
        return line,
    ```

    4. Создаём объект анимации:
    ```python
    ani = animation.FuncAnimation(fig, update, frames=50, interval=60, blit=True)
    ```

    🔹 **Параметры `FuncAnimation`:**
    * `fig` — фигура для анимации,
    * `update` — функция обновления,
    * `frames` — количество кадров,
    * `interval` — время между кадрами в мс,
    * `blit=True` — ускоряет обновление, перерисовывая только изменённые элементы.
    """)

# -------------------- АНИМАЦИЯ СИНУСОИДЫ --------------------
if menu == "Пример живой анимации функции":
    st.markdown(r"""
    ##### 🌊 Пример живой анимации функции

    Синусоида "движется" вдоль оси X, демонстрируя принцип обновления данных по кадрам.
    """)

    x = np.linspace(0, 2 * np.pi, 200)
    fig, ax = plt.subplots()
    line, = ax.plot(x, np.sin(x), color="royalblue", lw=2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\sin(x+t)$")
    ax.set_title(r"График $\sin(x + t)$", pad=15)

    def update(frame):
        line.set_ydata(np.sin(x + frame / 5))
        return line,

    ani = animation.FuncAnimation(fig, update, frames=50, interval=60, blit=True)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        temp_filename = tmpfile.name
    ani.save(temp_filename, writer="pillow", fps=20)
    with open(temp_filename, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    os.remove(temp_filename)

    st.markdown(f'<img src="data:image/gif;base64,{data}" alt="animation">', unsafe_allow_html=True)
    st.caption("Живая анимация синусоиды, созданная средствами Matplotlib")

# -------------------- АНИМАЦИЯ СЛУЧАЙНЫХ ДАННЫХ --------------------
if menu == "Анимация случайных данных":
    st.markdown("""
    ##### 🔄 Анимация случайных данных

    Точки на плоскости движутся случайным образом — имитация динамической системы.
    """)

    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=60, color="orange")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Случайное движение точек", pad=15)

    def update_points(frame):
        scat.set_offsets(np.c_[np.random.rand(20) * 10, np.random.rand(20) * 10])
        return scat,

    ani = animation.FuncAnimation(fig, update_points, frames=40, interval=200, blit=True)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        temp_filename = tmpfile.name
    ani.save(temp_filename, writer="pillow", fps=10)
    with open(temp_filename, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    os.remove(temp_filename)

    st.markdown(f'<img src="data:image/gif;base64,{data}" alt="animation">', unsafe_allow_html=True)
    st.caption("Анимация случайного движения точек — пример динамических данных.")
