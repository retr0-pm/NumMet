import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from io import BytesIO
import base64
import tempfile
import os

st.set_page_config(page_title="Анимация в Matplotlib", layout="centered")

menu = st.sidebar.radio("***",
    (
    "Что такое анимация",
    "Пример живой анимации функции",
    "Анимация случайных данных",
    )
)

if menu == "Что такое анимация":
    st.markdown("""
    ##### 🎬 Что такое анимация в Matplotlib

    **Анимация** — это последовательное изменение данных или параметров графика во времени.  
    В `Matplotlib` за неё отвечает модуль `matplotlib.animation`.

    💡 Основные типы применения:
    * Визуализация изменения **функций во времени** — например, колебания, волны, осцилляции;
    * Отображение **движения точек** или **динамики процессов**;
    * Создание **презентаций и обучающих демонстраций**.

    Ниже показаны два реальных примера живой анимации, встроенной прямо в Streamlit.
    """)

# === Анимация синусоиды ===
if menu == "Пример живой анимации функции":
    st.markdown("""
    ##### 🌊 Пример живой анимации функции

    Синусоида "движется" вдоль оси X, демонстрируя принцип обновления данных по кадрам.
    """)

    x = np.linspace(0, 2 * np.pi, 200)
    fig, ax = plt.subplots()
    line, = ax.plot(x, np.sin(x), color="royalblue", lw=2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Анимация sin(x + t)")

    def update(frame):
        line.set_ydata(np.sin(x + frame / 5))
        return line,

    ani = animation.FuncAnimation(fig, update, frames=50, interval=60, blit=True)

    # --- Сохраняем во временный GIF-файл ---
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        temp_filename = tmpfile.name
    ani.save(temp_filename, writer="pillow", fps=20)

    # --- Читаем содержимое и удаляем файл ---
    with open(temp_filename, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    os.remove(temp_filename)

    st.markdown(f'<img src="data:image/gif;base64,{data}" alt="animation">', unsafe_allow_html=True)
    st.caption("Живая анимация синусоиды, созданная средствами Matplotlib")

# === Анимация случайных данных ===
if menu == "Анимация случайных данных":
    st.markdown("""
    ##### 🔄 Анимация случайных данных

    Точки на плоскости движутся случайным образом — имитация динамической системы.
    """)

    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=60, color="orange")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Случайное движение точек")

    def update_points(frame):
        scat.set_offsets(np.c_[np.random.rand(20) * 10, np.random.rand(20) * 10])
        return scat,

    ani = animation.FuncAnimation(fig, update_points, frames=40, interval=200, blit=True)

    # --- Сохраняем во временный GIF-файл ---
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        temp_filename = tmpfile.name
    ani.save(temp_filename, writer="pillow", fps=10)

    # --- Читаем содержимое и удаляем файл ---
    with open(temp_filename, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    os.remove(temp_filename)

    st.markdown(f'<img src="data:image/gif;base64,{data}" alt="animation">', unsafe_allow_html=True)
    st.caption("Анимация случайного движения точек — пример динамических данных.")
