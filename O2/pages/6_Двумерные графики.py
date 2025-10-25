import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Двумерные графики", layout="wide")

menu = st.sidebar.radio("***",
    (
        "Общее определение принадлежности графика к двумерным",
        "Псевдоцветные графики",
        "Контурные графики",
        "Векторные поля",
        "Диаграммы рассеяния 2D",
        "Гистограммы 2D",
    )
)

# -------------------- ОБЩЕЕ ОПРЕДЕЛЕНИЕ --------------------
if menu == "Общее определение принадлежности графика к двумерным":
    st.markdown("""
    #### 📘 Общее определение принадлежности графика к двумерным

    * Двумерные графики используются для отображения функций **двух переменных**, то есть зависимостей вида:
      \[
      z = f(x, y)
      \]
    * Такие визуализации позволяют анализировать распределения, экстремумы и изменения функции на всей плоскости.
    * В `Matplotlib` двумерные зависимости могут быть показаны через **цвет**, **контуры**, **векторы** или **плотность точек**.
    * Типовые функции визуализации: `imshow`, `pcolormesh`, `contour`, `quiver`, `hist2d`.
    """)

    st.markdown("#### Демонстрация функции z = sin(x) * cos(y):")

    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    fig, ax = plt.subplots(figsize=(3, 2))
    im = ax.imshow(Z, extent=[-3, 3, -3, 3], origin="lower", cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Пример двумерной зависимости z = sin(x)*cos(y)", pad=15)
    plt.colorbar(im, ax=ax, label="z")
    st.pyplot(fig, use_container_width=False)

# -------------------- ПСЕВДОЦВЕТНЫЕ ГРАФИКИ --------------------
if menu == "Псевдоцветные графики":
    st.markdown("""
    #### 🌈 Псевдоцветные графики

    * Отображают значения функции *z = f(x, y)* через **цвет** в каждой точке плоскости.
    * Применяются для представления температурных полей, плотностей, интенсивностей сигналов.
    * Основные функции: `imshow()`, `pcolormesh()`, `matshow()`.
    """)

    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X**2 + Y**2)

    fig, ax = plt.subplots(figsize=(3, 2))
    pcm = ax.pcolormesh(X, Y, Z, cmap='plasma', shading='auto')
    plt.colorbar(pcm, ax=ax, label="z = sin(x² + y²)")
    ax.set_title("Псевдоцветное представление функции", pad=15)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    st.pyplot(fig, use_container_width=False)

# -------------------- КОНТУРНЫЕ ГРАФИКИ --------------------
if menu == "Контурные графики":
    st.markdown("""
    #### 🌀 Контурные графики

    * Отображают **линии равных значений** функции *f(x, y)*.
    * Аналог изолиний на географических картах.
    * Помогают выявить экстремумы и форму поверхности.
    * В `Matplotlib` используются функции `contour()` и `contourf()`.
    """)

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    fig, ax = plt.subplots(figsize=(3, 2))
    cs = ax.contourf(X, Y, Z, cmap='coolwarm', levels=20)
    plt.colorbar(cs, ax=ax, label="z")
    ax.set_title("Контурное отображение z = sin(x) * cos(y)", pad=15)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    st.pyplot(fig, use_container_width=False)

# -------------------- ВЕКТОРНЫЕ ПОЛЯ --------------------
if menu == "Векторные поля":
    st.markdown("""
    #### 🧭 Векторные поля

    * Используются для отображения распределения направленных величин (например, скорости или силы).
    * В каждой точке (x, y) задаётся вектор с компонентами (u, v).
    * В `Matplotlib` такие поля строятся функцией `quiver()` или `streamplot()`.
    """)

    Y, X = np.mgrid[-3:3:20j, -3:3:20j]
    U = -1 - X**2 + Y
    V = 1 + X - Y**2

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.quiver(X, Y, U, V, color='teal')
    ax.set_title("Пример векторного поля", pad=15)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig, use_container_width=False)

# -------------------- ДИАГРАММЫ РАССЕЯНИЯ 2D --------------------
if menu == "Диаграммы рассеяния 2D":
    st.markdown("""
    #### 🔹 Диаграммы рассеяния 2D

    * Показывают множество точек, заданных парами координат (x, y).
    * Цвет или размер точки может отражать дополнительный параметр *z = f(x, y)*.
    * Используются для анализа распределений и закономерностей.
    """)

    np.random.seed(42)
    x = np.random.randn(300)
    y = np.random.randn(300)
    z = x**2 + y**2

    fig, ax = plt.subplots(figsize=(3, 2))
    sc = ax.scatter(x, y, c=z, cmap='viridis', alpha=0.8)
    plt.colorbar(sc, ax=ax, label="z = x² + y²")
    ax.set_title("2D диаграмма рассеяния", pad=15)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig, use_container_width=False)

# -------------------- ГИСТОГРАММЫ 2D --------------------
if menu == "Гистограммы 2D":
    st.markdown("""
    #### 📊 Гистограммы 2D

    * Отображают плотность распределения точек на плоскости (x, y).
    * Делят пространство на «ячейки» (бины), в которых считается количество точек.
    * В `Matplotlib` используются функции `hist2d()` и `hexbin()`.
    """)

    x = np.random.randn(2000)
    y = x * 0.5 + np.random.randn(2000) * 0.5

    fig, ax = plt.subplots(figsize=(3, 2))
    h = ax.hist2d(x, y, bins=40, cmap='inferno')

    # Цветовая шкала
    plt.colorbar(h[3], ax=ax, label='Частота')

    # Настройка внешнего вида
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D гистограмма распределения", pad=15)

    # Равномерное заполнение без пустых зон
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect('auto')
    fig.tight_layout()

    st.pyplot(fig, use_container_width=False)

