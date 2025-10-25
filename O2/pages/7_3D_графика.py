import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # необходимо для 3D-графиков

st.set_page_config(page_title="3D графика в Matplotlib", layout="wide")

menu = st.sidebar.radio("***",
    (
    "Для чего нужна 3D графика в программировании",
    "Активация 3D",
    "Поверхности (3D surfaces)",
    "Сетки",
    "Точечные 3D графики",
    "Линейные 3D графики",
    "Оформление 3D сцен",
    )
)

# -------------------- ДЛЯ ЧЕГО НУЖНА 3D --------------------
if menu == "Для чего нужна 3D графика в программировании":
    st.markdown(r"""
    #### 🎯 Для чего нужна 3D графика в программировании

    * Трёхмерная визуализация используется, когда данные зависят **от двух независимых переменных**, а результат выражен как третья — \( z = f(x, y) \).
    * Такие графики позволяют:
        - анализировать формы поверхностей и экстремумы функции;
        - визуализировать физические поля (потенциалы, давления, температуры);
        - представлять результаты инженерных расчётов и моделирования;
        - демонстрировать многомерные зависимости в наглядном виде.
    * В `Matplotlib` трёхмерные графики строятся через модуль `mpl_toolkits.mplot3d`.
    """)

    # Демонстрация
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(r"$z = \sin(\sqrt{x^2 + y^2})$", pad=25)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    plt.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=r"$z$")
    st.pyplot(fig, use_container_width=False)

# -------------------- АКТИВАЦИЯ 3D --------------------
if menu == "Активация 3D":
    st.markdown(r"""
    #### 🧩 Активация 3D-графики

    * Для работы с 3D-графиками необходимо создать ось с параметром `projection='3d'`.
    * Это активирует модуль **mpl_toolkits.mplot3d**, который добавляет методы для построения 3D объектов:
    ```python
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ```
    * После этого можно использовать методы `plot_surface`, `scatter`, `plot_wireframe` и др.
    """)

    st.info("Ось с проекцией '3d' создаёт пространство для трёх координатных осей X, Y, Z.")

# -------------------- ПОВЕРХНОСТИ --------------------
if menu == "Поверхности (3D surfaces)":
    st.markdown(r"""
    #### 🏔 Поверхности (3D Surfaces)

    * Используются для отображения **непрерывных функций** двух переменных: \( z = f(x, y) \).
    * Цветовая карта (colormap) помогает визуализировать изменения по высоте.
    * Основная функция — `ax.plot_surface(X, Y, Z, cmap='...')`.
    """)

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(np.sqrt(X**2 + Y**2))

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')
    ax.set_title(r"$z = \cos(\sqrt{x^2 + y^2})$", pad=25)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    plt.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=r"$z$")
    st.pyplot(fig, use_container_width=False)

# -------------------- СЕТКИ --------------------
if menu == "Сетки":
    st.markdown(r"""
    #### 🕸 Сетки (Wireframes)

    * Сетки — облегчённый вариант поверхностей, показывающий только **каркас** графика.
    * Используются для структурного анализа формы функции.
    * Функция: `ax.plot_wireframe(X, Y, Z, color='...')`
    """)

    x = np.linspace(-5, 5, 40)
    y = np.linspace(-5, 5, 40)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, color='navy', linewidth=0.7)
    ax.set_title(r"$z = \sin(\sqrt{x^2 + y^2})$", pad=25)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    st.pyplot(fig, use_container_width=False)

# -------------------- ТОЧЕЧНЫЕ 3D --------------------
if menu == "Точечные 3D графики":
    st.markdown(r"""
    #### 💠 Точечные 3D графики

    * Отображают набор точек в трёхмерном пространстве.
    * Используются для визуализации облаков данных, кластеров, результатов моделирования.
    * Функция: `ax.scatter(x, y, z, ...)`.
    """)

    np.random.seed(0)
    x = np.random.randn(200)
    y = np.random.randn(200)
    z = np.random.randn(200)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=z, cmap='plasma', alpha=0.8)
    plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=r"$z$")
    ax.set_title(r"3D-точечный график: $(x, y, z)$", pad=25)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    st.pyplot(fig, use_container_width=False)

# -------------------- ЛИНЕЙНЫЕ 3D --------------------
if menu == "Линейные 3D графики":
    st.markdown(r"""
    #### 📈 Линейные 3D графики

    * Используются для отображения пространственных траекторий.
    * Каждая линия определяется последовательностью точек \((x, y, z)\).
    * Функция: `ax.plot3D(x, y, z, ...)`.
    """)

    theta = np.linspace(-4 * np.pi, 4 * np.pi, 200)
    z = np.linspace(-2, 2, 200)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(x, y, z, color='darkorange', linewidth=2)
    ax.set_title(r"Пространственная кривая: $r(z) = z^2 + 1$", pad=25)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    st.pyplot(fig, use_container_width=False)

# -------------------- ОФОРМЛЕНИЕ --------------------
if menu == "Оформление 3D сцен":
    st.markdown(r"""
    #### 🎨 Оформление 3D сцен

    * 3D-графики можно настраивать аналогично 2D:
        - Изменять углы обзора (`view_init`);
        - Устанавливать подписи и сетку;
        - Изменять масштаб и пропорции осей.
    * Углы обзора задаются параметрами `elev` (высота) и `azim` (азимут).
    """)

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.view_init(elev=40, azim=45)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.set_title(r"Настроенная сцена: $z = \sin(x)\cos(y)$", pad=25)
    st.pyplot(fig, use_container_width=False)
