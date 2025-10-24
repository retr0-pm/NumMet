import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Элементы оформления", layout="wide")

menu = st.sidebar.radio("***",
    (
        "Что такое элементы оформления в программировании",
        "Настройка осей",
        "Легенда с параметрами расположения",
        "Сетка",
        "Текст и аннотации",
        "Математические формулы",
        "Стили оформления",
    )
)

# -------------------- ВВЕДЕНИЕ --------------------
if menu == "Что такое элементы оформления в программировании":
    st.markdown("""
    #### 🎨 Что такое элементы оформления в программировании

    **Элементы оформления графика** — это параметры, которые определяют, **как данные отображаются** на графике:
    * оси координат и их подписи,
    * сетка,
    * легенда,
    * аннотации (текстовые пометки),
    * использование формул,
    * стиль оформления.

    Эти элементы делают визуализацию **понятной и профессиональной**,  
    помогая выделить ключевые моменты и улучшить читаемость графика.
    """)

    st.info("Matplotlib предоставляет множество инструментов для гибкой настройки внешнего вида визуализаций.")

# -------------------- ОСИ --------------------
if menu == "Настройка осей":
    st.markdown("""
    #### 📏 Настройка осей

    В Matplotlib можно легко управлять:
    * диапазоном значений осей (`plt.xlim`, `plt.ylim`);
    * подписями осей (`plt.xlabel`, `plt.ylabel`);
    * названием графика (`plt.title`);
    * делениями и метками (`plt.xticks`, `plt.yticks`).
    """)

    x = np.linspace(-5, 5, 200)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, color="royalblue", linewidth=2)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Ось X")
    ax.set_ylabel("Ось Y")
    ax.set_title("Пример настройки осей")
    st.pyplot(fig)

    st.caption("Настройка осей помогает сосредоточить внимание на нужной части данных.")

# -------------------- ЛЕГЕНДА --------------------
if menu == "Легенда с параметрами расположения":
    st.markdown("""
    #### 🗂️ Легенда с параметрами расположения

    **Легенда** объясняет, что означает каждая линия, точка или столбец.  
    Её можно разместить в разных местах с помощью параметра `loc`:
    * `'upper left'`, `'lower right'`, `'center'` и др.
    """)

    x = np.linspace(0, 10, 100)
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x), label='sin(x)', color='tomato')
    ax.plot(x, np.cos(x), label='cos(x)', color='royalblue')
    ax.legend(loc='upper right', fontsize=10, frameon=True)
    ax.set_title("Легенда в правом верхнем углу")
    st.pyplot(fig)

    st.caption("Легенда делает график понятным, особенно при множестве линий.")

# -------------------- СЕТКА --------------------
if menu == "Сетка":
    st.markdown("""
    #### 🔢 Сетка

    **Сетка** облегчает восприятие данных, помогая соотнести значения по осям.  
    Её можно включить функцией `plt.grid(True)` и настроить стиль линий.
    """)

    x = np.linspace(0, 2*np.pi, 200)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, color='mediumseagreen', linewidth=2)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title("График с включённой сеткой")
    st.pyplot(fig)

    st.caption("Сетка добавляет визуальные ориентиры и делает график аккуратным.")

# -------------------- ТЕКСТ И АННОТАЦИИ --------------------
if menu == "Текст и аннотации":
    st.markdown("""
    #### 📝 Текст и аннотации

    Аннотации позволяют добавлять пояснения к ключевым точкам графика.  
    Используется метод `plt.text()` или `plt.annotate()`.
    """)

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, color='purple', linewidth=2)
    ax.annotate('Максимум', xy=(np.pi/2, 1), xytext=(2, 1.3),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('Минимум', xy=(3*np.pi/2, -1), xytext=(5, -1.3),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax.set_title("Пример аннотаций")
    st.pyplot(fig)

    st.caption("Аннотации помогают подчеркнуть важные особенности данных.")

# -------------------- МАТЕМАТИЧЕСКИЕ ФОРМУЛЫ --------------------
if menu == "Математические формулы":
    st.markdown("""
    #### ∑ Математические формулы

    Matplotlib поддерживает отображение формул в **LaTeX-синтаксисе**.  
    Это позволяет красиво оформлять математические выражения на графике.
    """)

    x = np.linspace(-2, 2, 100)
    y = x**2

    fig, ax = plt.subplots()
    ax.plot(x, y, color='darkorange', linewidth=2)
    ax.text(0.2, 3, r"$y = x^2$", fontsize=14, color='black')
    ax.set_title("Использование математических формул")
    st.pyplot(fig)

    st.caption("Любое математическое выражение можно встроить с помощью синтаксиса LaTeX.")

# -------------------- СТИЛИ ОФОРМЛЕНИЯ --------------------
if menu == "Стили оформления":
    st.markdown("""
    #### 🎨 Стили оформления

    В Matplotlib можно выбрать готовые стили визуализации:  
    ```python
    plt.style.use('ggplot')
    plt.style.use('seaborn-v0_8')
    plt.style.use('dark_background')
    ```
    """)

    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)

    styles = ['default', 'ggplot', 'seaborn-v0_8']
    for s in styles:
        plt.style.use(s)
        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=2)
        ax.set_title(f"Стиль оформления: {s}")
        st.pyplot(fig)

    st.caption("Стили помогают быстро адаптировать внешний вид графика под задачу или бренд.")
