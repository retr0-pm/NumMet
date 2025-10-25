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


# -------------------- НАСТРОЙКА ОСЕЙ --------------------
if menu == "Настройка осей":
    st.markdown("#### 📏 Настройка осей — интерактивная демонстрация")

    col1, col2 = st.columns([1, 2])
    with col1:
        func = st.selectbox("Функция", ["sin(x)", "cos(x)", "exp(-x²)"])
        xmin, xmax = st.slider("Диапазон X", -10, 10, (-5, 5))
        ymin, ymax = st.slider("Диапазон Y", -5, 5, (-2, 2))
        show_labels = st.checkbox("Показывать подписи осей", value=True)

    with col2:
        x = np.linspace(xmin, xmax, 400)
        y = np.sin(x) if func == "sin(x)" else np.cos(x) if func == "cos(x)" else np.exp(-x**2)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, y, color="royalblue", linewidth=2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if show_labels:
            ax.set_xlabel("Ось X")
            ax.set_ylabel("Ось Y")
        ax.set_title(f"График функции {func}")
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

    st.code("""
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.grid(True, linestyle="--", alpha=0.4)
    """, language="python")

# -------------------- ЛЕГЕНДА --------------------
if menu == "Легенда с параметрами расположения":
    st.markdown("#### 🗂️ Легенда — настрой расположение и рамку")

    col1, col2 = st.columns([1, 2])
    with col1:
        loc = st.selectbox("Положение легенды",
                           ["upper left", "upper right", "lower left", "lower right", "center"])
        frame = st.checkbox("Показывать рамку", value=True)

    with col2:
        x = np.linspace(0, 10, 100)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, np.sin(x), label='sin(x)', color='tomato')
        ax.plot(x, np.cos(x), label='cos(x)', color='royalblue')
        ax.legend(loc=loc, frameon=frame)
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

    st.code("""
ax.legend(loc=loc, frameon=frame)
    """, language="python")

# -------------------- СЕТКА --------------------
if menu == "Сетка":
    st.markdown("#### 🔢 Настройка сетки")

    col1, col2 = st.columns([1, 2])
    with col1:
        show_grid = st.checkbox("Показать сетку", value=True)
        linestyle = st.selectbox("Тип линии", ["--", "-.", ":", "-"])
        alpha = st.slider("Прозрачность", 0.1, 1.0, 0.7, 0.1)

    with col2:
        x = np.linspace(0, 2*np.pi, 200)
        y = np.sin(x)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, y, color='mediumseagreen', linewidth=2)
        if show_grid:
            ax.grid(True, linestyle=linestyle, alpha=alpha)
        ax.set_title("Пример настройки сетки")
        st.pyplot(fig)

    st.code("""
ax.grid(True, linestyle=linestyle, alpha=alpha)
    """, language="python")

# -------------------- ТЕКСТ И АННОТАЦИИ --------------------
if menu == "Текст и аннотации":
    st.markdown("#### 📝 Добавление аннотаций")

    col1, col2 = st.columns([1, 2])
    with col1:
        text = st.text_input("Текст аннотации", "Максимум")
        x_coord = st.slider("Координата X", 0.0, 10.0, 1.5, 0.1)
        y_coord = np.sin(x_coord)
        st.write(f"sin({x_coord:.1f}) = {y_coord:.2f}")

    with col2:
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, y, color='purple', linewidth=2)
        ax.annotate(text, xy=(x_coord, y_coord),
                    xytext=(x_coord+1, y_coord+0.3),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

    st.code("""
ax.annotate(text, xy=(x_coord, y_coord),
            xytext=(x_coord+1, y_coord+0.3),
            arrowprops=dict(facecolor='black'))
    """, language="python")

# -------------------- МАТЕМАТИЧЕСКИЕ ФОРМУЛЫ --------------------
if menu == "Математические формулы":
    st.markdown("#### ∑ Математические формулы (LaTeX)")

    col1, col2 = st.columns([1, 2])
    with col1:
        formula = st.selectbox("Выражение", [
            r"$y = x^2$",
            r"$y = \sin(x)$",
            r"$y = e^{-x^2}$",
            r"$y = \sqrt{|x|}$"
        ])

    with col2:
        x = np.linspace(-3, 3, 200)
        y = {
            r"$y = x^2$": x**2,
            r"$y = \sin(x)$": np.sin(x),
            r"$y = e^{-x^2}$": np.exp(-x**2),
            r"$y = \sqrt{|x|}$": np.sqrt(np.abs(x))
        }[formula]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, y, color='darkorange', linewidth=2)
        ax.text(0, max(y)/1.5, formula, fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

    st.code("""
ax.text(0, max(y)/1.5, formula, fontsize=16)
    """, language="python")

# -------------------- СТИЛИ ОФОРМЛЕНИЯ --------------------
if menu == "Стили оформления":
    st.markdown("#### 🎨 Стили оформления — попробуй разные темы")

    col1, col2 = st.columns([1, 2])
    with col1:
        style = st.selectbox("Выберите стиль", plt.style.available)

    with col2:
        plt.style.use(style)
        x = np.linspace(0, 2*np.pi, 200)
        y = np.sin(x)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, y, linewidth=2)
        ax.set_title(f"Стиль оформления: {style}")
        st.pyplot(fig)

    st.code("""
plt.style.use(style)
ax.plot(x, y)
    """, language="python")
