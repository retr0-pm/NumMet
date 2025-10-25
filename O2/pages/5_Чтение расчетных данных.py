import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io

st.set_page_config(page_title="Чтение расчётных данных", layout="wide")

# Вспомогательная функция — компактный рендер, не растягивается под ширину страницы
def show_plot(fig, width_inches=5, height_inches=3):
    fig.set_size_inches(width_inches, height_inches)
    fig.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=False)

# Добавить padding по Y, чтобы линия не упиралась в границы
def apply_ylim_with_padding(ax, y, pad_factor=0.08, fixed=None):
    """y: np.array или список значений; pad_factor — доля диапазона для отступа.
       fixed: (ymin, ymax) если нужно применить фиксированные пределы"""
    if fixed is not None:
        ax.set_ylim(fixed)
        return
    y = np.asarray(y)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    if np.isclose(y_min, y_max):
        # если константа — задаём небольшой симметричный диапазон
        delta = max(0.5, abs(y_min) * 0.1)
        ax.set_ylim(y_min - delta, y_max + delta)
    else:
        pad = (y_max - y_min) * pad_factor
        ax.set_ylim(y_min - pad, y_max + pad)


menu = st.sidebar.radio("***",
    (
        "Прямая работа с массивами Numpy",
        "Пример полного цикла с Numpy",
        "Чтение из файлов",
    )
)

# -------------------- ПРЯМАЯ РАБОТА С МАССИВАМИ --------------------
if menu == "Прямая работа с массивами Numpy":
    st.markdown("""
    #### 🔢 Прямая работа с массивами NumPy

    `NumPy` — основа численных вычислений в Python.  
    Массивы (`ndarray`) позволяют хранить и обрабатывать большие объёмы данных эффективно.

    **Создание массивов:**
    ```python
    import numpy as np
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ```
    """)

    st.markdown("#### Визуализация массива")

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, color="royalblue", linewidth=2)
    ax.set_title("Пример прямой работы с массивом NumPy")
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")
    ax.grid(True, linestyle='--', alpha=0.6)
    apply_ylim_with_padding(ax, y)          # добавляем запас по Y
    show_plot(fig)

    st.info("NumPy позволяет выполнять векторные операции без циклов, ускоряя расчёты в десятки раз.")

# -------------------- ПРИМЕР ПОЛНОГО ЦИКЛА --------------------
if menu == "Пример полного цикла с Numpy":
    st.markdown("""
    #### 🔁 Пример полного цикла с NumPy

    Здесь показан **полный цикл обработки данных**:
    1. Создание массива с исходными данными.  
    2. Расчёт производных величин.  
    3. Визуализация результатов с помощью Matplotlib.
    """)

    # Генерация данных
    x = np.linspace(0, 2*np.pi, 200)
    y = np.sin(x)
    y_derivative = np.gradient(y, x)

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(x, y, color='mediumseagreen', label='y = sin(x)')
        ax1.legend(fontsize=8)
        ax1.set_title("Исходные данные")
        ax1.grid(True, linestyle='--', alpha=0.6)
        apply_ylim_with_padding(ax1, y)
        show_plot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(x, y_derivative, color='tomato', label="dy/dx")
        ax2.legend(fontsize=8)
        ax2.set_title("Рассчитанная производная")
        ax2.grid(True, linestyle='--', alpha=0.6)
        apply_ylim_with_padding(ax2, y_derivative)
        show_plot(fig2)

    st.success("""
    ➤ Полный цикл расчёта:  
    данные → вычисление → визуализация.
    """)

# -------------------- ЧТЕНИЕ ИЗ ФАЙЛОВ --------------------
if menu == "Чтение из файлов":
    st.markdown("""
    #### 📂 Чтение данных из файлов

    NumPy поддерживает простые и быстрые методы загрузки данных из текстовых и бинарных файлов:
    * `np.loadtxt()` — чтение из текстового файла (CSV, TXT);
    * `np.genfromtxt()` — более гибкая версия, поддерживает пропуски;
    * `np.load()` / `np.save()` — работа с бинарными `.npy` файлами.
    """)

    st.code("""
# Пример чтения из текстового файла
data = np.loadtxt("data.txt")
x = data[:, 0]
y = data[:, 1]

plt.plot(x, y)
plt.show()
    """, language="python")

    st.markdown("#### 💡 Пример с искусственным файлом (демонстрация):")

    # Создание искусственного CSV в памяти
    csv_data = "x,y\n0,0\n1,1\n2,4\n3,9\n4,16"
    data = np.genfromtxt(io.StringIO(csv_data), delimiter=",", skip_header=1)

    x = data[:, 0]
    y = data[:, 1]

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', color='darkorange', linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y = x²")
    ax.set_title("Демонстрация чтения данных из файла")
    ax.grid(True, linestyle='--', alpha=0.6)
    apply_ylim_with_padding(ax, y)
    show_plot(fig)

    st.caption("В реальных задачах данные могут поступать из CSV, Excel, датчиков или расчётных модулей.")
