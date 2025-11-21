import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


st.set_page_config(layout="wide", page_title="Метод кубических сплайнов")

menu = st.sidebar.radio(
    "Выберите раздел:",
    ["Постановка задачи", "Описание алгоритма", "Интерактивный пример", "Сравнение сходимости"]
)


def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)


def natural_cubic_spline_second_derivs(x, y):
    n = len(x) - 1
    h = np.diff(x)
    if n == 1:
        return np.zeros(2)

    A = np.zeros((n - 1, n - 1))
    rhs = np.zeros(n - 1)

    for i in range(n - 1):
        if i > 0:
            A[i, i - 1] = h[i]
        A[i, i] = 2.0 * (h[i] + h[i + 1])
        if i < n - 2:
            A[i, i + 1] = h[i + 1]
        rhs[i] = 6.0 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i])

    m = np.zeros(n + 1)
    m_inner = np.linalg.solve(A, rhs)
    m[1:n] = m_inner
    return m


def evaluate_natural_cubic_spline(x, y, m, t):

    x = np.asarray(x)
    t = np.asarray(t)
    s = np.zeros_like(t, dtype=float)
    inds = np.searchsorted(x, t) - 1
    inds = np.clip(inds, 0, len(x) - 2)

    for j, idx in enumerate(inds):
        xi = x[idx]
        xi1 = x[idx + 1]
        hi = xi1 - xi
        if hi == 0:
            s[j] = y[idx]
            continue
        a = (xi1 - t[j]) / hi
        b = (t[j] - xi) / hi
        s[j] = (
            a * y[idx]
            + b * y[idx + 1]
            + ((a**3 - a) * m[idx] + (b**3 - b) * m[idx + 1]) * (hi**2) / 6.0
        )
    return s


if menu == "Постановка задачи":
    st.markdown(r"""
##### 6 Интерполирование и приближение функций

**Задание 6.2**

Напишите программу для интерполирования
данных на основе естественного интерполяционного кубического сплайна.

С помощью этой программы интерполируйте
данные в равноотстоящих узлах для функции Рунге

$$
f(x) = \frac{1}{1+25 x^2}
$$

на интервале $[-1,1]$ при различных $n$.

Решите также эту задачу с помощью библиотеки SciPy.
""")


elif menu == "Описание алгоритма":
    st.header("Описание метода — натуральный кубический сплайн")

    st.markdown(r"""
Натуральный кубический сплайн — кусочно-полиномиальная функция $S(x)$, которая:

1. На каждом интервале $[x_i,x_{i+1}]$ является многочленом степени ≤ 3:
$$
S_i(x) = a_i + b_i (x-x_i) + c_i (x-x_i)^2 + d_i (x-x_i)^3
$$

2. $S(x)$ непрерывна и дважды непрерывно дифференцируема:
$$
S \in C^2[x_0,x_n]
$$

3. Удовлетворяет естественным граничным условиям:
$$
S''(x_0) = 0, \quad S''(x_n) = 0
$$

4. Интерполяция:
$$
S(x_i) = y_i, \quad i=0,\dots,n
$$
""")

    st.subheader("Основная формула на каждом отрезке")
    st.latex(r"""
S(x)=
\frac{M_i (x_{i+1}-x)^3}{6h_i}
+\frac{M_{i+1}(x-x_i)^3}{6h_i}
+\left( \frac{y_i}{h_i} - \frac{M_i h_i}{6} \right)(x_{i+1}-x)
+\left( \frac{y_{i+1}}{h_i} - \frac{M_{i+1} h_i}{6} \right)(x-x_i)
""")

    st.subheader("Система для вторых производных")
    st.latex(r"""
h_{i-1}M_{i-1} + 2(h_{i-1}+h_i)M_i + h_i M_{i+1} = 
6\left(\frac{y_{i+1}-y_i}{h_i} - \frac{y_i - y_{i-1}}{h_{i-1}}\right),\quad i=1..n-1
""")

    st.subheader("Формальное описание алгоритма")
    st.markdown("Натуральный кубический сплайн для данных $(x_i, y_i)$:")
    st.latex(r"M_0 = M_n = 0")
    st.latex(
        r"\text{Для } i=1..n-1: \quad h_{i-1} M_{i-1} + 2(h_{i-1}+h_i) M_i + h_i M_{i+1} = 6 \left( \frac{y_{i+1}-y_i}{h_i} - \frac{y_i - y_{i-1}}{h_{i-1}} \right)")
    st.latex(r"S(x) \text{ вычисляется по формуле на отрезках } [x_i,x_{i+1}]")


elif menu == "Интерактивный пример":
    st.header("Интерполяция кубическим сплайном")

    st.subheader(r"Используем функцию Рунге:")
    st.subheader(r"$$f(x) = \frac{1}{1 + 25 x^2}$$")

    n = st.number_input("Выберите число интервалов n", min_value=2, max_value=200, value=10)
    dense_x = np.linspace(-1, 1, 1000)

    nodes = np.linspace(-1, 1, n + 1)
    vals = runge(nodes)

    st.subheader("Код функции самописного натурального сплайна")
    st.code("""
def natural_cubic_spline(x, y):
    n = len(x)
    h = np.diff(x)

    # Сборка матрицы для второй производной
    A = np.zeros((n, n))
    b = np.zeros(n)

    A[0, 0] = 1
    A[-1, -1] = 1

    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1] / 6
        A[i, i] = (h[i - 1] + h[i]) / 3
        A[i, i + 1] = h[i] / 6
        b[i] = (y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]

    M = np.linalg.solve(A, b)
    return M
    """)

    m = natural_cubic_spline_second_derivs(nodes, vals)
    spline_vals = evaluate_natural_cubic_spline(nodes, vals, m, dense_x)

    true_vals = runge(dense_x)

    from scipy.interpolate import CubicSpline
    cs_scipy = CubicSpline(nodes, vals, bc_type="natural")
    spline_vals_scipy = cs_scipy(dense_x)

    err_custom = np.max(np.abs(spline_vals - true_vals))
    err_scipy  = np.max(np.abs(spline_vals_scipy - true_vals))

    st.subheader("Максимальная ошибка методов")
    st.write(f"**Самописный натуральный сплайн:** {err_custom:.6e}")
    st.write(f"**SciPy натуральный сплайн:** {err_scipy:.6e}")

    df_err = pd.DataFrame({
        "Ошибка самописного": [err_custom],
        "Ошибка SciPy": [err_scipy]
    })

    st.subheader("Сравнение ошибок")
    st.table(df_err)

    import plotly.graph_objects as go
    st.subheader("График функции Рунге и сплайнов")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dense_x, y=true_vals,
        mode="lines",
        name="f(x)",
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dense_x, y=spline_vals,
        mode="lines",
        name="Самописный сплайн",
        line=dict(width=2, dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=dense_x, y=spline_vals_scipy,
        mode="lines",
        name="SciPy сплайн",
        line=dict(width=3, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=nodes, y=vals,
        mode="markers",
        name="Узлы",
        marker=dict(size=8, color="black")
    ))

    fig.update_layout(
        width=900, height=500,
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.0
        )
    )

    st.plotly_chart(fig, use_container_width=True)


else:
    st.header("Сравнение точности сплайна при разных n")

    from scipy.interpolate import CubicSpline

    n_values = [5, 10, 15, 20, 40, 80]
    dense_x = np.linspace(-1, 1, 2001)
    true_vals = runge(dense_x)

    errors_manual = []
    errors_scipy = []

    for n in n_values:
        nodes = np.linspace(-1, 1, n + 1)
        vals = runge(nodes)

        m = natural_cubic_spline_second_derivs(nodes, vals)
        spline_vals = evaluate_natural_cubic_spline(nodes, vals, m, dense_x)
        err_manual = np.max(np.abs(spline_vals - true_vals))

        cs = CubicSpline(nodes, vals, bc_type="natural")
        spline_scipy = cs(dense_x)
        err_scipy = np.max(np.abs(spline_scipy - true_vals))

        errors_manual.append(err_manual)
        errors_scipy.append(err_scipy)

    df = pd.DataFrame({
        "n": n_values,
        "Ошибка (самописный)": errors_manual,
        "Ошибка (SciPy)": errors_scipy
    })

    st.subheader("Таблица ошибок")
    st.dataframe(df)


    st.subheader("График ошибки для всех n от 2 до 80")

    n_all = np.arange(2, 81)
    all_err_manual = []
    all_err_scipy = []

    for n in n_all:
        nodes = np.linspace(-1, 1, n + 1)
        vals = runge(nodes)

        m = natural_cubic_spline_second_derivs(nodes, vals)
        spline_vals = evaluate_natural_cubic_spline(nodes, vals, m, dense_x)
        err_manual = np.max(np.abs(spline_vals - true_vals))

        cs = CubicSpline(nodes, vals, bc_type="natural")
        spline_scipy = cs(dense_x)
        err_scipy = np.max(np.abs(spline_scipy - true_vals))

        all_err_manual.append(err_manual)
        all_err_scipy.append(err_scipy)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(n_all, all_err_manual, '-', label="Самописный сплайн")
    ax.plot(n_all, all_err_scipy, '--', label="SciPy CubicSpline (natural)")
    ax.set_xlabel("n (число интервалов)")
    ax.set_ylabel("Максимальная ошибка интерполяции")
    ax.set_title("Сравнение ошибок для всех n от 2 до 80")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


