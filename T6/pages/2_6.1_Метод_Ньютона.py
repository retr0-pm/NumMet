import streamlit as st
import numpy as np
from scipy.interpolate import BarycentricInterpolator
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide", page_title="Интерполяция Ньютона")

def divided_diff(x, y):
    n = len(x)
    coef = np.copy(y)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1:n - 1]) / (x[j:n] - x[0:n - j])
    return coef

def newton_eval(x, x_nodes, coef):
    n = len(coef)
    p = coef[-1]
    for k in range(n - 2, -1, -1):
        p = p * (x - x_nodes[k]) + coef[k]
    return p

menu = st.sidebar.radio(
    "Выберите раздел:",
    ["Постановка задачи", "Описание алгоритма", "Интерактивный пример", "Сравнение сходимости"]
)

f = lambda x: 1 / (1 + 25 * x ** 2)

def natural_cubic_spline(x, y):
    n = len(x)
    h = np.diff(x)

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

    def spline_eval(x_eval):
        x_eval = np.array(x_eval)
        res = np.zeros_like(x_eval)
        for j, xv in enumerate(x_eval):
            i = np.searchsorted(x, xv) - 1
            i = np.clip(i, 0, n - 2)
            h_i = x[i + 1] - x[i]
            a = (x[i + 1] - xv) / h_i
            b = (xv - x[i]) / h_i
            res[j] = (
                    a * y[i] + b * y[i + 1]
                    + ((a ** 3 - a) * M[i] + (b ** 3 - b) * M[i + 1]) * h_i ** 2 / 6
            )
        return res

    return spline_eval


if menu == "Постановка задачи":
    st.markdown(r"""

Напишите программу для интерполирования данных на основе интерполяционного многочлена Ньютона.

С помощью этой программы интерполируйте данные в равноотстоящих узлах для функции Рунге

$$ f(x) = (1+25 x^2)^{-1} $$

на интервале $[-1,1]$ при различных $n$.

Решите также эту задачу с помощью библиотеки SciPy.
    """)

elif menu == "Описание алгоритма":

    st.header("6.1. Описание алгоритма интерполяции Ньютона")

    st.markdown(r"""
### Интерполяционный многочлен Ньютона

Пусть заданы узлы интерполяции $x_0, x_1, \dots, x_n$  
и значения функции $f_0 = f(x_0), \dots, f_n = f(x_n)$.

Интерполяционный многочлен Ньютона имеет вид:

$$
P_n(x) = a_0 
       + a_1 (x - x_0) 
       + a_2 (x - x_0)(x - x_1) 
       + \dots 
       + a_n \prod_{k=0}^{n-1} (x - x_k)
$$

где коэффициенты $a_k$ вычисляются через **разделённые разности**.

---

### Разделённые разности

Разделённая разность нулевого порядка:

$$
f[x_i] = f_i
$$

Разделённая разность первого порядка:

$$
f[x_i, x_{i+1}] = \frac{f_{i+1} - f_i}{x_{i+1} - x_i}
$$

Разделённая разность второго порядка:

$$
f[x_i, x_{i+1}, x_{i+2}] = \frac{f[x_{i+1}, x_{i+2}] - f[x_i, x_{i+1}]}{x_{i+2} - x_i}
$$

И так далее для более высоких порядков.

Алгоритм заполнения вектора коэффициентов заключается в последовательном вычислении разностей по столбцам:

""")

    st.code("""
def divided_differences(x, y):
    n = len(x)
    coef = np.copy(y)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1]) / (x[j:n] - x[j - 1])
    return coef
""")

    st.markdown(r"""
---

### Вычисление значения многочлена (схема Горнера для формы Ньютона)

После вычисления коэффициентов значение полинома находится по обратной схеме:

$$
p = a_n, \quad
p = a_k + (x - x_k)p, \quad k = n-1, \dots, 0
$$

Реализация на Python:
""")

    st.code("""
def newton_eval(x_nodes, coef, x):
    p = coef[-1]
    for k in range(len(coef) - 2, -1, -1):
        p = coef[k] + (x - x_nodes[k]) * p
    return p
""")

    st.markdown(r"""
---

### Интерполяция с помощью SciPy

Для сравнения используется барицентрическая интерполяция Лагранжа:

```python
from scipy.interpolate import BarycentricInterpolator

interp = BarycentricInterpolator(x_nodes, y_nodes)
y_interp = interp(x_exact)
""")


elif menu == "Интерактивный пример":

    st.subheader("Интерактивная интерполяция — многочлен Ньютона")

    st.markdown(r"Используем функцию Рунге:")
    st.markdown(r"$$f(x) = \frac{1}{1 + 25 x^2}$$")

    n = st.slider("Число узлов n", 5, 40, 10)

    x_nodes = np.linspace(-1, 1, n)
    y_nodes = 1 / (1 + 25 * x_nodes**2)

    def divided_diff(x, y):
        n = len(x)
        coef = np.copy(y).astype(float)
        for j in range(1, n):
            coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
        return coef

    def newton_eval(x_eval, x_nodes, coef):
        p = coef[-1]
        for k in range(len(coef)-2, -1, -1):
            p = coef[k] + (x_eval - x_nodes[k]) * p
        return p

    coef = divided_diff(x_nodes, y_nodes)

    x_plot = np.linspace(-1, 1, 1000)
    y_newton = np.array([newton_eval(xx, x_nodes, coef) for xx in x_plot])

    from scipy.interpolate import BarycentricInterpolator
    interp_scipy = BarycentricInterpolator(x_nodes, y_nodes)
    y_scipy = interp_scipy(x_plot)

    y_true = 1 / (1 + 25 * x_plot**2)

    st.subheader("График интерполяции")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_plot, y_true, label="f(x)", linewidth=2)
    ax.plot(x_plot, y_newton, '--', label="Ньютон (самописный)")
    ax.plot(x_plot, y_scipy, ':', label="SciPy")
    ax.scatter(x_nodes, y_nodes, color="black", s=20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    err_newton = np.max(np.abs(y_newton - y_true))
    err_scipy = np.max(np.abs(y_scipy - y_true))

    st.subheader("Максимальная ошибка интерполяции")
    st.write(f"**Самописный Ньютон:** {err_newton:.6e}")
    st.write(f"**SciPy:** {err_scipy:.6e}")

    df_err = pd.DataFrame({
        "Ошибка самописного": [err_newton],
        "Ошибка SciPy": [err_scipy]
    })
    st.subheader("Сравнение ошибок")
    st.table(df_err)

    st.markdown(r"""
**Комментарий:**  
Даже при больших n интерполирующий многочлен сильно осциллирует на концах интервала.  
Это **эффект Рунге** — полиномиальная интерполяция равноотстоящих узлов плохо аппроксимирует сильно изогнутые функции.
""")



elif menu == "Сравнение сходимости":

    st.subheader("Сравнение ошибок интерполяции — многочлен Ньютона")

    n_plot = np.arange(2, 41)
    x_test = np.linspace(-1, 1, 1000)
    f_true = 1 / (1 + 25 * x_test**2)

    errors_newton = []
    errors_scipy = []

    for n in n_plot:
        x_nodes = np.linspace(-1, 1, n)
        y_nodes = 1 / (1 + 25 * x_nodes**2)

        coef = divided_diff(x_nodes, y_nodes)
        yN = np.array([newton_eval(xx, x_nodes, coef) for xx in x_test])
        errN = np.max(np.abs(yN - f_true))
        errors_newton.append(errN)

        from scipy.interpolate import BarycentricInterpolator
        interp_scipy = BarycentricInterpolator(x_nodes, y_nodes)
        ySc = interp_scipy(x_test)
        errSc = np.max(np.abs(ySc - f_true))
        errors_scipy.append(errSc)

    st.subheader("График максимальной ошибки")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(n_plot, errors_newton, 'o--', label="Ньютон (самописный)")
    ax.plot(n_plot, errors_scipy, 's:', label="SciPy")
    ax.set_xlabel("n (число узлов)")
    ax.set_ylabel("Максимальная ошибка")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Таблица максимальных ошибок")
    selected_n = [5, 10, 15, 20, 30, 40]
    table_data = []
    for n in selected_n:
        x_nodes = np.linspace(-1, 1, n)
        y_nodes = 1 / (1 + 25 * x_nodes**2)

        coef = divided_diff(x_nodes, y_nodes)
        yN = np.array([newton_eval(xx, x_nodes, coef) for xx in x_test])
        errN = np.max(np.abs(yN - f_true))

        interp_scipy = BarycentricInterpolator(x_nodes, y_nodes)
        ySc = interp_scipy(x_test)
        errSc = np.max(np.abs(ySc - f_true))

        table_data.append({
            "n": n,
            "Ошибка самописного Ньютона": errN,
            "Ошибка SciPy": errSc
        })

    df_errors = pd.DataFrame(table_data)
    st.dataframe(df_errors)

