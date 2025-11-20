import streamlit as st
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide", page_title="Метод Зейделя")

menu = st.sidebar.radio(
    "Выберите раздел:",
    ["Постановка задачи", "Описание алгоритма", "Интерактивный пример", "Сравнение сходимости"]
)
if menu == "Постановка задачи":
    st.markdown(r"""
    **Задание 2.1**

    Напишите программу, реализующую приближенное
    решение системы линейных алгебраических
    уравнений итерационным методом Зейделя. 

    С ее помощью найдите решение
    системы уравнений с трехдиагональной матрицей

    $\begin{aligned}
      A x = f,
    \end{aligned}$

    в которой

    $\begin{aligned}
      a_{ii} = 2,
      \quad a_{i,i+1} = - 1 - \alpha,
      \quad a_{i,i-1} = - 1 + \alpha,
    \end{aligned}$

    а правая часть

    $\begin{aligned}
      f_0 = 1 - \alpha,
      \quad f_i = 0, \quad i = 2,3, ..., n-1,
      \quad f_{n} = 1 + \alpha ,
    \end{aligned}$

    определяет точное решение $x_i = 1, \quad i = 1,2, ..., n$.

    Исследуйте зависимость числа итераций от $n$ и параметра
    $\alpha$ при $0 \leq \alpha \leq 1$.

    Решите также эту задачу с помощью библиотеки SciPy.
    """)

elif menu == "Описание алгоритма":
    st.header("Обобщение (каноническая формула Самарского)")
    st.latex(r"(D + L)\,\frac{x^{k+1} - x^k}{\tau_{k+1}} + A x^k = f,\qquad k = 0,1,\ldots")
    st.latex(r"""
        \sum_{j=1}^{i-1} a_{ij} x_j^{k+1}
        + a_{ii} x_i^{k+1}
        + \sum_{j=i+1}^{n} a_{ij} x_j^{k}
        = f_i,\qquad i = 1,2,\ldots,n
    """)
    st.subheader("Условие диагонального преобладания")
    st.latex(r"|a_{ii}| > \sum_{\substack{j=1 \\ j\ne i}}^{n} |a_{ij}|,\qquad i = 1,\ldots,n")
    st.subheader("Положительная определённость")
    st.latex(r"x^T A x > 0\quad \text{для любого } x\ne 0")
    st.markdown("При выполнении этих условий гарантируется **сходимость метода Гаусса–Зейделя**.")

elif menu == "Интерактивный пример":
    st.header("Интерактивный пример — трёхдиагональная матрица из условия задачи")
    st.markdown(
        r"""
    ### Реализация метода Зейделя (Gauss–Seidel)

    ```python
    def gauss_seidel():
    x = [0] * len(A[0])

    for iteration in range(1, max_iteration + 1):
        diff = 0
        for i in range(len(A)):
            x_temp = 0
            for j in range(len(A[i])):
                if j == i:
                    continue
                x_temp += -A[i][j] * x[j] / A[i][i]
            x_temp += b[i]/A[i][i]
            if abs(x[i] - x_temp) > diff:
                diff = abs(x[i] - x_temp)
            x[i] = x_temp
        if diff <= eps:
            return x, iteration
    print(f"Достигнут лимит итераций {max_iteration}")
    return 0, 0

    """
    )
    st.markdown(
        "**Матрица A создаётся по условию задачи:**\n\n"
        "- aᵢᵢ = 2\n"
        "- aᵢ,ᵢ₊₁ = −1 − α\n"
        "- aᵢ,ᵢ₋₁ = −1 + α\n\n"
        "**Правая часть f также задаётся по правилу:**\n\n"
        "- f₁ = 1 − α\n"
        "- fₙ = 1 + α\n"
        "- fᵢ = 0 для i = 2..n−1"
    )

    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Размерность n", min_value=3, max_value=300, value=5)
    with col2:
        alpha = st.slider("Параметр α", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    main_diag = 2.0 * np.ones(n)
    upper_diag = (-1 - alpha) * np.ones(n - 1)
    lower_diag = (-1 + alpha) * np.ones(n - 1)

    A = diags([main_diag, upper_diag, lower_diag], [0, 1, -1]).toarray()

    b = np.zeros(n)
    b[0] = 1 - alpha
    b[-1] = 1 + alpha

    df_A = pd.DataFrame(np.round(A, 6),
                        index=range(1, n + 1),
                        columns=range(1, n + 1))

    html_A = df_A.to_html(classes='table table-striped', border=1)
    html_A = html_A.replace(
        '<table border="1" class="dataframe table table-striped">',
        '<table border="1" class="dataframe table table-striped" style="text-align:center;">'
    )
    html_A = html_A.replace('<th>', '<th style="text-align:center;">')
    html_A = html_A.replace('<td>', '<td style="text-align:center;">')

    st.subheader("Матрица A")
    st.markdown(html_A, unsafe_allow_html=True)

    df_b = pd.DataFrame(np.round(b.reshape(-1, 1), 6),
                        index=range(1, n + 1),
                        columns=["b"])

    html_b = df_b.to_html(classes='table table-striped', border=1)
    html_b = html_b.replace(
        '<table border="1" class="dataframe table table-striped">',
        '<table border="1" class="dataframe table table-striped" style="text-align:center;">'
    )
    html_b = html_b.replace('<th>', '<th style="text-align:center;">')
    html_b = html_b.replace('<td>', '<td style="text-align:center;">')

    st.subheader("Вектор b")
    st.markdown(html_b, unsafe_allow_html=True)

    def gauss_seidel(A_mat, b_vec, eps_local=1e-6, max_iter=20000):
        n = len(b_vec)
        x = np.zeros(n)

        for iteration in range(1, max_iter + 1):
            diff = 0.0
            for i in range(n):
                x_temp = b_vec[i] / A_mat[i, i]
                for j in range(n):
                    if j == i:
                        continue
                    x_temp -= A_mat[i, j] * x[j] / A_mat[i, i]

                diff = max(diff, abs(x[i] - x_temp))
                x[i] = x_temp

            if diff <= eps_local:
                return x, iteration

        print(f"Достигнут лимит итераций {max_iter}")
        return x, max_iter

    def gmres_with_count(A_mat, b_vec, rtol=1e-8):
        it = [0]
        def cb(xk):
            it[0] += 1
        x, _ = gmres(A_mat, b_vec, callback=cb, rtol=rtol)
        return np.array(x), it[0]

    if st.button("Решить (GS и GMRES)"):
        x_gs, its_gs = gauss_seidel(A, b)
        x_gmres, its_gmres = gmres_with_count(A, b)

        res_gs = np.linalg.norm(A @ x_gs - b)
        res_gmres = np.linalg.norm(A @ x_gmres - b)

        st.subheader("Результаты")
        c1, c2 = st.columns(2)
        c1.metric("Итераций GS", f"{its_gs}")
        c2.metric("Итераций GMRES", f"{its_gmres}")

        st.write("Решение GS:")
        st.dataframe(np.round(x_gs.reshape(-1, 1), 8))

        st.write("Решение GMRES:")
        st.dataframe(np.round(x_gmres.reshape(-1, 1), 8))

        st.subheader("Нормы невязки")
        st.write(f"||A x_gs - b|| = {res_gs:.3e}")
        st.write(f"||A x_gmres - b|| = {res_gmres:.3e}")

        st.subheader("График решений")
        st.line_chart({"Gauss-Seidel": x_gs, "GMRES": x_gmres})

else:
    st.header("Сравнение сходимости: GS vs GMRES (исходный анализ)")
    st.markdown(
        "Два графика: зависимость числа итераций от n при фиксированном `a = 1`, "
        "и зависимость от параметра `a` при фиксированном `n = 20`."
    )

    a_param = 1.0
    a_vector = np.linspace(0.0, 1.0, 9)
    n_values = list(range(4, 21))

    gauss_seidel_iterations_n = []
    gmres_iterations_n = []

    gauss_seidel_iterations_a = []
    gmres_iterations_a = []

    def gs_iter_count(A_mat, b_vec, eps_local=1e-6, max_it=10000):
        x = np.zeros(len(b_vec))
        for k in range(1, max_it + 1):
            x_old = x.copy()
            for i in range(len(A_mat)):
                s = 0.0
                if i > 0:
                    s += A_mat[i, :i] @ x[:i]
                if i < len(A_mat) - 1:
                    s += A_mat[i, i + 1:] @ x_old[i + 1:]
                x[i] = (b_vec[i] - s) / A_mat[i, i]
            if np.linalg.norm(x - x_old, ord=np.inf) < eps_local:
                return k
        return max_it

    def gmres_count(A_mat, b_vec, rtol=1e-8):
        cnt = [0]
        def cb(xk): cnt[0] += 1
        gmres(A_mat, b_vec, callback=cb, rtol=rtol)
        return cnt[0]

    for n in n_values:
        main_diag = 2.0 * np.ones(n)
        upper_diag = (-1 - a_param) * np.ones(n - 1)
        lower_diag = (-1 + a_param) * np.ones(n - 1)
        b_vec = np.zeros(n)
        b_vec[0] = 1 - a_param
        b_vec[-1] = 1 + a_param
        A_mat = diags([main_diag, upper_diag, lower_diag], [0, 1, -1]).toarray()

        gauss_seidel_iterations_n.append(gs_iter_count(A_mat, b_vec))
        gmres_iterations_n.append(gmres_count(A_mat, b_vec))


    for av in a_vector:
        n_fix = 20
        main_diag = 2.0 * np.ones(n_fix)
        upper_diag = (-1 - av) * np.ones(n_fix - 1)
        lower_diag = (-1 + av) * np.ones(n_fix - 1)
        b_vec = np.zeros(n_fix)
        b_vec[0] = 1 - av
        b_vec[-1] = 1 + av
        A_mat = diags([main_diag, upper_diag, lower_diag], [0, 1, -1]).toarray()

        gauss_seidel_iterations_a.append(gs_iter_count(A_mat, b_vec))
        gmres_iterations_a.append(gmres_count(A_mat, b_vec))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(n_values, gauss_seidel_iterations_n, 'bo-', label='Gauss-Seidel')
    axes[0].plot(n_values, gmres_iterations_n, 'ro-', label='GMRES')
    axes[0].set_xlabel('Размер матрицы n')
    axes[0].set_ylabel('Количество итераций')
    axes[0].set_title('Итерации от n (a = 1)')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(a_vector, gauss_seidel_iterations_a, 'bo-', label='Gauss-Seidel')
    axes[1].plot(a_vector, gmres_iterations_a, 'ro-', label='GMRES')
    axes[1].set_xlabel('Параметр a')
    axes[1].set_ylabel('Количество итераций')
    axes[1].set_title('Итерации от a (n = 20)')
    axes[1].grid(True)
    axes[1].legend()

    st.pyplot(fig)
