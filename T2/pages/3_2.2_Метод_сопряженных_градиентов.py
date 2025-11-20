import streamlit as st
import numpy as np
import scipy.linalg
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt

st.title("Метод сопряжённых градиентов")

menu = st.sidebar.radio(
    "Выберите раздел:",
    ["Постановка задачи", "Описание алгоритма", "Интерактивный пример","Сравнение сходимости"]
)

if menu == "Постановка задачи":
    st.markdown(r"""
    **Задание 2.2**

    Напишите программу, реализующую приближенное
    решение системы линейных алгебраических уравнений
    с симметричной положительно определенной матрицей
    методом сопряженных градиентов. 

    С ее помощью найдите решение
    системы

    $\begin{aligned}
      A x = f
    \end{aligned}$

    с матрицей Гильберта

    $\begin{aligned}
      a_{ij} = \frac{1}{i+j-1},
      \quad i = 1,2, ..., n ,
      \quad j = 1,2, ..., n
    \end{aligned}$

    и правой частью

    $\begin{aligned}
      f_i =  \sum_{j=1}^{n} a_{ij}, \quad i = 1,2, ..., n ,
    \end{aligned}$

    для которой точное решение есть $x_i = 1, \quad i = 1,2, ..., n$.

    Исследуйте зависимость числа итераций от $n$.

    Решите также эту задачу с помощью библиотеки SciPy.
    """)

elif menu == "Описание алгоритма":
    st.header("Описание метода")
    st.markdown("""
Метод сопряженных градиентов — это итерационный метод оптимизации, который используется для нахождения локального экстремума функции.
""")
    st.subheader("Основные формулы метода")
    st.latex(r"x_{j+1} = x_j + \alpha_j p_j")
    st.markdown("### Скалярный шаг (спуск):")
    st.latex(r"\alpha_j = \frac{(r_j, r_j)}{(A p_j, p_j)}")
    st.markdown("### Направление спуска:")
    st.latex(r"p_{j+1} = r_{j+1} + \beta_j p_j")
    st.markdown("### Формула Флетчера–Ривса:")
    st.latex(r"\beta_j = \frac{(r_{j+1}, r_{j+1})}{(r_j, r_j)}")
    st.markdown("### Новая невязка (градиент):")
    st.latex(r"r_{j+1} = r_j - \alpha_j A p_j")
    st.subheader("Условия применения метода")
    st.markdown("Метод сопряжённых градиентов гарантированно сходится, если матрица **симметричная и положительно определённая**:")
    st.latex(r"A = A^T")
    st.latex(r"x^T A x > 0 \quad \text{для любого } x \ne 0")

    st.subheader("Формальное описание алгоритма")
    st.markdown("Метод сопряжённых градиентов (CG):")

    st.latex(r"r_0 := b - A x_0, \quad p_0 := r_0")

    st.latex(r"\text{Для } j = 0, 1, 2, \dots \text{ до выполнения критерия остановки:}")

    st.latex(r"\alpha_j := \frac{(r_j, r_j)}{(A p_j, p_j)}")
    st.latex(r"x_{j+1} := x_j + \alpha_j p_j")
    st.latex(r"r_{j+1} := r_j - \alpha_j A p_j")
    st.latex(r"\beta_j := \frac{(r_{j+1}, r_{j+1})}{(r_j, r_j)}")
    st.latex(r"p_{j+1} := r_{j+1} + \beta_j p_j")

elif menu == "Интерактивный пример":
    st.header("Интерактивный пример — метод сопряжённых градиентов")
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Размерность n", min_value=2, max_value=50, value=5)
    with col2:
        eps = st.number_input("Допустимая погрешность ε", min_value=1e-10, max_value=1e-3, value=1e-6, format="%.0e")

    st.markdown("Матрица `A` — матрица Гильберта, вектор `b` задается суммой строк матрицы.")
    A = scipy.linalg.hilbert(n)
    b = np.sum(A, axis=1)

    def render_matrix_html(matrix, name="A"):
        html = f'<table border="1" style="border-collapse:collapse;text-align:center;">'
        html += '<tr><th></th>' + ''.join([f'<th>{j+1}</th>' for j in range(matrix.shape[1])]) + '</tr>'
        for i, row in enumerate(matrix):
            html += f'<tr><th>{i+1}</th>'
            html += ''.join([f'<td>{v:.6f}</td>' for v in row])
            html += '</tr>'
        html += '</table>'
        st.markdown(f"<b>Матрица {name}:</b>", unsafe_allow_html=True)
        st.markdown(html, unsafe_allow_html=True)

    def render_vector_html(vec, name="b"):
        html = f'<table border="1" style="border-collapse:collapse;text-align:center;">'
        html += '<tr><th>i</th><th>Value</th></tr>'
        for i, v in enumerate(vec):
            html += f'<tr><td>{i+1}</td><td>{v:.6f}</td></tr>'
        html += '</table>'
        st.markdown(f"<b>Вектор {name}:</b>", unsafe_allow_html=True)
        st.markdown(html, unsafe_allow_html=True)

    render_matrix_html(A)
    render_vector_html(b)

    if st.button("Решить (CG и SciPy)"):
        # Самописная CG
        def conjugate_gradient():
            x = np.zeros(n)
            r = b - A @ x
            p = r
            for i in range(1, 1000+1):
                a = (r @ r)/((A @ p) @ p)
                x += a * p
                r_prev = r.copy()
                r -= a * (A @ p)
                if (np.linalg.norm(r)/np.linalg.norm(b)) < eps:
                    return x, i
                B = (r @ r)/(r_prev @ r_prev)
                p = r + B * p
            print("Достигнут лимит итераций 1000")
            return 0, 0

        # SciPy CG
        count = [0]
        def callback(xk): count[0] += 1

        x_self, its_self = conjugate_gradient()
        x_scipy, info = cg(A, b, np.zeros(n), rtol=eps, callback=callback)
        its_scipy = count[0]

        st.subheader("Итерации")
        c1, c2 = st.columns(2)
        c1.metric("CG (вручную)", its_self)
        c2.metric("CG (SciPy)", its_scipy)

        render_vector_html(x_self, "Решение CG (вручную)")
        render_vector_html(x_scipy, "Решение CG (SciPy)")

        st.subheader("График решений")
        st.line_chart({"CG (вручную)": x_self, "CG (SciPy)": x_scipy})

else:
    st.header("Сравнение сходимости методов")
    st.write("Нажмите кнопку, чтобы запустить сравнение реализации CG и SciPy CG.")

    if st.button("Запустить пример"):
        n_values = list(range(5, 150))
        iter_self = []
        iter_scipy = []
        eps = 1e-6

        for n in n_values:
            A = scipy.linalg.hilbert(n)
            b = np.sum(A, axis=1)

            # Самописная CG
            def conjugate_gradient():
                x = np.zeros(n)
                r = b - A @ x
                p = r
                for i in range(1, 1000+1):
                    a = (r @ r)/((A @ p) @ p)
                    x += a * p
                    r_prev = r.copy()
                    r -= a * (A @ p)
                    if (np.linalg.norm(r)/np.linalg.norm(b)) < eps:
                        return x, i
                    B = (r @ r)/(r_prev @ r_prev)
                    p = r + B * p
                return 0, 0

            _, its_self = conjugate_gradient()
            iter_self.append(its_self)

            # SciPy CG
            count = [0]
            def callback(xk): count[0] += 1
            x_scipy, info = cg(A, b, np.zeros(n), rtol=eps, callback=callback)
            iter_scipy.append(count[0])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(n_values, iter_self, 'o-', label='CG (самописный)')
        ax.plot(n_values, iter_scipy, 's-', label='CG (SciPy)')
        ax.set_xlabel('Размерность n')
        ax.set_ylabel('Количество итераций')
        ax.set_title('Сравнение количества итераций двух реализаций метода CG')
        ax.set_xticks(np.arange(5, 150, 10))
        ax.set_yticks(np.arange(min(min(iter_self), min(iter_scipy)),
                                max(max(iter_self), max(iter_scipy)) + 1))
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
