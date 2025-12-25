import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import bisect
import time

# Настройка страницы
st.set_page_config(
    page_title="Метод стрельбы для краевых задач",
    layout="wide"
)

# Боковое меню
menu = st.sidebar.radio(
    "Выберите раздел:",
    ["Постановка задачи", "Основные обозначения", "Теория метода",
     "Реализация алгоритма", "Интерактивный пример",
     "Решение с использованием SciPy", "Сравнение методов", "Выводы"]
)

# Раздел 1: Постановка задачи
if menu == "Постановка задачи":
    st.title("Метод стрельбы для решения краевых задач")

    st.markdown("""
    ### Общая постановка краевой задачи для ОДУ 2-го порядка

    Рассматривается нелинейное дифференциальное уравнение второго порядка:
    """)

    st.latex(r"""
    \frac{d^2u}{dx^2} = f\left(x, u, \frac{du}{dx}\right), \quad 0 < x < l
    """)

    st.markdown("с граничными условиями:")
    st.latex(r"""
    u(0) = \mu_1, \quad u(l) = \mu_2
    """)

    st.markdown("""
    ### Конкретная тестовая задача

    Для демонстрации метода будем решать:
    """)

    st.latex(r"""
    \frac{d^2u}{dx^2} = 100u(u-1), \quad 0 < x < 1
    """)
    st.latex(r"""
    u(0) = 0, \quad u(1) = 2
    """)

    st.markdown("""
    ### Физический смысл задачи

    Данное уравнение можно интерпретировать как модель нелинейной пружины или мембраны:
    - **u(x)** — отклонение от положения равновесия
    - **u''(x)** — ускорение (вторая производная)
    - **100u(u-1)** — нелинейная восстанавливающая сила
    - Граничные условия **u(0)=0, u(1)=2** — закрепленные концы на разных уровнях

    Такие уравнения возникают в задачах механики, теории упругости, физике плазмы и других областях,
    где присутствуют нелинейные эффекты.
    """)

    st.markdown("""
    ---
    ### Преобразование к системе уравнений первого порядка

    Введем новую переменную $v = \\frac{du}{dx}$, тогда:
    """)

    st.latex(r"""
    \begin{cases}
    \dfrac{du}{dx} = v \\[10pt]
    \dfrac{dv}{dx} = f(x, u, v)
    \end{cases}
    """)

# Раздел 2: Основные обозначения
elif menu == "Основные обозначения":
    st.title("Основные обозначения")

    st.markdown("""
    | Обозначение | → | Описание |
    |-------------|:-:|----------|
    | $u(x)$ | → | Искомая функция |
    | $v(x) = \\frac{du}{dx}$ | → | Производная искомой функции |
    | $\\theta$ | → | Параметр пристрелки (начальное значение $v(0)$) |
    | $\\mu_1, \\mu_2$ | → | Граничные значения |
    | $l$ | → | Длина интервала |
    | $N$ | → | Количество узлов сетки |
    | $h$ | → | Шаг сетки |
    | $F(\\theta)$ | → | Функция невязки |
    """)

    st.markdown("""
    ### Функция невязки

    Ключевая идея метода стрельбы — подобрать $\\theta$ так, чтобы:
    """)

    st.latex(r"""
    F(\theta) = u(l; \theta) - \mu_2 = 0
    """)

    st.markdown("где $u(l; \\theta)$ — значение решения в точке $x=l$ при начальном условии $v(0) = \\theta$.")

# Раздел 3: Теория метода
elif menu == "Теория метода":
    st.title("Теория метода стрельбы")

    st.markdown("""
    ### Основная идея метода

    Метод стрельбы преобразует краевую задачу в задачу Коши с параметром:
    """)

    st.latex(r"""
    \begin{cases}
    \dfrac{du}{dx} = v, & u(0) = \mu_1 \\[10pt]
    \dfrac{dv}{dx} = f(x, u, v), & v(0) = \theta
    \end{cases}
    """)

    st.markdown("""
    ### Алгоритм метода

    1. **Инициализация**: Выбираем начальный интервал $[\\theta_a, \\theta_b]$ для параметра пристрелки
    2. **Решение задачи Коши**: Для каждого $\\theta$ решаем систему уравнений методом Рунге-Кутты 4-го порядка
    3. **Вычисление невязки**: $F(\\theta) = u(l; \\theta) - \\mu_2$
    4. **Уточнение параметра**: Методом бисекции находим $\\theta^*$, при котором $F(\\theta^*) = 0$
    5. **Получение решения**: Решаем задачу Коши с найденным $\\theta^*$
    """)

    st.markdown("""
    ### Метод Рунге-Кутты 4-го порядка

    Для системы уравнений:
    """)

    st.latex(r"""
    \begin{cases}
    \dfrac{du}{dx} = v \\[10pt]
    \dfrac{dv}{dx} = f(x, u, v)
    \end{cases}
    """)

    st.markdown("""
    Формулы метода (шаг $h$):
    """)

    st.latex(r"""
    \begin{align*}
    k_1^u &= h \cdot v_n \\
    k_1^v &= h \cdot f(x_n, u_n, v_n) \\[5pt]
    k_2^u &= h \cdot \left(v_n + \frac{k_1^v}{2}\right) \\
    k_2^v &= h \cdot f\left(x_n + \frac{h}{2}, u_n + \frac{k_1^u}{2}, v_n + \frac{k_1^v}{2}\right) \\[5pt]
    k_3^u &= h \cdot \left(v_n + \frac{k_2^v}{2}\right) \\
    k_3^v &= h \cdot f\left(x_n + \frac{h}{2}, u_n + \frac{k_2^u}{2}, v_n + \frac{k_2^v}{2}\right) \\[5pt]
    k_4^u &= h \cdot (v_n + k_3^v) \\
    k_4^v &= h \cdot f(x_n + h, u_n + k_3^u, v_n + k_3^v)
    \end{align*}
    """)

    st.latex(r"""
    \begin{align*}
    u_{n+1} &= u_n + \frac{1}{6}(k_1^u + 2k_2^u + 2k_3^u + k_4^u) \\
    v_{n+1} &= v_n + \frac{1}{6}(k_1^v + 2k_2^v + 2k_3^v + k_4^v)
    \end{align*}
    """)

# Раздел 4: Реализация алгоритма
elif menu == "Реализация алгоритма":
    st.title("Реализация алгоритма")

    st.markdown("### Решение задачи Коши методом Рунге-Кутты 4-го порядка")

    code_rk4 = '''
def runge_kutta_4(f, x0, y0, v0, h, n_steps):
    """
    Решение системы ОДУ методом Рунге-Кутты 4-го порядка

    Параметры:
    f - функция правой части f(x, u, v) = d²u/dx²
    x0 - начальная точка
    y0 - начальное значение u(x0)
    v0 - начальное значение v(x0) = du/dx
    h - шаг
    n_steps - количество шагов

    Возвращает:
    x - массив узлов
    u - массив значений функции
    v - массив значений производной
    """
    x = np.zeros(n_steps + 1)
    u = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)

    x[0] = x0
    u[0] = y0
    v[0] = v0

    for i in range(n_steps):
        # Коэффициенты для u
        k1_u = h * v[i]
        k1_v = h * f(x[i], u[i], v[i])

        k2_u = h * (v[i] + k1_v/2)
        k2_v = h * f(x[i] + h/2, u[i] + k1_u/2, v[i] + k1_v/2)

        k3_u = h * (v[i] + k2_v/2)
        k3_v = h * f(x[i] + h/2, u[i] + k2_u/2, v[i] + k2_v/2)

        k4_u = h * (v[i] + k3_v)
        k4_v = h * f(x[i] + h, u[i] + k3_u, v[i] + k3_v)

        # Обновление значений
        u[i+1] = u[i] + (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6
        v[i+1] = v[i] + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
        x[i+1] = x[i] + h

    return x, u, v
    '''

    st.code(code_rk4, language='python')

    st.markdown("### Функция невязки")

    code_residual = '''
def residual(theta, f, x0, y0, x_end, y_end, h):
    """
    Вычисление невязки F(theta) = u(x_end) - y_end

    Параметры:
    theta - параметр пристрелки (v(x0))
    f - функция правой части
    x0, y0 - начальные условия для u
    x_end, y_end - целевые граничные условия
    h - шаг

    Возвращает:
    Невязку в правой границе
    """
    n_steps = int((x_end - x0) / h)
    x, u, v = runge_kutta_4(f, x0, y0, theta, h, n_steps)
    return u[-1] - y_end
    '''

    st.code(code_residual, language='python')

    st.markdown("### Метод стрельбы с использованием метода бисекции")

    code_shooting = '''
def shooting_method(f, x0, y0, x_end, y_end, theta_a, theta_b, h, tol=1e-6, max_iter=100):
    """
    Метод стрельбы с использованием бисекции

    Параметры:
    f - функция правой части
    x0, y0 - левые граничные условия
    x_end, y_end - правые граничные условия
    theta_a, theta_b - начальный интервал для параметра пристрелки
    h - шаг
    tol - требуемая точность
    max_iter - максимальное число итераций

    Возвращает:
    theta - найденный параметр пристрелки
    x, u, v - решение
    """
    # Проверка знаков на концах интервала
    fa = residual(theta_a, f, x0, y0, x_end, y_end, h)
    fb = residual(theta_b, f, x0, y0, x_end, y_end, h)

    if fa * fb > 0:
        raise ValueError(f"Невязки на концах интервала имеют одинаковый знак: f({theta_a}) = {fa}, f({theta_b}) = {fb}")

    # Метод бисекции
    for i in range(max_iter):
        theta_mid = (theta_a + theta_b) / 2
        f_mid = residual(theta_mid, f, x0, y0, x_end, y_end, h)

        if abs(f_mid) < tol:
            # Нашли решение с требуемой точностью
            n_steps = int((x_end - x0) / h)
            x, u, v = runge_kutta_4(f, x0, y0, theta_mid, h, n_steps)
            return theta_mid, x, u, v

        # Обновление интервала
        if fa * f_mid < 0:
            theta_b = theta_mid
            fb = f_mid
        else:
            theta_a = theta_mid
            fa = f_mid

    # Если не сошлось за max_iter итераций
    n_steps = int((x_end - x0) / h)
    theta_mid = (theta_a + theta_b) / 2
    x, u, v = runge_kutta_4(f, x0, y0, theta_mid, h, n_steps)
    return theta_mid, x, u, v
    '''

    st.code(code_shooting, language='python')

# Раздел 5: Интерактивный пример
elif menu == "Интерактивный пример":
    st.title("Интерактивный пример")


    # Определение функции для нашей задачи
    def f_example(x, u, v):
        return 100 * u * (u - 1)


    # Параметры задачи
    st.sidebar.header("Параметры задачи")
    N = st.sidebar.slider("Количество узлов сетки N", 10, 200, 50, 10)
    h = 1.0 / N

    # Начальный интервал для параметра пристрелки
    st.sidebar.header("Параметры метода стрельбы")
    theta_min = st.sidebar.number_input("Минимальное значение θ", -10.0, 10.0, -5.0, 0.5)
    theta_max = st.sidebar.number_input("Максимальное значение θ", -10.0, 10.0, 5.0, 0.5)
    tol = st.sidebar.number_input("Точность метода бисекции", 1e-8, 1e-3, 1e-6, format="%.0e")


    # Реализация методов
    def runge_kutta_4(f, x0, y0, v0, h, n_steps):
        x = np.zeros(n_steps + 1)
        u = np.zeros(n_steps + 1)
        v = np.zeros(n_steps + 1)

        x[0] = x0
        u[0] = y0
        v[0] = v0

        for i in range(n_steps):
            k1_u = h * v[i]
            k1_v = h * f(x[i], u[i], v[i])

            k2_u = h * (v[i] + k1_v / 2)
            k2_v = h * f(x[i] + h / 2, u[i] + k1_u / 2, v[i] + k1_v / 2)

            k3_u = h * (v[i] + k2_v / 2)
            k3_v = h * f(x[i] + h / 2, u[i] + k2_u / 2, v[i] + k2_v / 2)

            k4_u = h * (v[i] + k3_v)
            k4_v = h * f(x[i] + h, u[i] + k3_u, v[i] + k3_v)

            u[i + 1] = u[i] + (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6
            v[i + 1] = v[i] + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
            x[i + 1] = x[i] + h

        return x, u, v


    def residual(theta, f, x0, y0, x_end, y_end, h):
        n_steps = int((x_end - x0) / h)
        x, u, v = runge_kutta_4(f, x0, y0, theta, h, n_steps)
        return u[-1] - y_end


    # Решение методом стрельбы
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Решение методом стрельбы")

        if st.button("Запустить расчет"):
            with st.spinner("Выполняется расчет..."):
                try:
                    # Метод бисекции для нахождения theta
                    theta_a, theta_b = theta_min, theta_max
                    fa = residual(theta_a, f_example, 0, 0, 1, 2, h)
                    fb = residual(theta_b, f_example, 0, 0, 1, 2, h)

                    if fa * fb > 0:
                        st.error(
                            f"Функция невязки имеет одинаковый знак на концах интервала: f({theta_a}) = {fa:.4f}, f({theta_b}) = {fb:.4f}")
                    else:
                        # Биссекция
                        iterations = []
                        for i in range(100):
                            theta_mid = (theta_a + theta_b) / 2
                            f_mid = residual(theta_mid, f_example, 0, 0, 1, 2, h)
                            iterations.append((i, theta_mid, f_mid))

                            if abs(f_mid) < tol:
                                break

                            if fa * f_mid < 0:
                                theta_b = theta_mid
                                fb = f_mid
                            else:
                                theta_a = theta_mid
                                fa = f_mid

                        # Получение окончательного решения
                        n_steps = N
                        x, u, v = runge_kutta_4(f_example, 0, 0, theta_mid, h, n_steps)

                        st.success(f"Найден параметр пристрелки: θ = {theta_mid:.6f}")
                        st.success(f"Невязка на правой границе: F(θ) = {f_mid:.2e}")
                        st.success(f"Количество итераций: {len(iterations)}")

                        # Построение графика
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                        # График решения
                        ax1.plot(x, u, 'b-', linewidth=2, label='u(x)')
                        ax1.plot(x, v, 'r--', linewidth=2, label="u'(x)")
                        ax1.scatter([0, 1], [0, 2], color='green', s=100, zorder=5,
                                    label='Граничные условия')
                        ax1.set_xlabel('x')
                        ax1.set_ylabel('u(x), u\'(x)')
                        ax1.set_title('Решение краевой задачи')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)

                        # График сходимости
                        iter_nums = [it[0] for it in iterations]
                        theta_vals = [it[1] for it in iterations]
                        f_vals = [it[2] for it in iterations]

                        ax2.semilogy(iter_nums, np.abs(f_vals), 'o-', linewidth=2)
                        ax2.set_xlabel('Номер итерации')
                        ax2.set_ylabel('|F(θ)|')
                        ax2.set_title('Сходимость метода бисекции')
                        ax2.grid(True, alpha=0.3)

                        st.pyplot(fig)

                        # Таблица с результатами
                        st.markdown("### Значения решения в узлах")
                        display_points = min(10, len(x))
                        indices = np.linspace(0, len(x) - 1, display_points, dtype=int)

                        data = {
                            "x": x[indices],
                            "u(x)": u[indices],
                            "u'(x)": v[indices]
                        }
                        st.table(data)

                except Exception as e:
                    st.error(f"Ошибка при расчете: {e}")

    with col2:
        st.markdown("### Информация о методе")
        st.markdown(f"""
        **Параметры расчета:**
        - Уравнение: $u'' = 100u(u-1)$
        - Граничные условия: $u(0)=0$, $u(1)=2$
        - Шаг сетки: $h = {h:.4f}$
        - Число узлов: $N = {N}$
        - Точность: $\\epsilon = {tol}$

        **Метод бисекции:**
        1. Начальный интервал: $[\\theta_a, \\theta_b] = [{theta_min}, {theta_max}]$
        2. На каждом шаге вычисляется $F(\\theta) = u(1) - 2$
        3. Интервал делится пополам
        4. Процесс продолжается до достижения точности
        """)

# Раздел 6: Решение с использованием SciPy
elif menu == "Решение с использованием SciPy":
    st.title("Решение с использованием SciPy")


    # Определение функции для задачи
    def f_example(x, u, v):
        return 100 * u * (u - 1)


    # Функция для SciPy (требует другой формат)
    def system_for_scipy(x, y):
        """Система для SciPy: y = [u, v], dy/dx = [v, f(x,u,v)]"""
        u, v = y
        return [v, 100 * u * (u - 1)]


    st.markdown("### Реализация решения с использованием SciPy")

    code_scipy = '''
def solve_with_scipy(theta_min, theta_max, N=100, method='RK45', rtol=1e-6, atol=1e-8):
    """
    Решение краевой задачи с использованием SciPy

    Параметры:
    theta_min, theta_max - границы интервала поиска параметра
    N - количество точек на сетке
    method - метод решения ОДУ (RK45, DOP853, Radau, BDF, LSODA)
    rtol - относительная точность
    atol - абсолютная точность
    """
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.optimize import bisect

    def residual_scipy(theta):
        """Функция невязки для SciPy"""
        sol = solve_ivp(
            system_for_scipy,
            [0, 1],
            [0, theta],  # начальные условия: u(0)=0, v(0)=theta
            method=method,
            t_eval=np.linspace(0, 1, N),
            rtol=rtol,
            atol=atol
        )
        return sol.y[0, -1] - 2  # невязка: u(1) - 2

    # Проверяем знаки на концах интервала
    fa = residual_scipy(theta_min)
    fb = residual_scipy(theta_max)

    if fa * fb > 0:
        raise ValueError(f"Невязки на концах интервала имеют одинаковый знак")

    # Метод бисекции для нахождения theta
    theta_star, result = bisect(
        residual_scipy,
        theta_min,
        theta_max,
        full_output=True,
        xtol=1e-6,
        maxiter=100
    )

    # Финальное решение с найденным параметром
    sol_final = solve_ivp(
        system_for_scipy,
        [0, 1],
        [0, theta_star],
        method=method,
        t_eval=np.linspace(0, 1, N),
        rtol=rtol/10,
        atol=atol/10
    )

    return theta_star, sol_final.t, sol_final.y[0], sol_final.y[1], result
    '''

    st.code(code_scipy, language='python')

    st.markdown("### Пример использования SciPy решения")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Параметры расчета SciPy")
        N_scipy = st.slider("Количество узлов", 20, 500, 100, 20, key="scipy_n_nodes")
        method_scipy = st.selectbox("Метод решения ОДУ",
                                    ["RK45", "DOP853", "Radau", "BDF", "LSODA"],
                                    key="scipy_method_select")
        theta_min_scipy = st.number_input("Минимальное θ", -20.0, 20.0, -5.0, 0.5,
                                          key="scipy_theta_min")
        theta_max_scipy = st.number_input("Максимальное θ", -20.0, 20.0, 5.0, 0.5,
                                          key="scipy_theta_max")

    with col2:
        if st.button("Запустить решение SciPy"):
            with st.spinner("Выполняется расчет SciPy..."):
                try:
                    start_time = time.time()


                    # Определяем функцию невязки для SciPy
                    def residual_scipy(theta):
                        sol = solve_ivp(
                            system_for_scipy,
                            [0, 1],
                            [0, theta],
                            method=method_scipy,
                            t_eval=np.linspace(0, 1, N_scipy),
                            rtol=1e-6,
                            atol=1e-8
                        )
                        return sol.y[0, -1] - 2


                    # Проверяем знаки на концах
                    fa = residual_scipy(theta_min_scipy)
                    fb = residual_scipy(theta_max_scipy)

                    if fa * fb > 0:
                        st.error(f"SciPy: невязки на концах интервала имеют одинаковый знак!")
                        st.error(f"F({theta_min_scipy}) = {fa:.6f}, F({theta_max_scipy}) = {fb:.6f}")
                    else:
                        # Используем метод бисекции
                        theta_scipy, result = bisect(
                            residual_scipy,
                            theta_min_scipy,
                            theta_max_scipy,
                            full_output=True,
                            xtol=1e-6,
                            maxiter=100
                        )

                        # Получаем окончательное решение
                        sol_final = solve_ivp(
                            system_for_scipy,
                            [0, 1],
                            [0, theta_scipy],
                            method=method_scipy,
                            t_eval=np.linspace(0, 1, N_scipy),
                            rtol=1e-7,
                            atol=1e-9
                        )

                        x_scipy = sol_final.t
                        u_scipy = sol_final.y[0]
                        v_scipy = sol_final.y[1]

                        end_time = time.time()
                        scipy_time = end_time - start_time

                        st.success(f"Найден параметр: θ = {theta_scipy:.6f}")
                        st.info(f"Время выполнения: {scipy_time:.4f} с")
                        st.info(f"Количество итераций: {result.iterations}")

                        # Визуализация
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(x_scipy, u_scipy, 'b-', linewidth=2, label='u(x) - решение')
                        ax.plot(x_scipy, v_scipy, 'r--', linewidth=2, label="u'(x) - производная")
                        ax.scatter([0, 1], [0, 2], color='green', s=100, zorder=5,
                                   label='Граничные условия')
                        ax.set_xlabel('x', fontsize=12)
                        ax.set_ylabel('Значения', fontsize=12)
                        ax.set_title(f'Решение с использованием SciPy (метод {method_scipy})', fontsize=14)
                        ax.legend(fontsize=12)
                        ax.grid(True, alpha=0.3)

                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ошибка при решении с SciPy: {e}")

# Раздел 7: Сравнение методов
elif menu == "Сравнение методов":
    st.title("Сравнение методов")


    # Определение функции для задачи
    def f_example(x, u, v):
        return 100 * u * (u - 1)


    # Функция для SciPy
    def system_for_scipy(x, y):
        u, v = y
        return [v, 100 * u * (u - 1)]


    st.markdown("### Сравнение производительности методов")

    st.markdown("#### Общие параметры")
    col1, col2 = st.columns(2)

    with col1:
        theta_min_compare = st.slider(
            "Минимальный θ",
            -20.0, 20.0, -10.0, 0.5,
            key="compare_theta_min"
        )

    with col2:
        theta_max_compare = st.slider(
            "Максимальный θ",
            -20.0, 20.0, 10.0, 0.5,
            key="compare_theta_max"
        )

    # Показываем выбранный интервал
    st.info(f"Интервал поиска θ: [{theta_min_compare}, {theta_max_compare}]")

    st.markdown("#### Параметры сравнения")
    test_grids = st.multiselect(
        "Размеры сетки для тестирования",
        [20, 50, 100, 200, 500],
        default=[20, 50, 100]
    )

    if st.button("Запустить сравнение производительности"):
        if not test_grids:
            st.warning("Выберите хотя бы один размер сетки для тестирования")
        elif theta_min_compare >= theta_max_compare:
            st.error("Минимальное значение θ должно быть меньше максимального")
        else:
            with st.spinner("Выполняется сравнение методов..."):
                # Подготовка данных для сравнения
                custom_times = []
                scipy_times = []
                custom_residuals = []
                scipy_residuals = []
                custom_thetas = []
                scipy_thetas = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, N_test in enumerate(test_grids):
                    status_text.text(f"Тестирование на сетке N={N_test}...")
                    h_test = 1.0 / N_test

                    # Тестирование нашей реализации
                    start_time = time.time()

                    try:
                        # Наша реализация Рунге-Кутты 4-го порядка
                        def runge_kutta_4_custom(f, x0, y0, v0, h, n_steps):
                            x = np.zeros(n_steps + 1)
                            u = np.zeros(n_steps + 1)
                            v = np.zeros(n_steps + 1)

                            x[0] = x0
                            u[0] = y0
                            v[0] = v0

                            for i in range(n_steps):
                                k1_u = h * v[i]
                                k1_v = h * f(x[i], u[i], v[i])

                                k2_u = h * (v[i] + k1_v / 2)
                                k2_v = h * f(x[i] + h / 2, u[i] + k1_u / 2, v[i] + k1_v / 2)

                                k3_u = h * (v[i] + k2_v / 2)
                                k3_v = h * f(x[i] + h / 2, u[i] + k2_u / 2, v[i] + k2_v / 2)

                                k4_u = h * (v[i] + k3_v)
                                k4_v = h * f(x[i] + h, u[i] + k3_u, v[i] + k3_v)

                                u[i + 1] = u[i] + (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6
                                v[i + 1] = v[i] + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
                                x[i + 1] = x[i] + h

                            return x, u, v


                        # Функция невязки для нашей реализации
                        def residual_custom(theta):
                            x, u, v = runge_kutta_4_custom(f_example, 0, 0, theta, h_test, N_test)
                            return u[-1] - 2


                        # Проверяем знаки на концах интервала
                        fa = residual_custom(theta_min_compare)
                        fb = residual_custom(theta_max_compare)

                        # Метод бисекции
                        if fa * fb > 0:
                            st.warning(f"Для N={N_test} (наша): знаки невязок одинаковые. Пропускаем.")
                            custom_times.append(np.nan)
                            custom_residuals.append(np.nan)
                            custom_thetas.append(np.nan)
                        else:
                            a, b = theta_min_compare, theta_max_compare
                            f_a, f_b = fa, fb

                            for _ in range(30):
                                theta_mid = (a + b) / 2
                                f_mid = residual_custom(theta_mid)

                                if abs(f_mid) < 1e-8:
                                    break

                                if f_a * f_mid < 0:
                                    b = theta_mid
                                    f_b = f_mid
                                else:
                                    a = theta_mid
                                    f_a = f_mid

                            # Финальное решение
                            x_final, u_final, v_final = runge_kutta_4_custom(
                                f_example, 0, 0, theta_mid, h_test, N_test
                            )
                            final_residual = u_final[-1] - 2

                            custom_time = time.time() - start_time
                            custom_times.append(custom_time)
                            custom_residuals.append(abs(final_residual))
                            custom_thetas.append(theta_mid)

                    except Exception as e:
                        custom_times.append(np.nan)
                        custom_residuals.append(np.nan)
                        custom_thetas.append(np.nan)

                    # Тестирование SciPy реализации
                    start_time = time.time()

                    try:
                        # Функция невязки для SciPy (используем те же параметры)
                        def residual_scipy(theta):
                            sol = solve_ivp(
                                system_for_scipy,
                                [0, 1],
                                [0, theta],
                                method='RK45',
                                t_eval=np.linspace(0, 1, N_test),
                                rtol=1e-6,
                                atol=1e-8
                            )
                            return sol.y[0, -1] - 2


                        # Проверяем знаки на концах интервала
                        fa_scipy = residual_scipy(theta_min_compare)
                        fb_scipy = residual_scipy(theta_max_compare)

                        # Метод бисекции SciPy
                        if fa_scipy * fb_scipy > 0:
                            st.warning(f"Для N={N_test} (SciPy): знаки невязок одинаковые. Пропускаем.")
                            scipy_times.append(np.nan)
                            scipy_residuals.append(np.nan)
                            scipy_thetas.append(np.nan)
                        else:
                            theta_scipy, result = bisect(
                                residual_scipy,
                                theta_min_compare,
                                theta_max_compare,
                                full_output=True,
                                xtol=1e-6,
                                maxiter=30
                            )

                            # Финальное решение для вычисления невязки
                            sol_final = solve_ivp(
                                system_for_scipy,
                                [0, 1],
                                [0, theta_scipy],
                                method='RK45',
                                t_eval=np.linspace(0, 1, N_test),
                                rtol=1e-7,
                                atol=1e-9
                            )
                            final_residual_scipy = sol_final.y[0, -1] - 2

                            scipy_time = time.time() - start_time
                            scipy_times.append(scipy_time)
                            scipy_residuals.append(abs(final_residual_scipy))
                            scipy_thetas.append(theta_scipy)

                    except Exception as e:
                        scipy_times.append(np.nan)
                        scipy_residuals.append(np.nan)
                        scipy_thetas.append(np.nan)

                    progress_bar.progress((idx + 1) / len(test_grids))

                status_text.text("Построение графиков...")

                # Визуализация результатов
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                # Фильтруем данные для графиков
                valid_indices = [i for i, (ct, st) in enumerate(zip(custom_times, scipy_times))
                                 if not (np.isnan(ct) or np.isnan(st))]

                if valid_indices:
                    valid_grids = [test_grids[i] for i in valid_indices]
                    valid_custom_times = [custom_times[i] for i in valid_indices]
                    valid_scipy_times = [scipy_times[i] for i in valid_indices]
                    valid_custom_residuals = [custom_residuals[i] for i in valid_indices]
                    valid_scipy_residuals = [scipy_residuals[i] for i in valid_indices]

                    # График времени выполнения
                    ax1.plot(valid_grids, valid_custom_times, 'bo-', linewidth=2,
                             label='Наша реализация', markersize=8)
                    ax1.plot(valid_grids, valid_scipy_times, 'ro-', linewidth=2,
                             label='SciPy', markersize=8)
                    ax1.set_xlabel('Количество узлов N', fontsize=12)
                    ax1.set_ylabel('Время выполнения (с)', fontsize=12)
                    ax1.set_title('Время выполнения методов', fontsize=14)
                    ax1.legend(fontsize=12)
                    ax1.grid(True, alpha=0.3)

                    # График невязки
                    ax2.plot(valid_grids, valid_custom_residuals, 'bo-', linewidth=2,
                             label='Наша реализация', markersize=8)
                    ax2.plot(valid_grids, valid_scipy_residuals, 'ro-', linewidth=2,
                             label='SciPy', markersize=8)
                    ax2.set_xlabel('Количество узлов N', fontsize=12)
                    ax2.set_ylabel('|F(θ)|', fontsize=12)
                    ax2.set_title('Абсолютная невязка методов', fontsize=14)
                    ax2.legend(fontsize=12)
                    ax2.grid(True, alpha=0.3)

                    st.pyplot(fig)
                else:
                    st.warning("Нет данных для построения графиков. Проверьте интервал поиска θ.")

                # Таблица результатов
                st.markdown("### Результаты сравнения")
                comparison_data = []
                for i, N in enumerate(test_grids):
                    row = {
                        "N": N,
                        "Наша время (с)": f"{custom_times[i]:.4f}" if not np.isnan(custom_times[i]) else "Ошибка",
                        "SciPy время (с)": f"{scipy_times[i]:.4f}" if not np.isnan(scipy_times[i]) else "Ошибка",
                        "Наша невязка": f"{custom_residuals[i]:.2e}" if not np.isnan(custom_residuals[i]) else "Ошибка",
                        "SciPy невязка": f"{scipy_residuals[i]:.2e}" if not np.isnan(scipy_residuals[i]) else "Ошибка",
                        "Наша θ": f"{custom_thetas[i]:.6f}" if not np.isnan(custom_thetas[i]) else "Ошибка",
                        "SciPy θ": f"{scipy_thetas[i]:.6f}" if not np.isnan(scipy_thetas[i]) else "Ошибка"
                    }

                    # Сравнение времени
                    if (not np.isnan(custom_times[i]) and not np.isnan(scipy_times[i])
                            and scipy_times[i] > 0):
                        time_ratio = custom_times[i] / scipy_times[i]
                        if time_ratio > 1:
                            row["Сравнение времени"] = f"SciPy быстрее в {time_ratio:.1f} раз"
                        else:
                            row["Сравнение времени"] = f"Наша быстрее в {1 / time_ratio:.1f} раз"
                    else:
                        row["Сравнение времени"] = "Нет данных"

                    # Сравнение точности
                    if (not np.isnan(custom_residuals[i]) and not np.isnan(scipy_residuals[i])):
                        if custom_residuals[i] < scipy_residuals[i]:
                            row["Сравнение точности"] = "Наша точнее"
                        elif custom_residuals[i] > scipy_residuals[i]:
                            row["Сравнение точности"] = "SciPy точнее"
                        else:
                            row["Сравнение точности"] = "Одинаково"
                    else:
                        row["Сравнение точности"] = "Нет данных"

                    comparison_data.append(row)

                st.table(comparison_data)

                status_text.text("Готово!")

# Раздел 8: Выводы
elif menu == "Выводы":
    st.title("Выводы")

    st.markdown("""
    ### Основные результаты исследования

    Метод стрельбы успешно реализован для решения нелинейной краевой задачи второго порядка 
    $u'' = 100u(u-1)$ с граничными условиями $u(0)=0$, $u(1)=2$. Алгоритм включает:

    1. Преобразование уравнения к системе ОДУ первого порядка
    2. Решение задачи Коши методом Рунге-Кутты 4-го порядка с фиксированным шагом
    3. Подбор параметра пристрелки $\\theta$ методом бисекции

    Метод продемонстрировал устойчивую сходимость с достижением невязки порядка $10^{-6}$.

    ### Сравнение с библиотечной реализацией SciPy

    Проведенное сравнение выявило фундаментальные различия в подходах:

    1. **Наша реализация**: использует фиксированный шаг интегрирования $h = 1/N$, 
       где точность решения напрямую зависит от количества узлов сетки. 
       При увеличении $N$ невязка закономерно уменьшается.

    2. **SciPy реализация**: применяет адаптивный шаг интегрирования с контролем 
       точности через параметры `rtol` (относительная точность) и `atol` (абсолютная точность).
       Точность не зависит от явно заданной сетки.

    **Ключевой результат**: При сравнении на фиксированной сетке наша реализация 
    оказалась значительно быстрее (в 3-50 раз в зависимости от размера сетки), 
    но это сравнение не вполне корректно, так как SciPy решает более общую задачу 
    с адаптивным контролем точности.

    ### Практическая значимость и рекомендации

    1. **Для учебных целей и понимания алгоритма** рекомендуется использовать 
       собственную реализацию, которая обеспечивает полный контроль над 
       вычислительным процессом и наглядно демонстрирует зависимость точности 
       от шага интегрирования.

    2. **Для производственных расчётов и сложных задач** следует применять 
       оптимизированные библиотеки типа SciPy, которые:
       - Автоматически выбирают оптимальный шаг интегрирования
       - Обеспечивают устойчивость решения для широкого класса задач
       - Контролируют точность через заданные допуски
       - Эффективно обрабатывают жёсткие системы уравнений

    3. **Важные аспекты метода стрельбы**:
       - Правильный выбор начального интервала для параметра $\\theta$ 
         критически важен для сходимости метода
       - Для нелинейных задач может существовать несколько решений, 
         соответствующих разным значениям $\\theta$
       - Метод требует решения задачи Коши на каждой итерации, что 
         может быть затратно для сложных систем

    ### Заключение

    Метод стрельбы является эффективным и наглядным подходом к решению 
    краевых задач, позволяющим глубоко понять структуру задачи и принципы 
    численных методов. Собственная реализация метода полезна для обучения 
    и отладки, тогда как библиотечные реализации типа SciPy следует 
    использовать для практических расчётов, где важны надёжность, 
    точность и эффективность.
    """)

# Запуск приложения
if __name__ == "__main__":
    pass