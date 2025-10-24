import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Введение — Matplotlib", layout="centered")

st.markdown("""
# 🎨 Введение в Matplotlib
---
""")

col1, col2 = st.columns([2, 3])

with col1:
    st.image(
        "https://matplotlib.org/stable/_static/logo_light.svg",
        width=180,
        caption="Официальный логотип Matplotlib"
    )

with col2:
    st.markdown("""
    #### 📊 Что такое Matplotlib?

    **Matplotlib** — это мощная библиотека Python для **визуализации данных**.  
    Она позволяет создавать **наглядные, настраиваемые и публикационно-качественные** графики всего в несколько строк кода.

    💡 Используется в науке, аналитике данных, машинном обучении и инженерии.
    """)

st.divider()

st.markdown("""
#### 📘 Основная идея
> Сделать визуализацию данных простой, гибкой и понятной — чтобы каждый мог показать *данные в действии*.
""")

st.divider()

# Мини-график-пример
st.markdown("#### Пример визуализации:")

x = np.linspace(0, 10, 300)
y = np.sin(x) * np.exp(-x / 5)

fig, ax = plt.subplots()
ax.plot(x, y, color="crimson", linewidth=2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Демонстрация возможностей Matplotlib")
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)

st.caption("Matplotlib объединяет простоту и мощь: от базовых графиков до сложных визуальных композиций.")
