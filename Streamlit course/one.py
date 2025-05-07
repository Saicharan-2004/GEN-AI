import streamlit as st

st.title("hello")
st.subheader("hi")
st.text("This is a text")
st.write("find the best person")

temp = st.select_slider("Your fav programming language:",["C","C++","Python","Java","Javascript"])


st.success(f"Your fav language is {temp}")


