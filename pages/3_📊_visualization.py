from functions import *

Streamlit_Funcs.write_align(text="Visualization / Simple Anlaysis", pos="left", style="h1")

if 'city_obj' in st.session_state:
    city_obj = st.session_state["city_obj"]
    Streamlit_Funcs.write_align(text="Plotting Cities", pos="left")

    fig = city_obj.visualize_city_or_sol()
    st.pyplot(fig)


