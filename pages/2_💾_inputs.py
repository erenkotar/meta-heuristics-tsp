from functions import *

Streamlit_Funcs.write_align(text="Input Page", pos="left", style="h1")

submit_2_1 = st.button("CLEAR STATE")
if submit_2_1:
    for key in st.session_state.keys():
        del st.session_state[key]
    st.write("All state variables successfully deleted!")

data_source = st.radio("**Select the preffered data source and click the **SUBMIT** button at the end of the page:**", options=("Insert data", "Use pre-downloaded data", "Randomly generate data"))

if data_source=="Insert data":
    uploaded_file = st.file_uploader("Please upload a .csv file here")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=0)

elif data_source=="Use pre-downloaded data":
    df = default_df.copy()

elif data_source=="Randomly generate data":
    N_ = st.number_input('Number of cities:',
                            min_value=10,
                            max_value=65)

    x_1, x_2 = st.select_slider(
        'Select a range of X axis',
        options= list(np.arange(1,101)),
        value=(1, 100),
        key="x_range_k")
    y_1, y_2 = st.select_slider(
        'Select a range of Y axis',
        options= list(np.arange(1,101)),
        value=(1, 100),
        key="y_range_k")
    x_r = [int(x_1), int(x_2)]
    y_r = [int(y_1), int(y_2)]

    random_seed_ = st.number_input('Random seed:',
                            min_value=1,
                            max_value=99)
    df = Cities.generate_cities(N=N_, x_range=x_r, y_range=y_r, random_seed=random_seed_)

submit_2_2 = st.button("SUBMIT")
if (df is not None) & (submit_2_2):
    st.write("""Preview of Dataframe:""")
    st.dataframe(df)
    # It also checks the data validity 
    city_obj = Cities(df)

if submit_2_2 and city_obj.data_validity:
    st.session_state["city_obj"] = city_obj
    st.write("Successfully submitted!")
elif submit_2_2 and ~city_obj.data_validity:
    st.write("Please select one of the options above to download data!")