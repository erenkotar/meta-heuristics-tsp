from functions import *

st.set_page_config(page_title="Meta-heuristics on TSP", layout="wide")

tabs= ["Welcome","Inputs","Visualization / Simple Anlaysis","About"]

# page = st.sidebar.radio("Tabs",tabs)
st.sidebar.success("Select a page above.")

st.markdown("<h3 style='text-align:center;'>Heuristic Optimization Approaches to Traveling Salesman Problem (TSP)</h3>",unsafe_allow_html=True)

st.write("""**This project is allows you to:**""")
st.write("""    - Conduct Analysis on uploaded or randomly generated location points""")
st.write("""    - Optimize the solution through using various meta-heuristics with flexibility""")
st.write("""    - Analyze outputs to comphrend algorithm's behaviours""")

st.write("""**Current Constructive Heuristics:**""")
st.write("""    - Random Insertion""")
st.write("""    - Nearest Neighbor""")
st.write("""    - Nearest Insertion""")
st.write("""    - Farthest Insertion""")
st.write("""    - Arbitrary Insertion""")

st.write("""**Current Improvement Heuristics:**""")
st.write("""    - Two Opt""")
st.write("""    - City Swap""")

st.write("""**Current Meta Heuristic Algorithms:**""")
st.write("""    - Simulated Annealing (SA)""")
st.write("""    - Stochastic Hill Climbing (SHC)""")
st.write("""    - Tabu Search -> Coming Soon""")
st.write("""    - Genetic Algorithm -> Coming Soon""")
st.write("""    - Particle Swarm Optimziation (PSO) -> Coming Soon""")
st.write("""    - Ant Colony Optimization (ACO) -> Coming Soon""")



    
    # if button==True:
    #     with st.spinner("Tahmin yapılıyor, lütfen bekleyiniz..."):
    #         start_date="2016-01-01"
    #         end_date=datetime.date.today()
    #         df=get_consumption_data(start_date=str(start_date),end_date=str(end_date)).iloc[:-1]
    #         fig1,fig2 = forecast_func(df,select_period(fh_selection))
    #         st.markdown("<h3 style='text-align:center;'>Tahmin sonuçları</h3>",unsafe_allow_html=True)
    #         st.plotly_chart(fig1)
    #         st.markdown("<h3 style='text-align:center;'>Model için en önemli değişkenler</h3>",unsafe_allow_html=True)
    #         st.plotly_chart(fig2)
    


    
    # start_date=st.sidebar.date_input(label="Başlangıç Tarihi",value=datetime.date.today()-datetime.timedelta(days=10),max_value=datetime.date.today())
    # end_date=st.sidebar.date_input(label="Bitiş Tarihi",value=datetime.date.today())
    # df_vis = get_consumption_data(start_date=str(start_date),end_date=str(end_date))
    # df_describe=pd.DataFrame(df_vis.describe())
    # st.markdown("<h3 style='text-align:center;'>Tüketim-Tanımlayıcı İstatistikler</h3>",unsafe_allow_html=True)
    # st.table(df_describe)

    # fig3=go.Figure()
    # fig3.add_trace(go.Scatter(x=df_vis.date,y=df_vis.consumption,mode='lines',name='Tüketim (MWh)'))
    # fig3.update_layout(xaxis_title='Date',yaxis_title="Consumption")
    # st.markdown("<h3 style='text-align:center;'>Saatlik Tüketim (MWh)</h3>",unsafe_allow_html=True)
    # st.plotly_chart(fig3)

