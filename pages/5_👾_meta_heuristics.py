from functions import *

Streamlit_Funcs.write_align(text="Meta-heuristics", pos="left", style="h1")


if 'city_obj' in st.session_state:
    df = st.session_state["city_obj"].coord_df
    ConsH = Construction_Heuristics_TSP(df)
    Streamlit_Funcs.write_align(text="", pos="left")
    random_seed_ = st.number_input('Random seed (for constructive-heuristic):',
                            min_value=1,
                            max_value=99,
                            key="random_seed_cons_heu")
    cons_heu = st.radio("**Select an initial solution with one of the Constructive Heuristic algortihms below and click the **SUBMIT CONSTRUCTIVE-HEURISTIC** button at the end of the page:**", 
                           options=("Random Insertion", "Nearest Neighbor", "Nearest Insertion", "Farthest Insertion", "Arbitrary Insertion"), key="cons_heu")
    
    submit_cons_heu = st.radio("**Submit your Constructive Heuristic algortihm to proceed**", 
                           options=("DONT SUBMIT", "SUBMIT CONSTRUCTIVE-HEURISTIC"), key="submit_cons_heu")
    if submit_cons_heu=="SUBMIT CONSTRUCTIVE-HEURISTIC":
        if cons_heu=="Random Insertion":
            with st.spinner(f"{cons_heu} is running, please wait..."):
                sol_dict=ConsH.random_insertion(random_seed=random_seed_)
        elif cons_heu=="Nearest Neighbor":
            with st.spinner(f"{cons_heu} is running, please wait..."):
                sol_dict=ConsH.nearest_neighbor(random_seed=random_seed_, take_best=True)
        elif cons_heu=="Nearest Insertion":
            with st.spinner(f"{cons_heu} is running, please wait..."):
                sol_dict=ConsH.nearest_insertion()
        elif cons_heu=="Farthest Insertion":
            with st.spinner(f"{cons_heu} is running, please wait..."):
                sol_dict=ConsH.farthest_insertion()
        elif cons_heu=="Arbitrary Insertion":
            with st.spinner(f"{cons_heu} is running, please wait..."):
                sol_dict=ConsH.arbitrary_insertion(random_seed=random_seed_)

        Streamlit_Funcs.write_align(text = f"Visualization of {cons_heu}", pos="left")
        fig = ConsH.visualize_city_or_sol(solution=sol_dict["sol"])
        st.pyplot(fig)

        Streamlit_Funcs.write_align(text = f"Once you satisfied with the initial solution, select the main meta-heuristic method:", pos="left")
        
        main_heu = st.radio("Select the preffered meta-heuristic", options=("Simulated Annealing (SA)", "Stochastic Hill Climbing (SHC)"), key="main_heu")

        if main_heu=="Simulated Annealing (SA)":
            
            T_schedule_ = st.radio("Schedule of Temperature", options=("Geometric", "Logarithmic", "Linear Multiplicative", "Exponential Multiplicative", "Logarithmical Multiplicative", "Quadratic Multiplicative"), key="T_schedule_")
            
            alpha_ = st.number_input(
            'Alpha (cooling parameter). Not affected in Logarithmic schedule!',
            min_value=0.8,
            max_value=1.0,
            key="alpha_")

            T_schedule_dict_ = {
                "Geometric":T_schedules.Geometric(alpha=alpha_),
                "Logarithmic":T_schedules.Logarithmic(),
                "Linear Multiplicative":T_schedules.Linear_Multiplicative(alpha=alpha_),
                "Exponential Multiplicative":T_schedules.Exponential_Multiplicative(alpha=alpha_),
                "Logarithmical Multiplicative":T_schedules.Logarithmical_Multiplicative(alpha=alpha_),
                "Quadratic Multiplicative":T_schedules.Quadratic_Multiplicative(alpha=alpha_)
            }

            T_start_ = st.number_input('Inital Temperature',
                                    min_value=10,
                                    max_value=2500,
                                    key="T_start_")
           
            T_thresh_ = st.number_input(
                'Threshold of Temperature',
                min_value=0.001,
                max_value=1.0,
                key="T_thresh_")
            
            k_ = st.slider(
                'Boltzmann Constant (in most of cases, it is 1)',
                min_value=0.5,
                max_value=1.0,
                step=0.1,
                key="k")

            random_seed_ = st.number_input('Random seed:',
                                    min_value=1,
                                    max_value=99,
                                    key="random_seed_main_heu")
            
            submit_cons_heu_params = st.radio("**Confirm the parameters to proceed**", 
                           options=("DONT CONFIRM PARAMETERS", "CONFIRM PARAMETERS"), key="submit_cons_heu_params")
            if submit_cons_heu_params=="CONFIRM PARAMETERS":
                sa = Meta_Heuristics.Simulated_Annealing(df, init_sol=sol_dict["sol"], 
                                                            T_schedule=T_schedule_dict_[T_schedule_], 
                                                            T_start=T_start_, T_thres=T_thresh_, k=k_)
                submit_5_3 = st.button("Anneal")
                if submit_5_3:
                    with st.spinner(text="Annealing..."):
                        results, best = sa.anneal(return_=True)
                        Streamlit_Funcs.write_align(text="SA Results", pos="left")
                        fig1 = sa.visualize_city_or_sol(solution=best["sol_s"])
                        st.pyplot(fig1)
                        Streamlit_Funcs.write_align(text="SA Report", pos="left")
                        fig2 = sa.visualize()
                        st.pyplot(fig2)

        elif main_heu=="Stochastic Hill Climbing (SHC)":
            st.write("Nothing yet")
            #     N_ = st.number_input('Number of cities:',
            #                             min_value=10,
            #                             max_value=65)

            #     x_1, x_2 = st.select_slider(
            #         'Select a range of X axis',
            #         options= list(np.arange(1,101)),
            #         value=(1, 100),
            #         key="x_range_k")
            #     y_1, y_2 = st.select_slider(
            #         'Select a range of X axis',
            #         options= list(np.arange(1,101)),
            #         value=(1, 100),
            #         key="y_range_k")
            #     x_r = [int(x_1), int(x_2)]
            #     y_r = [int(y_1), int(y_2)]

            #     random_seed_ = st.number_input('Random seed:',
            #                             min_value=1,
            #                             max_value=99)
            #     sa = Meta_Heuristics.Simulated_Annealing(df, init_sol=result_dict["sol"], 
            #                                                 T_schedule=T_schedules.Geometric(alpha=0.999), 
            #                                                     T_start=1500, T_thres=0.1, k=1)
            #     results, best = sa.anneal(return_=True)

