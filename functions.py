import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-darkgrid")

default_df = pd.read_csv("Coords.csv",index_col=0)
df = None

class Streamlit_Funcs:
    @staticmethod
    def write_align(text, pos="left", style="strong"):
        st.markdown(f"<{style} style='text-align:{pos};'>{text}</{style}>",unsafe_allow_html=True)

class Cities:

    def __init__(self, coord_df = False):

        self.coord_df = coord_df
        self.x_range = None
        self.y_range = None
        self.distance_matrix = None
        self.data_validity = self.check_data_val(coord_df)

        if (type(self.coord_df) != bool) & (self.data_validity):
            print("Input Data is Valid!")
            self.x_range = [self.coord_df.iloc[:,0].min(),self.coord_df.iloc[:,0].max()]
            self.y_range = [self.coord_df.iloc[:,1].min(),self.coord_df.iloc[:,1].max()]
            self.calculate_distance_matrix(self.coord_df, return_=False)
            self.N = self.coord_df.shape[0]

    @staticmethod
    def check_data_val(df):
        type_val = isinstance(df, pd.DataFrame)
        col_val = df.shape[1]==2
        nan_val = ~df.isna().any().any()

        if type_val==False:
            raise Exception("Type cannot be other than pd.DataFrame")

        if col_val==False:
            raise Exception("Columns of DataFrame needs to be exactly 2") 

        if nan_val==False:
            raise Exception("All data needs to be filled with number") 
        
        return type_val & col_val & nan_val

    @staticmethod
    def generate_cities(N:int, x_range:list, y_range:list, random_seed=None):
        """
        Generates N number of cities in eucledian 
        space coordinates with deserved range

        Parameters
        ----------
        N : int
            Number of cities
        x_range : list
            Range of x coordinate
        y_range : list
            Range of y coordinate
        random_seed : int, optional
            Random seed

        Returns
        -------
        df : pd.DataFrame
            Tabular DataFrame that consists of city
            numbers and their coordinates
        """     
        np.random.seed(random_seed)
        x_range.sort() 
        y_range.sort()
        x_s = np.random.randint(x_range[0] , x_range[1] , (N,1))
        y_s = np.random.randint(y_range[0] , y_range[1] , (N,1))
        coords = np.concatenate([x_s,y_s], axis=1)
        df = pd.DataFrame(coords, columns=["x","y"], index=range(1,N+1))
        return df
    
    def update_coords(self, df):
        self.coord_df = df
        self.x_range = [self.coord_df.iloc[:,0].min(),self.coord_df.iloc[:,0].max()]
        self.y_range = [self.coord_df.iloc[:,1].min(),self.coord_df.iloc[:,1].max()]
        self.calculate_distance_matrix(self.coord_df, return_=False)
        self.current_sol = None
        self.best_sol = None

    def visualize_city_or_sol(self, solution=False, save_fig=False):
        # """
        # Generates N number of cities in eucledian 
        # space coordinates with deserved range

        # Parameters
        # ----------
        # N : int
        #     Number of cities
        # x_range : list
        #     Range of x coordinate
        # y_range : list
        #     Range of y coordinate
        # random_seed : int, optional
        #     Random seed

        # Returns
        # -------
        # df : pd.DataFrame
        #     Tabular DataFrame that consists of city
        #     numbers and their coordinates
        # """

        # if (self.coord_df)==None:
        #     pass  

        dataframe = self.coord_df
        
        fig = plt.figure(figsize=(15,8), dpi=150)
        ax1 = fig.add_subplot(1,1,1)
        xs = dataframe.iloc[:,0]
        ys = dataframe.iloc[:,1]

        ax1.set_title("COORDINATES AND CITY NUMBERS")
        ax1.scatter(xs,ys,color="g",label="$Cities$")

        for i in range(0,len(xs)):
            ax1.annotate("C"+str(i+1),(dataframe.iloc[i,0],dataframe.iloc[i,1]))
        ax1.set_xlabel("$X$ $Axis$")
        ax1.set_ylabel("$Y$ $Axis$")

        x_lims = [round(self.x_range[0]-10,0),round(self.x_range[1]+10,0)]
        ax1.set_xlim(x_lims[0], x_lims[1])
        y_lims = [round(self.y_range[0]-10,0),round(self.y_range[1]+10,0)]
        ax1.set_ylim(y_lims[0], y_lims[1])

        ax1.legend()

        if type(solution)!=bool:
            
            if solution[0]!=solution[-1]:
                solution = np.append(solution, solution[0])

            dataframe_ = dataframe.loc[solution,:].copy()
            xs = dataframe_.iloc[:,0]
            ys = dataframe_.iloc[:,1]
            ax1.plot(xs,ys)

            obj = round(Construction_Heuristics_TSP.calculate_objective(dataframe , solution),3)
            x_lims = np.array(x_lims)
            x_text = x_lims[1] - (x_lims[1]-x_lims[0])*0.1
            y_lims = np.array(y_lims)
            y_text = y_lims[0] + (y_lims[1]-y_lims[0])*0.05
            ax1.text(x_text, y_text, f'Obj: {obj}', style='normal')  
        else:
            print("Solution is not given!")

        if save_fig == True:
            fig.savefig("initial.png",dpi=150)
            
        return fig
               
    @staticmethod
    def calculate_distance(c1 , c2):
        distance = (sum((c1 - c2)**2))**(1/2) # Eucledian distance
        return distance
    
    def calculate_distance_matrix(self, coord_df_=False, return_=False):
        if type(coord_df_)!=bool:
            coord_df = coord_df_
        elif type(self.coord_df)!=bool:
            coord_df = self.coord_df
        else:
            pass

        distance_df = pd.DataFrame(index=coord_df.index , columns=coord_df.index, dtype= "float")
        ij_s = []
        for i in coord_df.index:
            for j in coord_df.index:
                if {i,j} in ij_s:
                    continue
                ij_s.append({i,j})
                
                if i == j:
                    distance_df.loc[i,j] = np.inf # if i equals j, distance taken as an infinity to prevent picking this node
                else:
                    coord_i = coord_df.loc[i] # coordinate of city i
                    coord_j = coord_df.loc[j] # coordinate of city j
                    distance = self.calculate_distance(coord_i , coord_j)
                    distance_df.loc[i,j] = distance
                    distance_df.loc[j,i] = distance
                    
        self.distance_matrix = distance_df

        if return_:
            return self.distance_matrix

class Construction_Heuristics_TSP(Cities):

    def __init__(self, coord_df):
        super().__init__(coord_df)
        self.best_sol = False
        self.best_obj = np.inf
        self.best_result = False


    def update_bests(self, result_dict):
        if type(self.best_result) == bool:
            result_dict = {
            "sol":result_dict["sol"],
            "obj":result_dict["obj"]
                            }
            self.best_result = result_dict
        else:
            if result_dict["obj"] < self.best_result["obj"]:
                self.best_sol = result_dict["sol"]
                self.best_obj = result_dict["obj"]
        
    def random_insertion(self, random_seed=None):
        """
        Removes the unnecessary string characters, blanks
        from columns and englarging all the letters

        Parameters
        ----------
        df : pd.DataFrame
            Tabular DataFrame
        chars : list, optional
            String characters to be removed from columns, by default ["-","*",">"]

        Returns
        -------
        df : pd.DataFrame
            Tabular DataFrame
        """     
        np.random.seed(random_seed)
        df = self.coord_df
        N = self.N

        c_sol = np.random.permutation(range(1,N+1)) # random permutatiton w all city nums 
        c_obj = self.calculate_objective(df, c_sol)

        result_dict = {
            "sol":c_sol,
            "obj":c_obj
        }

        self.update_bests(result_dict)

        return result_dict
    
    def nearest_neighbor(self, mode="random_one", random_seed=None, take_best=True):
        # random or iterate all
        
        df = self.coord_df
        N = self.N
        distance_matrix = self.distance_matrix

        result_dict = {}

        if mode=="random_one":
            np.random.seed(random_seed)
            fc = np.random.choice(df.index.to_list(),1)[0]
            c_sol = [fc,fc]

            while len(c_sol) != (N+1):
    
                c_i = c_sol[-2]
                k_ = self.from_is_to_js(c_sol, distance_matrix, c_i, method="min")
                best_pos = len(c_sol)-1
                c_sol.insert(best_pos, k_)
            
            c_obj = self.calculate_objective(df, c_sol)
            
            algo_dict = {
            "sol":np.array(c_sol),
            "obj":c_obj 
        }
            result_dict[fc] = algo_dict
        
        elif mode=="iterate_all":
                    
            for fc in distance_matrix.index.to_list():
                # fc: first city
                
                c_sol = [fc,fc]
                while len(c_sol) != (N+1):

                    c_i = c_sol[-2]
                    k_ = self.from_is_to_js(c_sol, distance_matrix, c_i, method="min")
                    best_pos = len(c_sol)-1
                    c_sol.insert(best_pos, k_)
                
                c_obj = self.calculate_objective(df, c_sol)
            
                algo_dict = {
                "sol":np.array(c_sol),
                "obj":c_obj 
            }
                result_dict[fc] = algo_dict

        
        if take_best:
            result_df = pd.DataFrame(result_dict)
            result_dict = result_df.loc[:,result_df.loc["obj"].astype(float).idxmin()].to_dict()
        
        self.update_bests(result_dict)

        return result_dict
    
    def nearest_insertion(self):
        df = self.coord_df
        N = self.N
        distance_matrix = self.distance_matrix

        fc,sc = self.initial_coord(distance_matrix, method="min")
        c_sol = [fc,sc,fc]
        c_cost = distance_matrix.loc[fc,sc] + distance_matrix.loc[sc,fc]

        while len(c_sol) != (len(distance_matrix.index)+1):

            k_ = self.from_is_to_js(c_sol, distance_matrix,  method="min")
 
            arc_cost = np.inf
            best_pos = None

            for idx in range(len(c_sol)-1):

                i_ = c_sol[idx]
                j_ = c_sol[idx+1]

                i_j = distance_matrix.loc[i_,j_]
                i_k_j = distance_matrix.loc[i_,k_] + distance_matrix.loc[k_,j_]
                c_arc_cost = i_k_j - i_j

                if c_arc_cost < arc_cost:
                    best_pos = idx+1
                    arc_cost = c_arc_cost
            
            c_sol.insert(best_pos, k_)
            c_cost += arc_cost
        

        c_obj = self.calculate_objective(df , c_sol)
        # even though c_cost is all objective, I calculated again

        result_dict = {
            "sol":np.array(c_sol),
            "obj":c_obj
            
        }

        self.update_bests(result_dict)
        
        return result_dict

    def farthest_insertion(self):
        df = self.coord_df
        N = self.N
        distance_matrix = self.distance_matrix

        fc,sc = self.initial_coord(distance_matrix, method="max")
        c_sol = [fc,sc,fc]
        c_cost = distance_matrix.loc[fc,sc] + distance_matrix.loc[sc,fc]

        while len(c_sol) != (len(distance_matrix.index)+1):
            self.from_is_to_js
            k_ = self.from_is_to_js(c_sol, distance_matrix,  method="min")

            arc_cost = -1
            best_pos = None

            for idx in range(len(c_sol)-1):

                i_ = c_sol[idx]
                j_ = c_sol[idx+1]

                i_j = distance_matrix.loc[i_,j_]
                i_k_j = distance_matrix.loc[i_,k_] + distance_matrix.loc[k_,j_]
                c_arc_cost = i_k_j - i_j

                if c_arc_cost > arc_cost:
                    best_pos = idx+1
                    arc_cost = c_arc_cost

            
            c_sol.insert(best_pos, k_)
            c_cost += arc_cost
        

        c_obj = self.calculate_objective(df , c_sol)
        # even though c_cost is all objective, I calculated again

        result_dict = {
            "sol":np.array(c_sol),
            "obj":c_obj
        }
        
        self.update_bests(result_dict)

        return result_dict
    
    def arbitrary_insertion(self, random_seed=None):
        np.random.seed(random_seed)
        df = self.coord_df
        N = self.N
        distance_matrix = self.distance_matrix

        fc,sc = self.initial_coord(distance_matrix, method="min")
        c_sol = [fc,sc,fc]
        c_cost = distance_matrix.loc[fc,sc] + distance_matrix.loc[sc,fc]

        while len(c_sol) != (len(distance_matrix.index)+1):

            leftovers = list(set(distance_matrix.index.to_list()) - set(c_sol))
            k_ = np.random.choice(leftovers,1)[0]

            arc_cost = np.inf
            best_pos = None

            for idx in range(len(c_sol)-1):

                i_ = c_sol[idx]
                j_ = c_sol[idx+1]

                i_j = distance_matrix.loc[i_,j_]
                i_k_j = distance_matrix.loc[i_,k_] + distance_matrix.loc[k_,j_]
                c_arc_cost = i_k_j - i_j

                if c_arc_cost < arc_cost:
                    best_pos = idx+1
                    arc_cost = c_arc_cost

            
            c_sol.insert(best_pos, k_)
            c_cost += arc_cost
        

        c_obj = self.calculate_objective(df , c_sol)
        # even though c_cost is all objective, I calculated again

        result_dict = {
            "sol":np.array(c_sol),
            "obj":c_obj
        }

        self.update_bests(result_dict)
        
        return result_dict
    
    @staticmethod
    def from_is_to_js(c_sol, distance_matrix, i=None, method="min"):
        
        if i != None:
            if method == "min":
                js = list(set(distance_matrix.index.to_list())-set(c_sol))
                j = distance_matrix.loc[i,js].idxmin()
            elif method == "max":
                js = list(set(distance_matrix.index.to_list())-set(c_sol))
                j = distance_matrix.loc[i,js].idxmax()
        else:
            if method == "min":
                js = list(set(distance_matrix.index.to_list())-set(c_sol))
                j = distance_matrix.loc[c_sol,js].min().idxmin()
            elif method == "max":
                js = list(set(distance_matrix.index.to_list())-set(c_sol))
                j = distance_matrix.loc[c_sol,js].min().idxmax()
        return j        
    
    @staticmethod
    def initial_coord(distance_matrix, method="min"):
        if method=="min":
            i = distance_matrix.min(axis=1).idxmin()
            j = distance_matrix.loc[i].idxmin() 
        elif method=="max":
            distance_matrix__ = distance_matrix.replace({np.inf:-np.inf}).copy()
            i = distance_matrix__.max(axis=1).idxmax()
            j = distance_matrix__.loc[i].idxmax()         
        return i , j
    
    @staticmethod
    def calculate_objective(df , solution):

        if solution[0]==solution[-1]:
            if isinstance(solution, list):
                del solution[-1]
            elif isinstance(solution, np.ndarray):
                solution = solution[:-1]

        a1 = df.loc[solution,:].to_numpy()
        altered_sol = np.append(solution[1:] , solution[0])
        a2 = df.loc[altered_sol,:].to_numpy()
        return float(np.sum(np.sqrt(np.sum((a1 - a2)**2 , axis=1))))

class Improvement_Heuristics_TSP:
    
    @staticmethod
    def two_opt(solution):
        n_sol = solution.copy()
        i = np.random.randint(0, len(n_sol)-3)
        j = np.random.randint(i+2, len(n_sol)-1)
        reversed_part = n_sol[i+1:j+1] # piece of solution will be reversed
        reverse = reversed_part[::-1] # reversed the piece of solution
        n_sol[i+1:j+1] = reverse # the space will be reversed filled with reversed solution piece
        return n_sol # solution returned

    @staticmethod
    def city_swap(solution):
        n_sol = solution.copy() # main solution copied
        i,j = np.random.choice(len(n_sol) , 2 , replace=False) # 2 random different indixes selected
        n_sol[i] , n_sol[j] = n_sol[j] , n_sol[i] # cities swapped
        return n_sol

class T_schedules:
    class Geometric:
        def __init__(self, alpha):
            self.sched_name = "geometric"
            self.alpha = alpha
        def update(self, T_start, c_T, iter_):
            return c_T * self.alpha
        
    class Logarithmic:
        def __init__(self):
            self.sched_name = "logarithmic"
        def update(self, T_start, c_T, iter_):
            return c_T/(1 + np.exp(1+iter_))
        
    class Linear_Multiplicative:
        """
        alpha>0
        """
        def __init__(self, alpha):
            self.sched_name = "linear"
            self.alpha = alpha
        def update(self, T_start, c_T, iter_):
            return T_start/ (1+self.alpha * iter_)
        
    class Exponential_Multiplicative:
        """
        0.8<=alpha<=0.9
        """
        def __init__(self, alpha):
            self.sched_name = "Exponential_Multiplicative"
            self.alpha = alpha
        def update(self, T_start, c_T, iter_):
            return T_start * self.alpha**iter_  
            
    class Logarithmical_Multiplicative:
        """
        alpha>1
        """
        def __init__(self, alpha):
            self.sched_name = "Logarithmical_Multiplicative"
            self.alpha = alpha
        def update(self, T_start, c_T, iter_):
            return T_start/(1 + self.__getattribute__alpha*np.exp(1+self.iter_))
    
    class Quadratic_Multiplicative:
        """
        alpha>1
        """
        def __init__(self, alpha):
            self.sched_name = "Logarithmical_Multiplicative"
            self.alpha = alpha
        def update(self, T_start, c_T, iter_):
            return T_start/ (1+self.alpha * iter_**2)

class Meta_Heuristics:
    
    class Simulated_Annealing(Construction_Heuristics_TSP):
        def __init__(self, coord_df, init_sol, T_schedule=T_schedules.Geometric(alpha=0.99), 
                     T_start=100, T_thres=0.1, k=1):
            super().__init__(coord_df)
            
            self.init_sol = init_sol
            self.T_schedule = T_schedule
            self.T_start = T_start
            self.T_thres = T_thres
            self.k = 1 # boltzmann constant

            self.T_s = None
            self.sol_s = None
            self.obj_s = None

            self.best_sol = None
            self.best_obj = None
            self.batch_dict = None
    
        def update_params(self, T_schedule, T_start, T_thres, p_accept, print_=False):
            self.T_schedule = T_schedule
            self.T_start = T_start
            self.T_thres  = T_thres
            self.p_accept = p_accept

        def anneal(self, return_=False):

            df = self.coord_df
            T_sched = self.T_schedule
            c_T = self.T_start
            c_sol = self.init_sol
            T_thres = self.T_thres
            k = self.k

            c_obj = Construction_Heuristics_TSP.calculate_objective(df, c_sol)

            T_s = []
            sol_s = []
            obj_s = []
            iter_= 1

            while c_T > T_thres:
            
                two_opt_sol = Improvement_Heuristics_TSP.two_opt(c_sol)
                city_swap_sol = Improvement_Heuristics_TSP.city_swap(c_sol)

                two_opt_obj = Construction_Heuristics_TSP.calculate_objective(df, two_opt_sol)
                city_swap_obj = Construction_Heuristics_TSP.calculate_objective(df, city_swap_sol)

                if two_opt_obj<city_swap_obj:
                    cand_sol = two_opt_sol
                    cand_obj = two_opt_obj
                else:
                    cand_sol = city_swap_sol
                    cand_obj = city_swap_obj

                if cand_obj<c_obj:
                    c_sol = cand_sol
                    c_obj = cand_obj
                    
                else:
                    power = (c_obj-cand_obj)/(k*c_T)
                    p = np.exp(power)

                    if p > np.random.random():
                        c_sol = cand_sol
                        c_obj = cand_obj

                T_s.append(c_T)
                sol_s.append(c_sol)
                obj_s.append(c_obj)

                c_T = T_sched.update(T_start=self.T_start, c_T=c_T, iter_=iter_)
                iter_ += 1

            result_dict = {
                        "T_s":T_s,
                        "sol_s":sol_s,
                        "obj_s":obj_s
                        }
            
            result_df = pd.DataFrame(result_dict)
            best_idx = result_df["obj_s"].idxmin()
            best_dict = result_df.loc[best_idx].to_dict()

            self.T_s = T_s
            self.sol_s = sol_s
            self.obj_s = obj_s

            self.best_result = best_dict
            
            if return_:
                return result_dict, best_dict
        
        # def batch_anneal(self, times=5):
        #     for i in range(1, times + 1):
        #         print(f"Iteration {i}/{times} -------------------------------")
        #         self.T = self.T_save
        #         self.iteration = 1
        #         self.cur_solution, self.cur_fitness = self.initial_solution()
        #         self.anneal()

        def visualize(self):

            T_s = self.T_s
            obj_s = self.obj_s

            fig, ax = plt.subplots()
            plt.figure(dpi = 80)

            ax.plot(obj_s, color='C0')
            ax.tick_params(axis='y', labelcolor='C0')
            # ax.text(0, obj_s[0], "START")
            ax.scatter(0,obj_s[0], c="red")
            ax.plot((0,len(T_s)-1), (obj_s[0],obj_s[0]), c="red")
            # ax.text(len(obj_s)-1, obj_s[-1], "FINISH")
            ax.scatter(len(obj_s)-1,obj_s[-1], c="green")
            ax.plot((0,len(T_s)-1), (obj_s[-1],obj_s[-1]), c="green")
            # ax.legend("Objective Value")
            ax.set_ylabel('Objective Values', color="C0")
            ax.set_xlabel('Iterations')

            ax2 = ax.twinx()
            ax2.plot(T_s, color='C1')
            ax2.tick_params(axis='y', labelcolor='C1')
            ax2.set_ylabel('Temperature', color="C1")

            plt.show()
            return fig

    class Stochastic_Hill_Climbing(Construction_Heuristics_TSP):
      
        def __init__(self, coord_df, init_sol, T_schedule=T_schedules.Geometric(alpha=0.99), 
                     T_start=100, T_thres=0.1, k=1):
            super().__init__(coord_df)
            
            self.init_sol = init_sol
            self.T_schedule = T_schedule
            self.T_start = T_start
            self.T_thres = T_thres
            self.k = 1 # boltzmann constant

            self.T_s = None
            self.sol_s = None
            self.obj_s = None

            self.best_sol = None
            self.best_obj = None
            self.batch_dict = None

        def update_params(self, T_schedule, T_start, T_thres, p_accept, print_=False):
            self.T_schedule = T_schedule
            self.T_start = T_start
            self.T_thres  = T_thres
            self.p_accept = p_accept
        
        def climb(self, return_=False):
            
            df = self.coord_df
            T_sched = self.T_schedule
            c_T = self.T_start
            c_sol = self.init_sol
            T_thres = self.T_thres
            k = self.k

            c_obj = Construction_Heuristics_TSP.calculate_objective(df, c_sol)

            T_s = []
            sol_s = []
            obj_s = []
            iter_= 1

            while c_T > T_thres:
            
                two_opt_sol = Improvement_Heuristics_TSP.two_opt(c_sol)
                city_swap_sol = Improvement_Heuristics_TSP.city_swap(c_sol)

                two_opt_obj = Construction_Heuristics_TSP.calculate_objective(df, two_opt_sol)
                city_swap_obj = Construction_Heuristics_TSP.calculate_objective(df, city_swap_sol)

                if two_opt_obj<city_swap_obj:
                    cand_sol = two_opt_sol
                    cand_obj = two_opt_obj
                else:
                    cand_sol = city_swap_sol
                    cand_obj = city_swap_obj

                power = (cand_obj-c_obj)/c_T
                p = 1/(1+np.exp(power))

                if p > np.random.random():
                    c_sol = cand_sol
                    c_obj = cand_obj

                T_s.append(c_T)
                sol_s.append(c_sol)
                obj_s.append(c_obj)

                c_T = T_sched.update(T_start=self.T_start, c_T=c_T, iter_=iter_)
                iter_ += 1

            result_dict = {
        
                        "T_s":T_s,
                        "sol_s":sol_s,
                        "obj_s":obj_s
                    }
            
            result_df = pd.DataFrame(result_dict)
            best_idx = result_df["obj_s"].idxmin()
            best_dict = result_df.loc[best_idx].to_dict()

            self.T_s = T_s
            self.sol_s = sol_s
            self.obj_s = obj_s

            self.best_result = best_dict
            
            if return_:
                return result_dict, best_dict

        def visualize(self):

            T_s = self.T_s
            obj_s = self.obj_s

            fig, ax = plt.subplots()
            plt.figure(dpi = 80)

            ax.plot(obj_s, color='C0')
            ax.tick_params(axis='y', labelcolor='C0')
            # ax.text(0, obj_s[0], "START")
            ax.scatter(0,obj_s[0], c="red")
            ax.plot((0,len(T_s)-1), (obj_s[0],obj_s[0]), c="red")
            # ax.text(len(obj_s)-1, obj_s[-1], "FINISH")
            ax.scatter(len(obj_s)-1,obj_s[-1], c="green")
            ax.plot((0,len(T_s)-1), (obj_s[-1],obj_s[-1]), c="green")
            # ax.legend("Objective Value")
            ax.set_ylabel('Objective Values', color="C0")
            ax.set_xlabel('Iterations')

            ax2 = ax.twinx()
            ax2.plot(T_s, color='C1')
            ax2.tick_params(axis='y', labelcolor='C1')
            ax2.set_ylabel('Temperature', color="C1")

            plt.show()


