import numpy as np
import pickle
import sys
import types
import warnings
from .prompts import GetPrompts
from .get_instance import GetData
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os

class TSPCONST():
    def __init__(self) -> None:
        # ABS_PATH = os.path.dirname(os.path.abspath(__file__))
        # sys.path.append(ABS_PATH)  # This is for finding all the modules
        # Construct the absolute path to the pickle file
        #pickle_file_path = os.path.join(ABS_PATH, 'instances.pkl')

        # with open("./instances.pkl" , 'rb') as f:
        #     self.instance_data = pickle.load(f)
        self.taskname = 'tsp'
        self.can_visualize = True
        self.ndelay = 1
        self.problem_size = 50
        self.neighbor_size = np.minimum(50, self.problem_size)
        self.n_instance = 8  
        self.running_time = 10



        self.prompts = GetPrompts()

        getData = GetData(self.n_instance,self.problem_size)
        self.instance_data = getData.generate_instances()
        

    def tour_cost(self,instance, solution, problem_size):
        cost = 0
        for j in range(problem_size - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost

    def generate_neighborhood_matrix(self,instance):
        instance = np.array(instance)
        n = len(instance)
        neighborhood_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            distances = np.linalg.norm(instance[i] - instance, axis=1)
            sorted_indices = np.argsort(distances)  # sort indices based on distances
            neighborhood_matrix[i] = sorted_indices

        return neighborhood_matrix


    #@func_set_timeout(5)
    def greedy(self,eva):

        dis = np.ones(self.n_instance)
        instances = []
        routes = []
        n_ins = 0
        for instance, distance_matrix in self.instance_data:

            # get neighborhood matrix
            neighbor_matrix = self.generate_neighborhood_matrix(instance)


            destination_node = 0

            current_node = 0

            route = np.zeros(self.problem_size)
            #print(">>> Step 0 : select node "+str(instance[0][0])+", "+str(instance[0][1]))
            for i in range(1,self.problem_size-1):

                near_nodes = neighbor_matrix[current_node][1:]

                mask = ~np.isin(near_nodes,route[:i])

                unvisited_near_nodes = near_nodes[mask]

                unvisited_near_size = np.minimum(self.neighbor_size,unvisited_near_nodes.size)

                unvisited_near_nodes = unvisited_near_nodes[:unvisited_near_size]

                next_node = eva.select_next_node(current_node, destination_node, unvisited_near_nodes, distance_matrix)

                if next_node in route:
                    #print("wrong algorithm select duplicate node, retrying ...")
                    return None

                current_node = next_node

                route[i] = current_node

                #print(">>> Step "+str(i)+": select node "+str(instance[current_node][0])+", "+str(instance[current_node][1]))

            mask = ~np.isin(np.arange(self.problem_size),route[:self.problem_size-1])

            last_node = np.arange(self.problem_size)[mask]

            current_node = last_node[0]

            route[self.problem_size-1] = current_node

            #print(">>> Step "+str(self.problem_size-1)+": select node "+str(instance[current_node][0])+", "+str(instance[current_node][1]))
            routes.append(route)
            instances.append(instance)

            LLM_dis = self.tour_cost(instance,route,self.problem_size)
            dis[n_ins] = LLM_dis

            n_ins += 1
            if n_ins == self.n_instance:
                break
            #self.route_plot(instance,route,self.oracle[n_ins])

        ave_dis = np.average(dis)
        #print("average dis: ",ave_dis)
        return ave_dis, instances, routes

    def visualization_save(self, save_path, instances, solutions):
        """
        Visualize the TSP solution.

        Parameters:
        - save_path: Path to save the plot.
        - instances: List of TSP instances, where each instance contains city coordinates.
        - solutions: List of solutions, where each solution is the order of cities.
        """
        # 如果save_path存在，则清除该目录下的所有png文件
        if os.path.exists(save_path):
            for file in os.listdir(save_path):
                if file.endswith('.png'):
                    os.remove(os.path.join(save_path, file))
        else:
            os.makedirs(save_path)  # 如果目录不存在，则创建

        plt.figure(figsize=(10, 8))  # 在循环外创建图形

        # 开始保存表现的instance
        for index, instance in enumerate(instances):
            plt.clf()  # 清空当前图形
            coordinates = instance  # 获取城市坐标
            solution = solutions[index].tolist()
            solution.append(solution[0])  # 回到起始点
            solution = np.array(solution)

            # 获取解的坐标
            solution_coords = coordinates[solution.astype(int)]

            # 绘制路线
            plt.plot(solution_coords[:, 0], solution_coords[:, 1], marker='o', linestyle='-')

            # 标记起始点
            plt.text(solution_coords[0, 0], solution_coords[0, 1], 'Start', fontsize=12, ha='right')

            # 设置标签和标题
            plt.title('TSP Solution Visualization')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(True)

            # 保存图形
            plt.savefig(os.path.join(save_path, f"{index}.png"))

        plt.close()  # 在循环结束后关闭图形

    def visualize(self, code_string, branch_save_path):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")

                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                # Now you can use the module as you would any other
                fitness, instances, solutions = self.greedy(heuristic_module)
                self.visualization_save(branch_save_path, instances, solutions)
                return fitness
        except Exception as e:
            print("Errora at visualization_save:", str(e))
            return None

    def evaluate(self, code_string, **kwargs):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")
                
                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                # Now you can use the module as you would any other
                fitness, _, _ = self.greedy(heuristic_module)
                return fitness
        except Exception as e:
            #print("Error:", str(e))
            return None
        # try:
        #     heuristic_module = importlib.import_module("ael_alg")
        #     eva = importlib.reload(heuristic_module)   
        #     fitness = self.greedy(eva)
        #     return fitness
        # except Exception as e:
        #     print("Error:",str(e))
        #     return None
            


