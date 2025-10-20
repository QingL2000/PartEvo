class GetPrompts():
    def __init__(self):
        self.prompt_task = ("In a cloud-edge collaborative system consisting of K mobile devices, multiple edge nodes, and a cloud data center, you need to propose a novel metaheuristic optimization algorithm to solve an NP-hard problem. The goal is to select the appropriate edge node for each mobile device and determine the proportion of tasks to be executed locally, at the edge, and in the cloud, in order to achieve low-cost task computation within a specified service delay.\n"
                            "The mathematical model for this problem has been constructed as a Python function named 'objective_function', which can evaluate the iterative solutions of the metaheuristic optimization algorithm and return a fitness value.\n"
                            "The decision variables for this problem are designed as follows: For K mobile devices, the total number of decision variables is 3*K. Each mobile device has three decision variables—Alpha, Beta, and Gamma.\n"
                            "Specifically, Alpha represents the proportion of tasks executed locally on the mobile device, with a float value ranging from 0 to 1.\n"
                            "Beta represents the proportion of non-local tasks executed at the edge node, with a float value ranging from 0 to 1.\n"
                            "Gamma represents the number of the selected edge node, with an integer value ranging from 0 to the number of edge noeds (exclusive).\n"
                            "These decision variables are encoded into the solution, which can be computed by the metaheuristic algorithm. \n"
                            "Please help me design a novel metaheuristic algorithm that is distinct from existing algorithms in the literature to achieve an optimal solution for this NP-hard problem.")
        self.prompt_solution_embedding = ("Each solution is encoded as a 1-dimensional numpy array of length (3*K)."
                                          "For one solution: the indices from 0 to (K−1) store the Alpha values for the K mobile devices, the indices from K to (2*K−1) store the Beta values for the K mobile devices, and the indices from (2*K) to (3*K−1) store the Gamma values for the K mobile devices.")
        # self.prompt_solution_embedding = ("Each solution is encoded as a 1-dimensional numpy array of length (3*K+1)."
        #                                   "For one solution: the indices from 0 to (K−1) store the Alpha values for the K mobile devices, the indices from K to (2*K−1) store the Beta values for the K mobile devices, the indices from (2*K) to (3*K−1) store the Gamma values for the K mobile devices, and the index 3*K stores the fitness value corresponding to the solution.")
        self.prompt_func_name = "algo"
        self.prompt_func_inputs = ["initial_population", "individual_upper", "individual_lower", "objective_function"]
        self.prompt_func_outputs = ["best_solution"]
        self.prompt_inout_inf = "'initial_population' is a set of pre-initalized solutions. 'individual_upper' and 'individual_lower' define the upper and lower bounds for the decision variables in these solutions. 'objective_function' is a python function that can be used to evaluate the fitness value of a solution, taking the solution as its input."
        self.prompt_other_inf = ("The value of K can be deduced from the length of the solution, specifically, K=(length of the solution)//3.\n"
                                 "'initial_population' is a 2-dimensional Numpy array.\n"
                                 "'individual_upper' and 'individual_lower' are 1-dimensional Numpy arrays, each with a length of 3*K.\n"
                                 "'objective_function' takes a solution as its input parameter to evaluate its fitness value."
                                 "The output 'best_solution' should be a 1-dimensional Numpy array"
                                 "The algorithm allows for 1000 iterations.")

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf

    def get_individual_embedding_inf(self):
        return  self.prompt_solution_embedding


if __name__ == "__main__":
    getprompts = GetPrompts()
    print(getprompts.get_task())
