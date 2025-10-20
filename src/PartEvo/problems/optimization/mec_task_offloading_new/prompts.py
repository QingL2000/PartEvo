class GetPrompts():
    def __init__(self):
        self.prompt_task = (
            "In a cloud-edge collaborative system consisting of S mobile devices, multiple edge nodes, and a cloud data center, you are tasked with proposing a novel metaheuristic optimization algorithm to solve an NP-hard problem. The goal is to select the appropriate edge node for each mobile device and determine the proportion of tasks to be executed locally, at the edge, and in the cloud, in order to minimize the cost of task computation while ensuring the service delay remains within a specified threshold.\n"
            "The mathematical model for this problem is implemented as a Python function named 'objective_function', which evaluates the iterative solutions of the metaheuristic optimization algorithm and returns a fitness value.\n"
            "The decision variables for this problem are as follows: For S mobile devices, there are a total of 3*S decision variables. Each mobile device has three decision variables—Alpha, Beta, and Gamma.\n"
            "Specifically, Alpha represents the proportion of tasks executed locally on the mobile device, with a float value ranging from 0 to 1.\n"
            "Beta represents the proportion of non-local tasks executed at the edge node, with a float value ranging from 0 to 1.\n"
            "Gamma is a value between 0 and 1, which, after mapping, represents the edge node connected to the mobile device.\n"
            "These decision variables are encoded into the solution, which can be processed by the metaheuristic algorithm.\n"
            "Please help design a novel metaheuristic algorithm to achieve the optimal solution for this NP-hard problem."
        )

        self.prompt_solution_embedding = ("Each solution is encoded as a 1-dimensional numpy array of length (3*S)."
                                          "For one solution: the indices from 0 to (S−1) store the Alpha values for the S mobile devices, the indices from K to (2*S−1) store the Beta values for the S mobile devices, and the indices from (2*S) to (3*S−1) store the Gamma values for the S mobile devices.")
        # self.prompt_solution_embedding = ("Each solution is encoded as a 1-dimensional numpy array of length (3*K+1)."
        #                                   "For one solution: the indices from 0 to (K−1) store the Alpha values for the K mobile devices, the indices from K to (2*K−1) store the Beta values for the K mobile devices, the indices from (2*K) to (3*K−1) store the Gamma values for the K mobile devices, and the index 3*K stores the fitness value corresponding to the solution.")
        self.prompt_func_name = "algo"
        self.prompt_func_inputs = ["initial_population", "individual_upper", "individual_lower", "objective_function"]
        self.prompt_func_outputs = ["best_solution"]
        self.prompt_inout_inf = "'initial_population' is a set of pre-initalized solutions. 'individual_upper' and 'individual_lower' define the upper and lower bounds for the decision variables in these solutions. 'objective_function' is a python function that can be used to evaluate the fitness value of a solution, taking the solution as its input."
        self.prompt_other_inf = (
            "The value of S can be derived from the length of the solution, specifically, S = (length of the solution) // 3.\n"
            "'initial_population' is a 2-dimensional numpy array.\n"
            "'individual_upper' and 'individual_lower' are 1-dimensional numpy arrays, each with a length of 3*S.\n"
            "'objective_function' takes a solution as its input parameter to evaluate its fitness value.\n"
            "The output, 'best_solution', should be a 1-dimensional numpy array representing the final solution obtained by the algorithm.\n"
            "The algorithm allows for a maximum of 1000 iterations."
        )

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
