class GetPrompts():
    def __init__(self):
        self.prompt_task = (
            "You are tasked with solving a black-box integer nonlinear programming (INLP) optimization problem using a metaheuristic optimization algorithm. "
            "The objective is to minimize the value of the objective function.\n"
            "Please design an effective metaheuristic algorithm that can solve this optimization problem and return the optimal solution."
        )

        self.prompt_solution_embedding = (
            "Each solution is represented as a 1-dimensional numpy array."
        )

        self.prompt_func_name = "algo"
        self.prompt_func_inputs = ["initial_population", "individual_upper", "individual_lower", "objective_function"]
        self.prompt_func_outputs = ["best_solution"]

        self.prompt_inout_inf = (
            "'initial_population' is a set of pre-initialized solutions.\n"
            "'individual_upper' and 'individual_lower' define the upper and lower bounds for the decision variables.\n"
            "'objective_function' is a Python function that can be called to compute the objective value of candidate solutions."
        )

        self.prompt_other_inf = (
            "'initial_population' is a 2-dimensional numpy array.\n"
            "'individual_upper' and 'individual_lower' are 1-dimensional numpy arrays, each with the same length as the solution.\n"
            "'objective_function' takes a solution as its input and returns its fitness value.\n"
            "The output 'best_solution' should be a 1-dimensional numpy array.\n"
            "The algorithm is allowed a maximum of 30,000 evaluations by calling the 'objective_function'."
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
        return self.prompt_solution_embedding


if __name__ == "__main__":
    getprompts = GetPrompts()
    print(getprompts.get_task())
