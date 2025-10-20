class GetPrompts():
    def __init__(self):
        self.prompt_task = "Task: Given an edge distance matrix and a local optimal route, please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance. \
You should create a heuristic for me to update the edge distance matrix."
        self.prompt_func_name = "update_edge_distance"
        self.prompt_func_inputs = ['edge_distance', 'local_opt_tour', 'edge_n_used']
        self.prompt_func_outputs = ['updated_edge_distance']
        self.prompt_inout_inf = "'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of each edge used during permutation."
        self.prompt_other_inf = "All are Numpy arrays."
        self.prompt_solution_embedding = ""

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
