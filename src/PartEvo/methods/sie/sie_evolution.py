import re
import time
from ...llm.interface_LLM import InterfaceLLM


class Evolution():
    """
    Evolution这个类主要定义进化过程中的遗传操作对应的prompt，从而指导大模型进行算法设计

    InterfaceLLM 是与大模型交互的类

    主要函数包括：
    1. get_prompt_i1(self) -- 定义初始化算法种群pop需要的prompt
    2. get_prompt_crossover(self,indivs) -- 根据输入的indivs来产生交叉产生后代需要的prompt
    3. get_prompt_mutation(self,indiv1) -- 根据输入的indivs来产生变异产生后代需要的prompt
    4. _get_alg(self,prompt_content) -- 通过prompt与大模型交互并返回代码和算法block
        ** 调用 self.interface_llm.get_response(prompt_content)与大模型交互 **
        return [code_all, algorithm]
    5. i1(self) -- 种群中单个个体初始化的顶层模块，用于获取定义的prompt，并调用_get_alg(self,prompt_content)获得代码和算法
    6. crossover(self,parents) -- crossover的顶层模块，通过get_prompt_crossover获取prompt，并调用_get_alg(self,prompt_content)获得代码和算法
    7. mutation(self,parents) -- mutation的顶层模块，通过get_prompt_mutation获取prompt，并调用_get_alg(self,prompt_content)获得代码和算法
    """

    def __init__(self, api_endpoint, api_endpoint_url, api_key, model_LLM, debug_mode, prompts, **kwargs):
        # -------------------- RZ: use local LLM --------------------
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        self._use_local_llm = kwargs.get('use_local_llm')
        self._url = kwargs.get('url')
        # -----------------------------------------------------------

        # set prompt interface
        # getprompts = GetPrompts()
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()
        self.prompt_solution_embedding = prompts.get_individual_embedding_inf()
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_endpoint_url = api_endpoint_url
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode  # close prompt checking

        # -------------------- RZ: use local LLM --------------------
        if self._use_local_llm:
            self.interface_llm = LocalLLM(self._url)
        else:
            self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_endpoint_url, self.api_key, self.model_LLM,
                                              debug_mode=self.debug_mode)

        # set Branch
        self.branch_novelty = kwargs.get('branch_novelty', 50)
        self.stepbystep_flag = kwargs.get('stepbystep_flag', False)
        self.step_cue = " You can first analyze the information you currently know, and then proceed with the algorithm design."

    def get_prompt_init(self, current_pop):
        """
        Option-1 Initialization one by one
        """
        if not isinstance(current_pop, list):
            raise ValueError("Error from Initialization: Input must be a list.")

        current_size = len(current_pop)

        if len(current_pop) > 0:
            init_cue = f"There are {current_size} algorithms available to solve this problem."
            init_task = f"Please help me create a new algorithm that differs from the existing algorithms by at least {self.branch_novelty}%."

        else:
            init_cue = f""
            init_task = f"Please, based on your expertise, help me create a novel and efficient algorithm to solve this problem."

        prompt_indiv = ""
        for idx, indiv in enumerate(current_pop):
            prompt_indiv = prompt_indiv + "No." + str(idx + 1) + " algorithm's idea is : \n" + \
                           indiv.algorithm + "\n"

        prompt_content = self.prompt_task + "\n" + init_cue + prompt_indiv + init_task + \
                         " First, describe your concept for the new algorithm and its main steps in as few words as possible while ensuring clarity." + \
                         " The description must be enclosed in braces. Next, implement it in Python as a runnable function named '" + self.prompt_func_name + \
                         "'. This function should accept " + str(len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + \
                         ". The function should return " + str(len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + self.prompt_solution_embedding + \
                         "Do not include any comments in the code." + (self.step_cue if self.stepbystep_flag else "")
        return prompt_content

    def get_prompt_branch_explore_independent(self, individual):
        """
        Option-2 Like Mutation
        """
        reflection = individual.reflection
        prompt_indiv = individual.algorithm + "\n" + individual.code + "\n"

        prompt_content = "An intelligent agent is currently executing the following design task: \"" + self.prompt_task + "\" \n "
        prompt_content += "The agent has designed an algorithm with the following ideas and code: " + prompt_indiv

        if reflection:
            prompt_content += "\nAn expert has provided some suggestions for this algorithm. You can decide whether to incorporate the expert's feedback, and then create a new algorithm that differs from the given one but motivated by it.\n"
        else:
            prompt_content += "\nPlease help me create a new algorithm that is different from the given one but motivated by it.\n"

        prompt_content += (
                    "First, describe your concept for the new algorithm and its main steps in as few words as possible while ensuring clarity. "
                    "The description must be enclosed in braces. Next, implement it in Python as a runnable function named '" + self.prompt_func_name + "'.")
        prompt_content += " This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " + self.joined_inputs + ". "
        prompt_content += "The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " + self.joined_outputs + ". "
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf + "\n" + self.prompt_solution_embedding + " Do not include any comments in the code." + (
            self.step_cue if self.stepbystep_flag else "")
        return prompt_content

    def get_prompt_branch_explore_cooperation(self, indivs):
        """
        Option-3 Like Crossover
        :param indivs:  [Main branch, coop branch1, coop branch2, ..., coop branch K]
        """
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "The No." + str(i + 1) + " algorithm and the corresponding code are: \n" + \
                           indivs[i].algorithm + "\n" + indivs[i].code + "\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n" \
                         + prompt_indiv + \
                         "\n" + ".Please help me create a better algorithm that has different form the given ones but can be motivated by the first algorithm (No. 1). \n" \
                                "First, describe your concept for the new algorithm and its main steps in as few words as possible while ensuring clarity. The description must be enclosed in braces. Next, implement it in Python as a runnable function named \'" + self.prompt_func_name + "\'. This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + self.prompt_solution_embedding + " Do not include any comments in the code." + (self.step_cue if self.stepbystep_flag else "")
        return prompt_content

    def get_prompt_branch_guided_God_view(self, individual, External_sorting_set, Summary):
        """
        Option-4 God view
        individual: Branch
        External_sorting_set: List [{algorithm:, code:, fitness:}, {}]
        """
        Esssize = len(External_sorting_set)
        prompt_God_view = f"Currently, {Esssize} algorithms have been explored for this problem, with their effectiveness decreasing from No. 1 to No. {Esssize}. The concepts and code for these methods are as follows:"
        for idx, cont in enumerate(External_sorting_set):
            prompt_God_view += f"No. {idx + 1} algorithm and the corresponding code are: \n{cont['algorithm']}\n{cont['code']}\n"

        prompt_summary = ""
        if Summary:
            prompt_summary += f"Based on the validation of various algorithms for this problem, I have summarized the following insights: {Summary}"

        prompt_God_view_task = f"Please analyze the current situation and modify the following algorithms to create a more promising algorithm for this problem. The thoughts and code for the algorithms to be modified are as follows:\n{individual.algorithm}\n{individual.code}\n"

        prompt_content = (
                self.prompt_task + "\n" +
                prompt_God_view + prompt_summary + "\n" +prompt_God_view_task +
                "First, describe your concept for the new algorithm and its main steps in as few words as possible while ensuring clarity. The description must be enclosed in braces. Next, implement it in Python as a runnable function named '" + self.prompt_func_name + "'. " +
                f"This function should accept {len(self.prompt_func_inputs)} input(s): " + self.joined_inputs + ". " +
                f"The function should return {len(self.prompt_func_outputs)} output(s): " + self.joined_outputs + ". " +
                self.prompt_inout_inf + " " + self.prompt_other_inf + "\n" +
                self.prompt_solution_embedding + " Do not include any comments in the code." +
                (self.step_cue if self.stepbystep_flag else "")
        )
        return prompt_content

    def get_prompt_branch_consider_best_local(self, individual, best_solution):
        """
        Option-5: Similar to PSO (Particle Swarm Optimization)
        individual: Branch
        best_solution: dict {'algorithm', 'code', 'fitness'}
        """
        prompt_indiv = f" On the algorithm branch starting with {individual.branch} as the initial key element, after several iterations, we arrived at the current algorithm (idea and the corresponding code):\n" + individual.algorithm + "\n" + individual.code + "\n"

        if individual.code == individual.local_code:
            prompt_local = ""
        else:
            prompt_local = f"During the iterations on this branch, a better-performing algorithm appeared, and its idea and code are as follows:\n" + individual.local_algorithm + "\n" + individual.local_code + "\n"

        prompt_cue_best_local = "In addition, among all the algorithms tested (including other branches), the best-performing algorithm's idea and code are as follows:\n" + best_solution['algorithm'] + "\n" + best_solution['code'] + "\n"

        prompt_content = (
                self.prompt_task + "\n" +
                "For this problem, we defined several algorithm branches, allowing each branch to incorporate different techniques while maintaining the main framework, thus discovering diverse solutions." +
                prompt_indiv + prompt_local + prompt_cue_best_local +
                "Please use the information above and follow the main framework of this algorithm branch to suggest any potential improvements that could enhance the algorithm's performance in solving this problem.\n" +
                "First, describe your concept for the new algorithm and its main steps in as few words as possible while ensuring clarity. The description must be enclosed in braces. Next, implement it in Python as a function named '" + self.prompt_func_name + "'. " +
                "This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " + self.joined_inputs + ". " +
                "The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " + self.joined_outputs + ". " +
                self.prompt_inout_inf + " " + self.prompt_other_inf + "\n" +
                self.prompt_solution_embedding + " Do not include any comments in the code." +
                (self.step_cue if self.stepbystep_flag else "")
        )
        return prompt_content

    def _get_alg(self, prompt_content):

        response = self.interface_llm.get_response(prompt_content)

        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry += 1

        algorithm = algorithm[0]
        code = code[0]

        code_all = code + " " + ", ".join(s for s in self.prompt_func_outputs)

        return [code_all, algorithm]

    def init_unit(self, **kwargs):
        current_pop = kwargs.get('current_pop')
        prompt_content = self.get_prompt_init(current_pop)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ init ] : \n", prompt_content)
            # print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            # print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]

    def independent_explore(self, **kwargs):
        individual = kwargs.get('individual')  # 一个分支
        prompt_content = self.get_prompt_branch_explore_independent(individual)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ independent_explore ] : \n", prompt_content)
            # print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            # print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]

    def cooperation_explore(self, **kwargs):
        indivs = kwargs.get('indivs')  # [多个分支]
        prompt_content = self.get_prompt_branch_explore_cooperation(indivs)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ cooperation_explore ] : \n", prompt_content)
            # print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            # print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]

    def God_guide_explore(self, **kwargs):

        individual, External_sorting_set, Summary = kwargs.get('individual'), kwargs.get(
            'External_sorting_set'), kwargs.get('summary')
        prompt_content = self.get_prompt_branch_guided_God_view(individual, External_sorting_set, Summary)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ God_guide_explore ] : \n", prompt_content)
            # print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            # print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]

    def PSO_explore(self, **kwargs):
        individual, best_solution = kwargs.get('individual'), kwargs.get('best_solution')
        prompt_content = self.get_prompt_branch_consider_best_local(individual, best_solution)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ PSO_explore ] : \n", prompt_content)
            # print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            # print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]
