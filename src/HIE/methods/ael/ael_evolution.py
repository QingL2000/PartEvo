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

    def __init__(self, api_endpoint, api_endpoint_url, api_key, model_LLM, debug_mode,prompts, **kwargs):
        # -------------------- RZ: use local LLM --------------------
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        self._use_local_llm = kwargs.get('use_local_llm')
        self._url = kwargs.get('url')
        # -----------------------------------------------------------

        # set prompt interface
        #getprompts = GetPrompts()
        self.prompt_task         = prompts.get_task()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()
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
        self.debug_mode = debug_mode # close prompt checking

        # -------------------- RZ: use local LLM --------------------
        if self._use_local_llm:
            self.interface_llm = LocalLLM(self._url)
        else:
            self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_endpoint_url, self.api_key, self.model_LLM, debug_mode=self.debug_mode)

    def get_prompt_i1(self):

        prompt_content = self.prompt_task+"\n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named '\
"+self.prompt_func_name +"'. This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+self.prompt_solution_embedding+"Do not include additional explanations in the code."
        return prompt_content

    def get_prompt_crossover(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(i + 1) + " algorithm and the corresponding code are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i]['code'] + "\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n" \
                         + prompt_indiv + \
                         "Please help me create a new algorithm that has different form the given ones but can be motivated by them. \n" \
                         "First, describe your new algorithm and main steps in one sentence. \
                         The description must be inside a brace. Next, implement it in Python as a function named '\
                         " + self.prompt_func_name + "'. This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + self.prompt_solution_embedding + "Do not include additional explanations in the code."
        return prompt_content

    def get_prompt_psocrossover(self,indivs, infos):
        prompt_indiv = ""
        indiv_index = 1

        prompt_cue_best_local = "Through testing on instances of the specific problem, "
        if infos['best']:
            prompt_cue_best_local += f"the No. {indiv_index} algorithm has proven to be the best-performing method to date."
            indiv_index += 1
        else:
            prompt_cue_best_local += ""

        local_number = len(indivs) - int(infos['best']) - infos['oriparnum']
        if local_number == 0:
            prompt_cue_best_local += ""
        elif local_number == 1:
            prompt_cue_best_local += f"The No. {indiv_index} algorithms {'also ' if infos['best'] else ''}demonstrated good performance."
            indiv_index += 1
        else:
            local_no_list = [w for w in range(indiv_index, indiv_index+local_number)]
            prompt_cue_best_local += f"The No. {', '.join(map(str, local_no_list[:-1]))} and No. {local_no_list[-1]} algorithms {'also ' if infos['best'] else ''}demonstrated good performance."
            indiv_index += len(local_no_list)

        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"\n"+prompt_cue_best_local+".Please help me create a better algorithm that has different form the given ones but can be motivated by them. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named '\
"+self.prompt_func_name +"'. This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+self.prompt_solution_embedding+"Do not include additional explanations in the code."
        return prompt_content


    def get_prompt_mutation(self,indiv1):
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please assist me in creating a new algorithm that is a modified version of the algorithm provided. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named '\
"+self.prompt_func_name +"'. This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+self.prompt_solution_embedding+"Do not include additional explanations in the code."
        return prompt_content



    def _get_alg(self,prompt_content):

        response = self.interface_llm.get_response(prompt_content)

        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

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
                    algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry +=1

        algorithm = algorithm[0]
        code = code[0]

        code_all = code+" "+", ".join(s for s in self.prompt_func_outputs)


        return [code_all, algorithm]


    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            # print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            # print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]

    def crossover(self,parents):

        prompt_content = self.get_prompt_crossover(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            # print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            # print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]

    def crossover_plus_pso(self, parents, infos):
        prompt_content = self.get_prompt_psocrossover(parents, infos)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            # print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            # print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]


    def mutation(self,parents):

        prompt_content = self.get_prompt_mutation(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            # print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            # print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]

