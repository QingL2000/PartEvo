import re
import time
from ...llm.interface_LLM import InterfaceLLM
import warnings
from joblib import Parallel, delayed
import re
import concurrent.futures
import sys


class Summarizer():
    """
    Summarizer这个类主要负责branch初始化时确定分支属性，并在进化过程中对外部解集作总结
    get_tags 是最上层接口，接受所有branch
    identify_branch 是实际为每个branch打标签的function

    """

    def __init__(self, api_endpoint, api_endpoint_url, api_key, model_LLM, debug_mode, prob, **kwargs):

        self._use_local_llm = False

        prompts = prob.prompts
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
        # self.branch_novelty = kwargs.get('branch_novelty', 50)
        self.stepbystep_flag = kwargs.get('stepbystep_flag', False)
        self.n_p = kwargs.get('n_p', 4)
        self.timeout = kwargs.get('timeout', 15)
        self.use_numba = kwargs.get('use_numba', False)
        self.step_cue = " You can first analyze the information you currently know, and then provide the final answer in the required format."

    def get_prompt_identify_branch(self, branch):
        """
        use key word to identify the branch
        """
        prompt_indiv = "The concept and code of the algorithm are as follows: \n" + branch.algorithm + "\n" + branch.code + "\n"
        prompt_identify = "I designed an algorithm. Please select as few keywords as possible to tag this algorithm based on its concept and code. Note that you should summarize the characteristics of the operations employed in the algorithm. " + prompt_indiv + "The answer must be enclosed in braces, with each keyword separated by a comma."

        prompt_content = prompt_identify + (self.step_cue if self.stepbystep_flag else "")
        return prompt_content

    def get_prompt_sum_up(self, external_set_new, summary_old=""):
        prompt_indiv = ""
        for idx, indiv in enumerate(external_set_new):
            prompt_indiv += (f"No. {idx + 1} algorithm's objective function value is {indiv['objective']}, "
                             f"and its idea is:\n{indiv['algorithm']}\n")

        if summary_old:
            prompt_content = (self.prompt_task + "\n"
                                                 "Based on previous design methods, we have the following summary:\n" +
                              summary_old +
                              f"\nWe have now explored an additional {len(external_set_new)} algorithms to tackle this problem, "
                              f"from No. 1 to No. {len(external_set_new)}, with performance gradually declining "
                              "(the smaller the objective function value, the better the performance).\n" +
                              prompt_indiv +
                              "Please review past experiences and current algorithms, analyzing which techniques within the algorithms are effective in solving this problem and which are not. "
                              "Finally, summarize the effective and ineffective techniques, ensuring that the summary is enclosed in braces. "+
                              (self.step_cue if self.stepbystep_flag else ""))



        else:
            prompt_content = (self.prompt_task + "\n" +
                              f"We can now solve this problem using {len(external_set_new)} algorithms, "
                              f"from No. 1 to No. {len(external_set_new)}, with performance gradually declining "
                              "(the smaller the objective function value, the better the performance).\n" +
                              prompt_indiv +
                              "Please review these algorithms, analyzing which techniques within the algorithms are effective in solving this problem and which are not. "
                              "Finally, summarize the effective and ineffective techniques, ensuring that the summary is enclosed in braces. "+
                              (self.step_cue if self.stepbystep_flag else ""))

        return prompt_content

    def _get_tags(self, prompt_content):
        response = self.interface_llm.get_response(prompt_content)

        tags_string = re.findall(r"\{(.*)\}", response, re.DOTALL)

        tags = []
        if tags_string:
            # 分割标签并去除空白字符，存储到列表中
            tags = [tag.strip() for tag in tags_string[0].split(",")]

        n_retry = 1
        while len(tags) == 0:
            if self.debug_mode:
                print("Error: tags not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)
            tags_string = re.findall(r"\{(.*)\}", response, re.DOTALL)

            if tags_string:
                tags = [tag.strip() for tag in tags_string[0].split(",")]

            if n_retry > 3:
                break
            n_retry += 1

        return tags

    def _get_summary(self, prompt_content):
        response = self.interface_llm.get_response(prompt_content)

        summary_string = re.findall(r"\{(.*)\}", response, re.DOTALL)

        n_retry = 1
        while len(summary_string) == 0:
            if self.debug_mode:
                print("Error: Summary not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)
            summary_string = re.findall(r"\{(.*)\}", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry += 1

        # 返回第一个匹配项，如果没有匹配项则返回空字符串
        return summary_string[0] if summary_string else ""

    def identify_branch(self, **kwargs):
        branch = kwargs.get('branch_wait_identification')
        prompt_content = self.get_prompt_identify_branch(branch)
        if self.debug_mode:
            print("\n >>> check prompt for identifying algorithm using : \n", prompt_content)
        tags = self._get_tags(prompt_content)
        return tags

    def sum_up(self, external_set_new, summary_old, **kwargs):
        prompt_content = self.get_prompt_sum_up(external_set_new, summary_old)
        if self.debug_mode:
            print("\n >>> check prompt for identifying algorithm using : \n", prompt_content)
        summary = self._get_summary(prompt_content)
        return summary

    def get_tags(self, population, **kwargs):
        """
        :param population: List [branch, branch, ..., branch]
        :param kwargs:
        :return:
        """

        branches_tags = []

        timeout = self.timeout + 15
        if sys.gettrace() is not None:  # 检查是否在调试模式下
            timeout = None  # 取消超时限制

        try:
            branches_tags = Parallel(n_jobs=self.n_p, timeout=timeout)(
                delayed(self.identify_branch)(branch_wait_identification=branch, **kwargs) for branch in population)
        except Exception as e:
            print('Identification Error:', e)
            # print("Parallel time out .")

        return branches_tags

    def get_summary(self, external_set_new, summary_old, **kwargs):
        """
        :param external_set_new: class
        :param kwargs:
        :return:
        """

        try:
            summary_new = self.sum_up(external_set_new, summary_old)
        except Exception as e:
            print('Summary Error:', e)
            summary_new = ""
            # print("Parallel time out .")
        return summary_new
