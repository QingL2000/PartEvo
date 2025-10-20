from ..llm.api_general import InterfaceAPI
from ..llm.api_local_llm import InterfaceLocalLLM


class InterfaceLLM:
    """
    这个类用于与LLM进行交互，包括检验llm是否可用，以及输入prompt并得到输出return

    主要功能是：
    get_response(self, prompt_content) -- 负责传出prompt并返回
        1. chatgpt：json_data["choices"][0]["message"]["content"]
        2. 通义千问：

    InterfaceAPI 是用于交互的API的类，在InterfaceLLM类中作为负责与大模型沟通的插件
    """

    def __init__(self, api_endpoint, api_endpoint_url, api_key, model_LLM, llm_use_local=None, llm_local_url=None, debug_mode=False):
        self.api_endpoint = api_endpoint
        self.api_endpoint_url = api_endpoint_url
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.llm_use_local = llm_use_local
        self.llm_local_url = llm_local_url

        print("- check LLM API")

        if self.llm_use_local:
            print('local llm delopyment is used ...')

            if self.llm_local_url == None or self.llm_local_url == 'xxx':
                print(">> Stop with empty url for local llm !")
                exit()

            self.interface_llm = InterfaceLocalLLM(
                self.llm_local_url
            )

        else:
            print('remote llm api is used ...')

            if self.api_key == None or self.api_endpoint == None or self.api_key == 'xxx' or self.api_endpoint == 'xxx':
                print(
                    ">> Stop with wrong API setting: Set api_endpoint (e.g., api.chat...) and api_key (e.g., kx-...) !")
                exit()

            self.interface_llm = InterfaceAPI(
                self.api_endpoint,
                self.api_endpoint_url,
                self.api_key,
                self.model_LLM,
                self.debug_mode,
            )

        res = self.interface_llm.get_response("1+1=?")

        if res == None:
            print(">> Error in LLM API, wrong endpoint, key, model or local deployment!")
            exit()

        # choose LLMs
        # if self.type == "API2D-GPT":
        #     self.interface_llm = InterfaceAPI2D(self.key,self.model_LLM,self.debug_mode)
        # else:
        #     print(">>> Wrong LLM type, only API2D-GPT is available! \n")

    def get_response(self, prompt_content):
        response = self.interface_llm.get_response(prompt_content)

        return response

    def get_response_M(self, prompt_content, images_path):
        response = self.interface_llm.get_response_M(prompt_content, images_path)

        return response
