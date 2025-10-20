import http.client
import json
import base64


class InterfaceAPI:
    def __init__(self, api_endpoint, api_endpoint_url, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_endpoint_url = api_endpoint_url
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def get_response(self, prompt_content):
        """
        目前支持GPT和通义千问
        """
        if self.api_endpoint == 'api.bltcy.ai':  # api.openai.com
            payload_explanation = json.dumps(
                {
                    "model": self.model_LLM,
                    "messages": [
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_content}
                    ],
                }
            )

            headers = {
                "Authorization": "Bearer " + self.api_key,
                "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                "Content-Type": "application/json",
                "x-api2d-no-cache": 1,
            }

            response = None
            n_trial = 1
            while True:
                n_trial += 1
                if n_trial > self.n_trial:
                    return response
                try:
                    conn = http.client.HTTPSConnection(self.api_endpoint)
                    conn.request("POST", self.api_endpoint_url, payload_explanation, headers)
                    res = conn.getresponse()
                    data = res.read()
                    # print(data)
                    json_data = json.loads(data)
                    # print(json_data)
                    response = json_data["choices"][0]["message"]["content"]
                    break
                except:
                    if self.debug_mode:
                        print("Error in API. Restarting the process...")
                    continue
        # 通义千问
        elif self.api_endpoint == 'dashscope.aliyuncs.com':
            payload_explanation = json.dumps(
                {
                    "model": self.model_LLM,
                    "input": {"messages": [
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_content}
                    ]},
                    "parameters": {
                        "result_format": "message"
                    }
                }
            )

            headers = {
                "Authorization": "Bearer " + self.api_key,
                # "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                "Content-Type": "application/json",
                # "x-api2d-no-cache": 1,
            }

            response = None
            n_trial = 1
            while True:
                n_trial += 1
                if n_trial > self.n_trial:
                    return response
                try:
                    conn = http.client.HTTPSConnection(self.api_endpoint)
                    conn.request("POST", self.api_endpoint_url, body=payload_explanation, headers=headers)
                    res = conn.getresponse()
                    data = res.read()
                    json_data = json.loads(data)
                    response = json_data['output']["choices"][0]["message"]["content"]
                    break
                except:
                    if self.debug_mode:
                        print("Error in API. Restarting the process...")
                    continue
        else:
            print('sorry, the endpoint is not supported')
            response = None
        return response

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_response_M(self, prompt_content, images):
        content = [
            {
                "type": "text",
                "text": prompt_content
            }
        ]

        for index, image in enumerate(images):
            image_base64 = self.encode_image(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })

            # 计算并显示进度
            # progress = (index + 1) / len(images) * 100
            # print(f"\rProgress: {progress:.2f}%", end="")

        payload_explanation = json.dumps(
            {
                "model": self.model_LLM,
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content}
                ],
            }
        )

        headers = {
            "Authorization": "Bearer " + self.api_key,
            # "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            # "x-api2d-no-cache": 1,
        }

        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                conn = http.client.HTTPSConnection(self.api_endpoint)
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data["choices"][0]["message"]["content"]
                break
            except:
                if self.debug_mode:
                    print("Error in API. Restarting the process...")
                continue

        return response


if __name__ == "__main__":
    llm_api_endpoint = "api.bltcy.ai"  # set your LLM endpoint
    llm_api_url = '/v1/chat/completions'
    llm_api_key = "sk-0hCjhh3wBUP7H2TQF9B6D290Ee604cAc88633dDc5f68B0Ed"  # set your key
    llm_model = "gpt-4o-mini"

    Interfaceapi = InterfaceAPI(llm_api_endpoint, llm_api_url, llm_api_key, llm_model, True)
    print(Interfaceapi.get_response('1+100=?'))
