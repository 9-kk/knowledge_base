from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import openai
import requests
"""
调用大模型api接口
"""


class AIModel:
    # 在QASystem中传入参数
    def __init__(self, model_type="api", model_name="gpt_4o_mini"):
        self.model_type = model_type
        self.model_name = model_name
        # 根据不同模型获取url
        self.url = None
        self.local_model = None
        self.local_tokenizer = None

    #     if model_type == "local":
    #         self._load_local_model(model_name)
    #
    # def _load_local_model(self, model_name):
    #     """加载本地模型"""
    #     try:
    #         self.local_tokenizer = AutoTokenizer.from_pretrained(
    #             model_name,
    #             trust_remote_code=True
    #         )
    #         self.local_model = AutoModelForCausalLM.from_pretrained(
    #             model_name,
    #             device_map="auto",
    #             trust_remote_code=True
    #         )
    #         print(f"本地模型 {model_name} 加载成功")
    #     except Exception as e:
    #         print(f"本地模型加载失败: {str(e)}")
    #         self.model_type = "api"  # 回退到API

    def get_url(self):
        if self.model_name == 'gpt_4o_mini':
            # input:0.15$/million_token output:0.6$/million_token
            self.url = 'http://8.211.137.78:6661/openai_chat_non_reasoning_4o_mini/'
        elif self.model_name == 'gpt_4o':
            # input:2.5$/million_token output:10$/million_token
            self.url = 'http://8.211.137.78:6661/openai_chat_non_reasoning_4o/'
        elif self.model_name == 'gpt_o3_mini':
            # input:1.1$/million_token output:4.4$/million_token
            self.url = 'http://8.211.137.78:6661/openai_chat_reasoning_o3_mini/'
        elif self.model_name == 'gpt_o1':
            # input:15$/million_token output:60$/million_token
            self.url = 'http://8.211.137.78:6661/openai_chat_reasoning_o1/'
            messages = []
        elif self.model_name == 'gpt_o1_mini':
            # input:1.1$/million_token output:4.4$/million_token
            self.url = 'http://8.211.137.78:6661/openai_chat_reasoning_o1_mini/'
            messages = []
        elif self.model_name == 'deepseek':
            # input:2RMB/million_token output:8RMB/million_token
            messages = []
            pass
        else:
            raise ValueError("model is not exist")

    def generate(self, prompt, context="", max_tokens=1000, temperature=0.7):
        """
        生成AI响应
        prompt：用户输入的问题
        context：整合的网络与知识库的信息
        """
        full_prompt = self._build_prompt(prompt, context)
        print("full_prompt", full_prompt)

        if self.model_type == "local" and self.local_model:
            return self._generate_local(full_prompt, max_tokens, temperature)
        else:
            return self._generate_api(full_prompt, max_tokens, temperature)

    def _build_prompt(self, prompt, context):
        """构建提示词"""
        if not context:
            return prompt

        return f"""
        基于以下上下文信息回答问题：
        {context}

        问题：{prompt}

        回答：
        """

    def _generate_local(self, prompt, max_tokens, temperature):
        """使用本地模型生成响应"""
        try:
            inputs = self.local_tokenizer(prompt, return_tensors="pt").to(self.local_model.device)
            outputs = self.local_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )
            response = self.local_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            # 移除提示词部分
            return response.replace(prompt, "").strip()
        except Exception as e:
            return f"本地模型生成失败: {str(e)}"

    def _generate_api(self, prompt, max_tokens, temperature):
        """使用API生成响应"""

        messages = [
            {"role": "system",
             "content": "你现在是一个文件检索与搜集资料的系统"},  # 给gpt身份
        ]

        try:
            self.get_url()
            # 获取url并发送请求
            messages.append({"role": "user", "content": prompt})
            r = requests.post(self.url, json=messages)
            response = r.json()
            print('result', response)
            return response
            # response = openai.ChatCompletion.create(
            #     model=self.model_name,
            #     messages=[
            #         {"role": "user", "content": prompt}
            #     ],
            #     max_tokens=max_tokens,
            #     temperature=temperature
            # )
            # return response.choices[0].message['content'].strip()
        except Exception as e:
            return f"API请求失败: {str(e)}"

    def summarize(self, text, max_length=300):
        """文本摘要"""
        prompt = f"请为以下文本生成摘要（不超过{max_length}字）：\n\n{text}"
        return self.generate(prompt, max_tokens=max_length, temperature=0.3)

    def translate(self, text, target_language="英文"):
        """文本翻译"""
        prompt = f"将以下文本翻译成{target_language}：\n\n{text}"
        return self.generate(prompt, temperature=0.5)
