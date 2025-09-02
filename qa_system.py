import time
from knowledge_base import KnowledgeBase
from web_search import WebSearcher
from gpt_model import AIModel


class QASystem:
    def __init__(self, config):
        self.config = config
        self.knowledge_base = KnowledgeBase(config["knowledge_base_dir"])
        self.web_searcher = WebSearcher()
        self.ai_model = AIModel(
            model_type=config.get("model_type", "api"),
            model_name=config.get("model_name", "gpt_4o_mini")
        )
        self.history = []

    def upload_file(self, file_path):
        """上传文件到知识库"""
        return self.knowledge_base.upload_file(file_path)

    def query(self, question, use_knowledge=True, use_web=True, use_web_content=False):
        """回答问题"""
        start_time = time.time()

        # 收集上下文
        context_parts = []
        sources = set()

        # 1. 知识库查询
        if use_knowledge:
            kb_results = self.knowledge_base.search(question)
            if kb_results:
                context_parts.append("知识库相关内容：")
                for i, res in enumerate(kb_results, 1):
                    # 检索到的信息
                    context_parts.append(f"{i}. [{res['source']}] {res['content']}")
                    # 检索的文件来源
                    sources.add(res['source'])

        # 2. 网络搜索
        web_results = []
        if use_web:
            web_results = self.web_searcher.search(question)
            print(web_results)
            if web_results:
                context_parts.append("\n网络搜索结果：")
                for i, res in enumerate(web_results, 1):
                    if use_web_content:
                        # 获取完整页面内容
                        content = self.web_searcher.get_page_content(res['url'])
                        if content:
                            context_parts.append(
                                f"{i}. {res['title']}\n   URL: {res['url']}\n   内容摘要：{content[:500]}...")
                        else:
                            context_parts.append(f"{i}. {res['title']}\n   URL: {res['url']}\n   摘要：{res['snippet']}")
                    else:
                        context_parts.append(f"{i}. {res['title']}\n   URL: {res['url']}\n   摘要：{res['snippet']}")
                    sources.add(res['url'])
                    # context_parts.append(f"{i}. {res['title']}\n   URL: {res['url']}")
                    # if res['snippet']:
                    #     context_parts.append(f"   摘要：{res['snippet']}")
                    # sources.add(res['url'])

        # 3. 整合上下文
        full_context = "\n".join(context_parts)
        print(full_context)
        response = ""

        # 4. 生成AI响应
        response = self.ai_model.generate(question, full_context)

        # 5. 添加来源信息
        if sources:
            response += "\n\n信息来源："
            response += "\n" + "\n".join(f"- {src}" for src in sources)

        print('response', response)

        # 记录历史
        self.history.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "response": response,
            "time_taken": time.time() - start_time,
            "sources": list(sources)
        })

        return response

    def summarize_document(self, file_path):
        """文档摘要"""
        # 上传文档（如果尚未存在）
        upload_result = self.upload_file(file_path)
        if upload_result["status"] != "success" and "已存在" not in upload_result["message"]:
            return f"文档处理失败: {upload_result['message']}"

        # 获取文档内容
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(10000)  # 读取前10000字符

        return self.ai_model.summarize(content)

    def translate_text(self, text, target_language="英文"):
        """文本翻译"""
        return self.ai_model.translate(text, target_language)

    def get_history(self):
        """获取查询历史"""
        return self.history
