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

    def query(self, question, use_knowledge=True, use_web=False, use_web_content=False):
        # 解析问题中的过滤条件
        filters = self._parse_filters(question)
        filtered_question = filters["filtered_question"]
        """回答问题"""
        start_time = time.time()

        # 收集上下文
        context_parts = []
        sources = set()

        # 1. 知识库查询
        if use_knowledge:
            # 检查是否存在明确的过滤条件
            has_explicit_filter = any([
                filters["chapter"],
                filters["table"],
                filters["page"],
                filters["file"]
            ])

            # 构建文件过滤器
            file_filter = None
            if filters["file"]:
                # 查找匹配的文件名
                matching_files = []
                for file_hash, meta in self.knowledge_base.metadata.items():
                    if filters["file"].lower() in meta["filename"].lower():
                        matching_files.append(meta["filename"])

                if matching_files:
                    file_filter = matching_files

            # 如果有明确的过滤条件，只返回匹配的内容
            if has_explicit_filter:
                # 设置较小的top_k，因为我们只需要匹配过滤条件的内容
                kb_results = self.knowledge_base.search(
                    filtered_question,
                    top_k=10,  # 稍微增加一点以应对可能的多个匹配
                    chapter_filter=filters["chapter"],
                    table_filter=filters["table"],
                    page_filter=filters["page"],
                    file_filter=file_filter
                )

                # 如果没有找到匹配过滤条件的内容，添加提示
                if not kb_results:
                    context_parts.append("未找到符合过滤条件的相关内容。")
                else:
                    context_parts.append("知识库相关内容（已应用过滤条件）：")
                    for i, res in enumerate(kb_results, 1):
                        context_parts.append(f"{i}. [{res['source']}] {res['content']}")
                        sources.add(res['source'])
            else:
                # 如果没有明确的过滤条件，使用常规搜索
                kb_results = self.knowledge_base.search(
                    filtered_question,
                    top_k=5
                )

                if kb_results:
                    context_parts.append("知识库相关内容：")
                    for i, res in enumerate(kb_results, 1):
                        context_parts.append(f"{i}. [{res['source']}] {res['content']}")
                        sources.add(res['source'])

            # 添加过滤提示（只有在有过滤条件且有结果时才显示）
            if has_explicit_filter and kb_results:
                filter_notes = []
                if filters["chapter"]:
                    filter_notes.append(f"章节'{filters['chapter']}'")
                if filters["table"]:
                    filter_notes.append(f"表格'{filters['table']}'")
                if filters["page"]:
                    filter_notes.append(f"第{filters['page']}页")
                if filters["file"]:
                    filter_notes.append(f"文件'{filters['file']}'")

                if filter_notes:
                    context_parts.append(f"\n注意：已应用过滤条件：{', '.join(filter_notes)}")

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
        print('full_context', full_context)
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
            "sources": list(sources),
            "filters": filters
        })

        return response

    def _parse_filters(self, question):
        """解析问题中的过滤条件（支持数字编号格式）"""
        import re

        filters = {
            "chapter": None,
            "table": None,
            "page": None,
            "file": None,
            "filtered_question": question
        }

        # 匹配数字编号章节过滤条件（如"3.0.1"、"1.2.3"）
        number_chapter_patterns = [
            r'第?(\d+(\.\d+)*)[章节条]',
            r'第?(\d+(\.\d+)*)',
            r'章节?(\d+(\.\d+)*)',
            r'条(\d+(\.\d+)*)'
        ]

        for pattern in number_chapter_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                filters["chapter"] = match.group(1)
                filters["filtered_question"] = re.sub(pattern, '', filters["filtered_question"]).strip()
                break

        # 如果没有匹配到数字编号，再尝试匹配传统章节格式
        if not filters["chapter"]:
            chapter_patterns = [
                r'在第?([一二三四五六七八九十百千万零]+)章',
                r'在(.+?)章节',
                r'关于第?([一二三四五六七八九十百千万零]+)章',
                r'第?([一二三四五六七八九十百千万零]+)章中?'
            ]

            for pattern in chapter_patterns:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    filters["chapter"] = match.group(1) or match.group(2)
                    filters["filtered_question"] = re.sub(pattern, '', filters["filtered_question"]).strip()
                    break

        # 匹配表格过滤条件
        table_patterns = [
            r'表([\d\.]+)',
            r'表格([\d\.]+)',
            r'表(\d+)[\.\s]?(\d*)',
            r'Table\s*([\d\.]+)'
        ]

        for pattern in table_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                filters["table"] = match.group(1)
                if match.lastindex > 1 and match.group(2):
                    filters["table"] += "." + match.group(2)
                filters["filtered_question"] = re.sub(pattern, '', filters["filtered_question"]).strip()
                break

        # 匹配页面过滤条件
        page_patterns = [
            r'第?(\d+)页',
            r'在第?(\d+)页',
            r'page\s*(\d+)',
            r'页码?(\d+)'
        ]

        for pattern in page_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                filters["page"] = int(match.group(1))
                filters["filtered_question"] = re.sub(pattern, '', filters["filtered_question"]).strip()
                break

        # 匹配文件过滤条件
        file_patterns = [
            r'在(.+?)文件中',
            r'从(.+?)文件',
            r'文件(.+?)中',
            r'文档(.+?)的'
        ]

        for pattern in file_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                filters["file"] = match.group(1).strip()
                filters["filtered_question"] = re.sub(pattern, '', filters["filtered_question"]).strip()
                break

        return filters

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
