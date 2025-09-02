import os
import argparse
from qa_system import QASystem


def main():
    # 配置参数
    config = {
        "knowledge_base_dir": "knowledge_base",
        "model_type": "api",  # "local" 或 "api"
        "model_name": "gpt_4o_mini",
        # "api_key": os.getenv("OPENAI_API_KEY", "your_api_key_here")
    }
    # print(0)
    # 初始化系统
    qa_system = QASystem(config)

    # 示例用法
    print("本地知识库与AI问答系统")
    print("=" * 50)

    # 上传文件到知识库
    print("\n上传文件中...")
    upload_result = qa_system.upload_file(r"E:\WeChat Files\WXWork\1688856623467349\Cache\File\2025-07\《市政工程勘察规范》CJJ 56-2012.pdf")
    print(upload_result["message"])

    # # 查询知识库
    # print("\n知识库查询中...")
    # question = "如何实现岩⼟勘察项⽬的自动化"
    # response = qa_system.query(question, use_web=False)
    # print(f"问题: {question}")
    # print(f"回答: {response}")

    # # 网络查询
    # print("\n网络查询中...")
    # question = "2023年诺贝尔物理学奖得主是谁？"
    # response = qa_system.query(question, use_knowledge=False, use_web_content=True)
    # print(f"问题: {question}")
    # print(f"回答: {response}")

    # # 综合查询
    # print("\n综合查询中...")
    # question = "如何申请美国研究生院？"
    # response = qa_system.query(question)
    # print(f"问题: {question}")
    # print(f"回答: {response}")

    # # 文档摘要
    # print("\n文档摘要中...")
    # summary = qa_system.summarize_document("example_document.pdf")
    # print(f"文档摘要: {summary}")

    # # 文本翻译
    # print("\n文本翻译中...")
    # text_to_translate = "人工智能是未来科技发展的核心驱动力"
    # translation = qa_system.translate_text(text_to_translate, "英文")
    # print(f"原文: {text_to_translate}")
    # print(f"英文: {translation}")


if __name__ == "__main__":
    main()
