import os
import json
import hashlib
import time
import numpy as np
import faiss
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# from paddlex import create_pipeline
# from sentence_transformers import SentenceTransformer
import torch


class TableProcessor:
    def __init__(self):
        from transformers import DetrImageProcessor, DetrForObjectDetection
        from paddleocr import PaddleOCR
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用所有GPU
        # 初始化表格检测模型 (DETR)
        self.processor = DetrImageProcessor.from_pretrained("models/detr-doc-table-detection")
        self.detection_model = DetrForObjectDetection.from_pretrained("models/detr-doc-table-detection",
                                                                      low_cpu_mem_usage=False  # 禁用 meta device
                                                                      )
        # 初始化 PP-TableMagic 表格识别管道
        # self.table_pipeline = create_pipeline(pipeline="table_recognition_v2")
        print(self.detection_model.device)  # 应显示实际设备（如cuda:0）
        print(self.detection_model.config)  # 应显示模型配置

        # 初始化OCR引擎 (PaddleOCR)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')

    def detect_tables(self, image):
        """检测图像中的表格位置"""
        # 预处理图像
        inputs = self.processor(images=image, return_tensors="pt")

        # 模型推理
        with torch.no_grad():
            outputs = self.detection_model(**inputs)

        # 后处理
        target_sizes = torch.tensor([image.shape[:2]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5
        )[0]

        # 提取表格边界框
        table_bboxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if self.detection_model.config.id2label[label.item()] == "table":
                box = [int(i) for i in box.tolist()]
                table_bboxes.append(box)

        return table_bboxes

    def extract_table_content(self, image, bbox):
        try:

            """提取表格内容（带结构识别）"""
            # 裁剪表格区域
            x_min, y_min, x_max, y_max = bbox
            h, w = image.shape[:2]
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(w, int(x_max))
            y_max = min(h, int(y_max))

            # 检查bbox是否有效
            if x_max <= x_min or y_max <= y_min:
                # print(f"无效的bbox: {bbox}")
                return []

            table_img = image[y_min:y_max, x_min:x_max]
            # print('table_img', table_img.shape)

            # 识别表格结构
            print('table_img', type(table_img), table_img.shape, table_img.shape[2])
            structure = self.recognize_table_structure(table_img)
            # print('structure', structure)

            # 提取单元格内容
            table_data = []

            # 获取行列边界
            rows = structure["rows"]
            cols = structure["columns"]

            # 确保行列边界在表格图像范围内
            rows = [r for r in rows if r < table_img.shape[0]]
            cols = [c for c in cols if c < table_img.shape[1]]

            # 遍历每个单元格
            for i in range(len(rows) - 1):
                row_data = []
                for j in range(len(cols) - 1):
                    # 裁剪单元格图像
                    cell_y1 = rows[i]
                    cell_y2 = rows[i + 1]
                    cell_x1 = cols[j]
                    cell_x2 = cols[j + 1]

                    # 确保单元格坐标有效
                    if cell_y1 >= cell_y2 or cell_x1 >= cell_x2:
                        row_data.append("")
                        continue

                    cell_img = table_img[cell_y1:cell_y2, cell_x1:cell_x2]

                    # print('cell_img', cell_img)
                    # print(f'cell_img shape: {cell_img.shape}')  # 打印单元格图像形状

                    # 如果单元格图像为空，则添加空字符串并跳过OCR
                    # 检查单元格图像是否过小，跳过过小的单元格
                    if cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
                        row_data.append("")
                        continue

                    # 对单元格进行OCR
                    print('shape', cell_img.shape)
                    results = self.ocr.ocr(cell_img)
                    result = results[0]
                    print('result', result)

                    # 提取文本内容
                    cell_text = ""
                    if result and 'rec_texts' in result:
                        for text in result['rec_texts']:
                            cell_text += text + " "

                    row_data.append(cell_text.strip().replace(' ', '').replace('\n', ''))

                table_data.append(row_data)
            print('table_data', table_data)

            return table_data

        except Exception as e:
            import traceback
            print(f"表格提取失败: {str(e)}")
            print(traceback.format_exc())  # 打印堆栈跟踪

    def recognize_table_structure(self, table_image):
        """识别表格结构（使用深度学习方法）"""
        # 使用更先进的表格结构识别方法
        # try:
            # 方法1：使用深度学习模型（如果有）
        # self._recognize_with_deep_learning(table_image)
        # except:
            # 方法2：回退到传统方法
        return self._recognize_with_traditional(table_image)

    def _recognize_with_deep_learning(self, table_image):
        """使用深度学习识别表格结构"""
        """处理页面图像"""
        # 使用 PP-TableMagic 进行端到端的表格识别
        # input 可以是图像路径或 numpy 数组（HWC, BGR格式）
        results = self.table_pipeline.predict(
            input=table_image)  # 默认会自动进行方向校正和扭曲矫正，如需关闭可传参 use_doc_orientation_classify=False, use_doc_unwarping=False

        # page_content = f"第{page_num + 1}页:\n"
        page_content = ''

        # 假设我们主要处理第一个识别结果（对于单页单表的情况）
        if results:
            # 获取第一个结果对象
            table_result = results[0]

            # 1. 可以直接输出或保存结构化结果
            # 打印识别结果的基本信息
            print('---------------------------')
            table_result.print()
            # 将表格保存为 HTML、Excel、JSON 等多种格式
            # table_result.save_to_html("./output/")
            # table_result.save_to_xlsx("./output/")
            # table_result.save_to_json("./output/")
            #
            # # 2. 获取标记down字符串（例如HTML格式）并添加到页面内容中
            # # 如果你需要将表格的Markdown或HTML字符串整合到你的文本输出中
            # html_str = table_result.get_html()  # 获取HTML字符串
        #     # 或者，如果你需要Markdown格式，可能需要从HTML转换，或查看API是否直接支持
        #     # 假设你有一个将HTML表格转换为Markdown的函数
        #     table_md = self.html_table_to_markdown(html_str)
        #     page_content += f"\n{table_md}\n"
        #
        # return page_content

    def _recognize_with_traditional(self, table_image):
        """使用传统方法识别表格结构"""
        import cv2
        import numpy as np

        gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)

        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 检测水平线
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # 检测垂直线
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # 合并线
        grid = cv2.add(horizontal, vertical)

        # 找到交点
        intersections = cv2.bitwise_and(horizontal, vertical)

        # 提取行和列的位置
        # 水平线y坐标
        y_coords = []
        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            y = contour[0][0][1]
            if y not in y_coords:
                y_coords.append(y)

        # 垂直线x坐标
        x_coords = []
        contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x = contour[0][0][0]
            if x not in x_coords:
                x_coords.append(x)

        # 排序坐标
        rows = sorted(y_coords)
        cols = sorted(x_coords)

        # 添加边界
        rows = [0] + rows + [table_image.shape[0]]
        cols = [0] + cols + [table_image.shape[1]]

        return {"rows": rows, "columns": cols}

    def convert_to_markdown(self, table_data):
        """将表格数据转换为Markdown格式"""
        if not table_data or len(table_data) < 1:
            return ""

        # 创建表头
        markdown = "| " + " | ".join(table_data[0]) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(table_data[0])) + " |\n"

        # 添加行数据
        for row in table_data[1:]:
            markdown += "| " + " | ".join(row) + " |\n"

        return markdown


class KnowledgeBase:
    def __init__(self, base_dir="knowledge_base"):
        self.base_dir = base_dir
        self.documents_dir = os.path.join(base_dir, "documents")
        self.chunks_dir = os.path.join(base_dir, "chunks")
        self.embeddings_dir = os.path.join(base_dir, "embeddings")
        self.structure_dir = os.path.join(base_dir, "structures")  # 新增结构信息目录
        self.metadata_file = os.path.join(base_dir, "metadata.json")

        # 创建目录
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.structure_dir, exist_ok=True)  # 创建结构信息目录

        # 初始化嵌入模型
        '''
        BAAI/bge-small-zh-v1.5 是由北京智源人工智能研究院（BAAI）开源的一种中文嵌入模型（Embedding Model），
        专门用于将文本映射为低维稠密向量，广泛应用于检索、分类、聚类或语义匹配等任务。
        '''
        self.embedding_model = HuggingFaceEmbeddings(
            # model_name="BAAI/bge-small-zh-v1.5",
            model_name="models/bge-small-zh-v1.5",  # 替换为本地路径
            model_kwargs={'device': 'cpu'}
        )

        # 初始化FAISS索引
        self.index = faiss.IndexFlatL2(384)  # 假设嵌入维度为384
        self.faiss_index_path = os.path.join(base_dir, "faiss_index")

        # 加载元数据
        self.metadata = self._load_metadata()

        # 初始化FAISS索引
        self.index = self._init_faiss_index()

        # 初始化表格处理器
        self.table_processor = TableProcessor()

    def _load_metadata(self):
        """加载元数据文件"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """保存元数据文件"""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _init_faiss_index(self):
        """初始化FAISS索引"""
        embedding_size = 1024  # 根据嵌入模型调整
        index_path = os.path.join(self.base_dir, "faiss_index")

        if os.path.exists(index_path):
            return faiss.read_index(index_path)

        # 创建新索引
        return faiss.IndexFlatL2(embedding_size)

    def _is_image_pdf(self, file_path):
        """检测PDF是否为图片型（扫描版）"""
        try:
            from PyPDF2 import PdfReader

            # 打开PDF文件
            with open(file_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)

                # 检查前3页（或所有页）
                num_pages_to_check = min(3, len(reader.pages))
                text_content = ""

                for i in range(num_pages_to_check):
                    page = reader.pages[i]
                    text_content += page.extract_text()

                # 如果提取的文本很少或为空，很可能是图片型PDF
                if len(text_content.strip()) < 75:  # 阈值可根据实际情况调整
                    return True

            return False
        except Exception as e:
            print(f"PDF检测失败: {str(e)}")
            return False

    def upload_file(self, file_path):
        """上传并处理文件"""
        # 1.检查文件是否存在
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"文件不存在: {file_path}"}

        # 2.计算文件哈希（用于唯一标识文件）
        with open(file_path, "rb") as f:
            file_content = f.read()
            file_hash = hashlib.md5(file_content).hexdigest()

        # 3.检查文件是否已存在知识库中
        if file_hash in self.metadata:
            return {"status": "info", "message": "文件已在知识库中存在"}

        # 4. 保存原始文件到知识库文档目录
        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.documents_dir, filename)
        with open(dest_path, "wb") as f:
            f.write(file_content)

        # 5. 根据文件类型选择加载器
        ext = os.path.splitext(file_path)[1].lower()

        loader_map = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
            ".txt": TextLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".ppt": UnstructuredPowerPointLoader,
            ".html": UnstructuredHTMLLoader,
            ".htm": UnstructuredHTMLLoader,
            ".md": UnstructuredMarkdownLoader,
        }
        loader_class = loader_map.get(ext)

        if not loader_class:
            return {"status": "error", "message": f"不支持的文件类型: {os.path.splitext(file_path)[1]}"}
        # print(loader_class)

        if ext == '.pdf':
            # 检测是否为图片型PDF
            # print(self._is_image_pdf(file_path))
            if self._is_image_pdf(file_path):
                # 初始化
                # self.tables_dir = os.path.join(base_dir, "tables")
                # os.makedirs(self.tables_dir, exist_ok=True)
                self.table_processor = TableProcessor()  # 表格处理器
                return self._process_image_pdf_with_structure(file_path, file_hash, filename, dest_path)

        try:
            # 6. 使用加载器加载文档内容
            loader = loader_class(file_path)
            document = loader.load()
        except Exception as e:
            return {"status": "error", "message": f"文件加载失败: {str(e)}"}

        # 7. 分割文本为多个块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(document)

        # 8. 保存文本块到chunks目录
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            chunk_filename = f"{file_hash}_chunk{i}.txt"
            chunk_path = os.path.join(self.chunks_dir, chunk_filename)

            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk_text)

            chunk_files.append(chunk_path)

        # 9. 生成文本嵌入向量
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(texts)
        embeddings_np = np.array(embeddings).astype("float32")

        # 10. 保存嵌入向量
        embedding_path = os.path.join(self.embeddings_dir, f"{file_hash}.npy")
        np.save(embedding_path, embeddings_np)

        # 11. 更新FAISS索引
        if self.index.ntotal == 0:
            # 如果是第一个文档，重置索引维度
            self.index = faiss.IndexFlatL2(embeddings_np.shape[1])
        self.index.add(embeddings_np)
        faiss.write_index(self.index, os.path.join(self.base_dir, "faiss_index"))

        # 12. 保存索引
        faiss.write_index(self.index, os.path.join(self.base_dir, "faiss_index"))

        # 13. 更新元数据
        self.metadata[file_hash] = {
            "filename": filename,
            "file_path": dest_path,
            "chunk_files": chunk_files,
            "embedding_path": embedding_path,
            "chunk_count": len(chunks),
            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_ocr": False,  # 标记为非OCR处理
            "has_structure": False  # 标记为无结构信息
        }
        self._save_metadata()

        return {"status": "success", "message": f"文件已成功添加到知识库: {filename}"}

    def _process_image_pdf_with_structure(self, file_path, file_hash, filename, dest_path):
        """处理图片型PDF（使用OCR提取文本）"""
        try:
            print(f"检测到图片型PDF，使用OCR处理: {filename}")

            # OCR提取文本
            ocr_result = self._ocr_pdf_with_structure(file_path)
            # print('ocr_text', ocr_result)
            if not ocr_result or "content" not in ocr_result:
                return {"status": "error", "message": "OCR提取文本失败"}

            # 创建文档对象
            from langchain_core.documents import Document
            document = [Document(page_content=ocr_result["content"])]

            # 分割文本
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=50,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(document)

            # 保存文本块
            chunk_files = []
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.page_content
                chunk_filename = f"{file_hash}_chunk{i}.txt"
                chunk_path = os.path.join(self.chunks_dir, chunk_filename)

                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write(chunk_text)

                chunk_files.append(chunk_path)

            # 生成嵌入向量
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embedding_model.embed_documents(texts)
            embeddings_np = np.array(embeddings).astype("float32")

            # 保存嵌入向量
            embedding_path = os.path.join(self.embeddings_dir, f"{file_hash}.npy")
            np.save(embedding_path, embeddings_np)

            # 更新索引
            if self.index.ntotal == 0:
                self.index = faiss.IndexFlatL2(embeddings_np.shape[1])
            self.index.add(embeddings_np)
            faiss.write_index(self.index, os.path.join(self.base_dir, "faiss_index"))

            # 保存结构信息（如果有）
            if "structure" in ocr_result:
                structure_path = os.path.join(self.structure_dir, f"{file_hash}.json")
                # JSON 序列化不支持ndarray 类型的数据,递归转换所有 ndarray 为列表
                structure = self.convert_ndarray_to_list(ocr_result["structure"])
                with open(structure_path, "w", encoding="utf-8") as f:
                    json.dump(structure, f, ensure_ascii=False, indent=2)

            # 更新元数据
            self.metadata[file_hash] = {
                "filename": filename,
                "file_path": dest_path,
                "chunk_files": chunk_files,
                "embedding_path": embedding_path,
                "chunk_count": len(chunks),
                "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "is_ocr": True,  # 标记为OCR处理
                "has_structure": "structure" in ocr_result  # 标记是否有结构信息
            }
            self._save_metadata()

            # 返回包含结构信息的结果
            return {
                "status": "success",
                "message": f"图片型PDF已通过OCR添加到知识库: {filename}"
            }

            # if "structure" in ocr_result:
            #     result["structure"] = ocr_result["structure"]
        except Exception as e:
            import traceback
            print(f"2OCR处理失败: {str(e)}")
            print(traceback.format_exc())  # 打印堆栈跟踪
            return {"status": "error", "message": f"2OCR处理失败: {str(e)}"}

    def convert_ndarray_to_list(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_ndarray_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ndarray_to_list(item) for item in obj]
        return obj

    def _ocr_pdf_with_structure(self, pdf_path):
        """使用OCR提取PDF中的文本（支持中英文）"""
        try:
            from pdf2image import convert_from_path
            import pytesseract
            import re
            import numpy as np

            # 配置Tesseract路径（Windows可能需要）
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

            # 转换PDF为图像列表
            images = convert_from_path(pdf_path)

            # 识别每页文本
            full_content = ""

            structure_info = {
                "chapters": [],
                "tables": [],
                "pages": []
            }
            for page_num, image in enumerate(images):
                img_np = np.array(image)

                # 1. 提取页面所有文本（包含位置信息）
                text_blocks = self._extract_text_with_position(img_np)
                # print('text_blocks', text_blocks)

                # 2. 检测是否为目录页
                is_toc_page = self._is_table_of_contents_page(text_blocks, page_num + 1)
                if is_toc_page:
                    continue

                # 3. 检测表格
                tables = self.table_processor.detect_tables(img_np)
                print('tables', tables)

                # 4. 识别章节标题（通过字体大小、位置等特征）
                chapter_titles = self._detect_chapter_titles(text_blocks)
                for title in chapter_titles:
                    structure_info["chapters"].append({
                        "title": title["text"],
                        "page": page_num + 1,
                        "y_position": title["y"],
                        "bbox": title["bbox"]
                    })

                # 3. 按垂直位置排序所有元素
                all_elements = []

                # 添加文本块
                for block in text_blocks:
                    all_elements.append({
                        "type": "text",
                        "y": block["y"],
                        "content": block["text"],
                        "is_table_caption": False
                    })
                    # print(block["text"])

                # 添加表格（检测表名）
                page_tables = []
                for table_idx, table_bbox in enumerate(tables):
                    table_id = f"page_{page_num + 1}_table_{table_idx + 1}"
                    print('table_id', table_id)

                    # 查找表格上方的表名
                    caption = self._find_table_caption(text_blocks, table_bbox)
                    print('caption', caption)

                    # 添加表名
                    if caption:
                        # 结构添加章节标题
                        structure_info["chapters"].append({
                            "title": caption["text"],
                            "page": page_num + 1,
                            "y_position": caption["y"],
                            "bbox": caption["bbox"]
                        })
                        all_elements.append({
                            "type": "text",
                            "y": caption["y"],
                            "content": caption["text"],
                            "is_table_caption": True
                        })

                        # 结构添加表格
                        table_info = {
                            "id": table_id,
                            "page": page_num + 1,
                            "caption": caption["text"],
                            "bbox": table_bbox
                        }
                        structure_info["tables"].append(table_info)
                        page_tables.append(table_info)

                        # 添加表格
                        all_elements.append({
                            "type": "table",
                            "y": table_bbox[1],  # 表格顶部y坐标
                            "table_id": table_id,
                            "bbox": table_bbox
                        })
                    # print('table_bbox', table_bbox)
                    # print('all_elements', all_elements)

                # 4. 按垂直位置排序
                all_elements.sort(key=lambda x: x["y"])

                # 5. 构建页面内容
                page_content = f"第{page_num + 1}页:\n"
                # print(page_content)
                for element in all_elements:
                    if element["type"] == "text":
                        # 如果是表名，已经单独处理
                        if not element["is_table_caption"]:
                            page_content += element["content"] + "\n"
                    else:
                        print('---------------------------------')
                        # 表格
                        # 提取表格内容
                        table_data = self.table_processor.extract_table_content(
                            img_np, element["bbox"]
                        )
                        table_md = self.table_processor.convert_to_markdown(table_data)
                        print('table_md', table_md)

                        # 添加表格
                        page_content += f"\n{table_md}\n"

                full_content += page_content + "\n\n"

                # 保存页面结构信息
                page_structure = {
                    "page_num": page_num + 1,
                    "text_blocks": text_blocks,
                    "tables": structure_info["tables"],
                    "chapter_titles": chapter_titles
                }
                structure_info["pages"].append(page_structure)

                # # 使用中文+英文识别
                # # 去掉多余的空格
                # text = pytesseract.image_to_string(image, lang='chi_sim+eng').replace(' ', '')
                # # 两个或以上换行符换成一个换行符，只有一个换行符删掉
                # text = re.sub(r'\n', ' ', text)
                # text = re.sub(r'\n\s*\n', '\n', text)
                # full_content += f"第{i + 1}页:\n{text}\n\n"
            print('full_content', full_content)
            print('structure', structure_info)

            # return full_content
            return {
                "content": full_content,
                "structure": structure_info
            }
        except ImportError:
            print("OCR依赖未安装，请运行: pip install pdf2image pytesseract")
            return None
        except Exception as e:
            import traceback
            print(f"1OCR处理失败: {str(e)}")
            print(traceback.format_exc())  # 打印堆栈跟踪
            return None

    def _is_table_of_contents_page(self, text_blocks, page_num):
        import re
        """检测是否为目录页"""
        # 检查页面是否包含"目录"、"CONTENTS"等关键词
        toc_keywords = ["目录", "CONTENTS", "目次", "TABLE OF CONTENTS"]

        for block in text_blocks:
            text = block["text"].strip()
            if any(keyword in text for keyword in toc_keywords):
                return True

        # 检查页面是否有典型的目录特征（大量数字和点）
        dot_pattern_count = 0
        number_pattern_count = 0

        for block in text_blocks:
            text = block["text"].strip()

            # 检查是否有大量点（目录中的引导点）
            if "..." in text or "…" in text or "．" in text:
                dot_pattern_count += 1

            # 检查是否有页码模式（数字在行尾）
            if re.search(r'\d+$', text):
                number_pattern_count += 1

        # 如果点的数量或页码模式数量超过阈值，认为是目录页
        if dot_pattern_count > 3 or number_pattern_count > 5:
            return True

        # 第一页很可能是目录页（但不是绝对）
        if page_num == 1 and (dot_pattern_count > 1 or number_pattern_count > 2):
            return True

        return False

    def _extract_text_with_position(self, image):
        """提取文本及其位置信息"""
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        results = ocr.ocr(image)
        result = results[0]

        text_blocks = []
        print(result)
        print(type(result))
        for line in result['rec_texts']:
            # 提取文本并进行 OCR 错误校正
            corrected_text = self._correct_ocr_errors(line)
            print('corrected_text', corrected_text)
            # 计算文本块的中心y坐标
            # 获取文本框坐标
            points = result['rec_polys'][result['rec_texts'].index(line)]
            y_coords = [point[1] for point in points]
            y_center = sum(y_coords) / len(y_coords)

            text_blocks.append({
                "text": corrected_text,
                "y": y_center,
                "bbox": points
            })

        return text_blocks

    def _correct_ocr_errors(self, text):
        import re
        """校正OCR识别错误"""
        if not text:
            return text

        # 常见OCR错误映射表
        error_mapping = {
            # 数字错误
            '【': '1', '】': '1', 'l': '1', 'I': '1', '丨': '1',
            '﹒': '2', 'ｚ': '2', 'Z': '2',
            '§': '3', 'з': '3', 'З': '3',
            'Ч': '4', 'ч': '4',
            'Б': '5', 'ь': '5',
            'б': '6',
            'Т': '7', 'т': '7',
            'В': '8', 'в': '8',
            'д': '9', 'g': '9',
            'о': '0', 'O': '0', '〇': '0',

            # 标点错误
            '，': '.', '。': '.', '、': '.', '·': '.', '•': '.',
            '；': ':', '：': ':',
            '‘': "'", '’': "'", '「': "'", '」': "'",
            '『': '"', '』': '"', '〝': '"', '〞': '"',

            # 括号错误
            '〔': '(', '〕': ')', '［': '[', '］': ']', '｛': '{', '｝': '}',

            # 其他常见错误
            '一': '-', '一一': '=', '二': '=', 'ニ': '='
        }

        # 逐步校正
        corrected_text = text

        # 第一步：直接替换已知错误映射
        for error, correction in error_mapping.items():
            corrected_text = corrected_text.replace(error, correction)

        # 第二步：处理数字编号模式错误
        # 匹配类似"3.0.【"的模式并校正为"3.0.1"
        number_error_patterns = [
            (r'(\d+)\.(\d+)\.【', r'\1.\2.1'),  # 3.0.【 → 3.0.1
            (r'(\d+)\.(\d+)\.】', r'\1.\2.1'),  # 3.0.】 → 3.0.1
            (r'(\d+)\.(\d+)\.l', r'\1.\2.1'),  # 3.0.l → 3.0.1
            (r'(\d+)\.(\d+)\.I', r'\1.\2.1'),  # 3.0.I → 3.0.1
            (r'(\d+)\.(\d+)\.丨', r'\1.\2.1'),  # 3.0.丨 → 3.0.1
        ]

        for pattern, replacement in number_error_patterns:
            corrected_text = re.sub(pattern, replacement, corrected_text)

        # # 第三步：处理缺失的点号
        # # 将"301"校正为"3.0.1"（需要谨慎使用，可能会误校正）
        # missing_dot_pattern = r'(\d)(\d)(\d)(?=\s|$)'
        #
        # def add_dots(match):
        #     # 只在看起来像编号的情况下添加点号
        #     numbers = match.groups()
        #     if len(numbers) == 3 and numbers[0] in '123456789' and numbers[1] in '0123456789' and numbers[
        #         2] in '0123456789':
        #         return f"{numbers[0]}.{numbers[1]}.{numbers[2]}"
        #     return match.group(0)
        #
        # corrected_text = re.sub(missing_dot_pattern, add_dots, corrected_text)

        # 第四步：处理空格问题
        # 将"3 . 0 . 1"校正为"3.0.1"
        space_pattern = r'(\d)\s*\.\s*(\d)\s*\.\s*(\d)'
        corrected_text = re.sub(space_pattern, r'\1.\2.\3', corrected_text).replace(' ', '').replace('\n', '')

        return corrected_text

    """检测章节标题"""

    def _detect_chapter_titles(self, text_blocks):
        import re
        """检测章节标题（支持数字编号格式）"""
        chapter_titles = []

        # 正则表达式模式匹配数字编号标题
        # 匹配如 "3.0.1"、"1.2"、"2.3.4.5" 等格式
        number_pattern = r'^\d+(\.\d+)*\s+'

        for block in text_blocks:
            text = block["text"].strip()

            # 检测数字编号章节标题
            if re.search(number_pattern, text):
                # 确保不是页码或其他数字（通过文本长度和内容判断）
                if len(text) > 3 and not text.isdigit():  # 排除纯数字（可能是页码）
                    chapter_titles.append(block)
                    continue

            # 检测传统章节标题（保留原有逻辑）
            if (("章" in text and "第" in text) or
                    (len(text) < 30 and any(c in text for c in ["概述", "引言", "背景", "结论", "摘要"])) or
                    (len(text) < 20 and text.endswith("章"))):
                chapter_titles.append(block)

        return chapter_titles

    # 检测表名
    def _find_table_caption(self, text_blocks, table_bbox):
        """在表格上方查找表名"""
        table_top = table_bbox[1]

        # 在表格上方50像素范围内查找文本
        caption_candidates = []
        for block in text_blocks:
            # 检查是否在表格上方
            if block["bbox"][0][1] < table_top and block["bbox"][3][1] < table_top:
                # 检查垂直距离是否在合理范围内
                distance = table_top - block["bbox"][3][1]
                if distance < 100:  # 最多100像素距离
                    caption_candidates.append((distance, block))

        # 如果没有候选，返回None
        if not caption_candidates:
            return None

        # 选择最近的文本块作为表名
        caption_candidates.sort(key=lambda x: x[0])
        return caption_candidates[0][1]

    def search(self, query, top_k=5, chapter_filter=None, table_filter=None, page_filter=None,
               file_filter=None):
        # TODO:修改搜索方法以支持结构过滤和文件过滤
        """在知识库中搜索相关内容，支持章节、表格、页面和文件过滤"""
        # 生成查询嵌入
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding_np = np.array([query_embedding]).astype("float32")

        # 搜索索引
        # 搜索索引（获取更多结果以便过滤）
        search_k = top_k * 5  # 获取更多结果用于过滤
        distances, indices = self.index.search(query_embedding_np, search_k)

        # 收集结果
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx < 0:
                continue

            # 查找包含该索引的文档
            current_idx = 0
            for file_hash, meta in self.metadata.items():
                start_idx = current_idx
                end_idx = current_idx + meta["chunk_count"]

                if start_idx <= idx < end_idx:
                    # 应用文件过滤
                    if file_filter and meta["filename"] not in file_filter:
                        current_idx = end_idx
                        continue

                    chunk_idx = idx - start_idx
                    chunk_path = meta["chunk_files"][chunk_idx]

                    with open(chunk_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    result_item = {
                        "source": f"知识库: {meta['filename']}",
                        "content": content,
                        "score": float(1 - distances[0][i]),  # 转换为相似度分数
                        "chunk_index": chunk_idx,
                        "file_hash": file_hash
                    }

                    # 加载结构信息（如果有）
                    if meta.get("has_structure", False):
                        structure_path = os.path.join(self.structure_dir, f"{file_hash}.json")
                        if os.path.exists(structure_path):
                            with open(structure_path, "r", encoding="utf-8") as f:
                                result_item["structure"] = json.load(f)

                    results.append(result_item)
                    break

                current_idx = end_idx

        # 应用结构过滤
        if chapter_filter or table_filter or page_filter:
            filtered_results = []
            for result in results:
                # 检查是否有结构信息
                if "structure" not in result:
                    continue

                # 应用章节过滤
                if chapter_filter:
                    chapter_match = False
                    for chapter in result["structure"].get("chapters", []):
                        if chapter_filter.lower() in chapter["title"].lower():
                            chapter_match = True
                            break
                    if not chapter_match:
                        continue

                # 应用表格过滤
                if table_filter:
                    table_match = False
                    for table in result["structure"].get("tables", []):
                        if (table_filter.lower() in table.get("caption", "").lower() or
                                table_filter.lower() in table.get("id", "").lower()):
                            table_match = True
                            break
                    if not table_match:
                        continue

                # # 应用页面过滤
                # if page_filter:
                #     # 检查chunk所在的页面
                #     chunk_page = self._estimate_chunk_page(
                #         result["chunk_index"],
                #         result["file_hash"]
                #     )
                #     if chunk_page != page_filter:
                #         continue

                filtered_results.append(result)

            # 按分数排序并取前top_k个结果
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            results = filtered_results[:top_k]
        else:
            # 如果没有过滤，取前top_k个结果
            results = results[:top_k]

        return results

    def _estimate_chunk_page(self, chunk_index, file_hash):
        """估计chunk所在的页面"""
        # 简化的实现：假设每个页面有固定数量的chunk
        meta = self.metadata.get(file_hash, {})
        total_chunks = meta.get("chunk_count", 1)

        # 假设文档有10页，均匀分布
        estimated_page = int((chunk_index / total_chunks) * 10) + 1
        return min(estimated_page, 10)  # 确保不超过10页

    def list_documents(self):
        """列出知识库中的所有文档"""
        return [
            {
                "filename": meta["filename"],
                "upload_time": meta["upload_time"],
                "chunk_count": meta["chunk_count"]
            }
            for meta in self.metadata.values()
        ]

    def delete_document(self, filename):
        """从知识库中删除文档"""
        file_hash = None
        for hash_val, meta in self.metadata.items():
            if meta["filename"] == filename:
                file_hash = hash_val
                break

        if not file_hash:
            return {"status": "error", "message": "文件不存在"}

        # 删除文件
        os.remove(meta["file_path"])

        # 删除片段文件
        for chunk_path in meta["chunk_files"]:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

        # 删除嵌入文件
        if os.path.exists(meta["embedding_path"]):
            os.remove(meta["embedding_path"])

        # 重建索引
        all_embeddings = []
        for hash_val, meta in self.metadata.items():
            if hash_val == file_hash:
                continue

            embeddings = np.load(meta["embedding_path"])
            all_embeddings.append(embeddings)

        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            self.index = faiss.IndexFlatL2(all_embeddings.shape[1])
            self.index.add(all_embeddings)
            faiss.write_index(self.index, os.path.join(self.base_dir, "faiss_index"))
        else:
            # 如果没有文档，创建空索引
            self.index = faiss.IndexFlatL2(1024)
            faiss.write_index(self.index, os.path.join(self.base_dir, "faiss_index"))

        # 更新元数据
        del self.metadata[file_hash]
        self._save_metadata()

        return {"status": "success", "message": f"已删除文档: {filename}"}
