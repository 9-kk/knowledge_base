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
            outputs, target_sizes=target_sizes, threshold=0.8
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
                print(f"无效的bbox: {bbox}")
                return []

            table_img = image[y_min:y_max, x_min:x_max]
            print('table_img', table_img.shape)

            # 识别表格结构
            structure = self.recognize_table_structure(table_img)
            print('structure', structure)

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
                    if cell_img.size == 0:
                        row_data.append("")
                        continue

                    # 对单元格进行OCR
                    result = self.ocr.ocr(cell_img, cls=True)
                    print('result', result)

                    # 提取文本内容
                    cell_text = ""
                    for line in result:
                        if line and line[0]:
                            for word_info in line:
                                cell_text += word_info[1][0] + " "
                    print('cell_text', cell_text)

                    row_data.append(cell_text.strip())

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
        try:
            # 方法1：使用深度学习模型（如果有）
            return self._recognize_with_deep_learning(table_image)
        except:
            # 方法2：回退到传统方法
            return self._recognize_with_traditional(table_image)

    def _recognize_with_deep_learning(self, table_image):
        """使用深度学习识别表格结构"""
        # 这里可以替换为实际模型，如TableNet
        # 暂时使用传统方法
        return self._recognize_with_traditional(table_image)

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
        self.metadata_file = os.path.join(base_dir, "metadata.json")

        # 创建目录
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)

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

        # 加载元数据
        self.metadata = self._load_metadata()

        # 初始化FAISS索引
        self.index = self._init_faiss_index()

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
                return self._process_image_pdf(file_path, file_hash, filename, dest_path)

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

        # 12. 保存索引
        faiss.write_index(self.index, os.path.join(self.base_dir, "faiss_index"))

        # 13. 更新元数据
        self.metadata[file_hash] = {
            "filename": filename,
            "file_path": dest_path,
            "chunk_files": chunk_files,
            "embedding_path": embedding_path,
            "chunk_count": len(chunks),
            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_metadata()

        return {"status": "success", "message": f"文件已成功添加到知识库: {filename}"}

    def _process_image_pdf(self, file_path, file_hash, filename, dest_path):
        """处理图片型PDF（使用OCR提取文本）"""
        try:
            print(f"检测到图片型PDF，使用OCR处理: {filename}")

            # OCR提取文本
            ocr_text = self._ocr_pdf(file_path)
            print('ocr_text', ocr_text)
            if not ocr_text:
                return {"status": "error", "message": "OCR提取文本失败"}

            # 创建文档对象
            from langchain_core.documents import Document
            document = [Document(page_content=ocr_text)]

            # 分割文本
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
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

            # 更新元数据
            self.metadata[file_hash] = {
                "filename": filename,
                "file_path": dest_path,
                "chunk_files": chunk_files,
                "embedding_path": embedding_path,
                "chunk_count": len(chunks),
                "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "is_ocr": True  # 标记为OCR处理
            }
            self._save_metadata()

            return {"status": "success", "message": f"图片型PDF已通过OCR添加到知识库: {filename}"}
        except Exception as e:
            return {"status": "error", "message": f"OCR处理失败: {str(e)}"}

    def _ocr_pdf(self, pdf_path):
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
            for page_num, image in enumerate(images):
                img_np = np.array(image)

                # 1. 提取页面所有文本（包含位置信息）
                text_blocks = self._extract_text_with_position(img_np)
                # print('text_blocks', text_blocks)

                # 2. 检测表格
                tables = self.table_processor.detect_tables(img_np)
                print('tables', tables)

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
                for table_idx, table_bbox in enumerate(tables):
                    table_id = f"page_{page_num + 1}_table_{table_idx + 1}"
                    print('table_id', table_id)

                    # 查找表格上方的表名
                    caption = self._find_table_caption(text_blocks, table_bbox)
                    print('caption', caption)

                    # 添加表名
                    if caption:
                        all_elements.append({
                            "type": "text",
                            "y": caption["y"],
                            "content": caption["text"],
                            "is_table_caption": True
                        })

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
                        print('table_data', table_data)
                        table_md = self.table_processor.convert_to_markdown(table_data)
                        print('table_md', table_md)

                        # 添加表格
                        page_content += f"\n{table_md}\n"

                full_content += page_content + "\n\n"

                # # 使用中文+英文识别
                # # 去掉多余的空格
                # text = pytesseract.image_to_string(image, lang='chi_sim+eng').replace(' ', '')
                # # 两个或以上换行符换成一个换行符，只有一个换行符删掉
                # text = re.sub(r'\n', ' ', text)
                # text = re.sub(r'\n\s*\n', '\n', text)
                # full_content += f"第{i + 1}页:\n{text}\n\n"
            print('full_content', full_content)

            return full_content
        except ImportError:
            print("OCR依赖未安装，请运行: pip install pdf2image pytesseract")
            return None
        except Exception as e:
            import traceback
            print(f"OCR处理失败: {str(e)}")
            print(traceback.format_exc())  # 打印堆栈跟踪
            return None

    def _extract_text_with_position(self, image):
        """提取文本及其位置信息"""
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        result = ocr.ocr(image)
        print(result)

        text_blocks = []
        for lines in result:
            for line in lines:
                if line and line[0]:
                    # 计算文本块的中心y坐标
                    # 文本框坐标
                    points = line[0]
                    text_info = line[1]  # 文本信息和置信度
                    y_coords = [point[1] for point in points]
                    y_center = sum(y_coords) / len(y_coords)

                    # 提取文本
                    text = line[1][0] if line[1] else ""

                    text_blocks.append({
                        "text": text,
                        "y": y_center,
                        "bbox": points
                    })

        return text_blocks

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

    def search(self, query, top_k=5):
        """在知识库中搜索相关内容"""
        # 生成查询嵌入
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding_np = np.array([query_embedding]).astype("float32")

        # 搜索索引
        distances, indices = self.index.search(query_embedding_np, top_k)

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
                    chunk_idx = idx - start_idx
                    chunk_path = meta["chunk_files"][chunk_idx]

                    with open(chunk_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    results.append({
                        "source": f"知识库: {meta['filename']}",
                        "content": content,
                        "score": float(1 - distances[0][i]),  # 转换为相似度分数
                        "chunk_index": chunk_idx
                    })
                    break

                current_idx = end_idx

        return results

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
