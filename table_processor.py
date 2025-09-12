import os
import torch

from paddlex import create_pipeline


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
        self.table_pipeline = create_pipeline(pipeline="table_recognition_v2")
        # print(self.detection_model.device)  # 应显示实际设备（如cuda:0）
        # print(self.detection_model.config)  # 应显示模型配置

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
            # print('table_img', type(table_img), table_img.shape, table_img.shape[2])
            # 识别并返回表格的markdown结构
            return self.recognize_table_md(table_img)

        except Exception as e:
            import traceback
            print(f"表格提取失败: {str(e)}")
            print(traceback.format_exc())  # 打印堆栈跟踪

    def recognize_table_md(self, table_image):
        """识别表格结构（使用深度学习方法）"""
        # 使用更先进的表格结构识别方法
        try:
            # 方法1：使用深度学习模型（如果有）
            return self._recognize_with_deep_learning(table_image)
        except Exception as e:
            print(f"深度学习模型提取表格失败: {str(e)}")
            # 方法2：回退到传统方法
            return self._recognize_with_traditional(table_image)

    def _recognize_with_deep_learning(self, table_image):
        """使用深度学习识别表格结构"""
        """处理页面图像"""
        # 使用 PP-TableMagic 进行端到端的表格识别
        # input 可以是图像路径或 numpy 数组（HWC, BGR格式）
        results = self.table_pipeline.predict(
            input=table_image,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )  # 默认会自动进行方向校正和扭曲矫正，如需关闭可传参 use_doc_orientation_classify=False, use_doc_unwarping=False

        # page_content = f"第{page_num + 1}页:\n"
        page_content = ''

        # 假设我们主要处理第一个识别结果（对于单页单表的情况）
        if results:
            import html2text
            results = list(results)

            # 获取结果对象
            # print('---------------------------')
            # print('table_results', results)

            table_data = results[0]

            # 如果存在html数据，通过html提取表格数据
            if table_data['table_res_list']:
                # 提取 HTML 表格并修复tdcolspan语法问题，以避免后续转换出错
                html_table = table_data['table_res_list'][0]['pred_html'].replace(' ', '').replace('tdcolspan', 'td colspan')
                # print('html', html_table)

                # 创建HTML到Markdown转换器
                h = html2text.HTML2Text()
                h.ignore_links = True
                h.ignore_images = True
                h.body_width = 0  # 不换行

                # 转换HTML表格为Markdown
                markdown_table = h.handle(html_table)

                lines = markdown_table.split('\n')
                for i, line in enumerate(lines):
                    lines[i] = f"|{line}|"
                markdown_table = '\n'.join(lines)

            # 如果没有，用rec_texts文本拼接
            else:
                # 通过rec_texts提取表格数据
                texts = table_data['overall_ocr_res']['rec_texts']
                boxes = table_data['overall_ocr_res']['rec_boxes']
                # print('rec_texts', texts)

                """
                先合并再排序：
                1、先判断有没有识别成两行的位置但其实在同一个单元格的——满足x1差距不大或后者x_center在前者x1x2之间，
                且y_center之间的距离减去行高的值小于行高
                2、合并后计算两者的平均y_center写入rows中
                3、按照y坐标排序行（不能按照四舍五入到10像素来排序，按照y的差值不超过行高的1/2）
                """
                # 计算每个文本的中心坐标和高度
                centers = []
                for i, (text, box) in enumerate(zip(texts, boxes)):
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    height = box[3] - box[1]
                    width = box[2] - box[0]
                    centers.append((x_center, y_center, height, width, text, box, i))  # 添加索引以便跟踪

                # 主合并过程
                merged_items = []
                original_items = []  # 保存原始文本和边界框，用于可能的回退
                processed_indices = set()

                for i in range(len(centers)):
                    if i in processed_indices:
                        continue

                    # 查找和合并属于同一单元格的所有文本
                    cell_texts, cell_boxes, processed_indices = self.find_and_merge_cell_texts(i, processed_indices, centers)

                    # 保存原始文本和边界框
                    original_items.append((cell_texts, cell_boxes))

                    # 合并文本和边界框
                    if cell_texts:
                        # 按Y坐标排序文本，确保正确的阅读顺序
                        text_box_pairs = list(zip(cell_texts, cell_boxes))
                        text_box_pairs.sort(key=lambda x: (x[1][1], x[1][0]))  # 先按Y坐标排序，再按X坐标排序

                        sorted_texts = [pair[0] for pair in text_box_pairs]
                        sorted_boxes = [pair[1] for pair in text_box_pairs]

                        merged_text = "".join(sorted_texts)
                        merged_box = [
                            min(box[0] for box in sorted_boxes),
                            min(box[1] for box in sorted_boxes),
                            max(box[2] for box in sorted_boxes),
                            max(box[3] for box in sorted_boxes)
                        ]

                        # 计算合并后的中心坐标
                        merged_y_center = (merged_box[1] + merged_box[3]) / 2
                        merged_height = merged_box[3] - merged_box[1]

                        merged_items.append((merged_text, merged_box, merged_y_center, merged_height, len(cell_texts)))

                # 修复误将上下两个单元格内容合并的问题
                final_items = self.find_and_repair_false_merge(merged_items, original_items)

                # 更新文本和边界框
                new_texts = [item[0] for item in final_items]
                new_boxes = [item[1] for item in final_items]

                # print('合并后的文本:', new_texts)

                # 重新计算中心坐标和高度
                centers = []
                for box in new_boxes:
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    height = box[3] - box[1]
                    centers.append((x_center, y_center, height))

                # 按Y坐标分组行
                new_rows = {}
                for i, (text, (x_center, y_center, height)) in enumerate(zip(new_texts, centers)):
                    # 查找是否已经存在相似Y坐标的行
                    matched_key = None
                    for key in new_rows.keys():
                        # 如果Y坐标差值不超过行高的1/2，认为是同一行
                        if abs(y_center - key) <= height / 2:
                            matched_key = key
                            break

                    if matched_key is not None:
                        # 添加到现有行
                        new_rows[matched_key].append((x_center, text))
                    else:
                        # 创建新行
                        new_rows[y_center] = [(x_center, text)]

                # 按Y坐标排序行
                sorted_rows = sorted(new_rows.items(), key=lambda x: x[0])

                # 对每行内的文本按X坐标排序
                for i, (y_center, cells) in enumerate(sorted_rows):
                    sorted_cells = sorted(cells, key=lambda x: x[0])
                    sorted_rows[i] = (y_center, sorted_cells)

                # print('最终的行结构:', sorted_rows)

                markdown_table = self.sorted_rows_to_md(sorted_rows)

            # print('markdown_table', markdown_table)

            return markdown_table

        else:
            raise

    # 递归函数，用于查找和合并属于同一单元格的所有文本
    def find_and_merge_cell_texts(self, start_idx, processed_indices, centers):
        """
        递归查找和合并属于同一单元格的所有文本
        """
        if start_idx in processed_indices:
            return [], [], processed_indices

        x_center_i, y_center_i, height_i, width_i, text_i, box_i, idx_i = centers[start_idx]
        merged_texts = [text_i]
        merged_boxes = [box_i]
        processed_indices.add(start_idx)

        # 查找可能与当前文本属于同一单元格的其他文本
        for j, (x_center_j, y_center_j, height_j, width_j, text_j, box_j, idx_j) in enumerate(centers):
            if j == start_idx or j in processed_indices:
                continue

            # 检查是否可能属于同一单元格的条件：
            # 1. X坐标相近（在同一个列中）
            x_diff = abs(x_center_i - x_center_j)
            # 检查X方向是否有重叠:右边界完全在左边或左边界完全在右边
            x_overlap = not (box_i[2] < box_j[0] or box_j[2] < box_i[0])

            # 2. Y坐标相差不大（在同一行或相邻行）
            y_diff = abs(y_center_i - y_center_j)

            # 如果满足条件，可能是同一单元格的多行文本
            if x_overlap and y_diff < (height_i + height_j):
                # 递归查找和合并
                sub_texts, sub_boxes, processed_indices = self.find_and_merge_cell_texts(j, processed_indices,
                                                                                    centers)
                merged_texts.extend(sub_texts)
                merged_boxes.extend(sub_boxes)

        return merged_texts, merged_boxes, processed_indices

    def find_and_repair_false_merge(self, merged_items, original_items):
        """
        由于有可能存在明明确实是上下两个单元格的内容，但是检测到的文本框高度较大，从而误识别为同一个单元格而合并，增加一个查漏补缺操作：
        理论上，不管是两行三行还是多行文本的单元格，合并后重新计算y_center时是整一个单元格的中点，都应该和该行其他单元格的y_center差不多，
        但如果是误把上下两个单元格的合并了，y_center就会和别的单元格不一样，因此增加一个判断：
        如果是合并后的y_center没有任何一个其他单元格的y_center与其差不多，则去掉该合并数据，采用原数据
        """
        final_items = []

        for i, (merged_text, merged_box, merged_y_center, merged_height, num_merged) in enumerate(merged_items):
            # 检查是否有其他单元格的y_center与当前合并后的y_center相近
            has_similar_y = False

            for j, (_, _, y_center_j, height_j, _) in enumerate(merged_items):
                if i == j:
                    continue

                # 如果y_center差值不超过较小高度的1/2，认为是相近的
                if abs(merged_y_center - y_center_j) <= min(merged_height, height_j) / 3:
                    has_similar_y = True
                    break

            # 如果没有找到相近的y_center，可能是误合并
            if not has_similar_y and num_merged > 1:
                print(f"检测到可能的误合并: {merged_text}")

                # 使用原始文本而不是合并后的文本
                original_texts, original_boxes = original_items[i]

                for text, box in zip(original_texts, original_boxes):
                    y_center = (box[1] + box[3]) / 2
                    height = box[3] - box[1]
                    final_items.append((text, box, y_center, height, 1))
            else:
                # 使用合并后的文本
                final_items.append((merged_text, merged_box, merged_y_center, merged_height, num_merged))

        return final_items

    def sorted_rows_to_md(self, sorted_rows):

        # 创建表格
        markdown_table = []

        # 添加表头（假设前两个文本是表头）
        if len(sorted_rows) > 0 and len(sorted_rows[0][1]) >= 2:
            # 按x坐标排序第一行的文本
            sorted_first_row = sorted(sorted_rows[0][1], key=lambda x: x[0])
            header = [text for _, text in sorted_first_row]
            markdown_table.append("| " + " | ".join(header) + " |")
            markdown_table.append("|" + "|".join(["---"] * len(header)) + "|")

            # 处理剩余行
            for y, row_texts in sorted_rows[1:]:
                # 按x坐标排序文本
                sorted_row = sorted(row_texts, key=lambda x: x[0])
                row_cells = [text for _, text in sorted_row]

                # 确保行中的单元格数量与表头一致
                if len(row_cells) < len(header):
                    # 如果单元格数量不足，用空字符串填充
                    row_cells.extend([""] * (len(header) - len(row_cells)))
                elif len(row_cells) > len(header):
                    # 如果单元格数量过多，合并多余的单元格
                    merged_text = " ".join(row_cells[len(header) - 1:])
                    row_cells = row_cells[:len(header) - 1] + [merged_text]

                markdown_table.append("| " + " | ".join(row_cells) + " |")

        markdown_table = "\n".join(markdown_table)

        return markdown_table

    def _recognize_with_traditional(self, table_img):

        # 用传统方式提取表格结构
        structure = self._traditional_get_table_content(table_img)

        try:
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
                    results = self.ocr.ocr(cell_img)
                    result = results[0]
                    # print('result', result)

                    # 提取文本内容
                    cell_text = ""
                    if result and 'rec_texts' in result:
                        for text in result['rec_texts']:
                            cell_text += text + " "

                    row_data.append(cell_text.strip().replace(' ', '').replace('\n', ''))

                table_data.append(row_data)
            # print('table_data', table_data)

            # 转为markdown
            table_md = self.convert_to_markdown(table_data)

            return table_md

        except Exception as e:
            import traceback
            print(f"表格提取失败: {str(e)}")
            print(traceback.format_exc())  # 打印堆栈跟踪

    def _traditional_get_table_content(self, table_image):

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
