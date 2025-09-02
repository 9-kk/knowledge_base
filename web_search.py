import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import time

import requests
import json
from urllib.parse import urlparse
import os


class WebSearcher:
    def __init__(self, search_engine="bing", max_results=5):
        self.search_engine = search_engine
        self.max_results = max_results
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # self.headers = {
        #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        #     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        #     "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        #     "Accept-Encoding": "gzip, deflate, br",
        #     "Connection": "keep-alive",
        #     "Upgrade-Insecure-Requests": "1",
        #     "Cache-Control": "max-age=0"
        # }

    # def _is_relevant_result(self, query, title, snippet):
    #     """检查结果是否与查询相关"""
    #     # 基本质量检查
    #     if not self._is_quality_result(None, title, snippet):
    #         return False
    #
    #     # 计算查询词覆盖率
    #     query_words = set(query.lower().split())
    #     content = f"{title} {snippet}".lower()
    #     content_words = set(re.findall(r'\w+', content))
    #
    #     # 至少包含50%的查询词
    #     match_ratio = len(query_words & content_words) / len(query_words)
    #     return match_ratio >= 0.5

    def search(self, query):
        """执行网络搜索"""
        if self.search_engine == "bing":
            return self._search_bing(query)
        elif self.search_engine == "google":
            return self._search_google(query)
        else:
            return self._search_bing(query)  # 默认使用Bing

    def _search_bing(self, query):
        """使用Bing搜索"""
        try:
            search_url = f"https://www.bing.com/search?q={requests.utils.quote(query)}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            # 解析搜索结果
            for result in soup.select('li.b_algo'):
                title_elem = result.select_one('h2')
                link_elem = result.select_one('a')
                desc_elem = result.select_one('div.b_caption p')

                if title_elem and link_elem:
                    title = title_elem.get_text().strip()
                    url = link_elem.get('href')
                    domain = urlparse(url).netloc if url else ""

                    snippet = ""
                    if desc_elem:
                        snippet = desc_elem.get_text().strip()

                    # 过滤低质量结果
                    if not self._is_quality_result(url, title, snippet):
                        continue

                    results.append({
                        'title': title,
                        'url': url,
                        'domain': domain,
                        'snippet': snippet
                    })

                    if len(results) >= self.max_results:
                        break

            return results
        except Exception as e:
            print(f"网络搜索失败: {str(e)}")
            return []
        #         # 获取标题和链接
        #         title_elem = result.select_one('h2 a')
        #         if not title_elem:
        #             continue
        #
        #         title = title_elem.get_text().strip()
        #         url = title_elem.get('href')
        #
        #         # 获取摘要
        #         snippet_elem = result.select_one('.b_caption p')
        #         snippet = snippet_elem.get_text().strip() if snippet_elem else ""
        #
        #         # 获取域名
        #         domain = urlparse(url).netloc if url else ""
        #
        #         # 过滤结果
        #         if not self._is_relevant_result(query, title, snippet):
        #             continue
        #
        #         results.append({
        #             'title': title,
        #             'url': url,
        #             'domain': domain,
        #             'snippet': snippet
        #         })
        #
        #         if len(results) >= self.max_results:
        #             break
        #
        #     return results
        # except Exception as e:
        #     print(f"Bing搜索失败: {str(e)}")
        #     return []

    def _search_google(self, query):
        """使用Google搜索（需要API Key）"""
        # 实际实现需要Google Search API Key
        # 这里简化为返回空结果
        return []

    def _is_quality_result(self, url, title, snippet):
        """判断结果质量"""
        if not url or not title:
            return False

        # 过滤广告
        if "ad" in url or "ad" in title.lower():
            return False

        # 过滤社交媒体
        social_domains = ['facebook.com', 'twitter.com', 'instagram.com', 'youtube.com']
        if any(domain in url for domain in social_domains):
            return False

        # 过滤短内容
        if len(snippet) < 30:
            return False

        return True

    def get_page_content(self, url):
        """获取网页内容"""
        try:
            response = requests.get(url, headers=self.headers, timeout=20)
            response.raise_for_status()

            # 使用专业库提取主要内容
            try:
                # 方法1：使用newspaper3k（推荐）
                from newspaper import Article
                article = Article(url)
                article.download()
                article.parse()
                return article.text[:8000]  # 获取更多字符

            except ImportError:
                # 方法2：回退到BeautifulSoup（增强选择器）
                soup = BeautifulSoup(response.text, 'html.parser')

                # 尝试查找常见内容容器
                selectors = [
                    'article',
                    'div.article-content',
                    'div.post-content',
                    'div.content',
                    'div.main-content',
                    'div.entry-content',
                    'div[itemprop="articleBody"]',
                    'div#content',
                    'div#article-body',
                    'div#main-content'
                ]

                # 提取主要内容
                content = ""
                # print(0)

                for selector in selectors:
                    elements = soup.select(selector)
                    if elements:
                        content = "\n\n".join([e.get_text().strip() for e in elements])
                        if len(content) > 1000:
                            return re.sub(r'\s+', ' ', content)[:8000]

                # 如果未找到特定容器，使用整个正文
                if not content:
                    body = soup.find('body')
                    print(body)
                    if body:
                        content = body.get_text().strip()

                # 清理内容
                content = re.sub(r'\n\s*\n', '\n\n', content)  # 移除多余空行
                content = re.sub(r'\s{2,}', ' ', content)  # 移除多余空格

                # 截取前5000字符
                return content[:5000]
        except Exception as e:
            print(f"获取页面内容失败: {str(e)}")
            return ""

    # def get_page_content(self, url):
    #     """获取网页主要内容（改进版）"""
    #     try:
    #         response = requests.get(url, headers=self.headers, timeout=15)
    #         response.raise_for_status()
    #
    #         # 使用更智能的内容提取库
    #         try:
    #             from readability import Document
    #             doc = Document(response.text)
    #             content = doc.summary()
    #         except ImportError:
    #             # 回退到BeautifulSoup
    #             soup = BeautifulSoup(response.text, 'html.parser')
    #             # 移除不需要的元素
    #             for element in soup(['script', 'style', 'footer', 'nav', 'aside']):
    #                 element.decompose()
    #             content = soup.get_text()
    #
    #         # 清理内容
    #         content = re.sub(r'\s+', ' ', content)  # 合并空白字符
    #         content = re.sub(r'\n{3,}', '\n\n', content)  # 减少空行
    #
    #         # 截取前8000字符（约1500字）
    #         return content[:8000]
    #     except Exception as e:
    #         print(f"获取页面内容失败: {str(e)}")
    #         return ""


# class WebSearcher:
#     def __init__(self, max_results=5):
#         self.max_results = max_results
#         self.headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
#             "Accept": "application/json"
#         }
#         # 使用公开可用的SearXNG实例（开源元搜索引擎）
#         self.searxng_url = "https://searx.work"  # 可替换为其他公共实例
#
#         # 备选SearXNG实例列表
#         self.backup_instances = [
#             "https://search.us.projectsegfau.lt",
#             "https://searx.be",
#             "https://searx.nixnet.services"
#         ]
#
#     def search(self, query):
#         """执行网络搜索"""
#         try:
#             # 尝试主实例
#             results = self._search_searxng(query, self.searxng_url)
#             if results:
#                 return results
#
#             # 如果主实例失败，尝试备选实例
#             for instance in self.backup_instances:
#                 results = self._search_searxng(query, instance)
#                 if results:
#                     return results
#
#             return []
#         except Exception as e:
#             print(f"搜索失败: {str(e)}")
#             return []
#
#     def _search_searxng(self, query, base_url):
#         """使用SearXNG API搜索"""
#         try:
#             search_url = f"{base_url}/search"
#             params = {
#                 "q": query,
#                 "format": "json",
#                 "language": "zh-CN",
#                 "safesearch": 1,
#                 "pageno": 1
#             }
#
#             response = requests.get(
#                 search_url,
#                 params=params,
#                 headers=self.headers,
#                 timeout=15
#             )
#             response.raise_for_status()
#
#             data = response.json()
#             results = []
#
#             # 提取结果
#             for result in data.get('results', [])[:self.max_results]:
#                 # 跳过无效结果
#                 if not result.get('url') or not result.get('title'):
#                     continue
#
#                 # 添加到结果列表
#                 results.append({
#                     'title': result['title'],
#                     'url': result['url'],
#                     'snippet': result.get('content', '')[:200],
#                     'domain': urlparse(result['url']).netloc
#                 })
#
#             return results
#         except:
#             return []
#
#     def get_page_content(self, url):
#         """获取网页内容（简化版）"""
#         try:
#             response = requests.get(
#                 url,
#                 headers=self.headers,
#                 timeout=15,
#                 allow_redirects=True
#             )
#             response.raise_for_status()
#
#             # 只返回文本内容的前2000字符
#             return response.text[:2000]
#         except Exception as e:
#             print(f"获取页面内容失败: {str(e)}")
#             return ""

