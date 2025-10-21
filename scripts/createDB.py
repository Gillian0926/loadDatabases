import json
import requests
import time
import os
import glob
import re
import sqlite3
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue


class PoetryDatabase:
    def __init__(self, db_path="poetry_embedding.db"):
        self.db_path = db_path
        self._lock = threading.Lock()  # 添加线程锁
        self.init_database()

    def init_database(self):
        """初始化数据库表结构"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS Poems
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               author
                               TEXT
                               NOT
                               NULL,
                               title
                               TEXT
                               NOT
                               NULL,
                               content
                               TEXT
                               NOT
                               NULL,
                               dynasty
                               TEXT
                               NOT
                               NULL,
                               original_id
                               TEXT,
                               created_time
                               TIMESTAMP
                               DEFAULT
                               CURRENT_TIMESTAMP
                           )
                           ''')

            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS Tags
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               poem_id
                               INTEGER
                               NOT
                               NULL,
                               flower
                               TEXT,
                               emotion
                               TEXT,
                               tag
                               TEXT,
                               embedding
                               BLOB,
                               FOREIGN
                               KEY
                           (
                               poem_id
                           ) REFERENCES Poems
                           (
                               id
                           )
                               )
                           ''')

            conn.commit()
            conn.close()

    def insert_poem(self, author, title, content, dynasty, original_id=None):
        """插入诗歌数据 - 线程安全"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           INSERT INTO Poems (author, title, content, dynasty, original_id)
                           VALUES (?, ?, ?, ?, ?)
                           ''', (author, title, content, dynasty, original_id))

            poem_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return poem_id

    def insert_tag(self, poem_id, flower=None, emotion=None, tag=None, embedding=None):
        """插入标签数据 - 线程安全"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            embedding_blob = None
            if embedding is not None:
                embedding_blob = embedding.tobytes()

            cursor.execute('''
                           INSERT INTO Tags (poem_id, flower, emotion, tag, embedding)
                           VALUES (?, ?, ?, ?, ?)
                           ''', (poem_id, flower, emotion, tag, embedding_blob))

            conn.commit()
            conn.close()


class PoetryAnalyzer:
    def __init__(self, api_key, max_workers=3):
        self.api_key = api_key
        self.max_workers = max_workers  # 最大线程数
        self.total_tokens = 0
        self.processed_count = 0
        self.success_count = 0
        self._lock = threading.Lock()  # 计数器锁

        # 初始化数据库
        self.db = PoetryDatabase("poetry_embedding.db")

        # 加载标签库
        self.standard_tag_categories = self._build_standard_tag_categories()

        # 诗人年代数据库
        self.poet_timeline = self._build_poet_timeline()

    def analyze_poems_multithreaded(self, poems):
        """多线程分析诗歌"""
        print(f"开始多线程分析，使用 {self.max_workers} 个线程...")
        start_time = time.time()

        # 使用线程池
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_poem = {
                executor.submit(self._analyze_single_poem, poem): poem
                for poem in poems
            }

            # 使用进度条跟踪完成情况
            with tqdm(total=len(poems), desc="分析进度") as pbar:
                for future in as_completed(future_to_poem):
                    poem = future_to_poem[future]
                    try:
                        success = future.result()
                        if success:
                            with self._lock:
                                self.success_count += 1
                    except Exception as e:
                        print(f"分析诗歌《{poem['title']}》时出错: {e}")

                    pbar.update(1)
                    pbar.set_postfix({
                        '成功': self.success_count,
                        '总数': len(poems),
                        '成功率': f"{(self.success_count / len(poems)) * 100:.1f}%"
                    })

        elapsed_time = time.time() - start_time
        print(f"\n多线程分析完成!")
        print(f"成功分析: {self.success_count}/{len(poems)}")
        print(f"总用时: {elapsed_time / 60:.1f} 分钟")
        print(f"平均速度: {len(poems) / elapsed_time:.2f} 首/分钟")

    def _analyze_single_poem(self, poem_data):
        """分析单首诗词 - 线程安全版本"""
        print(f"线程 {threading.current_thread().name} 分析: {poem_data['title']} - {poem_data['author']}")

        prompt = self._build_analysis_prompt(poem_data['content'], poem_data['author'])

        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1500
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                cleaned_content = self._clean_response(content)
                parsed_result = self._parse_json_result(cleaned_content)

                if parsed_result:
                    standardized = self._standardize_result(parsed_result, poem_data['author'])

                    # 保存到数据库（线程安全）
                    self._save_to_database(poem_data, standardized)

                    with self._lock:
                        self.processed_count += 1

                    print(f"✓ {poem_data['title']} - 分析成功")
                    return True

            else:
                print(f"✗ {poem_data['title']} - API请求失败: {response.status_code}")

        except Exception as e:
            print(f"✗ {poem_data['title']} - 分析失败: {e}")

        return False

    def _build_analysis_prompt(self, content, author):
        """构建分析提示词"""
        poet_info = ""
        if author in self.poet_timeline:
            birth, death = self.poet_timeline[author]
            poet_info = f"（诗人{author}生卒年：{birth}-{death}年）"

        return f"""请严格按以下JSON格式输出分析结果：

{{
    "date": 年代,
    "date_rationale": "时间推断理由",
    "mentioned_flowers": ["花卉1", "花卉2"],
    "main_flower": "主要花卉",
    "emotion": "情感标签",
    "free_tags": ["标签1", "标签2", "标签3"]
}}

诗歌内容：
{content}
作者：{author}{poet_info}

请只输出JSON格式，不要有任何其他文字。"""

    def _save_to_database(self, poem_data, analysis):
        """保存分析结果到数据库"""
        dynasty = self._determine_dynasty(analysis['date'], poem_data['author'])

        # 插入诗歌数据
        poem_id = self.db.insert_poem(
            author=poem_data['author'],
            title=poem_data['title'],
            content=poem_data['content'],
            dynasty=dynasty,
            original_id=poem_data.get('id')
        )

        # 生成 embedding
        embedding = self._generate_simple_embedding(poem_data['content'])

        # 插入标签数据
        self.db.insert_tag(
            poem_id=poem_id,
            flower=analysis['main_flower'],
            emotion=analysis['emotion'],
            tag=','.join(analysis['free_tags']),
            embedding=embedding
        )

    # 以下方法保持不变（需要添加线程安全考虑）
    def _determine_dynasty(self, year, author):
        """根据年份确定朝代"""
        if 618 <= year <= 907:
            return "唐"
        elif 960 <= year <= 1279:
            return "宋"
        elif author in self.poet_timeline:
            birth, death = self.poet_timeline[author]
            if 618 <= birth <= 907:
                return "唐"
            elif 960 <= birth <= 1279:
                return "宋"
        return "未知"

    def _generate_simple_embedding(self, text):
        """生成简单的文本 embedding"""
        dimension = 384
        embedding = np.zeros(dimension, dtype=np.float32)

        text_hash = hashlib.md5(text.encode()).hexdigest()
        for i in range(min(dimension, 16)):
            embedding[i] = int(text_hash[i * 2:i * 2 + 2], 16) / 255.0

        return embedding

    def _standardize_result(self, result, author):
        """标准化分析结果"""
        mentioned_flowers = result.get('mentioned_flowers', [])
        if not isinstance(mentioned_flowers, list):
            mentioned_flowers = [mentioned_flowers]

        standardized_mentioned = [self._standardize_flower(flower) for flower in mentioned_flowers if flower]

        original_date = result.get('date', 0)
        date_rationale = result.get('date_rationale', '')
        validated_date = self._validate_date(original_date, author, date_rationale)

        return {
            "date": validated_date['date'],
            "original_date": original_date,
            "date_rationale": validated_date['rationale'],
            "date_correction": validated_date['correction'],
            "mentioned_flowers": standardized_mentioned,
            "main_flower": self._standardize_flower(result.get('main_flower', '无')),
            "emotion": result.get('emotion', '未知'),
            "free_tags": result.get('free_tags', []),
            "date_confidence": self._get_date_confidence(validated_date['date'], author)
        }

    def _validate_date(self, date_val, author, original_rationale):
        """验证和修正年代"""
        try:
            if isinstance(date_val, str) and '-' in date_val:
                year_match = re.search(r'(\d{3,4})-(\d{3,4})', str(date_val))
                if year_match:
                    start_year = int(year_match.group(1))
                    end_year = int(year_match.group(2))
                    date_int = (start_year + end_year) // 2
                else:
                    date_int = self._extract_year_from_string(str(date_val))
            else:
                date_int = self._extract_year_from_string(str(date_val))

            correction_info = ""
            corrected = False

            if author in self.poet_timeline:
                birth, death = self.poet_timeline[author]

                if date_int == 0 or date_int < birth - 20 or date_int > death + 20:
                    corrected_date = birth + (death - birth) // 2
                    if date_int == 0:
                        correction_info = f"无法解析时间，基于{author}生卒年推断为{corrected_date}年"
                    else:
                        correction_info = f"原推断{date_int}年，基于{author}生卒年修正为{corrected_date}年"
                    date_int = corrected_date
                    corrected = True

            if corrected:
                final_rationale = f"{original_rationale} | {correction_info}"
            else:
                final_rationale = original_rationale

            return {
                'date': date_int,
                'rationale': final_rationale,
                'correction': corrected
            }

        except Exception as e:
            return {
                'date': 0,
                'rationale': '时间解析失败',
                'correction': True
            }

    def _extract_year_from_string(self, date_str):
        """从字符串中提取年份"""
        try:
            clean_str = re.sub(r'[^\d-]', '', date_str)
            if clean_str and clean_str != '-':
                return int(clean_str)

            year_match = re.search(r'\b(\d{3,4})\b', date_str)
            if year_match:
                return int(year_match.group(1))

            return 0
        except:
            return 0

    def _get_date_confidence(self, date, author):
        """评估年代可信度"""
        if date == 0:
            return "低"

        if author in self.poet_timeline:
            birth, death = self.poet_timeline[author]
            if birth <= date <= death:
                return "高"
            elif birth - 10 <= date <= death + 10:
                return "中"
            else:
                return "低"
        else:
            return "中"

    def _standardize_flower(self, flower):
        """标准化花卉名称"""
        mapping = {
            '梅': '梅花', '菊': '菊花', '莲': '莲花', '荷': '莲花',
            '桃': '桃花', '杏': '杏花', '牡丹': '牡丹', '桂': '桂花',
            '无': '无', '未知': '无'
        }

        if flower in mapping:
            return mapping[flower]

        for key in mapping:
            if key in flower and key != '无':
                return mapping[key]

        return flower

    def _clean_response(self, content):
        """清理响应内容"""
        content = content.replace('```json', '').replace('```', '').strip()
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            if not line.strip().startswith('注：') and not line.strip().startswith('//'):
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def _parse_json_result(self, content):
        """解析JSON结果"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                content = content.replace("'", '"')
                content = re.sub(r',\s*}', '}', content)
                content = re.sub(r',\s*]', ']', content)
                return json.loads(content)
            except:
                return None
        except Exception:
            return None

    def _build_standard_tag_categories(self):
        """构建标准标签体系"""
        return {
            "emotion": ["喜悦", "忧伤", "思念", "孤独", "豪迈", "闲适", "愤懑", "恬淡"],
            "imagery_type": ["自然意象", "人物意象", "动物意象", "植物意象", "时空意象", "器物意象"],
            "season": ["春", "夏", "秋", "冬"],
        }

    def _build_poet_timeline(self):
        """构建诗人年代参考数据库"""
        return {
            "王昌龄": (698, 757), "李白": (701, 762), "杜甫": (712, 770),
            "白居易": (772, 846), "李商隐": (813, 858), "杜牧": (803, 852),
            "王维": (701, 761), "孟浩然": (689, 740), "刘禹锡": (772, 842),
            "苏轼": (1037, 1101), "李清照": (1084, 1155), "辛弃疾": (1140, 1207),
            "陆游": (1125, 1210), "王安石": (1021, 1086), "黄庭坚": (1045, 1105),
        }


# DataLoader 类保持不变
class DataLoader:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir

    def load_from_databases(self):
        """从数据库加载数据"""
        poems = []

        tang_db_path = os.path.join(self.data_dir, 'tang_poems_simple.db')
        if os.path.exists(tang_db_path):
            print("正在从 tang_poems_simple.db 加载数据...")
            poems.extend(self._load_from_sqlite(tang_db_path, 'poems'))

        ci_db_path = os.path.join(self.data_dir, 'ci.db')
        if os.path.exists(ci_db_path):
            print("正在从 ci.db 加载数据...")
            poems.extend(self._load_from_sqlite(ci_db_path, 'poems'))

        print(f"总共加载 {len(poems)} 首诗歌")
        return poems

    def _load_from_sqlite(self, db_path, table_name):
        """从 SQLite 数据库加载诗歌数据"""
        poems = []
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()

            for row in rows:
                poem_data = {}
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]

                for i, col_name in enumerate(columns):
                    poem_data[col_name] = row[i]

                poem = self._convert_to_standard_format(poem_data)
                if poem:
                    poems.append(poem)

            conn.close()
            print(f"从 {os.path.basename(db_path)} 加载 {len(poems)} 首诗歌")

        except Exception as e:
            print(f"从 {db_path} 加载数据失败: {e}")

        return poems

    def _convert_to_standard_format(self, poem_data):
        """转换为标准格式"""
        content = ""
        if 'content' in poem_data:
            content = poem_data['content']
        elif 'paragraphs' in poem_data:
            if isinstance(poem_data['paragraphs'], str):
                content = poem_data['paragraphs']
            else:
                content = ' '.join(poem_data['paragraphs'])

        if not content or len(content.strip()) < 10:
            return None

        return {
            'id': poem_data.get('id'),
            'title': poem_data.get('title', '无题'),
            'author': poem_data.get('author', '未知'),
            'content': content,
            'source_file': 'database'
        }


def test_api_connection(api_key):
    """测试API连接"""
    print("测试API连接...")
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "测试"}],
                "temperature": 0.1,
                "max_tokens": 50
            },
            timeout=30
        )
        return response.status_code == 200
    except:
        return False


def main():
    """主函数"""
    API_KEY = "sk-5176c49ed09c4af98b5ee37598df4f91"

    if not test_api_connection(API_KEY):
        print("API连接失败，请检查网络和API密钥")
        return

    # 设置线程数（根据API限制调整）
    max_workers = 3  # 建议3-5个线程，避免触发API限制

    analyzer = PoetryAnalyzer(API_KEY, max_workers=max_workers)
    data_loader = DataLoader('.')

    print("正在从数据库加载诗歌数据...")
    poems = data_loader.load_from_databases()

    if not poems:
        print("没有找到诗歌数据")
        return

    print(f"\n开始多线程诗词分析...")
    print(f"线程数: {max_workers}")
    print(f"目标分析数量: {len(poems)} 首")

    # 使用多线程分析
    analyzer.analyze_poems_multithreaded(poems)

    print(f"\n最终结果:")
    print(f"成功分析: {analyzer.success_count}/{len(poems)} 首诗歌")
    print(f"数据库文件: poetry_embedding.db")


if __name__ == "__main__":
    main()