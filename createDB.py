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


class PoetryDatabase:
    def __init__(self, db_path="poetry_embedding.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
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
                               INTEGER
                               NOT
                               NULL, -- 改为整数类型，存储具体年份
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
        """插入诗歌数据 - dynasty 现在存储具体年份"""
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
        """插入标签数据"""
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


class DataLoader:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir

    def load_from_databases(self):
        """从数据库加载数据"""
        poems = []

        # 数据库文件路径
        db_files = [
            os.path.join(self.data_dir, 'tang_poems_simple.db'),
            os.path.join(self.data_dir, 'ci.db')
        ]

        for db_path in db_files:
            if os.path.exists(db_path):
                print(f"正在检查数据库: {os.path.basename(db_path)}")
                poems.extend(self._explore_database(db_path))

        print(f"总共加载 {len(poems)} 首诗歌")
        return poems

    def _explore_database(self, db_path):
        """探索数据库结构并加载数据"""
        poems = []
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"数据库 {os.path.basename(db_path)} 中的表: {[table[0] for table in tables]}")

            for table in tables:
                table_name = table[0]
                print(f"  检查表: {table_name}")

                # 获取表结构
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                print(f"    表结构: {columns}")

                # 尝试读取前几行数据
                try:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                    sample_rows = cursor.fetchall()
                    print(f"    样本数据:")
                    for i, row in enumerate(sample_rows):
                        print(f"      行{i + 1}: {dict(zip(columns, row))}")

                    # 加载所有数据
                    cursor.execute(f"SELECT * FROM {table_name}")
                    all_rows = cursor.fetchall()

                    table_poems = []
                    for row in all_rows:
                        poem_data = dict(zip(columns, row))
                        poem = self._convert_to_standard_format(poem_data, table_name)
                        if poem:
                            table_poems.append(poem)

                    print(f"    从表 {table_name} 加载 {len(table_poems)} 首诗歌")
                    poems.extend(table_poems)

                except Exception as e:
                    print(f"    读取表 {table_name} 失败: {e}")
                    continue

            conn.close()

        except Exception as e:
            print(f"探索数据库 {db_path} 失败: {e}")

        return poems

    def _convert_to_standard_format(self, poem_data, table_name):
        """转换为标准格式"""
        content = ""

        # 尝试不同的内容字段
        content_fields = ['content', 'paragraphs', 'body', 'text', 'poem']
        for field in content_fields:
            if field in poem_data and poem_data[field]:
                if field == 'paragraphs' and isinstance(poem_data[field], list):
                    content = ' '.join(poem_data[field])
                else:
                    content = str(poem_data[field])
                break

        # 如果还没有内容，尝试所有字段
        if not content:
            for key, value in poem_data.items():
                if isinstance(value, str) and len(value) > 20 and 'id' not in key.lower():
                    content = value
                    break

        if not content or len(content.strip()) < 10:
            return None

        # 获取作者和标题
        author = poem_data.get('author', '未知')
        title = poem_data.get('title', '无题')

        # 如果作者还是未知，尝试从其他字段获取
        if author == '未知':
            for key in ['writer', 'poet', 'name']:
                if key in poem_data:
                    author = poem_data[key]
                    break

        print(f"      转换: 《{title}》 - {author} (内容长度: {len(content)})")

        return {
            'id': poem_data.get('id'),
            'title': title,
            'author': author,
            'content': content,
            'source_file': table_name
        }


class PoetryAnalyzer:
    def __init__(self, api_key, max_workers=3):
        self.api_key = api_key
        self.max_workers = max_workers
        self.total_tokens = 0
        self.processed_count = 0
        self.success_count = 0
        self._lock = threading.Lock()

        self.db = PoetryDatabase("poetry_embedding.db")
        self.standard_tag_categories = self._build_standard_tag_categories()
        self.poet_timeline = self._build_poet_timeline()

    def _build_poet_timeline(self):
        """构建诗人年代参考数据库"""
        return {
            # 唐代诗人 (618-907)
            "李白": (701, 762), "杜甫": (712, 770), "白居易": (772, 846),
            "王维": (701, 761), "孟浩然": (689, 740), "李商隐": (813, 858),
            "杜牧": (803, 852), "王昌龄": (698, 757), "刘禹锡": (772, 842),
            "柳宗元": (773, 819), "韩愈": (768, 824), "岑参": (715, 770),
            "高适": (704, 765), "韦应物": (737, 792), "元稹": (779, 831),

            # 宋代诗人 (960-1279)
            "苏轼": (1037, 1101), "李清照": (1084, 1155), "辛弃疾": (1140, 1207),
            "陆游": (1125, 1210), "王安石": (1021, 1086), "黄庭坚": (1045, 1105),
            "柳永": (984, 1053), "晏殊": (991, 1055), "欧阳修": (1007, 1072),
            "范仲淹": (989, 1052), "杨万里": (1127, 1206), "范成大": (1126, 1193),
        }

    def analyze_poems_multithreaded(self, poems):
        """多线程分析诗歌"""
        if not poems:
            print("没有诗歌数据可分析")
            return

        print(f"开始多线程分析，使用 {self.max_workers} 个线程...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_poem = {
                executor.submit(self._analyze_single_poem, poem): poem
                for poem in poems
            }

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
                        '成功率': f"{(self.success_count / len(poems)) * 100:.1f}%"
                    })

        elapsed_time = time.time() - start_time
        print(f"\n分析完成! 成功: {self.success_count}/{len(poems)}")
        print(f"总用时: {elapsed_time / 60:.1f} 分钟")

    def _analyze_single_poem(self, poem_data):
        """分析单首诗词"""
        print(f"\n分析: 《{poem_data['title']}》 - {poem_data['author']}")

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
                print(f"API响应: {content}")

                cleaned_content = self._clean_response(content)
                parsed_result = self._parse_json_result(cleaned_content)

                if parsed_result:
                    print(f"解析结果: {parsed_result}")
                    standardized = self._standardize_result(parsed_result, poem_data['author'])

                    # 保存到数据库 - 现在 dynasty 字段存储具体年份
                    self._save_to_database(poem_data, standardized)

                    with self._lock:
                        self.processed_count += 1

                    print(f"✓ 《{poem_data['title']}》 - 分析成功")
                    return True
                else:
                    print(f"✗ 《{poem_data['title']}》 - JSON解析失败")

            else:
                print(f"✗ 《{poem_data['title']}》 - API请求失败: {response.status_code}")

        except Exception as e:
            print(f"✗ 《{poem_data['title']}》 - 分析失败: {e}")

        return False

    def _build_analysis_prompt(self, content, author):
        """构建分析提示词 - 强调具体年份"""
        poet_info = ""
        if author in self.poet_timeline:
            birth, death = self.poet_timeline[author]
            poet_info = f"（诗人{author}生卒年：{birth}-{death}年）"

        return f"""请分析这首诗歌并返回JSON格式结果：

{{
    "date": 具体创作年份,
    "date_rationale": "时间推断理由",
    "mentioned_flowers": ["花卉1", "花卉2"],
    "main_flower": "主要花卉", 
    "emotion": "情感标签",
    "free_tags": ["标签1", "标签2", "标签3"]
}}

诗歌内容：{content}
作者：{author}{poet_info}

重要要求：
1. date字段必须是具体的公元年份数字，如：755、1080、1205，不要使用朝代名称
2. 基于诗歌内容、作者生平、历史背景等具体信息推断创作年份
3. 如果无法确定确切年份，请给出最可能的年份并说明理由
4. 情感标签请从：喜悦、忧伤、思念、孤独、豪迈、闲适、愤懑、恬淡 中选择
5. 请确保输出是纯JSON格式，不要有其他文字
"""

    def _save_to_database(self, poem_data, analysis):
        """保存分析结果到数据库 - dynasty 字段现在存储具体年份"""
        # 直接使用分析得到的具体年份作为 dynasty 字段的值
        specific_year = analysis['date']
        print(f"保存到数据库: 具体年份 {specific_year}")

        # 插入诗歌数据 - dynasty 字段存储具体年份
        poem_id = self.db.insert_poem(
            author=poem_data['author'],
            title=poem_data['title'],
            content=poem_data['content'],
            dynasty=specific_year,  # 直接存储具体年份
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

    def _standardize_result(self, result, author):
        """标准化分析结果 - 专注于年份处理"""
        mentioned_flowers = result.get('mentioned_flowers', [])
        if not isinstance(mentioned_flowers, list):
            mentioned_flowers = [mentioned_flowers]

        standardized_mentioned = [self._standardize_flower(flower) for flower in mentioned_flowers if flower]

        # 专注于提取和验证具体年份
        original_date = result.get('date', 0)
        date_rationale = result.get('date_rationale', 'API分析结果')

        # 详细处理年份提取
        validated_date = self._extract_and_validate_year(original_date, author)

        return {
            "date": validated_date['year'],  # 这是最终的具体年份
            "original_date": original_date,
            "date_rationale": validated_date['rationale'],
            "date_correction": validated_date['corrected'],
            "mentioned_flowers": standardized_mentioned,
            "main_flower": self._standardize_flower(result.get('main_flower', '无')),
            "emotion": result.get('emotion', '未知'),
            "free_tags": result.get('free_tags', []),
            "date_confidence": validated_date['confidence']
        }

    def _extract_and_validate_year(self, date_value, author):
        """提取和验证具体年份"""
        original_rationale = "API分析结果"

        try:
            # 处理字符串类型的年份
            if isinstance(date_value, str):
                # 多种模式匹配年份
                patterns = [
                    r'\b(\d{3,4})\b',  # 纯数字
                    r'约?(\d{3,4})年',  # 带"年"字
                    r'公元(\d{3,4})年',  # 带"公元"
                ]

                for pattern in patterns:
                    match = re.search(pattern, str(date_value))
                    if match:
                        extracted_year = int(match.group(1))
                        break
                else:
                    # 如果没有匹配到，尝试直接转换
                    try:
                        extracted_year = int(date_value)
                    except:
                        extracted_year = 0
            else:
                # 直接处理数字
                extracted_year = int(date_value)

            # 验证年份合理性
            validation = self._validate_year_plausibility(extracted_year, author)

            return {
                'year': validation['year'],
                'rationale': f"{original_rationale} | {validation['rationale']}",
                'corrected': validation['corrected'],
                'confidence': validation['confidence']
            }

        except Exception as e:
            print(f"年份提取失败: {e}")
            # 使用作者信息推断年份
            inferred_year = self._infer_year_from_author(author)
            return {
                'year': inferred_year,
                'rationale': f"年份解析失败，基于作者信息推断",
                'corrected': True,
                'confidence': "低"
            }

    def _validate_year_plausibility(self, year, author):
        """验证年份的合理性"""
        if year == 0:
            inferred_year = self._infer_year_from_author(author)
            return {
                'year': inferred_year,
                'rationale': "年份无效，基于作者生卒年推断",
                'corrected': True,
                'confidence': "低"
            }

        # 检查年份是否在合理范围内
        if year < 500 or year > 2000:
            inferred_year = self._infer_year_from_author(author)
            return {
                'year': inferred_year,
                'rationale': f"原年份{year}超出合理范围，基于作者信息修正",
                'corrected': True,
                'confidence': "中"
            }

        # 如果作者在时间线中，验证年份是否在生卒年范围内
        if author in self.poet_timeline:
            birth, death = self.poet_timeline[author]
            if birth <= year <= death:
                return {
                    'year': year,
                    'rationale': f"年份在作者生卒年范围内",
                    'corrected': False,
                    'confidence': "高"
                }
            elif birth - 10 <= year <= death + 10:
                return {
                    'year': year,
                    'rationale': f"年份接近作者生卒年范围",
                    'corrected': False,
                    'confidence': "中"
                }
            else:
                # 年份明显不合理，进行修正
                corrected_year = birth + (death - birth) // 2
                return {
                    'year': corrected_year,
                    'rationale': f"原年份{year}与作者生卒年不符，修正为作者主要活动时期",
                    'corrected': True,
                    'confidence': "中"
                }

        # 作者不在时间线中，但年份看起来合理
        return {
            'year': year,
            'rationale': "年份在合理范围内",
            'corrected': False,
            'confidence': "中"
        }

    def _infer_year_from_author(self, author):
        """根据作者信息推断年份"""
        if author in self.poet_timeline:
            birth, death = self.poet_timeline[author]
            # 返回作者的主要活动时期（生年+30岁左右）
            return birth + 30

        # 默认返回一个合理的古代年份
        return 800  # 唐代中期

    def _generate_simple_embedding(self, text):
        """生成简单的文本 embedding"""
        dimension = 384
        embedding = np.zeros(dimension, dtype=np.float32)

        text_hash = hashlib.md5(text.encode()).hexdigest()
        for i in range(min(dimension, 16)):
            embedding[i] = int(text_hash[i * 2:i * 2 + 2], 16) / 255.0

        return embedding

    def _standardize_flower(self, flower):
        """标准化花卉名称"""
        mapping = {
            '梅': '梅花', '菊': '菊花', '莲': '莲花', '荷': '莲花',
            '桃': '桃花', '杏': '杏花', '牡丹': '牡丹', '桂': '桂花',
            '无': '无', '未知': '无', '': '无'
        }

        if flower in mapping:
            return mapping[flower]

        for key in mapping:
            if key in str(flower) and key != '无':
                return mapping[key]

        return str(flower) if flower else '无'

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

                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    return json.loads(json_match.group())

                return None
            except:
                return None

    def _build_standard_tag_categories(self):
        """构建标准标签体系"""
        return {
            "emotion": ["喜悦", "忧伤", "思念", "孤独", "豪迈", "闲适", "愤懑", "恬淡"],
            "season": ["春", "夏", "秋", "冬"],
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
                "messages": [{"role": "user", "content": "回复：OK"}],
                "temperature": 0.1,
                "max_tokens": 10
            },
            timeout=30
        )
        if response.status_code == 200:
            print("✓ API连接成功")
            return True
        else:
            print(f"✗ API连接失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ API连接异常: {e}")
        return False


def main():
    """主函数"""
    API_KEY = "sk-5176c49ed09c4af98b5ee37598df4f91"

    if not test_api_connection(API_KEY):
        print("API连接失败，请检查网络和API密钥")
        return

    # 创建数据加载器
    data_loader = DataLoader('.')  # 当前目录

    print("正在从数据库加载诗歌数据...")
    poems = data_loader.load_from_databases()

    if not poems:
        print("没有找到诗歌数据，请检查：")
        print("1. 数据库文件是否存在")
        print("2. 数据库文件是否在当前目录")
        print("3. 数据库文件是否包含诗歌数据")
        return

    # 创建分析器 - 根据数据量调整线程数
    total_poems = len(poems)
    if total_poems > 100:
        max_workers = 5  # 大数据量使用更多线程
    else:
        max_workers = 3  # 小数据量使用较少线程

    analyzer = PoetryAnalyzer(API_KEY, max_workers=max_workers)

    print(f"开始分析全部 {total_poems} 首诗歌...")

    # 计算预估成本
    def calculate_cost_estimate(total_poems):
        """计算成本估算"""
        tokens_per_poem = 1500
        total_tokens = total_poems * tokens_per_poem
        cost = total_tokens * 0.14 / 1000000

        print(f"\n成本估算:")
        print(f"  诗歌数量: {total_poems}")
        print(f"  估算tokens: {total_tokens:,}")
        print(f"  估算成本: {cost:.2f} 元")
        return cost

    cost = calculate_cost_estimate(total_poems)

    # 确认是否继续
    confirm = input(f"\n确认分析全部 {total_poems} 首诗歌？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消分析")
        return

    # 分析全部诗歌
    analyzer.analyze_poems_multithreaded(poems)

    print(f"\n分析完成!")
    print(f"成功分析: {analyzer.success_count}/{total_poems}")
    print(f"成功率: {(analyzer.success_count / total_poems) * 100:.1f}%")
    print(f"数据库文件: poetry_embedding.db")

    # 验证数据库内容
    print("\n验证数据库内容:")
    conn = sqlite3.connect("poetry_embedding.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM Poems")
    total_in_db = cursor.fetchone()[0]
    print(f"数据库中共有 {total_in_db} 首诗歌")

    cursor.execute("SELECT author, title, dynasty FROM Poems LIMIT 10")
    results = cursor.fetchall()
    print("前10首诗歌的创作年份:")
    for author, title, dynasty in results:
        print(f"  《{title}》 - {author}: 创作年份 {dynasty}")
    conn.close()


if __name__ == "__main__":
    main()