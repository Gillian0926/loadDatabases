import json
import requests
import time
import os
import glob
import re
from collections import defaultdict


class PoetryAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.total_tokens = 0
        self.analysis_results = []
        self.processed_count = 0

        # 加载标签库并提取标准标签体系
        self.standard_tag_categories = self._build_standard_tag_categories()

        # 统一的输出文件名
        self.output_filename = f'poetry_analysis_{time.strftime("%Y%m%d_%H%M%S")}.json'

        # 诗人年代数据库（简化版）
        self.poet_timeline = self._build_poet_timeline()

    def _build_poet_timeline(self):
        """构建诗人年代参考数据库"""
        return {
            # 唐代诗人
            "王昌龄": (698, 757), "李白": (701, 762), "杜甫": (712, 770),
            "白居易": (772, 846), "李商隐": (813, 858), "杜牧": (803, 852),
            "王维": (701, 761), "孟浩然": (689, 740), "刘禹锡": (772, 842),

            # 宋代诗人
            "苏轼": (1037, 1101), "李清照": (1084, 1155), "辛弃疾": (1140, 1207),
            "陆游": (1125, 1210), "王安石": (1021, 1086), "黄庭坚": (1045, 1105),
            "柳永": (984, 1053), "晏殊": (991, 1055), "欧阳修": (1007, 1072),

            # 五代至宋初文人
            "徐铉": (916, 991), "李昉": (925, 996), "宋庠": (996, 1066),
            "严蕊": (1140, 1180),

            # 宋代皇帝（作为诗人）
            "宋太宗": (939, 997), "宋真宗": (968, 1022), "宋仁宗": (1010, 1063),
        }

    def _build_standard_tag_categories(self):
        """从标签库文件中构建标准标签体系"""
        label_paths = [
            r"C:\database\myLabelBase\flower\flower_images_detailed.json",
            r"C:\database\myLabelBase\flower\flower_search_tool.json",
            r"C:\database\myLabelBase\flower\flower_seasonal_index.json",
            r"C:\database\myLabelBase\overall\complete_classical_images.json"
        ]

        tag_categories = {
            "emotion": set(),
            "imagery_type": set(),
            "symbolism": set(),
            "season": set(),
            "technique": set(),
            "atmosphere": set()
        }

        for path in label_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self._extract_tags_from_data(data, tag_categories)
                print(f"成功从 {os.path.basename(path)} 提取标签")

            except Exception as e:
                print(f"加载标签库失败 {path}: {e}")

        # 转换为列表并添加常用标签
        standardized_categories = {}
        for category, tags in tag_categories.items():
            tag_list = list(tags)
            # 添加一些基础标签确保覆盖
            if category == "emotion" and not tag_list:
                tag_list = ["喜悦", "忧伤", "思念", "孤独", "豪迈", "闲适", "愤懑", "恬淡"]
            elif category == "imagery_type" and not tag_list:
                tag_list = ["自然意象", "人物意象", "动物意象", "植物意象", "时空意象", "器物意象"]
            elif category == "technique" and not tag_list:
                tag_list = ["比喻", "拟人", "夸张", "对偶", "用典", "象征", "白描"]
            elif category == "atmosphere" and not tag_list:
                tag_list = ["雄浑", "婉约", "豪放", "清丽", "凄美", "空灵", "沉郁"]

            standardized_categories[category] = tag_list

        # 打印提取的标签统计
        print("\n=== 标签库统计 ===")
        for category, tags in standardized_categories.items():
            print(f"{category}: {len(tags)} 个标签")
            if tags:
                print(f"  示例: {', '.join(tags[:5])}")
        print("=================\n")

        return standardized_categories

    def _extract_tags_from_data(self, data, tag_categories):
        """从不同格式的标签库数据中提取标签"""
        if isinstance(data, dict):
            # 处理 complete_classical_images.json 格式
            if "images" in data:
                for image in data["images"]:
                    if "emotional_tone" in image:
                        emotions = str(image["emotional_tone"]).split("、")
                        tag_categories["emotion"].update([e.strip() for e in emotions])
                    if "core_symbolism" in image:
                        symbolisms = str(image["core_symbolism"]).split("、")
                        tag_categories["symbolism"].update([s.strip() for s in symbolisms])

            # 处理 flower_categories 格式
            if "flower_categories" in data:
                for category_key, category_data in data["flower_categories"].items():
                    # 提取季节信息
                    if "blooming_season" in category_data:
                        if isinstance(category_data["blooming_season"], list):
                            tag_categories["season"].update(category_data["blooming_season"])

                    # 提取特征信息
                    if "characteristics" in category_data:
                        if isinstance(category_data["characteristics"], list):
                            tag_categories["atmosphere"].update(category_data["characteristics"])

                    # 处理具体花卉
                    if "flowers" in category_data:
                        for flower in category_data["flowers"]:
                            if "emotional_tone" in flower:
                                emotions = str(flower["emotional_tone"]).split("、")
                                tag_categories["emotion"].update([e.strip() for e in emotions])
                            if "core_symbolism" in flower:
                                symbolisms = str(flower["core_symbolism"]).split("、")
                                tag_categories["symbolism"].update([s.strip() for s in symbolisms])
                            if "cultural_significance" in flower:
                                significances = str(flower["cultural_significance"]).split("，")
                                tag_categories["symbolism"].update([s.strip() for s in significances])

    def _build_analysis_prompt(self, content, author):
        """构建更精确的分析提示词，要求给出时间推断理由"""
        # 获取诗人年代信息用于提示
        poet_info = ""
        if author in self.poet_timeline:
            birth, death = self.poet_timeline[author]
            poet_info = f"（诗人{author}生卒年：{birth}-{death}年）"

        return f"""请严格按以下JSON格式输出分析结果，不要添加任何额外文字：

{{
    "date": 年代,
    "date_rationale": "时间推断理由",
    "mentioned_flowers": ["花卉1", "花卉2", "花卉3"],
    "main_flower": "主要花卉",
    "emotion": "情感标签",
    "free_tags": ["自由标签1", "自由标签2", "自由标签3"]
}}

诗歌内容：
{content}
作者：{author}{poet_info}

分析要求：
1. 创作年代：基于作者生平、诗歌风格、历史背景等推断创作年代，请给出具体的单一公元年份
2. 时间推断理由：详细说明推断该年代的理由和依据
3. 提及花卉：识别诗中明确提到的所有花卉名称
4. 主要花卉：确定诗歌最主要描写的花卉
5. 情感标签：从喜悦、忧伤、思念、孤独、豪迈、闲适、愤懑、恬淡中选择
6. 自由标签：生成3-5个有洞察力的分析标签

时间推断注意事项：
- 请给出具体的单一公元年份，不要使用年份范围
- 必须基于作者生卒年、诗歌风格成熟期、历史事件等具体依据
- 如果是宋代诗人，创作年代应在960-1279年之间
- 如果是唐代诗人，创作年代应在618-907年之间  
- 给出具体的推断理由，如"基于作者中年创作风格"、"根据诗中提到的历史事件"等

请只输出JSON格式，不要有任何其他文字。"""

    def analyze_poem(self, poem_data):
        """分析单首诗词"""
        print(f"正在分析: {poem_data['title']} - {poem_data['author']}")
        print(f"诗词内容: {poem_data['content'][:100]}...")

        prompt = self._build_analysis_prompt(poem_data['content'], poem_data['author'])
        print("发送API请求...")

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
                    "max_tokens": 1500  # 增加token限制以容纳理由
                },
                timeout=60
            )

            print(f"API响应状态码: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print("API请求成功")

                content = result['choices'][0]['message']['content']
                print(f"原始响应: {content}")

                cleaned_content = self._clean_response(content)
                parsed_result = self._parse_json_result(cleaned_content)

                if parsed_result:
                    print("JSON解析成功")
                    standardized = self._standardize_result(parsed_result, poem_data['author'])

                    analysis_record = {
                        'title': poem_data['title'],
                        'author': poem_data['author'],
                        'content': poem_data['content'],
                        'analysis': standardized,
                        'source_file': poem_data['source_file'],
                        'analysis_timestamp': time.time()
                    }

                    self.analysis_results.append(analysis_record)
                    self.processed_count += 1
                    return True
                else:
                    print("JSON解析失败")
                    # 尝试提取JSON部分
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        print("尝试提取JSON部分...")
                        json_content = json_match.group()
                        parsed_result = self._parse_json_result(json_content)
                        if parsed_result:
                            print("JSON提取成功")
                            standardized = self._standardize_result(parsed_result, poem_data['author'])

                            analysis_record = {
                                'title': poem_data['title'],
                                'author': poem_data['author'],
                                'content': poem_data['content'],
                                'analysis': standardized,
                                'source_file': poem_data['source_file'],
                                'analysis_timestamp': time.time()
                            }

                            self.analysis_results.append(analysis_record)
                            self.processed_count += 1
                            return True
            else:
                print(f"API请求失败: {response.status_code}")
                print(f"响应内容: {response.text}")

        except requests.exceptions.Timeout:
            print("API请求超时")
        except requests.exceptions.ConnectionError:
            print("网络连接错误")
        except Exception as e:
            print(f"分析失败: {e}")
            import traceback
            traceback.print_exc()

        return False

    def _standardize_result(self, result, author):
        """标准化分析结果，包含时间验证和理由"""
        mentioned_flowers = result.get('mentioned_flowers', [])
        if not isinstance(mentioned_flowers, list):
            mentioned_flowers = [mentioned_flowers]

        # 标准化所有提及的花卉
        standardized_mentioned = [self._standardize_flower(flower) for flower in mentioned_flowers if flower]

        # 验证和修正年代，同时保留原始理由
        original_date = result.get('date', 0)
        date_rationale = result.get('date_rationale', '')
        validated_date = self._validate_date(original_date, author, date_rationale)

        return {
            "date": validated_date['date'],
            "original_date": original_date,  # 保留原始推断
            "date_rationale": validated_date['rationale'],  # 可能修正后的理由
            "date_correction": validated_date['correction'],  # 是否经过修正
            "mentioned_flowers": standardized_mentioned,
            "main_flower": self._standardize_flower(result.get('main_flower', '无')),
            "emotion": result.get('emotion', '未知'),
            "free_tags": result.get('free_tags', []),
            "date_confidence": self._get_date_confidence(validated_date['date'], author)
        }

    def _validate_date(self, date_val, author, original_rationale):
        """验证和修正年代，支持年份范围解析"""
        try:
            # 处理年份范围格式（如"960-974年"）
            if isinstance(date_val, str) and '-' in date_val:
                # 提取年份范围中的第一个年份
                year_match = re.search(r'(\d{3,4})-(\d{3,4})', str(date_val))
                if year_match:
                    start_year = int(year_match.group(1))
                    end_year = int(year_match.group(2))
                    date_int = (start_year + end_year) // 2  # 取中间值
                    print(f"解析年份范围: {date_val} -> {date_int}")
                else:
                    date_int = self._extract_year_from_string(str(date_val))
            else:
                date_int = self._extract_year_from_string(str(date_val))

            correction_info = ""
            corrected = False

            # 如果作者在数据库中，验证年代合理性
            if author in self.poet_timeline:
                birth, death = self.poet_timeline[author]

                # 如果年代明显错误，修正为作者主要活动时期
                if date_int == 0 or date_int < birth - 20 or date_int > death + 20:
                    corrected_date = birth + (death - birth) // 2
                    if date_int == 0:
                        correction_info = f"无法解析时间，基于{author}生卒年({birth}-{death})推断为{corrected_date}年"
                    else:
                        correction_info = f"原推断{date_int}年，基于{author}生卒年({birth}-{death})修正为{corrected_date}年"
                    date_int = corrected_date
                    corrected = True

            # 构建最终的理由
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
            print(f"时间解析异常: {e}")
            # 如果无法解析，返回默认值
            return {
                'date': 0,
                'rationale': '时间解析失败',
                'correction': True
            }

    def _extract_year_from_string(self, date_str):
        """从字符串中提取年份"""
        try:
            # 移除中文和特殊字符
            clean_str = re.sub(r'[^\d-]', '', date_str)

            # 尝试直接转换
            if clean_str and clean_str != '-':
                return int(clean_str)

            # 尝试从字符串中匹配4位数字
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
            '梨': '梨花', '海棠': '海棠', '兰': '兰花', '芍药': '芍药',
            '芙蓉': '芙蓉', '杜鹃': '杜鹃', '蔷薇': '蔷薇', '茉莉': '茉莉',
            '石榴': '石榴', '水仙': '水仙', '栀子': '栀子', '茶花': '茶花',
            '竹': '竹', '松': '松', '柏': '柏', '柳': '柳',
            '丹沙': '朱砂',  # 丹沙是矿物，不是植物
            '无明确花卉': '无', '无明确花卉（主题为禽鸟意象）': '无',
            '无': '无', '未知': '无', '石衣': '苔藓'  # 石衣是苔藓类植物
        }
        # 移除可能的修饰词，只保留核心花卉名称
        if flower in mapping:
            return mapping[flower]

        # 处理包含修饰词的花卉名称
        for key in mapping:
            if key in flower and key != '无':
                return mapping[key]

        return flower

    def _clean_response(self, content):
        """清理响应内容"""
        # 移除代码块标记
        content = content.replace('```json', '').replace('```', '').strip()
        # 移除可能的注释行
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            if not line.strip().startswith('注：') and not line.strip().startswith('//'):
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def _parse_json_result(self, content):
        """解析JSON结果，增加容错处理"""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"解析内容: {content}")

            # 尝试修复常见的JSON格式问题
            try:
                # 修复单引号问题
                content = content.replace("'", '"')
                # 修复尾随逗号
                content = re.sub(r',\s*}', '}', content)
                content = re.sub(r',\s*]', ']', content)

                return json.loads(content)
            except:
                return None
        except Exception as e:
            print(f"解析异常: {e}")
            return None

    # 其他方法保持不变...
    def save_results(self, output_dir='analysis_output'):
        """保存结果到同一个JSON文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, self.output_filename)

        output_data = {
            'project': '唐宋词花卉意象分析-时间推断版',
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_processed': self.processed_count,
            'total_analyzed': len(self.analysis_results),
            'metadata': {
                'output_dir': output_dir,
                'file_created': time.strftime('%Y%m%d_%H%M%S'),
                'api_tokens_used': self.total_tokens,
                'standard_tag_categories': self.standard_tag_categories
            },
            'results': self.analysis_results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"分析结果已保存到: {filepath}")
        return filepath

    def load_previous_results(self, output_dir='analysis_output'):
        """加载之前结果 - 修改为加载最新的单个文件"""
        if not os.path.exists(output_dir):
            return set(), 0

        try:
            json_files = glob.glob(os.path.join(output_dir, 'poetry_analysis_*.json'))
            if not json_files:
                return set(), 0

            # 找到最新的文件
            latest_file = max(json_files, key=os.path.getctime)
            print(f"加载最新分析文件: {os.path.basename(latest_file)}")

            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            analyzed_poems = set()
            for result in data.get('results', []):
                key = f"{result['title']}_{result['author']}"
                analyzed_poems.add(key)

            # 恢复计数和结果
            self.processed_count = data.get('total_processed', 0)
            self.analysis_results = data.get('results', [])
            self.total_tokens = data.get('metadata', {}).get('api_tokens_used', 0)

            # 设置相同的输出文件名以便继续写入
            self.output_filename = os.path.basename(latest_file)

            print(f"已加载之前分析结果: {len(analyzed_poems)} 首诗词")
            return analyzed_poems, len(analyzed_poems)

        except Exception as e:
            print(f"加载之前结果失败: {e}")
            return set(), 0


# DataLoader 类和其他函数保持不变...
class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.databases = ['SongSongs', 'TangPoems', 'TangPoems2']

    def scan_databases(self):
        """扫描数据库统计信息"""
        print("扫描数据库文件...")
        stats = {}
        total_files = 0
        total_poems_estimate = 0

        for db_name in self.databases:
            db_path = os.path.join(self.data_dir, db_name)
            if os.path.exists(db_path):
                json_files = glob.glob(os.path.join(db_path, '*.json'))
                file_count = len(json_files)
                stats[db_name] = file_count
                total_files += file_count
                poems_estimate = file_count * 1000
                total_poems_estimate += poems_estimate
                print(f"  {db_name}: {file_count} 个文件 (约{poems_estimate}首)")
            else:
                stats[db_name] = 0
                print(f"  {db_name}: 目录不存在")

        print(f"文件总数: {total_files}")
        print(f"估算诗词总数: {total_poems_estimate}")
        return stats, total_poems_estimate

    def load_poems(self, mode='sample', sample_size=100, sample_rate=0.01, limit=None):
        """加载诗词数据"""
        print("正在加载诗词数据...")
        all_poems = []

        for db_name in self.databases:
            db_path = os.path.join(self.data_dir, db_name)
            if not os.path.exists(db_path):
                continue

            json_files = glob.glob(os.path.join(db_path, '*.json'))
            print(f"处理 {db_name} 数据库 ({len(json_files)} 个文件)...")

            files_to_process = json_files
            if mode == 'sample' and len(json_files) > 5:
                files_to_process = json_files[:5]

            for json_file in files_to_process:
                poems = self._load_from_file(json_file, mode, sample_rate, sample_size)
                all_poems.extend(poems)

                if limit and len(all_poems) >= limit:
                    all_poems = all_poems[:limit]
                    break

            if limit and len(all_poems) >= limit:
                break

        print(f"成功加载 {len(all_poems)} 首诗词")
        return all_poems

    def _load_from_file(self, file_path, mode, sample_rate, sample_size):
        """从单个文件加载数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            poems = []
            if isinstance(data, list):
                items_to_process = data

                if mode == 'sample' and len(data) > sample_size:
                    import random
                    items_to_process = random.sample(data, sample_size)
                elif mode == 'rate' and sample_rate < 1.0:
                    items_to_process = [item for i, item in enumerate(data) if i % int(1 / sample_rate) == 0]

                for item in items_to_process:
                    poem = self._extract_poem_data(item, file_path)
                    if poem:
                        poems.append(poem)

            return poems

        except Exception as e:
            print(f"文件读取失败 {file_path}: {e}")
            return []

    def _extract_poem_data(self, item, source_file):
        """提取诗词数据"""
        if not isinstance(item, dict):
            return None

        title = item.get('title') or '无题'
        author = item.get('author') or '未知'
        content = item.get('content') or ' '.join(item.get('paragraphs', []))

        if content and len(content.strip()) > 10:
            return {
                'title': str(title),
                'author': str(author),
                'content': str(content),
                'source_file': os.path.basename(source_file)
            }

        return None


def calculate_cost_estimate(total_poems):
    """计算成本估算"""
    tokens_per_poem = 1500  # 增加因为包含时间推断理由
    total_tokens = total_poems * tokens_per_poem
    cost = total_tokens * 0.14 / 1000000

    print(f"\n估算:")
    print(f"  诗词数量: {total_poems}")
    print(f"  估算tokens: {total_tokens:,}")
    print(f"  估: {cost:.2f}")

    return cost


def test_api_connection(api_key):
    """测试API连接"""
    print("测试API连接...")
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "请只回复JSON格式：{\"test\": \"success\"}"}],
                "temperature": 0.1,
                "max_tokens": 50
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✓ API连接测试成功，响应: {content}")
            return True
        else:
            print(f"✗ API连接测试失败: {response.status_code}")
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"✗ API连接测试异常: {e}")
        return False


def main():
    """主函数"""
    API_KEY = "sk-5176c49ed09c4af98b5ee37598df4f91"

    # 首先测试API连接
    if not test_api_connection(API_KEY):
        print("API连接失败，请检查网络和API密钥")
        return

    analyzer = PoetryAnalyzer(API_KEY)
    data_loader = DataLoader('data')

    stats, total_poems_estimate = data_loader.scan_databases()

    print("\n分析模式:")
    print("1. 抽样分析 (快速测试)")
    print("2. 完整分析 (处理所有数据)")
    print("3. 继续分析 (从上次停止处继续)")
    print("4. 单首测试 (调试模式)")

    choice = input("请输入选择 (1/2/3/4): ").strip()

    if choice == '1':
        sample_size = int(input("请输入抽样数量 (默认3): ") or "3")
        poems = data_loader.load_poems(mode='sample', sample_size=sample_size)
        calculate_cost_estimate(sample_size)

    elif choice == '2':
        limit = input("请输入处理数量限制 (enter表示无限制): ").strip()
        limit = int(limit) if limit else None
        poems = data_loader.load_poems(mode='full', limit=limit)
        poems_count = limit if limit else total_poems_estimate
        calculate_cost_estimate(poems_count)

    elif choice == '3':
        analyzed_poems, previous_count = analyzer.load_previous_results()
        poems = data_loader.load_poems(mode='full')
        original_count = len(poems)
        poems = [p for p in poems if f"{p['title']}_{p['author']}" not in analyzed_poems]
        print(f"过滤后待分析诗词: {len(poems)}/{original_count}")

    elif choice == '4':
        # 单首测试模式
        poems = data_loader.load_poems(mode='sample', sample_size=1)
        if poems:
            print(f"\n测试诗词: 《{poems[0]['title']}》 - {poems[0]['author']}")
            print(f"内容: {poems[0]['content']}")
        else:
            print("没有找到测试诗词")
            return

    else:
        print("无效选择")
        return

    if not poems:
        print("没有找到诗词数据")
        return

    print(f"\n开始诗词分析...")
    print(f"目标分析数量: {len(poems)} 首")
    print(f"结果将保存到: {analyzer.output_filename}")

    success_count = 0
    start_time = time.time()

    for i, poem in enumerate(poems, 1):
        print(f"\n进度: {i}/{len(poems)}")

        if analyzer.analyze_poem(poem):
            success_count += 1
            print("✓ 分析成功")

            # 显示最新分析结果
            latest_result = analyzer.analysis_results[-1]
            analysis = latest_result['analysis']
            print(f"  年代: {analysis['date']} (可信度: {analysis.get('date_confidence', '未知')})")
            if analysis.get('date_correction'):
                print(f"  时间修正: 是")
            print(f"  时间理由: {analysis['date_rationale'][:100]}...")
            print(f"  提及花卉: {analysis['mentioned_flowers']}")
            print(f"  主要花卉: {analysis['main_flower']}")

            # 每分析成功3首保存一次
            if success_count % 3 == 0:
                analyzer.save_results()
                print("已保存进度")
        else:
            print("✗ 分析失败")

            # 如果是单首测试模式，直接退出
            if choice == '4':
                break

        time.sleep(2)

    # 最终保存
    if success_count > 0:
        final_file = analyzer.save_results()
        elapsed_time = time.time() - start_time

        print(f"\n分析完成统计:")
        print(f"总处理诗词: {analyzer.processed_count}")
        print(f"成功分析: {success_count}")
        print(f"成功率: {(success_count / len(poems)) * 100:.1f}%")
        print(f"总用时: {elapsed_time / 60:.1f} 分钟")
        print(f"所有结果已保存到: {final_file}")
    else:
        print("\n所有分析都失败了，请检查以上错误信息")


if __name__ == "__main__":
    main()