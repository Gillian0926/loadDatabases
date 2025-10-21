import os
import json
import sqlite3
import glob


def simple_batch_import():
    """简化版的批量导入（不需要额外安装库）"""

    folder_path = r"C:\Users\30372\PycharmProjects\loadDataProject\data\TangPoems"
    db_directory = r"C:\Users\30372\Desktop\databases"
    db_file = os.path.join(db_directory, "tang_poems_simple.db")

    # 确保目录存在
    os.makedirs(db_directory, exist_ok=True)

    # 连接数据库
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # 创建表
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS poems
                   (
                       id
                       TEXT
                       PRIMARY
                       KEY,
                       author
                       TEXT,
                       title
                       TEXT,
                       content
                       TEXT,
                       source_file
                       TEXT
                   )
                   """)

    # 查找 JSON 文件
    json_files = glob.glob(os.path.join(folder_path, "**/*.json"), recursive=True)
    if not json_files:
        json_files = glob.glob(os.path.join(folder_path, "*.json"))

    print(f"找到 {len(json_files)} 个文件")

    total_count = 0
    for i, json_file in enumerate(json_files):
        print(f"处理文件 {i + 1}/{len(json_files)}: {os.path.basename(json_file)}")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            filename = os.path.basename(json_file)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        content = '\n'.join(item.get('paragraphs', []))
                        cursor.execute(
                            "INSERT OR IGNORE INTO poems VALUES (?, ?, ?, ?, ?)",
                            (item.get('id'), item.get('author'), item.get('title'), content, filename)
                        )
                        total_count += 1
            elif isinstance(data, dict):
                content = '\n'.join(data.get('paragraphs', []))
                cursor.execute(
                    "INSERT OR IGNORE INTO poems VALUES (?, ?, ?, ?, ?)",
                    (data.get('id'), data.get('author'), data.get('title'), content, filename)
                )
                total_count += 1

        except Exception as e:
            print(f"  错误: {e}")
            continue

    conn.commit()
    conn.close()

    print(f"\n完成！共导入 {total_count} 条诗歌记录")
    print(f"数据库文件: {db_file}")


# 运行简化版本
if __name__ == "__main__":
    simple_batch_import()