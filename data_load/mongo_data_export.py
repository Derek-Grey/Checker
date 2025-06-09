# 在文件开头添加导入语句
from datetime import datetime
import pymongo
import csv
from urllib.parse import quote_plus
import logging
import os

logger = logging.getLogger(__name__)

def get_client_U(m='r'):
    """
    获取带用户认证的MongoDB客户端连接
    :param m: 权限类型('r'/'rw'/'Neo')
    :return: MongoDB客户端实例
    """
    auth_config = {
        'r': ('Tom', 'tom'),
        'rw': ('Amy', 'amy'),
        'Neo': ('Neo', 'neox')
    }
    
    user, pwd = auth_config.get(m, ('Tom', 'tom'))
    if m not in auth_config:
        logger.warning(f'无效权限参数 {m}，使用默认只读权限')
        
    return pymongo.MongoClient(
        f"mongodb://{quote_plus(user)}:{quote_plus(pwd)}@192.168.1.99:29900/"
    )

def get_database(db_name='basic_wind', auth_type='r'):
    """
    获取数据库实例
    """
    client = get_client_U(auth_type)
    return client[db_name]

def export_data(start_date, end_date):
    # 创建输出目录
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, 'mongo_data_export')
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建完整文件路径
    output_path = os.path.join(output_dir, '250101-250501.csv')
    
    # 建立数据库连接（使用读写权限）
    db = get_database('basic_wind', 'rw')
    collection = db['ww_indexconstituent8841431']

    # 转换日期格式
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # 构建查询条件
    query = {
        'date': {
            '$gte': start_date,
            '$lte': end_date
        }
    }

    # 执行查询
    results = collection.find(query).sort('date', pymongo.ASCENDING)

    # 导出到CSV
    count = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 修改列名：date, code, weight(原i_weight)
        writer.writerow(['date', 'code', 'weight'])
        
        for doc in results:
            # 只导出指定字段
            writer.writerow([
                doc['date'],  # 直接使用字符串类型的日期
                doc.get('code', ''),
                doc.get('i_weight', 0)  # 默认值为0
            ])
            count += 1
            
    print(f'成功导出{count}条数据到 {output_path}')

if __name__ == '__main__':
    export_data('2025-01-01', '2025-05-01')