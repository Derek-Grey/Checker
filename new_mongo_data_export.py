from datetime import datetime, timedelta
import pymongo
import csv
from urllib.parse import quote_plus
import logging
import time
from multiprocessing import Pool, cpu_count

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


def export_data_for_date(date):
    db = get_database('minute_jq_new', 'rw')
    table_name = f"minute2024_jq_minute_{date.strftime('%Y%m%d')}"
    collection = db[table_name]
    query = {'date': date.strftime('%Y-%m-%d')}
    results = collection.find(query).sort('date', pymongo.ASCENDING)
    code_close_map = {}
    rows = []
    for doc in results:
        code = doc.get('code', '')
        close = doc.get('close', 0)
        prev_close = code_close_map.get(code, None)
        if prev_close is not None and prev_close != 0:
            return_value = (close - prev_close) / prev_close
        else:
            return_value = 0
        code_close_map[code] = close
        rows.append([doc.get('date', ''), code, return_value])
    return rows


def export_data(start_date, end_date):
    start_time = time.time()
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    with Pool(cpu_count()) as pool:
        results = pool.map(export_data_for_date, date_range)
    with open('240101-240630.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'code', 'return'])
        for rows in results:
            writer.writerows(rows)
    print(f'成功导出数据到240101-240630.csv')
    end_time = time.time()
    total_time = end_time - start_time
    print(f'数据导出耗时: {total_time} 秒')


if __name__ == '__main__':
    export_data('2024-01-01', '2024-06-30')