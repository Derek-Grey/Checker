import pymongo
from urllib.parse import quote_plus
import pandas as pd

def get_client_U(m='r'):
    """
    获取带用户认证的MongoDB客户端连接
    :param m: 权限类型('r'/'rw'/'Neo')
    :return: MongoDB客户端实例
    """
    # 用户权限配置
    auth_config = {
        'r': ('Tom', 'tom'),      # 只读权限
        'rw': ('Amy', 'amy'),     # 读写权限
        'Neo': ('Neo', 'neox'),   # 管理员权限
    }
    
    user, pwd = auth_config.get(m, ('Tom', 'tom'))  # 默认只读权限
    if m not in auth_config:
        print(f'传入的参数 {m} 有误，使用默认只读权限')
        
    return pymongo.MongoClient(
        "mongodb://%s:%s@%s" % (
            quote_plus(user),
            quote_plus(pwd),
            '192.168.1.99:29900/'
        )
    )

def get_pct_chg_from_db(start_date, end_date):
    """从MongoDB获取pct_chg数据"""
    client = get_client_U('r')  # 获取只读权限的MongoDB客户端
    db = client['basic_wind']
    collection = db['ww_index8841431Daily']
    
    # 将日期转换为字符串格式
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    query = {
        'date': {'$gte': start_date_str, '$lte': end_date_str}
    }
    
    projection = {'date': 1, 'pct_chg': 1}
    cursor = collection.find(query, projection)
    
    # 逐行读取数据并返回
    for document in cursor:
        date = document.get('date')
        pct_chg = document.get('pct_chg')
        print(f"date: {date}, pct_chg: {pct_chg}")
    
    client.close()

# 添加测试函数
if __name__ == "__main__":
    start_date = pd.to_datetime("2023-01-01")
    end_date = pd.to_datetime("2023-01-31")
    get_pct_chg_from_db(start_date, end_date)