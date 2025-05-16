import csv
from pymongo import MongoClient
from urllib.parse import quote_plus

def get_client_U(m='r'):
    """
    获取带用户认证的MongoDB客户端连接
    :param m: 权限类型('r'/'rw'/'Neo')
    :return: MongoDB客户端实例
    """
    auth_config = {
        'r': ('Tom', 'tom'),      # 只读权限
        'rw': ('Amy', 'amy'),     # 读写权限
        'Neo': ('Neo', 'neox'),   # 管理员权限
    }
    
    user, pwd = auth_config.get(m, ('Tom', 'tom'))  # 默认只读权限
    if m not in auth_config:
        print(f'传入的参数 {m} 有误，使用默认只读权限')
        
    return MongoClient(
        "mongodb://%s:%s@%s" % (
            quote_plus(user),
            quote_plus(pwd),
            '192.168.1.99:29900/'
        )
    )

def fetch_chg_pre_from_db(date, time, code):
    client = get_client_U()
    db = client['minute_jq']
    collection = db['jq_minute_none_2025']
    query = {"date": date, "time": time, "code": code}
    result = collection.find_one(query, {"chg_pre": 1, "_id": 0})
    return result['chg_pre'] if result else None
    print(result['chg_pre'])
    
def read_csv_and_fetch_data(file_path, output_file_path):
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.DictReader(file)
        fieldnames = ['date', 'time', 'code', 'return']
        with open(output_file_path, mode='w', newline='') as output_file:
            csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            print("正在读取数据，请稍等...")
            csv_writer.writeheader()
            for row in csv_reader:
                date = row['date']
                time = row['time']
                code = row['code']
                chg_pre = fetch_chg_pre_from_db(date, time, code)
                print(f"正在读取数据：{date} {time} {code}")
                csv_writer.writerow({'date': date, 'time': time, 'code': code, 'return': chg_pre})
                
# 调用函数读取CSV并获取数据，生成新的CSV文件
read_csv_and_fetch_data('output.csv', 'output_with_return.csv')