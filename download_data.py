import jqdatasdk as jq
import pandas as pd
from thriftpy2.transport import TTransportException
import time
from loguru import logger

class JQDataDownloader:
    def __init__(self, username, password):
        jq.auth(username, password)
        self.the_date = None  # 初始化 the_date 属性

    def set_date(self, date):
        """
        设置下载数据的日期
        :param date: 日期字符串，格式为 'YYYY-MM-DD'
        """
        self.the_date = date

    def download_minute_price(self, codes_jq=None):
        if codes_jq is None:
            raise Exception('没给定聚宽格式股票代码列表')
        if self.the_date is None:
            raise Exception('日期未设置')
        time_s = str(self.the_date) + ' ' + '09:30:00'
        time_e = str(self.the_date) + ' ' + '15:00:00'
        try:
            df_m = jq.get_price(codes_jq, fields=['close'],
                                fill_paused=True, panel=False, skip_paused=False, fq='none', frequency='1m',
                                start_date=time_s, end_date=time_e)
            df_m.dropna(inplace=True)
            df_m.insert(0, 'date', df_m.time.apply(lambda x: str(x)[:10]))
            df_m['time'] = df_m.time.apply(lambda x: str(x)[-8:])
            
            # 计算 return
            df_m['return'] = df_m['close'].pct_change().fillna(0)
            
            df_m.rename(columns={'code': 'code_jq'}, inplace=True)
            df_m.insert(2, 'code', df_m.code_jq.transform(lambda x: 'SH' + x[:6] if x.startswith('6') else 'SZ' + x[:6]))
            
            # 只保留所需的列
            df_m = df_m[['date', 'time', 'code', 'return']]
            return df_m
        except TTransportException:  # 没拉到数据，停止1m,再拉
            raise Exception("聚宽下载数据超时，等下再来！！")

    def convert_to_jq_code(self, common_code):
        """
        将普通股票代码转换为聚宽所需的代码格式。

        :param common_code: 普通股票代码，如 'SH688335'
        :return: 聚宽格式的股票代码，如 '688335.XSHG'
        """
        if common_code.startswith('SH'):
            return common_code[2:] + '.XSHG'
        elif common_code.startswith('SZ'):
            return common_code[2:] + '.XSHE'
        else:
            logger.warning(f"无法识别的股票代码格式: {common_code}")
            return common_code

    def read_csv_and_download(self, csv_file_path):
        """
        从CSV文件读取日期和代码，并下载数据
        :param csv_file_path: CSV文件路径
        :return: DataFrame
        """
        try:
            data = pd.read_csv(csv_file_path)
            print("CSV数据读取成功：")
            print(data.head())  # 打印前几行数据以确认读取成功
            
            # 去除重复的日期和代码组合
            data.drop_duplicates(subset=['date', 'code'], inplace=True)
            
            # 将日期格式转换为 YYYY-MM-DD
            data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')
            print("日期格式转换成功：")
            print(data['date'].head())  # 打印转换后的日期
            
            dates = data['date'].tolist()
            if not dates:
                logger.warning("CSV文件中没有日期数据")
                return pd.DataFrame()
            
            # 确保代码格式正确
            data['code'] = data['code'].apply(lambda x: x.strip())  # 去除可能的空格
            codes = data['code'].apply(self.convert_to_jq_code).tolist()
            print("代码格式转换成功：")
            print("转换后的代码列表：")
            for code in codes:
                print(code)
            
            all_data = pd.DataFrame()
            processed = set()  # 用于跟踪已处理的日期和代码组合
            for date, code in zip(dates, codes):
                if (date, code) not in processed:
                    try:
                        self.set_date(date)  # 设置日期
                        df = self.download_minute_price([code])
                        print(f"下载数据成功: {date} - {code}")
                        all_data = pd.concat([all_data, df], ignore_index=True)
                        processed.add((date, code))
                    except Exception as e:
                        logger.error(f"下载数据时出错: {e}")
                else:
                    print(f"跳过重复下载: {date} - {code}")
            return all_data
        except Exception as e:
            logger.error(f"读取CSV文件或下载数据时出错: {e}")
            return pd.DataFrame()
    
# 使用示例
if __name__ == '__main__':
    downloader = JQDataDownloader('13788991423', 'Shdq2024')
    csv_file_path = 'csv/aa.csv'  # 替换为你的CSV文件路径
    df = downloader.read_csv_and_download(csv_file_path)
    df.to_csv('output12.csv', index=False)
    print(df)