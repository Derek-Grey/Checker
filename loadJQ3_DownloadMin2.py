import datetime
import pandas as pd
import WindPy
import pymongo
from loguru import logger
from WindPy import w
import logging
from db import insert_db_from_df
import time
from collections import Counter
from thriftpy2.transport import TTransportException
from loguru import logger
import jqdatasdk as jq
from my_logger import MyLogger
import numpy as np
jq.auth('13788991423', 'Shdq2024')

class JQInit:
    def __init__(self):
        jq.auth('13788991423', 'Shdq2024')
        self.spare = 0  # 获得今天还有多少数据查询量
        self.jq = jq

    def get_spare_count(self):
        self.spare = self.jq.get_query_count()['spare']  # 获得今天还有多少数据查询量
        return self.spare

    def get_past_trade_dates(self, count=5, start_date=None) -> list:
        """
        默认返回过去 count = 5 个交易日期列表，不含当前交易日（今日）
        如果有给 start_date 默认会忽略掉 count
        :return:
        """
        today = datetime.date.today()
        if start_date:
            trades = self.jq.get_trade_days(start_date=start_date).tolist()
        else:
            trades = self.jq.get_trade_days(count=count).tolist()  # 返回截至到今日的前面交易日期（含今日）
            if today in trades:
                trades = self.jq.get_trade_days(count=count + 1).tolist()
        if today in trades:  # 如果今日是交易日，则把今日剔除。当天的数据要第二个交易日获取
            trades.pop(-1)
        days = [str(i) for i in trades]
        return days

    def get_the_prev_trade_day(self, the_trade_date=None, prev_days=1):
        """ 取 the_trade_date 之前 prev_days 个交易日的第一天"""
        if the_trade_date is None:
            raise Exception("请指定一个交易日！")
        days = self.jq.get_trade_days(count=prev_days + 1, end_date=the_trade_date).tolist()
        days = [str(i) for i in days]
        if the_trade_date in days:
            return days[0]
        else:
            raise Exception("指定的日期：{} ，不是交易日，请指定一个交易日".format(the_trade_date))

    def logout(self):
        self.jq.logout()



def get_client(c_from='local'):
    client_dict = {
        # 'local': '127.0.0.1:27017',
        
        # 'neo': '192.168.1.77:27017/',
        # 'db_w': 'Amy:amy@192.168.1.99:29900',
        
    }
    client_name = client_dict.get(c_from, None)
    if client_name is None:
        raise Exception(f'传入的数据库目标服务器有误 {client_name}，请检查 {client_dict}')
    return pymongo.MongoClient("mongodb://{}".format(client_name))




class JQData2DB:
    def __init__(self, mylog=None):
        self.today = datetime.date.today().isoformat()
        self.log = mylog

        client_U = get_client('local')
        self.t_base_info = client_U['basic_jq']['jq_base_info']
        self.t_daily_price_none = client_U['basic_jq']['jq_daily_price_none']
        self.t_daily_industries = client_U['basic_jq']['jq_daily_indusSWL1']
        self.t_valuation = client_U['basic_jq']['jq_daily_valuation']
        self.t_trade_date = client_U.economic.trade_dates
        self.db_minute = client_U['minute_jq']

    def _get_the_dates(self, date_: str):
        # df = pd.DataFrame(self.t_trade_date.find({'trade_date': {'$lte': self.today, '$gt': date_}},
        #                                          {'trade_date': 1, '_id': 0}).sort([('trade_date', 1), ]))
        # if df.empty:
        #     return []
        # dates = df.trade_date.tolist()
        # if (self.today in dates) and (datetime.datetime.now().hour < 1):
        #     dates.remove(self.today)
        
        
        dates = pd.DataFrame(client.economic.trade_dates.find(
                {'trade_date': {'$gte': "2025-03-08", '$lte':"2025-03-06"}},
                {'_id': 0})).sort_values('trade_date').trade_date.to_list()
        
        
        
        return dates

    def update_minute(self):
        t_minute = self.db_minute['jq_minute_none_2025']
        if self.today > '2025-01-01':
            t_minute = self.db_minute['jq_minute_none_2025']
        try:
            date_ = next(t_minute.find({}, {'date': 1, '_id': 0}).sort([('date', -1), ]).limit(1)).get('date')
        except Exception:
            date_ = '2025-01-01'
            logger.warning(f'到2025咯，开新表，自 {date_} 起')
        dates = self._get_the_dates(date_)

        for date in dates:
            codes_jq = self.t_base_info.distinct('code_jq', {'date': date})
            ljq = LoadJQ(the_date=date)
            codes_db = t_minute.distinct('code_jq', {'date': date})
            down_codes = list(set(codes_jq) - set(codes_db))
            if len(down_codes) > 0:
                self.log.info_logger(f"更新，JQ 分钟表 {date} 的数据，共 {len(down_codes)} 个票")
                df = ljq.download_minute_price(down_codes)
                if not df.empty:
                    insert_db_from_df(table=t_minute, df=df)
                    self.log.info_logger(f'完成 {date} 聚宽分钟数更新')
                else:
                    logger.error(f'交易日：{date}，票：{down_codes}，没拉到分钟数据，看下是不是停牌了！！')
        else:
            logger.warning(f'分钟数据已经全部有了呀')

    def update_industry(self):
        dates = pd.DataFrame(client.economic.trade_dates.find(
                {'trade_date': {'$gte': "2020-01-02", '$lte':"2025-02-28"}},
                {'_id': 0})).sort_values('trade_date').trade_date.to_list()
        logger.warning(f'数据下载区间: {dates[0]} ~ {dates[-1]}')

        for date in dates:
            lj = LoadJQ(the_date=date)
            df = lj.download_industries_and_stocks_SWL1()
            if not df.empty:
                insert_db_from_df(table=self.t_daily_industries, df=df)
                self.log.info_logger(f'完成 {date} 聚宽行业数更新')
            else:
                logger.error(f'交易日：{date}，聚宽行业数据为空！')
               
        else:
            logger.info(f'聚宽行业数据已经是最新的啦。。。')

        

    def update_base_info(self):
        date_ = next(self.t_base_info.find({}, {'date': 1, '_id': 0}).sort([('date', -1), ]).limit(1)).get('date')
        dates = self._get_the_dates(date_)

        for date in dates:
            ljq = LoadJQ(the_date=date)
            df = ljq.download_base_info()
            df.loc[(df.is_st == 1) & (~df.display_name.str.contains('ST')), 'is_st'] = 0  # 把股改的变回来
            insert_db_from_df(self.t_base_info, df)
            self.log.info_logger(f'完成 {date} 聚宽基础信息表更新')
        else:
            logger.info(f'聚宽基础信息表已经是最新的啦。。。')

    def update_price(self):
        date_ = next(self.t_daily_price_none.find({}, {'date': 1, '_id': 0}).sort([('date', -1), ]).limit(1)).get('date')
        dates = self._get_the_dates(date_)
        for date in dates:
            ljq = LoadJQ(the_date=date)
            codes = self.t_base_info.distinct('code_jq', {'date': date})
            df = ljq.download_daily_price(code_list=codes)
            insert_db_from_df(self.t_daily_price_none, df)
            self.log.info_logger(f'完成 {date} 聚宽量价(没复权)信息表更新')
        else:
            logger.info(f'聚宽量价信息表已经是最新的啦。。。')

    def update_valuation(self):
        """更新聚宽市值表数据"""
        try:
            date_ = next(self.t_valuation.find({}, {'date': 1, '_id': 0}).sort([('date', -1), ]).limit(1)).get('date')
        except Exception:
            self.t_valuation.create_index([("date", 1), ("code", 1)], background=True, unique=True)
            date_ = '2005-01-01'
            logger.warning(f'初始化 聚宽 市值表，自 {date_} 起')
        dates = self._get_the_dates(date_)
        for date in dates:
            ljq = LoadJQ(the_date=date)
            codes = self.t_base_info.distinct('code_jq', {'date': date})
            df = ljq.download_valuation(codes_jq=codes)
            insert_db_from_df(self.t_valuation, df)
            self.log.info_logger(f'{date} 聚宽 市值表 UPDATE DONE!')
        else:
            logger.info(f'聚宽量价信息表已经是最新的啦。。。')

    def update_factor(self):
        """一次性更新聚宽复权因子为后复权因子，时点：2023-11-24"""
        dates = self._get_the_dates(date_='2021-01-01')
        dates = sorted(list(filter(lambda x: x < '2024-01-01', dates)))
        ljq = LoadJQ(the_date='2023-11-24')
        for date in dates:
            stock_list = self.t_daily_price_none.distinct('code_jq', {'date': date})
            df = ljq.jq.get_price(stock_list, end_date=date, count=1, fq='post', fields=['factor', ])
            df.insert(1, 'mycode', df.code.transform(lambda x: 'SH' + x[:6] if x.startswith('6') else 'SZ' + x[:6]))
            df.set_index('mycode', inplace=True)
            requests = []
            for code in df.index.to_list():
                value = df.loc[code, 'factor'].squeeze()
                requests.append(UpdateOne({'date': str(date), 'code': code}, {'$set': {'factor': value}}))
            result = self.t_daily_price_none.bulk_write(requests=requests, ordered=False)
            logger.info(f'{date} ~ {df.shape[0]} ,modified_count = {result.modified_count}')



class LoadJQ(JQInit):
    def __init__(self, the_date=None):
        if the_date is None:
            raise Exception('请指定交易日！！')
        super().__init__()

        self.the_date = the_date

        xxx = self.get_spare_count()
        logger.info(f'当前聚宽接口余量：{xxx}，用于更新数据：{self.the_date}，')
        if xxx < 1000000 * 3:
            raise Exception('今日数据量不够了，留一点备用。明日再来吧。')
        self.my_logger = MyLogger()

    def download_base_info(self):
        for i in range(1, 6):
            try:
                df_all_stock = self.jq.get_all_securities(date=self.the_date)
            except TTransportException:
                time.sleep(60)
            else:
                break
        else:
            txt = "聚宽API:get_all_securities，连续5次都没拉到数据，{} 需要单独检查".format(self.the_date)
            self.my_logger.error_logger(txt)
            raise Exception(txt)
        df_all_stock['margin'] = 0
        stocks_sell = self.jq.get_marginsec_stocks(date=self.the_date)
        df_all_stock.loc[df_all_stock.index.isin(stocks_sell), 'margin'] = 1

        sec_info = df_all_stock.reset_index().rename(columns={'index': 'code_jq', })
        sec_info.insert(0, 'code', sec_info.code_jq.transform(lambda x: 'SH' + x[:6] if x.startswith('6') else 'SZ' + x[:6]))
        sec_info.insert(0, 'date', str(self.the_date))
        sec_info['trade_days'] = (pd.to_datetime(self.the_date) - sec_info['start_date']).dt.days

        all_codes_list = sec_info.code_jq.tolist()
        for i in range(1, 6):
            try:
                is_st = self.jq.get_extras('is_st', all_codes_list, end_date=self.the_date, count=1)
            except TTransportException:
                time.sleep(60)
            else:
                break
        else:
            txt = "聚宽API:get_extras，连续5次都没拉到数据，{} 需要单独检查".format(self.the_date)
            self.my_logger.error_logger(txt)
            raise Exception(txt)

        is_st.replace({False: 0, True: 1}, inplace=True)  # 将布尔值转换为数字
        is_st.index = ['is_st']
        is_st = is_st.T.reset_index()
        is_st.rename(columns={'index': 'code_jq'}, inplace=True)
        is_st.insert(0, 'date', str(self.the_date))
        # 将数据合并
        stocks = pd.merge(sec_info, is_st, on=['date', 'code_jq'])
        stocks['date'] = stocks.date.astype(str)
        stocks['start_date'] = stocks.start_date.astype(str)
        stocks['end_date'] = stocks.end_date.astype(str)
        return stocks

    def download_daily_price(self, code_list=list):
        """
        取得没进行复权计算的股票信息,后复权因子, 'factor'
        """
        # 最多可返回['open','close','low','high','volume','money','factor','high_limit','low_limit','avg','pre_close','paused','open_interest']
        columns = ['open', 'close', 'high', 'low', 'high_limit', 'low_limit', 'volume', 'money', 'paused']
        for i in range(1, 6):
            try:
                ps = self.jq.get_price(code_list, fields=columns, start_date=self.the_date, end_date=self.the_date, fq=None,
                                       panel=False, frequency='1d')
                fs = self.jq.get_price(code_list, fields=['factor', ], start_date=self.the_date, end_date=self.the_date, fq='post',
                                       panel=False, frequency='1d')
                stocks = pd.merge(ps, fs, on=['time', 'code'])
            except TTransportException:
                time.sleep(60)
            else:
                break
        else:
            txt = "聚宽API:get_price，连续5次都没拉到数据，{} 需要单独检查".format(self.the_date)
            self.my_logger.error_logger(txt)
            raise Exception(txt)
        stocks.rename(columns={'time': 'date', 'code': 'code_jq'}, inplace=True)
        stocks.insert(1, 'code', stocks.code_jq.transform(lambda x: 'SH' + x[:6] if x.startswith('6') else 'SZ' + x[:6]))
        stocks['date'] = stocks.date.astype(str)
        return stocks

    def download_industries_and_stocks_SWL1(self):
        """下载该日期下，证监会、聚宽、申万的行业分类，以及该分类下有哪些股票"""
        industry_name = 'sw_l1'  # 申万二级行业
        df_indus = pd.DataFrame()
        for t in range(1, 6):
            try:
                df_indus = self.jq.get_industries(name=industry_name, date=self.the_date)  # 获取该日期，该分类下的行业数据
            except TTransportException:  # 没拉到数据，停止1m,再拉
                logger.warning('没拉到数据，停止1m,再拉')
                time.sleep(60)
                df_indus = pd.DataFrame()
            else:
                break
        if df_indus.empty:
            txt = "聚宽数据：申万一级行业，日期：{} 试了5次都没拉到，需要单独进数据库检查".format(self.the_date)
            self.my_logger.error_logger(txt)
            return df_indus
        # 原索引为行业代码，进行重置，并添加日期和行业名称两列，并对字段顺序重新排序
        df_indus.reset_index(inplace=True)
        df_indus.rename(columns={'index': 'indus_code'}, inplace=True)
        # df_indus['stocks'] = np.nan  # 增加一列，默认nan
        df_indus['stocks'] = '[]'  # pandas 2.1.2 中 FutureWarning: Setting an item of incompatible dtype
        check_list = []
        # 获取该分裂下有哪些股票
        for ind in df_indus.indus_code:
            stock_list = self._download_indus_stocks(ind)
            df_indus.loc[df_indus['indus_code'] == ind, ['stocks']] = str(stock_list)
            check_list.extend(stock_list)
        if len(check_list) != len(set(check_list)):
            # 统计列表重复项的数量并转为字典
            dict1 = dict(Counter(check_list))
            # 列表推导式查找字典中值大于1的键值
            dict2 = {key: value for key, value in dict1.items() if value > 1}
            logger.error(f'交易日：{self.the_date}，聚宽返回的申万二级行业数据有重复项，具体(票：重复次数)：{dict2}')
            raise Exception('这个问题比较严重，先退出')
        df_indus['start_date'] = df_indus.start_date.astype(str)
        df_indus.insert(0, 'date', self.the_date)
        return df_indus

    def _download_indus_stocks(self, indus_code=None):
        res_list = []
        for t in range(1, 6):
            try:
                res_list = self.jq.get_industry_stocks(indus_code, date=self.the_date)  # 获取该日期，该行业代码下 成分股
                return res_list
            except TTransportException:  # 没拉到数据，停止1m,再拉
                logger.warning('没拉到数据，停止1m,再拉')
                time.sleep(60)
        if not res_list:
            txt = "聚宽数据：行业代码下成分股，日期：{} 试了5次都没拉到，需要单独进数据库检查".format(self.the_date)
            self.my_logger.error_logger(txt)
        return res_list

    def download_minute_price(self, codes_jq=None):
        if codes_jq is None:
            raise Exception('没给定聚宽格式股票代码列表')
        time_s = str(self.the_date) + ' ' + '09:30:00'
        time_e = str(self.the_date) + ' ' + '15:00:00'
        try:
            df_m = self.jq.get_price(codes_jq, fields=['open', 'close', 'low', 'high', 'volume', 'money', 'avg'],
                                     fill_paused=True, panel=False, skip_paused=False, fq='none', frequency='1m',
                                     start_date=time_s, end_date=time_e)
            df_m.dropna(inplace=True)
            df_m.insert(0, 'date', df_m.time.apply(lambda x: str(x)[:10]))
            df_m['time'] = df_m.time.apply(lambda x: str(x)[-8:])

            df_d = self.jq.get_price(codes_jq, fields=['pre_close', 'close'],
                                     fill_paused=True, panel=False, skip_paused=False, fq='none', frequency='1d',
                                     start_date=self.the_date, end_date=self.the_date,
                                     )
            df_d.dropna(inplace=True)
            df_d.rename(columns={'pre_close': 'pre_day_close', 'close': 'today_close'}, inplace=True)

            df = pd.merge(df_m, df_d[['code', 'pre_day_close', 'today_close']], on='code')
            df['chg_pre'] = (df.close / df.pre_day_close - 1).transform(lambda x: format(x, '.14f'))
            df['chg_close'] = (df.today_close / df.close - 1).transform(lambda x: format(x, '.14f'))
            df.rename(columns={'code': 'code_jq'}, inplace=True)
            df.insert(2, 'code', df.code_jq.transform(lambda x: 'SH' + x[:6] if x.startswith('6') else 'SZ' + x[:6]))
            return df
        except TTransportException:  # 没拉到数据，停止1m,再拉
            raise Exception("聚宽下载数据超时，等下再来！！")

    def download_valuation(self, codes_jq: list = None):
        """聚宽估值表"""
        fields = ['capitalization', 'circulating_cap', 'market_cap', 'circulating_market_cap',
                  'turnover_ratio', 'pe_ratio', 'pe_ratio_lyr', 'pb_ratio', 'ps_ratio', 'pcf_ratio']
        if len(codes_jq) > 4999:
            codes_ll = np.array_split(codes_jq, 2)
            df1 = self.jq.get_valuation(codes_ll[0].tolist(), end_date=self.the_date, count=1, fields=fields)
            df2 = self.jq.get_valuation(codes_ll[1].tolist(), end_date=self.the_date, count=1, fields=fields)
            df = pd.concat((df1, df2))
        else:
            df = self.jq.get_valuation(codes_jq, end_date=self.the_date, count=1, fields=fields)
        df.rename(columns={'code': 'code_jq', 'day': 'date'}, inplace=True)
        df.insert(1, 'code', df.code_jq.transform(lambda x: 'SH' + x[:6] if x.startswith('6') else 'SZ' + x[:6]))
        df['date'] = df.date.astype(str)
        return df




if __name__ == '__main__':
    client = get_client('local')

    log = MyLogger()
    xx = JQData2DB(mylog=log)
    xx.update_minute


        