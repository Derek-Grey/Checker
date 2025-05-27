'''
autor: Casper Dai 
time: 2025.5.19
version: 1.4
'''
import os
import pandas as pd
import numpy as np
import time
from loguru import logger
from pathlib import Path
import sys
import plotly.graph_objects as go

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 定义输出目录
OUTPUT_DIR = Path(__file__).parent / 'output'

# 导入自定义模块
from pre_import.DictionaryDTType import D1_11_dtype, D1_11_numpy_dtype
from db_client import get_client_U
import pymongo
from urllib.parse import quote_plus

def read_npq_file(file_path):
    """读取NPQ文件并返回DataFrame"""
    npq_data = np.fromfile(file_path, dtype=D1_11_numpy_dtype)
    quote = npq_data['quote']
    df = pd.DataFrame(quote)  
    df = df.astype(str)
    df = df[['date', 'code', 'pct_chg']]
    return df

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
        logger.warning(f'传入的参数 {m} 有误，使用默认只读权限')
        
    return pymongo.MongoClient(
        "mongodb://%s:%s@%s" % (
            quote_plus(user),
            quote_plus(pwd),
            '192.168.1.99:29900/'
        )
    )

class DataChecker:
    """数据检查器类"""
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.trading_dates = self._fetch_trading_dates()

    def _fetch_trading_dates(self):
        """从CSV文件获取交易日"""
        try:
            csv_path = os.path.join(self.data_directory, 'trade_dates_all.csv')
            trading_dates_df = pd.read_csv(csv_path)
            trading_dates_df['trade_date'] = pd.to_datetime(trading_dates_df['trade_date'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')
            return set(trading_dates_df['trade_date'])
        except Exception as e:
            print(f"\n=== CSV文件读取错误 ===\n错误信息: {str(e)}\n===================\n")
            raise

    def check_time_format(self, df):
        """检查时间列格式是否符合HH:MM:SS格式，并验证是否在交易时间内"""
        if 'time' not in df.columns:
            return
            
        print("检测到time列，开始检查时间")
        try:
            times = pd.to_datetime(df['time'], format='%H:%M:%S')
            invalid_times = df[pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').isna()]
            if not invalid_times.empty:
                raise ValueError(f"发现不符合格式的时间: \n{invalid_times['time'].unique()}")
            
            morning_start = pd.to_datetime('09:31:00').time()
            morning_end = pd.to_datetime('11:30:00').time()
            afternoon_start = pd.to_datetime('13:01:00').time()
            afternoon_end = pd.to_datetime('15:00:00').time()
            
            times_outside_trading = df[~(
                ((times.dt.time >= morning_start) & (times.dt.time <= morning_end)) |
                ((times.dt.time >= afternoon_start) & (times.dt.time <= afternoon_end))
            )]
            
            if not times_outside_trading.empty:
                non_trading_times = times_outside_trading['time'].unique()
                raise ValueError(
                    f"发现非交易时间数据：\n"
                    f"{non_trading_times}\n"
                    f"交易时间为 09:31:00-11:30:00 和 13:01:00-15:00:00"
                )
            
            print("时间格式和交易时间范围检查通过")
            
        except ValueError as e:
            print("时间检查失败")
            raise ValueError(f"时间检查错误: {str(e)}")

    def check_time_frequency(self, df):
        """检查时间切片的频率是否一致，并检查是否存在缺失的时间点"""
        if 'time' not in df.columns:
            date_code_counts = df.groupby(['date', 'code']).size()
            invalid_records = date_code_counts[date_code_counts > 1]
            if not invalid_records.empty:
                raise ValueError(
                    f"发现日频数据中存在重复记录：\n"
                    f"日期-股票对及其出现次数：\n{invalid_records}"
                )
            return
        
        print("开始检查时间频率一致性")
        
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        unique_times = sorted(df['datetime'].unique())
        
        time_diffs = []
        for i in range(1, len(unique_times)):
            curr_time = unique_times[i]
            prev_time = unique_times[i-1]
            
            if (curr_time.date() != prev_time.date() or
                (prev_time.time() <= pd.to_datetime('11:30:00').time() and 
                 curr_time.time() >= pd.to_datetime('13:01:00').time())):
                continue
            
            time_diffs.append((curr_time - prev_time).total_seconds())
        
        if not time_diffs:
            raise ValueError("没有足够的数据来确定时间频率")
        
        freq_seconds = pd.Series(time_diffs).mode()
        if len(freq_seconds) == 0:
            raise ValueError("无法确定标准时间频率")
        
        freq_minutes = freq_seconds[0] / 60
        if freq_minutes <= 0:
            raise ValueError(
                f"计算得到的时间频率异常: {freq_minutes} 分钟\n"
                f"时间差统计：{pd.Series(time_diffs).value_counts()}"
            )
        
        if not freq_minutes.is_integer():
            raise ValueError(f"时间频率必须是整数分钟，当前频率为: {freq_minutes} 分钟")
        
        freq_minutes = int(freq_minutes)
        print(f"检测到数据频率为: {freq_minutes} 分钟")
        
        invalid_diffs = [diff for diff in time_diffs if abs(diff - freq_seconds[0]) > 1]
        if invalid_diffs:
            raise ValueError(
                f"发现不规则的时间间隔：\n"
                f"标准频率为: {freq_minutes} 分钟\n"
                f"异常间隔（秒）：{invalid_diffs}"
            )
        
        all_dates = pd.to_datetime(df['date']).unique()
        expected_times = []
        
        for date in all_dates:
            try:
                morning_times = pd.date_range(
                    f"{date.strftime('%Y-%m-%d')} 09:31:00",
                    f"{date.strftime('%Y-%m-%d')} 11:30:00",
                    freq=f"{freq_minutes}min"
                )
                afternoon_times = pd.date_range(
                    f"{date.strftime('%Y-%m-%d')} 13:01:00",
                    f"{date.strftime('%Y-%m-%d')} 15:00:00",
                    freq=f"{freq_minutes}min"
                )
                expected_times.extend(morning_times)
                expected_times.extend(afternoon_times)
            except Exception as e:
                raise ValueError(f"生成时间序列时出错，日期: {date}, 频率: {freq_minutes}分钟\n错误信息: {str(e)}")
        
        expected_times = pd.DatetimeIndex(expected_times)
        actual_times = pd.DatetimeIndex(unique_times)
        
        missing_times = expected_times[~expected_times.isin(actual_times)]
        if len(missing_times) > 0:
            raise ValueError(
                f"发现缺失的时间点：\n"
                f"共计缺失 {len(missing_times)} 个时间点\n"
                f"部分缺失时间点示例（最多显示10个）：\n"
                f"{missing_times[:10].strftime('%Y-%m-%d %H:%M:%S').tolist()}"
            )
        
        print(f"时间频率检查通过，数据频率为: {freq_minutes} 分钟")

    def check_trading_dates(self, df):
        """检查数据是否包含非交易日"""
        dates = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').unique()
        invalid_dates = [d for d in dates if d not in self.trading_dates]
        if invalid_dates:
            raise ValueError(f"数据包含非交易日: {invalid_dates}")

class PortfolioMetrics:
    """投资组合指标计算器类"""
    def __init__(self, stock_path, return_path, use_equal_weights, data_directory, input_type, turn_loss):
        self.stock_path = stock_path
        self.return_path = return_path
        self.use_equal_weights = use_equal_weights
        self.data_directory = data_directory
        self.input_type = input_type
        self.weights = None
        self.returns = None
        self.index_cols = None
        self.is_minute = None
        self.prepare_data()
        self.turn_loss = turn_loss

    def prepare_data(self):
        """为投资组合指标计算准备数据。"""
        start_time = time.time()
        
        # 根据输入类型读取权重数据
        if self.input_type == 'csv':
            weights_df = pd.read_csv(self.stock_path)
        else:  
            weights_df = stock_path.copy()
            
        self._validate_weights(weights_df)
        self.weights = weights_df
        returns_df = self._fetch_returns(weights_df)
        self.returns = returns_df
        date_range = weights_df['date'].unique()
        self.dates, self.codes, self.weights_arr, self.returns_arr = self._convert_to_arrays(weights_df, returns_df)
        self.is_minute = 'time' in weights_df.columns
        
        print(f"数据准备总耗时: {time.time() - start_time:.2f}秒\n")

    def _fetch_returns(self, weights_df):
        """从文件或数据库获取收益率数据"""
        if self.return_path is None:
            print("\n未提供收益率数据文件，将从数据库获取收益率数据...")
            unique_dates = weights_df['date'].unique()
            unique_codes = weights_df['code'].unique()
            start_date = min(unique_dates)
            end_date = max(unique_dates)
            returns = self.get_returns_from_db(start_date, end_date, unique_codes)
            print(f"成功从数据库获取了 {len(returns)} 条收益率记录")
            return returns
        else:
            if self.input_type == 'csv':
                return pd.read_csv(self.return_path)
            else: 
                return self.return_path.copy()

    def get_returns_from_db(self, start_date, end_date, codes):
        """从数据库获取收益率数据"""
        start_time = time.time()
        try:
            returns_data = []
            for date in pd.date_range(start=start_date, end=end_date):
                npq_file_path = Path(self.data_directory) / date.strftime('%Y-%m-%d') / "1" / "11.npq"
                if not npq_file_path.exists():
                    continue
                df = read_npq_file(str(npq_file_path))
                daily_returns = df[df['code'].isin(codes)]
                for _, record in daily_returns.iterrows():
                    returns_data.append({'date': record['date'], 'code': record['code'], 'return': float(record['pct_chg'])})
            returns = pd.DataFrame(returns_data)
        
            print(f"获取收益率数据总耗时: {time.time() - start_time:.2f}秒\n")
            return returns
        except Exception as e:
            raise Exception(f"从NPQ文件获取收益率数据时出错: {str(e)}")

    def _validate_weights(self, weights_df):
        """验证权重数据"""
        required_weight_columns = ['date', 'code']
        missing_weight_columns = [col for col in required_weight_columns if col not in weights_df.columns]
        if missing_weight_columns:
            raise ValueError(f"权重表缺少必要的列: {missing_weight_columns}")

    def _convert_to_arrays(self, weights_df, returns_df):
        """将DataFrame转换为numpy数组，并处理等权重"""
        if 'time' in weights_df.columns:
            weights_df['datetime'] = pd.to_datetime(weights_df['date'] + ' ' + weights_df['time'])
            returns_df['datetime'] = pd.to_datetime(returns_df['date'] + ' ' + returns_df['time'])
            date_col = 'datetime'
        else:
            date_col = 'date'
        
        dates = weights_df[date_col].unique()
        codes = weights_df['code'].unique()
        
        n_dates = len(dates)
        n_codes = len(codes)
        weights_arr = np.zeros((n_dates, n_codes))
        returns_arr = np.zeros((n_dates, n_codes))
        
        date_idx = {date: i for i, date in enumerate(dates)}
        code_idx = {code: i for i, code in enumerate(codes)}
        
        if self.use_equal_weights:
            print("使用等权重")
            weights_per_date = 1 / weights_df.groupby(date_col)['code'].transform('count').values
            for idx, row in weights_df.iterrows():
                i = date_idx[row[date_col]]
                j = code_idx[row['code']]
                weights_arr[i, j] = weights_per_date[idx]
        elif 'weight' in weights_df.columns:
            print("使用提供的权重")
            for _, row in weights_df.iterrows():
                i = date_idx[row[date_col]]
                j = code_idx[row['code']]
                weights_arr[i, j] = row['weight']
        else:
            raise ValueError("权重列缺失，且未设置使用等权重")
        
        for _, row in returns_df.iterrows():
            i = date_idx[row[date_col]]
            j = code_idx[row['code']]
            returns_arr[i, j] = row['return']
        
        return dates, codes, weights_arr, returns_arr
    
    def calculate_portfolio_metrics(self, turn_loss):
        """计算投资组合的收益率、换手率及带成本的净值"""
        start_time = time.time()
        is_minute = self.is_minute
    
        weights_wide = pd.DataFrame(self.weights_arr, index=self.dates, columns=self.codes)
        returns_wide = pd.DataFrame(self.returns_arr, index=self.dates, columns=self.codes)
        weights_wide = weights_wide.fillna(0)
        returns_wide = returns_wide.fillna(0)
        portfolio_returns = (weights_wide * returns_wide).sum(axis=1)
        
        # 初始化换手率相关列
        turnover = pd.Series(index=weights_wide.index)
        passive_turnover = pd.Series(index=weights_wide.index)
        active_turnover = pd.Series(index=weights_wide.index)
        
        # 第一个时间点的换手率应该是从0到初始权重的变化
        turnover.iloc[0] = weights_wide.iloc[0][weights_wide.iloc[0] > 0].sum()
        active_turnover.iloc[0] = turnover.iloc[0]  # 初始建仓全部是主动换手
        passive_turnover.iloc[0] = 0  # 初始没有被动换手

        for i in range(1, len(weights_wide)):
            curr_weights = weights_wide.iloc[i]
            prev_weights = weights_wide.iloc[i-1]
            returns_t = returns_wide.iloc[i-1]
            
            # 计算理论权重（考虑前一时间点的权重和收益率）
            theoretical_weights = prev_weights * (1 + returns_t)
            # 处理可能的0或负值
            theoretical_sum = theoretical_weights.sum()
            if theoretical_sum > 0:
                theoretical_weights = theoretical_weights / theoretical_sum
            else:
                # 如果理论权重和不为正，使用前一时间点的权重
                theoretical_weights = prev_weights.copy()
                if prev_weights.sum() > 0:
                    theoretical_weights = theoretical_weights / prev_weights.sum()
            
            # 被动换手率：前一天权重与理论权重的差异
            passive_turnover.iloc[i] = np.abs(prev_weights - theoretical_weights).sum() / 2
            
            # 主动换手率：当前权重与理论权重的差异
            active_turnover.iloc[i] = np.abs(curr_weights - theoretical_weights).sum() / 2
            
            # 总换手率（使用主动换手率）
            turnover.iloc[i] = active_turnover.iloc[i]
    
        df = pd.DataFrame({
            'portfolio_return': portfolio_returns,
            'turnover': turnover,
            'passive_turnover': passive_turnover,
            'active_turnover': active_turnover
        })
        print(df)
        
        # 设置交易成本
        df['loss'] = 0.0013
        df.loc[df.index > '2023-08-31', 'loss'] = 0.0008
        df['loss'] += turn_loss
        df['chg_'] = df['portfolio_return'] - df['turnover'] * df['loss']
        
        # 修改净值计算，初始净值应该扣除初始建仓成本
        df['net_value'] = 1.0 - df['turnover'].iloc[0] * df['loss'].iloc[0] 
        
        # 从第二个时间点开始累积计算净值
        df['net_value'].iloc[1:] = (df['chg_'].iloc[1:] + 1).cumprod() * df['net_value'].iloc[0]
        
        df.index = pd.to_datetime(df.index)
        
        client = get_client_U('r')
        db = client['basic_wind']
        collection = db['ww_index8841431Daily']
        pct_chg_data = collection.find({'date': {'$in': df.index.strftime('%Y-%m-%d').tolist()}}, {'date': 1, 'pct_chg': 1})
        pct_chg_dict = {item['date']: float(item['pct_chg']) for item in pct_chg_data}
        client.close()
        
        df['pct_chg'] = df.index.strftime('%Y-%m-%d').map(pct_chg_dict)
        results = pd.DataFrame({
            'portfolio_return': df['portfolio_return'],
            'turnover': df['turnover'],
            'passive_turnover': df['passive_turnover'],
            'active_turnover': df['active_turnover'],
            'net_value': df['net_value'],
            'pct_chg': df['pct_chg']
        }, index=pd.to_datetime(self.dates))
        print(results)
        print(f"投资组合指标计算总耗时: {time.time() - start_time:.2f}秒\n")
        output_prefix = 'minute' if is_minute else 'daily'
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'output/{output_prefix}_portfolio_metrics_{timestamp}.csv'
        
        if is_minute:
            daily_results = results.copy()
            results['date'] = results.index.date
            results['time'] = results.index.time
            results = results.reset_index(drop=True)
            minute_results = results[['date', 'time', 'portfolio_return', 'turnover', 'net_value', 'pct_chg']]
            minute_results.to_csv(filename)
            print(f"已保存原始分钟频数据，共 {len(minute_results)} 行")
            
            # 计算每日收益率总和
            daily_sum_returns = results.groupby('date')['portfolio_return'].sum()
            
            daily_results['date'] = daily_results.index.date
            daily_results = daily_results.groupby('date').last().reset_index()
            # 添加每日收益率总和列
            daily_results['daily_sum_return'] = daily_results['date'].map(daily_sum_returns)
            daily_results.loc[daily_results.index[0], 'net_value'] = 1
            daily_filename = f'output/daily_summary_{timestamp}.csv'
            
            # 计算指数净值，保留第一天的实际收益
            daily_results['index_net_value'] = (daily_results['pct_chg'] + 1).cumprod()
            daily_results.loc[daily_results.index[0], 'index_net_value'] = 1
            daily_results.to_csv(daily_filename, index=False)
            print(f"已保存日频汇总数据，共 {len(daily_results)} 行")
        else:
            results['date'] = pd.to_datetime(results.index).date
            results = results.reset_index(drop=True)
            results = results[['date', 'portfolio_return', 'turnover', 'net_value', 'pct_chg']]
            
            # 计算指数净值，保留第一天的实际收益
            daily_results['index_net_value'] = (daily_results['pct_chg'] + 1).cumprod()
            daily_results.loc[daily_results.index[0], 'index_net_value'] = 1
            daily_results.to_csv(daily_filename, index=False)
            print(f"已保存日频汇总数据，共 {len(daily_results)} 行")
    
        print(f"已保存{output_prefix}频投资组合指标数据，共 {len(results)} 行")
        print(f"计算指标总耗时: {time.time() - start_time:.2f}秒\n")
        return daily_results, minute_results, daily_filename

class StrategyPlotter:
    """策略绘图类"""
    def __init__(self, output_dir='output'):
        """初始化绘图类，创建输出目录"""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_net_value(self, df: pd.DataFrame, strategy_name: str):
        """绘制净值曲线图"""
        df = df.copy()
        df.reset_index(inplace=True)
        df.set_index('date', inplace=True)

        # 确保索引为DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        start_date = df.index[0]

        # 检查必要的列是否存在
        if 'portfolio_return' not in df.columns:
            logger.error("DataFrame 不包含 'portfolio_return' 列。")
            return
        if 'turnover' not in df.columns:
            logger.error("DataFrame 不包含 'turnover' 列。")
            return

        # 计算回撤
        self._calculate_drawdown(df)
        
        # 计算统计指标
        stats = self._calculate_statistics(df)
        
        # 绘制图表
        self._create_plot(df, strategy_name, start_date, stats)

    def _calculate_drawdown(self, df: pd.DataFrame):
        """计算最大回撤（直接使用已有净值列）"""
        if 'net_value' not in df.columns:
            raise ValueError("DataFrame必须包含net_value列")
            
        # 计算最大净值和回撤
        dates = df.index.unique().tolist()
        for date in dates:
            df.loc[date, 'max_net'] = df.loc[:date].net_value.max()
        df['back_net'] = df['net_value'] / df['max_net'] - 1
    
    def _calculate_statistics(self, df: pd.DataFrame):
        """计算统计指标（基于已有净值）"""
        if 'net_value' not in df.columns:
            raise ValueError("需要net_value列进行统计计算")
            
        s_ = df.iloc[-1]
        return {
            'annualized_return': format(s_.net_value ** (252 / df.shape[0]) - 1, '.2%'),
            'monthly_volatility': format(df.net_value.pct_change().std() * 21 ** 0.5, '.2%'),
            'end_date': s_.name
        }
    
    def _create_plot(self, df, strategy_name, start_date, stats):
        """创建图表"""
        # 创建净值和回撤的plotly图形对象
        g1 = go.Scatter(x=df.index.unique().tolist(), y=df['net_value'], name='策略净值')
        g2 = go.Scatter(x=df.index.unique().tolist(), y=df['back_net'] * 100, name='回撤', xaxis='x', yaxis='y2', mode="none",
                        fill="tozeroy")
        
        g3 = go.Scatter(x=df.index.unique().tolist(), y=df['index_net_value'], name='指数净值')

        # 修正后的图表配置
        fig = go.Figure(
            data=[g1, g2, g3],
            layout={
                'height': 1122,
                "title": f"{strategy_name}策略，<br>净值（左）& 回撤（右），<br>全期：{start_date} ~ {stats['end_date']}，<br>年化收益：{stats['annualized_return']}，月波动：{stats['monthly_volatility']}",
                "font": {"size": 22},
                "yaxis": {"title": "净值", "side": "left"},
                "yaxis2": {
                    "title": "最大回撤", 
                    "side": "right", 
                    "overlaying": "y", 
                    "ticksuffix": "%",
                    "showgrid": False
                },
                "xaxis": {
                    "title": "交易日序列",
                    "type": "category",
                    "tickmode": "array",
                    "tickvals": df.index[::len(df)//10],
                    "ticktext": df.index.strftime('%Y-%m-%d')[::len(df)//10]
                },
                "legend": {"x": 0, "y": 1},
                "hovermode": "x unified"
            }
        )
        
        fig.show()

    def _save_plot(self, strategy_name: str):
        """保存图表到文件（如果需要）"""
        # TODO: 实现图表保存功能
        pass

def backtest(data_directory, frequency, stock_path, return_path, use_equal_weights, plot_results, turn_loss, input_type):
    """
    执行回测并计算投资组合指标
    
    参数:
    data_directory: 数据目录
    frequency: 频率，'daily'或'minute'
    stock_path: 股票权重文件路径或DataFrame对象
    return_path: 收益率文件路径或DataFrame对象，如果为None则从数据库获取
    use_equal_weights: 是否使用等权重
    plot_results: 是否绘制结果图表
    turn_loss: 换手损失率
    input_type: 输入类型，'csv'或'df'
    
    返回:
    portfolio_returns: 投资组合收益率
    turnover: 换手率
    """
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 根据频率设置检查器
    if frequency == 'minute':
        checker = DataChecker(data_directory)
    else:
        checker = None
    
    # 处理输入数据
    if input_type == 'csv':
        if stock_path is None:
            raise ValueError("未提供股票权重文件路径")
        if not os.path.exists(stock_path):
            raise FileNotFoundError(f"找不到股票权重文件: {stock_path}")
    elif input_type == 'df':
        if not isinstance(stock_path, pd.DataFrame):
            raise ValueError("当input_type为'df'时，stock_path必须是DataFrame对象")
    else:
        raise ValueError("input_type必须是'csv'或'df'")
    
    # 创建投资组合指标计算器
    metrics = PortfolioMetrics(
        stock_path=stock_path,
        return_path=return_path,
        use_equal_weights=use_equal_weights,
        data_directory=data_directory,
        input_type=input_type,
        turn_loss=turn_loss
    )
    
    # 计算投资组合指标
    portfolio_returns, turnover, daily_filename = metrics.calculate_portfolio_metrics(turn_loss=turn_loss)
    
    # 绘制结果
    if plot_results:
        plotter = StrategyPlotter()
        daily_results = pd.read_csv(daily_filename)
        plotter.plot_net_value(daily_results, "投资组合策略")
    
    return portfolio_returns, turnover

if __name__ == "__main__":
    portfolio_returns, turnover = backtest(
        data_directory='D:\\Data',
        turn_loss=0.003,
        frequency='minute',
        stock_path=r'D:\\Derek\\Code\\Checker\\output\\minute_weights_20250526_154852.csv',
        return_path=r'D:\\Derek\\Code\\Keven_wang\\output3411.csv',
        use_equal_weights=False,
        plot_results=True,
        input_type='csv'  
    )
    print(portfolio_returns)
    print(turnover)
    print('完成')