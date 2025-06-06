'''
autor: Casper Dai 
time: 2025.5.12
version: 1.1
'''
import os
import pandas as pd
import numpy as np
import time
from loguru import logger
from pathlib import Path
import sys
import plotly.graph_objects as go
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
OUTPUT_DIR = Path(__file__).parent / 'output'  # 使用当前文件所在目录下的output文件夹
from pre_import.DictionaryDTType import D1_11_dtype, D1_11_numpy_dtype
from urllib.parse import quote_plus

def read_npq_file(file_path):
    """读取NPQ文件并返回DataFrame"""
    npq_data = np.fromfile(file_path, dtype=D1_11_numpy_dtype)
    quote = npq_data['quote']
    df = pd.DataFrame(quote)  
    df = df.astype(str)
    df = df[['date', 'code', 'pct_chg']]
    return df

# 定义数据检查器
class DataChecker:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.trading_dates = self._fetch_trading_dates()

    def _fetch_trading_dates(self):
        """从CSV文件获取交易日"""
        try:
            # 使用 data_directory 构建 CSV 文件路径
            csv_path = os.path.join(self.data_directory, 'trade_dates_all.csv')
            trading_dates_df = pd.read_csv(csv_path)
            
            # 转换日期格式
            trading_dates_df['trade_date'] = pd.to_datetime(trading_dates_df['trade_date'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')
            
            trading_dates_set = set(trading_dates_df['trade_date'])
            return trading_dates_set
        except Exception as e:
            print(f"\n=== CSV文件读取错误 ===\n错误信息: {str(e)}\n===================\n")
            raise

    def check_time_format(self, df):
        """检查时间列格式是否符合HH:MM:SS格式，并验证是否在交易时间内"""
        if 'time' not in df.columns:
            return
            
        print("检测到time列，开始检查时间")
        try:
            # 转换时间格式并检查
            times = pd.to_datetime(df['time'], format='%H:%M:%S')
            invalid_times = df[pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').isna()]
            if not invalid_times.empty:
                raise ValueError(f"发现不符合格式的时间: \n{invalid_times['time'].unique()}")
            
            # 定义交易时间段
            morning_start = pd.to_datetime('09:30:00').time()
            morning_end = pd.to_datetime('11:30:00').time()
            afternoon_start = pd.to_datetime('13:00:00').time()
            afternoon_end = pd.to_datetime('15:00:00').time()
            
            # 检查是否在交易时间内
            times_outside_trading = df[~(
                ((times.dt.time >= morning_start) & (times.dt.time <= morning_end)) |
                ((times.dt.time >= afternoon_start) & (times.dt.time <= afternoon_end))
            )]
            
            if not times_outside_trading.empty:
                non_trading_times = times_outside_trading['time'].unique()
                raise ValueError(
                    f"发现非交易时间数据：\n"
                    f"{non_trading_times}\n"
                    f"交易时间为 09:30:00-11:30:00 和 13:00:00-15:00:00"
                )
            
            print("时间格式和交易时间范围检查通过")
            
        except ValueError as e:
            print("时间检查失败")
            raise ValueError(f"时间检查错误: {str(e)}")

    def check_time_frequency(self, df):
        """检查时间切片的频率是否一致，并检查是否存在缺失的时间点
        检查规则：
        1. 对于日频数据，每个交易日应该只有一条数据（每个股票）
        2. 对于分钟频数据：
           - 相邻时间点之间的间隔应该一致
           - 在交易时段内不应该有缺失的时间点
        """
        if 'time' not in df.columns:
            # 日频数据检查
            date_code_counts = df.groupby(['date', 'code']).size()
            invalid_records = date_code_counts[date_code_counts > 1]
            if not invalid_records.empty:
                raise ValueError(
                    f"发现日频数据中存在重复记录：\n"
                    f"日期-股票对及其出现次数：\n{invalid_records}"
                )
            return
        
        # 分钟频数据检查
        print("开始检查时间频率一致性")
        
        # 合并日期和时间列创建完整的时间戳
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        
        # 获取唯一的时间点
        unique_times = sorted(df['datetime'].unique())
        
        # 计算时间间隔
        time_diffs = []
        for i in range(1, len(unique_times)):
            # 只在同一个交易时段内计算时间差
            curr_time = unique_times[i]
            prev_time = unique_times[i-1]
            
            # 跳过跨天和午休时段的时间差
            if (curr_time.date() != prev_time.date() or  # 跨天
                (prev_time.time() <= pd.to_datetime('11:30:00').time() and 
                 curr_time.time() >= pd.to_datetime('13:00:00').time())):  # 跨午休
                continue
            
            time_diffs.append((curr_time - prev_time).total_seconds())
        
        if not time_diffs:
            raise ValueError("没有足够的数据来确定时间频率")
        
        # 计算众数作为标准频率
        freq_seconds = pd.Series(time_diffs).mode()
        if len(freq_seconds) == 0:
            raise ValueError("无法确定标准时间频率")
        
        freq_minutes = freq_seconds[0] / 60
        if freq_minutes <= 0:
            raise ValueError(
                f"计算得到的时间频率异常: {freq_minutes} 分钟\n"
                f"时间差统计：{pd.Series(time_diffs).value_counts()}"
            )
        
        # 确保频率是整数分钟
        if not freq_minutes.is_integer():
            raise ValueError(f"时间频率必须是整数分钟，当前频率为: {freq_minutes} 分钟")
        
        freq_minutes = int(freq_minutes)
        print(f"检测到数据频率为: {freq_minutes} 分钟")
        
        # 检查是否存在异常的时间间隔
        invalid_diffs = [diff for diff in time_diffs if abs(diff - freq_seconds[0]) > 1]
        if invalid_diffs:
            raise ValueError(
                f"发现不规则的时间间隔：\n"
                f"标准频率为: {freq_minutes} 分钟\n"
                f"异常间隔（秒）：{invalid_diffs}"
            )
        
        # 生成理论上应该存在的所有时间点
        all_dates = pd.to_datetime(df['date']).unique()
        expected_times = []
        
        for date in all_dates:
            try:
                # 生成上午的时间序列
                morning_times = pd.date_range(
                    f"{date.strftime('%Y-%m-%d')} 09:30:00",
                    f"{date.strftime('%Y-%m-%d')} 11:30:00",
                    freq=f"{freq_minutes}min"
                )
                # 生成下午的时间序列
                afternoon_times = pd.date_range(
                    f"{date.strftime('%Y-%m-%d')} 13:00:00",
                    f"{date.strftime('%Y-%m-%d')} 15:00:00",
                    freq=f"{freq_minutes}min"
                )
                expected_times.extend(morning_times)
                expected_times.extend(afternoon_times)
            except Exception as e:
                raise ValueError(f"生成时间序列时出错，日期: {date}, 频率: {freq_minutes}分钟\n错误信息: {str(e)}")
        
        expected_times = pd.DatetimeIndex(expected_times)
        actual_times = pd.DatetimeIndex(unique_times)
        
        # 找出缺失的时间点
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
    def __init__(self, weight_file, return_file, use_equal_weights,data_directory):
        """初始化投资组合指标计算器"""
        self.weight_file = weight_file
        self.return_file = return_file
        self.use_equal_weights = use_equal_weights
        self.data_directory = data_directory  # 确保使用用户提供的路径或默认路径
        self.weights = None
        self.returns = None
        self.index_cols = None
        self.is_minute = None
        self.prepare_data()

    def prepare_data(self):
        """为投资组合指标计算准备数据。"""
        start_time = time.time()
        
        # 读取并转换权重数据
        weights_df = pd.read_csv(self.weight_file)
        self._validate_weights(weights_df)
        self.weights = weights_df
        returns_df = self._fetch_returns(weights_df)
        self.returns = returns_df
        date_range = weights_df['date'].unique()  # 获取日期范围
        self.dates, self.codes, self.weights_arr, self.returns_arr = self._convert_to_arrays(weights_df, returns_df)  # 转换为numpy数组
        self.is_minute = 'time' in weights_df.columns  # 设置数据频率标志
        
        print(f"数据准备总耗时: {time.time() - start_time:.2f}秒\n")

    def _fetch_returns(self, weights_df):
        """从文件或数据库获取收益率数据"""
        if self.return_file is None:
            print("\n未提供收益率数据文件，将从数据库获取收益率数据...")
            unique_dates = weights_df['date'].unique()
            unique_codes = weights_df['code'].unique()
            # 修改此处，传递正确的参数
            start_date = min(unique_dates)
            end_date = max(unique_dates)
            returns = self.get_returns_from_db(start_date, end_date, unique_codes)
            print(f"成功从数据库获取了 {len(returns)} 条收益率记录")
            return returns
        else:
            return pd.read_csv(self.return_file)

    def get_returns_from_db(self, start_date, end_date, codes):
        """从数据库获取收益率数据"""
        start_time = time.time()
        try:
            returns_data = []
            # 查询数据
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
        # 如果存在time列，先将date和time列合并
        if 'time' in weights_df.columns:
            weights_df['datetime'] = pd.to_datetime(weights_df['date'] + ' ' + weights_df['time'])
            returns_df['datetime'] = pd.to_datetime(returns_df['date'] + ' ' + returns_df['time'])
            date_col = 'datetime'
        else:
            date_col = 'date'
        
        # 获取唯一的日期和股票代码
        dates = weights_df[date_col].unique()
        codes = weights_df['code'].unique()
        
        # 创建空的权重和收益率矩阵
        n_dates = len(dates)
        n_codes = len(codes)
        weights_arr = np.zeros((n_dates, n_codes))
        returns_arr = np.zeros((n_dates, n_codes))
        
        # 创建日期和代码的映射字典以加速查找
        date_idx = {date: i for i, date in enumerate(dates)}
        code_idx = {code: i for i, code in enumerate(codes)}
        
        # 判断是否使用等权重
        if self.use_equal_weights:
            print("使用等权重")
            weights_per_date = 1.0 / weights_df.groupby(date_col)['code'].transform('count').values
            for idx, row in weights_df.iterrows():
                i = date_idx[row[date_col]]
                j = code_idx[row['code']]
                weights_arr[i, j] = weights_per_date[idx]
        elif 'weight' in weights_df.columns:
            # 填充权重矩阵
            print("使用提供的权重")
            for _, row in weights_df.iterrows():
                i = date_idx[row[date_col]]
                j = code_idx[row['code']]
                weights_arr[i, j] = row['weight']
        else:
            raise ValueError("权重列缺失，且未设置使用等权重")
        
        # 填充收益率矩阵
        for _, row in returns_df.iterrows():
            i = date_idx[row[date_col]]
            j = code_idx[row['code']]
            returns_arr[i, j] = row['return']
        
        return dates, codes, weights_arr, returns_arr

    def calculate_portfolio_metrics(self):
        """计算投资组合的收益率和换手率"""
        start_time = time.time()
        is_minute = self.is_minute
    
        # 将 numpy 数组转换为 DataFrame
        weights_wide = pd.DataFrame(self.weights_arr, index=self.dates, columns=self.codes)
        returns_wide = pd.DataFrame(self.returns_arr, index=self.dates, columns=self.codes)
        print(weights_wide.head()) 
        print(returns_wide.head())
        # 计算组合收益率
        portfolio_returns = (weights_wide * returns_wide ).sum(axis=1)
    
        # 计算换手率
        turnover = pd.Series(index=weights_wide.index)
        turnover.iloc[0] = weights_wide.iloc[0].abs().sum()
        for i in range(1, len(weights_wide)):
            curr_weights = weights_wide.iloc[i]
            prev_weights = weights_wide.iloc[i-1]
            returns_t = returns_wide.iloc[i-1]
            theoretical_weights = prev_weights * (1 + returns_t)
            theoretical_weights /= theoretical_weights.sum()
            turnover.iloc[i] = np.abs(curr_weights - theoretical_weights).sum() / 2
    
        # 保存结果
        results = pd.DataFrame({'portfolio_return': portfolio_returns, 'turnover': turnover})
        output_prefix = 'minute' if is_minute else 'daily'
        
        # 添加时间戳到文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'output/test_{output_prefix}_portfolio_metrics_{timestamp}.csv'
        
        # 如果是分钟频数据，拆分 datetime 列为 date 和 time 列
        if is_minute:
            results['date'] = results.index.date
            results['time'] = results.index.time
            results = results.reset_index(drop=True)
            # 调整列顺序，将 date 和 time 放在前两列
            results = results[['date', 'time', 'portfolio_return', 'turnover']]
        else:
            # 对于日频数据，确保包含 date 列
            results['date'] = pd.to_datetime(results.index).date
            results = results.reset_index(drop=True)
            results = results[['date', 'portfolio_return', 'turnover']]
        
        results.to_csv(filename)
        
        print(f"已保存{output_prefix}频投资组合指标数据，共 {len(results)} 行")
        print(f"计算指标总耗时: {time.time() - start_time:.2f}秒\n")
        return portfolio_returns, turnover ,filename
        
# 绘图类
class StrategyPlotter:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_net_value(self, df: pd.DataFrame, strategy_name: str, turn_loss: float = 0.003):
        df = df.copy()  # 创建副本避免修改原始数据
        df.reset_index(inplace=True)

        # 根据数据频率设置索引
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
            df.set_index('datetime', inplace=True)
        else:
            df.set_index('date', inplace=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        start_date = df.index[0]

        # 确保必要的列存在
        if 'portfolio_return' not in df.columns:
            logger.error("DataFrame 不包含 'portfolio_return' 列。")
            return
        if 'turnover' not in df.columns:
            logger.error("DataFrame 不包含 'turnover' 列。")
            return

        # 计算成本和净值
        self._calculate_costs_and_returns(df, turn_loss)
        
        # 计算回撤
        self._calculate_drawdown(df)
        
        # 计算统计指标
        stats = self._calculate_statistics(df)
        
        # 绘制图表
        self._create_plot(df, strategy_name, start_date, stats)
    
    def _calculate_costs_and_returns(self, df: pd.DataFrame, turn_loss: float):
        """计算成本和收益"""
        # 设置固定成本，并针对特定日期进行调整
        df['loss'] = 0.0013  # 初始固定成本
        df.loc[df.index > '2023-08-31', 'loss'] = 0.0008  # 特定日期后的调整成本
        df['loss'] += float(turn_loss)  # 加上换手损失

        # 计算调整后的变动和累计净值
        df['chg_'] = df['portfolio_return'] - df['turnover'] * df['loss']
        df['net_value'] = (df['chg_'] + 1).cumprod()
    
    def _calculate_drawdown(self, df: pd.DataFrame):
        """计算最大回撤"""
        # 计算最大净值和回撤
        dates = df.index.unique().tolist()
        for date in dates:
            df.loc[date, 'max_net'] = df.loc[:date].net_value.max()
        df['back_net'] = df['net_value'] / df['max_net'] - 1
    
    def _calculate_statistics(self, df: pd.DataFrame):
        """计算统计指标"""
        s_ = df.iloc[-1]
        return {
            'annualized_return': format(s_.net_value ** (252 / df.shape[0]) - 1, '.2%'),
            'monthly_volatility': format(df.net_value.pct_change().std() * 21 ** 0.5, '.2%'),
            'end_date': s_.name
        }
    
    def _create_plot(self, df, strategy_name, start_date, stats):
        """创建图表"""
        # 创建净值和回撤的plotly图形对象
        g1 = go.Scatter(x=df.index.unique().tolist(), y=df['net_value'], name='净值')
        g2 = go.Scatter(x=df.index.unique().tolist(), y=df['back_net'] * 100, name='回撤', xaxis='x', yaxis='y2', mode="none",
                        fill="tozeroy")

        # 修正后的图表配置
        fig = go.Figure(
            data=[g1, g2],
            layout={
                'height': 1122,
                "title": f"{strategy_name}策略，<br>净值（左）& 回撤（右），<br>全期：{start_date} ~ {stats['end_date']}，<br>年化收益：{stats['annualized_return']}，月波动：{stats['monthly_volatility']}",
                "font": {"size": 22},
                "yaxis": {"title": "累计净值", "side": "left"},
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
        )  # 补充缺失的括号
        
        fig.show()

    def _save_plot(self, strategy_name: str):
        """保存图表到文件（如果需要）"""
        # TODO: 实现图表保存功能
        pass

def backtest(data_directory, frequency, stock_path, return_file, use_equal_weights, plot_results):
    """主函数，支持交互式和命令行参数两种调用方式"""
    try:
        print("=== 投资组合指标计算器 ===")
        
        # 初始化检查器，传递 data_directory 参数
        checker = DataChecker(data_directory)
        weights = pd.read_csv(stock_path)
        checker.check_trading_dates(weights)
        if frequency == 'minute':
            checker.check_time_frequency(weights)
        
        print("数据验证通过")
        
        # 初始化组合指标计算器
        portfolio = PortfolioMetrics(
            weight_file=stock_path,
            return_file=return_file,
            use_equal_weights=use_equal_weights,
            data_directory=data_directory
        )
        
        # 执行计算并保存结果
        portfolio_returns, turnover,filename = portfolio.calculate_portfolio_metrics()
        print(f"\n计算完成！结果已保存至 filename ”目录")
        print (filename)
        # 如果需要绘制结果
        if plot_results:
            try:
                results_df = pd.read_csv(filename)
                plotter = StrategyPlotter(output_dir='output')
                plotter.plot_net_value(results_df, f"Portfolio_{frequency}", turn_loss=0.003)
                print("已生成策略净值图表")
            except Exception as e:
                print(f"绘制图表时出错: {str(e)}")

        return portfolio_returns, turnover
        
    except Exception as e:
        print(f"\n执行过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    portfolio_returns, turnover= backtest(
        data_directory='D:\\Data',
        frequency='daily',
        stock_path=r'D:\Derek\Code\Checker\data\test_daily_weight.csv',
        return_file=None,
        use_equal_weights=True,
        plot_results=True
    )