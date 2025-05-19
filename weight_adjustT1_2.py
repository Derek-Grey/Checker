'''
autor: Casper Dai 
time: 2025.5.12
version: 1.2
'''
# %%
import os
import pandas as pd
import pymongo
import numpy as np
import time
import plotly.graph_objects as go
from plotly.offline import plot
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
OUTPUT_DIR = Path(__file__).parent / 'output'
from urllib.parse import quote_plus
from pre_import.DictionaryDTType import D1_11_dtype, D1_11_numpy_dtype, D1_6_numpy_dtype, D1_3_numpy_dtype
import datetime

def read_npq_file(file_path, dtype, columns):
    npq_data = np.fromfile(file_path, dtype=dtype)
    quote = npq_data['quote']
    df = pd.DataFrame(quote)
    df = df.astype(str)
    df = df[columns]
    return df

class DataSource:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def get_limit_status(self, date, codes):
        start_time = time.time()
        try:
            limit_status_data = []
            npq_file_path = Path(self.data_directory) / date.strftime('%Y-%m-%d') / "1" / "3.npq"
            if npq_file_path.exists():
                df = read_npq_file(str(npq_file_path), D1_3_numpy_dtype, ['date', 'code', 'close', 'high_limit', 'low_limit'])
                daily_limits = df[df['code'].isin(codes)]
                for _, record in daily_limits.iterrows():
                    limit_status_data.append({
                        'date': record['date'],
                        'code': record['code'],
                        'limit': 1 if record['close'] == record['high_limit'] else -1 if record['close'] == record['low_limit'] else 0
                    })
            limit_status = pd.DataFrame(limit_status_data)
            return limit_status
        except Exception as e:
            raise Exception(f"从NPQ文件获取涨跌停状态数据时出错: {str(e)}")

    def get_trade_status(self, date, codes):
        start_time = time.time()
        try: 
            trade_status_data = []
            npq_file_path = Path(self.data_directory) / date.strftime('%Y-%m-%d') / "1" / "6.npq"
            if npq_file_path.exists():
                df = read_npq_file(str(npq_file_path), D1_6_numpy_dtype, ['date', 'code', 'trade_status'])
                daily_trade_status = df[df['code'].isin(codes)]
                for _, record in daily_trade_status.iterrows():
                    trade_status_data.append({
                        'date': record['date'],
                        'code': record['code'],
                        'trade_status': record['trade_status']
                    })
            trade_status = pd.DataFrame(trade_status_data)
            return trade_status
        except Exception as e:
            raise Exception(f"从NPQ文件获取交易状态数据时出错: {str(e)}")

    def can_adjust_weight(self, code, weight_change, limit_status, trade_status):
        if trade_status.get(code, 1) == 0:
            return False
        status = limit_status.get(code, 0)
        if status == 1 and weight_change > 0: return False
        if status == -1 and weight_change < 0: return False
        return True

class PortfolioWeightAdjuster:
    def __init__(self, weights_array, dates, codes, change_limit, data_directory):
        self._start_time = time.time()
        self.weights = weights_array
        self.dates = pd.to_datetime(dates)
        self.codes = np.array(codes)
        self.change_limit = change_limit
        self.data_source = DataSource(data_directory)
        print(f"初始化耗时: {time.time() - self._start_time:.2f}秒")

    def validate_weights_sum(self) -> bool:
        _start = time.time()
        try:
            daily_sums = np.sum(self.weights, axis=1)
            valid_sums = np.logical_and(daily_sums >= 0.999, daily_sums <= 1.001)
            if not np.all(valid_sums):
                invalid_indices = np.where(~valid_sums)[0]
                print("数据验证失败：以下日期的权重和不为1:")
                for idx in invalid_indices:
                    print(f"{self.dates[idx]}: {daily_sums[idx]}")
                return False
            print("所有日期的权重和验证通过")
            print(f"权重验证耗时: {time.time() - _start:.2f}秒")
            return True
        except Exception as e:
            print(f"权重验证出错：{e}")
            print(f"权重验证耗时: {time.time() - _start:.2f}秒")
            return False

    def adjust_weights_over_days(self):
        _start = time.time()
        n_timestamps, n_codes = self.weights.shape
        adjusted_weights = np.zeros_like(self.weights)
        current_weights = self.weights[0].copy()
    
        for ts_idx in range(n_timestamps):
            _loop_start = time.time()
            current_ts = self.dates[ts_idx]  # 获取完整时间戳，包括日期和时间
            
            # 获取分钟级交易状态
            limit_status = self.data_source.get_limit_status(current_ts.date(), self.codes)
            trade_status = self.data_source.get_trade_status(current_ts.date(), self.codes)
            
            target_weights = self.weights[ts_idx]
            weight_changes = target_weights - current_weights
            
            # 创建可调整的掩码
            can_adjust_mask = np.array([
                self.data_source.can_adjust_weight(code, chg, limit_status, trade_status)
                for code, chg in zip(self.codes, weight_changes)
            ])
            
            # 限制权重变化幅度
            weight_changes_limited = np.zeros_like(weight_changes)
            for i in range(n_codes):
                if can_adjust_mask[i]:
                    # 严格限制在change_limit范围内
                    if weight_changes[i] > 0:
                        weight_changes_limited[i] = min(weight_changes[i], self.change_limit)
                    else:
                        weight_changes_limited[i] = max(weight_changes[i], -self.change_limit)
            
            # 更新当前权重
            current_weights = current_weights + weight_changes_limited
            adjusted_weights[ts_idx] = current_weights.copy()
            
            if ts_idx % 100 == 0:
                print(f"处理 {current_ts.strftime('%Y-%m-%d %H:%M')} 时间点, 单次耗时: {time.time() - _loop_start:.2f}秒")

        print(f"\n权重调整总耗时: {time.time() - _start:.2f}秒")
        print(f"平均每分钟耗时: {(time.time() - _start)/n_timestamps:.4f}秒")
        return adjusted_weights

    @staticmethod
    def load_data(data_source, source_type):
        if source_type == 'csv':
            return PortfolioWeightAdjuster._from_csv(data_source)
        elif source_type == 'df':
            weights_df = data_source
            dates = weights_df.index.to_list()
            codes = weights_df.columns.to_list()
            weights_array = weights_df.values
            return weights_array, dates, codes
        else:
            raise ValueError(f"不支持的数据源类型: {source_type}")

    @staticmethod
    def _from_csv(csv_file_path):
        df = pd.read_csv(csv_file_path)
        time_col = 'datetime' if 'datetime' in df.columns else 'date'
        df[time_col] = pd.to_datetime(df[time_col])
        dates = sorted(df[time_col].unique())
        all_codes = sorted(df['code'].unique())
        n_dates = len(dates)
        n_codes = len(all_codes)
        weights_array = np.zeros((n_dates, n_codes))
        code_to_idx = {code: idx for idx, code in enumerate(all_codes)}
        for date_idx, date in enumerate(dates):
            daily_data = df[df[time_col] == date]
            for _, row in daily_data.iterrows():
                code_idx = code_to_idx[row['code']]
                weights_array[date_idx, code_idx] = row['weight']
        return weights_array, dates, all_codes

def generate_minute_frequency_data(df):
    morning_time = pd.date_range(start='09:30', end='11:30', freq='min').time 
    afternoon_time = pd.date_range(start='13:00', end='15:00', freq='min').time  
    valid_times = list(morning_time) + list(afternoon_time)
    expanded_df = pd.DataFrame()
    for date in df['date'].unique():
        for code in df['code'].unique():
            temp_df = pd.DataFrame({'date': date, 'code': code, 'time': valid_times})
            weight_data = df[(df['date'] == date) & (df['code'] == code)]['weight']
            if not weight_data.empty:
                weight = weight_data.iloc[0]
                temp_df['weight'] = weight
                expanded_df = pd.concat([expanded_df, temp_df], ignore_index=True)
    return expanded_df

def adjust_weights_to_minute_frequency(source_type, change_limit, data_source, data_directory):
    """
    根据输入的权重数据，先生成分钟频率数据，再应用交易限制进行调整
    
    参数:
        source_type: 数据源类型，'csv'或'df'
        change_limit: 每次调整的最大权重变化
        data_source: 数据源路径或DataFrame
        data_directory: 市场数据目录
    """
    try:
        # 1. 数据加载与初始化
        weights_array, dates, codes = PortfolioWeightAdjuster.load_data(data_source, source_type)
        
        # 2. 转换为长格式并生成分钟频率数据
        long_format = pd.DataFrame(weights_array, index=dates, columns=codes).reset_index().melt(
            id_vars='index', var_name='code', value_name='weight'
        ).rename(columns={'index': 'date'})
        
        minute_df = generate_minute_frequency_data(long_format[long_format['weight'] > 0])
        minute_df = minute_df.drop_duplicates(['date', 'time', 'code']).sort_values(['date', 'time'])
        
        # 3. 创建分钟级别的时间戳
        minute_df['datetime'] = pd.to_datetime(
            minute_df['date'].astype(str) + ' ' + minute_df['time'].astype(str)
        )
        
        # 4. 转换为宽格式用于权重调整
        pivot_df = minute_df.pivot_table(
            index='datetime', columns='code', values='weight'
        ).fillna(0)
        
        # 保存调整前的权重数据，用于后续比较
        original_weights = pivot_df.copy()
        
        # 5. 创建调整器并验证权重
        minute_adjuster = PortfolioWeightAdjuster(
            pivot_df.values, 
            pivot_df.index, 
            pivot_df.columns, 
            change_limit, 
            data_directory
        )
        
        if not minute_adjuster.validate_weights_sum():
            raise ValueError("分钟频率数据权重和验证失败")
        
        # 6. 执行权重调整
        adjusted_weights = minute_adjuster.adjust_weights_over_days()
        
        # 7. 结果处理与输出
        result_df = pd.DataFrame(
            adjusted_weights, 
            columns=pivot_df.columns, 
            index=pivot_df.index
        )
        
        # 8. 统计调整前后的变化
        print("\n===== 权重调整前后变化统计 =====")
        
        # 计算每个时间点的权重变化总量
        total_change_by_time = np.abs(result_df.values - original_weights.values).sum(axis=1)
        avg_change_by_time = total_change_by_time.mean()
        max_change_by_time = total_change_by_time.max()
        min_change_by_time = total_change_by_time.min()
        
        print(f"每个时间点平均权重变化总量: {avg_change_by_time:.6f}")
        print(f"最大权重变化时间点变化总量: {max_change_by_time:.6f}")
        print(f"最小权重变化时间点变化总量: {min_change_by_time:.6f}")
        
        # 计算每只股票的权重变化总量
        total_change_by_stock = np.abs(result_df.values - original_weights.values).sum(axis=0)
        avg_change_by_stock = total_change_by_stock.mean()
        max_change_by_stock = total_change_by_stock.max()
        max_change_stock = pivot_df.columns[np.argmax(total_change_by_stock)]
        
        print(f"每只股票平均权重变化总量: {avg_change_by_stock:.6f}")
        print(f"最大权重变化股票: {max_change_stock}, 变化总量: {max_change_by_stock:.6f}")
        
        # 计算调整前后权重和的差异
        original_sum = original_weights.sum(axis=1)
        adjusted_sum = result_df.sum(axis=1)
        sum_diff = adjusted_sum - original_sum
        
        print(f"调整前权重和平均值: {original_sum.mean():.6f}")
        print(f"调整后权重和平均值: {adjusted_sum.mean():.6f}")
        print(f"权重和差异最大值: {sum_diff.abs().max():.6f}")
        
        # 统计调整幅度超过change_limit的次数
        changes = np.abs(result_df.values - original_weights.values)
        exceed_limit_count = (changes > change_limit).sum()
        
        if exceed_limit_count > 0:
            print(f"警告: 有 {exceed_limit_count} 次调整幅度超过了设定的change_limit ({change_limit})")
        else:
            print(f"所有调整幅度均未超过设定的change_limit ({change_limit})")
        
        # 9. 转换回长格式
        adjusted_long = result_df.reset_index().melt(
            id_vars='datetime', var_name='code', value_name='weight'
        )
        
        # 10. 提取日期和时间
        adjusted_long['date'] = adjusted_long['datetime'].dt.date
        adjusted_long['time'] = adjusted_long['datetime'].dt.time
        
        # 11. 过滤并排序
        final_df = adjusted_long[adjusted_long['weight'] > 0].sort_values(['date', 'time'])
        
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 生成输出文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = OUTPUT_DIR / f"adjusted_weights_minute_{timestamp}.csv"
        
        # 保存结果
        final_df.to_csv(file_name, columns=['date', 'time', 'code', 'weight'], index=False)
        print(f"结果已保存至: {file_name}")
        
        return final_df
        
    except Exception as e:
        print(f"处理流程异常终止: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    weight_list = adjust_weights_to_minute_frequency(
        source_type='csv',
        change_limit=0.0002,
        data_source='data/test_daily_weight.csv',
        data_directory='D:\\Data'
    )
    print(weight_list)