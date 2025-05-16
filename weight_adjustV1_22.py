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
            
            can_adjust_mask = [
                self.data_source.can_adjust_weight(code, chg, limit_status, trade_status)
                for code, chg in zip(self.codes, weight_changes)
            ]
            
            weight_changes_limited = np.clip(weight_changes, -self.change_limit, self.change_limit)
            current_weights[can_adjust_mask] += weight_changes_limited[can_adjust_mask]
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
    # 主流程控制
    try:
        weights_array, dates, codes = _load_initial_data(source_type, data_source)
        adjuster = _initialize_adjuster(weights_array, dates, codes, change_limit, data_directory)
        _validate_weights(adjuster)
        
        # 数据格式转换流水线
        long_format_df = _convert_to_long_format(weights_array, dates, codes)
        minute_frequency_df = _generate_initial_minute_data(long_format_df)
        filtered_df = _clean_and_filter_data(minute_frequency_df)
        
        # 权重调整核心逻辑
        adjusted_weights = _execute_weight_adjustment(adjuster, filtered_df)
        processed_df = _process_adjusted_weights(adjusted_weights, dates, codes)
        
        # 最终输出处理
        return _generate_final_output(processed_df)
    except Exception as e:
        print(f"处理流程异常终止: {str(e)}")
        return None

# 新定义的工具函数（保持原有功能）
def _load_initial_data(source_type, data_source):
    """加载初始权重数据"""
    return PortfolioWeightAdjuster.load_data(data_source, source_type)

def _initialize_adjuster(weights_array, dates, codes, change_limit, data_directory):
    """初始化权重调整器"""
    return PortfolioWeightAdjuster(weights_array, dates, codes, change_limit, data_directory)

def _validate_weights(adjuster):
    """执行权重验证"""
    if not adjuster.validate_weights_sum():
        raise ValueError("权重和验证失败")

def _convert_to_long_format(weights_array, dates, codes):
    """转换数据为长格式"""
    return pd.DataFrame(weights_array, index=dates, columns=codes).reset_index().melt(
        id_vars='index', var_name='code', value_name='weight').rename(columns={'index': 'date'})

def _generate_initial_minute_data(df):
    """生成初始分钟数据"""
    return generate_minute_frequency_data(df)

def _clean_and_filter_data(df):
    """数据清洗过滤"""
    return df[
        (df['weight'] != 0) & 
        (df['time'].between(pd.Timestamp('09:30').time(), pd.Timestamp('15:00').time()))
    ].assign(datetime=lambda x: pd.to_datetime(x['date'].astype(str) + ' ' + x['time'].astype(str)))

def _execute_weight_adjustment(adjuster, df):
    """执行权重调整"""
    return adjuster.adjust_weights_over_days()

def _process_adjusted_weights(adjusted_weights, dates, codes):
    """处理调整后的权重"""
    return pd.DataFrame(adjusted_weights, columns=codes, index=dates).reset_index().melt(
        id_vars='index', var_name='code', value_name='weight').rename(columns={'index': 'date'})

def _generate_final_output(df):
    """生成最终输出"""
    final_df = generate_minute_frequency_data(df[df['weight'] != 0])
    final_df = final_df.drop_duplicates(['date', 'time', 'code']).sort_values(['date', 'time'])
    
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"output/adjusted_weights_minute_{timestamp}.csv"
    
    final_df.to_csv(file_name, columns=['date', 'time', 'code', 'weight'], index=False)
    print(f"结果已保存至: {file_name}")
    return final_df

if __name__ == "__main__":
    weight_list = adjust_weights_to_minute_frequency(
        source_type='csv',
        change_limit=0.002,
        data_source='data/test_daily_weight.csv',
        data_directory='D:\\Data'
    )
    print(weight_list)