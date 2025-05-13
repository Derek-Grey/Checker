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
        n_dates, n_codes = self.weights.shape
        adjusted_weights = np.zeros_like(self.weights)
        current_weights = self.weights[0].copy()
    
        # 修改为按datetime时间点循环处理
        for time_idx in range(n_dates):
            _loop_start = time.time()
            # 获取当前时间点的datetime对象
            current_datetime = self.dates[time_idx]
            
            # 获取对应时间点的交易状态和涨跌停状态
            limit_status = self.data_source.get_limit_status(current_datetime.date(), self.codes)
            trade_status = self.data_source.get_trade_status(current_datetime.date(), self.codes)
            
            # 使用时间点维度的权重数据
            target_weights = self.weights[time_idx]
            weight_changes = target_weights - current_weights
            
            can_adjust_mask = np.zeros(n_codes, dtype=bool)
            for i, code in enumerate(self.codes):
                can_adjust_mask[i] = self.data_source.can_adjust_weight(
                    code, weight_changes[i], limit_status, trade_status)
            
            weight_changes_limited = np.clip(weight_changes, -self.change_limit, self.change_limit)
            current_weights[can_adjust_mask] += weight_changes_limited[can_adjust_mask]
            adjusted_weights[time_idx] = current_weights.copy()
            
            # 修改日志输出显示完整时间
            if time_idx % 50 == 0:
                print(f"处理 {self.dates[time_idx].strftime('%Y-%m-%d %H:%M')} 时间点, 单次耗时: {time.time() - _loop_start:.2f}秒")

        print(f"\n权重调整总耗时: {time.time() - _start:.2f}秒")
        print(f"平均每个时间点耗时: {(time.time() - _start)/n_dates:.2f}秒")
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
    morning_time = pd.date_range(start='09:30', end='11:30', freq='min').time  # 将'T'改为'min'
    afternoon_time = pd.date_range(start='13:00', end='15:00', freq='min').time  # 将'T'改为'min'
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

# 修复SettingWithCopyWarning（第232行附近）
def adjust_weights_to_minute_frequency(source_type, change_limit, data_source, data_directory):
    # 加载原始权重数据（从CSV或DataFrame）
    weights_array, dates, codes = PortfolioWeightAdjuster.load_data(data_source, source_type)
    
    # 初始化权重调整器
    adjuster = PortfolioWeightAdjuster(weights_array, dates, codes, change_limit, data_directory)
    
    # 步骤1：数据完整性验证（确保权重和为1）
    if not adjuster.validate_weights_sum():
        print("数据完整性检查失败，无法继续处理。")
        return None
    
    # 步骤2：将numpy数组转换为DataFrame长格式
    # 原始数据结构：dates x codes 的二维数组
    weights_df = pd.DataFrame(weights_array, index=dates, columns=codes)
    
    # 使用melt将宽表转换为长表（日期、股票代码、权重三列）
    weights_df = weights_df.reset_index().melt(
        id_vars='index',        # 保留日期列
        var_name='code',        # 股票代码列名
        value_name='weight'     # 权重值列名
    ).rename(columns={'index': 'date'})  # 重命名日期列

    # 步骤3：生成分钟频数据（将每日权重扩展为每分钟数据）
    # 生成交易时间段：上午09:30-11:30，下午13:00-15:00
    # 生成初始分钟频数据
    minute_frequency_df = generate_minute_frequency_data(weights_df)
    
    # 新增数据清洗与填充步骤
    # 步骤1：过滤无效数据（零权重和非交易时段）
    cleaned_df = minute_frequency_df[
        (minute_frequency_df['weight'] != 0) & 
        (minute_frequency_df['time'].between(pd.Timestamp('09:30').time(), pd.Timestamp('15:00').time()))
    ]
    
    # 步骤2：后向填充缺失数据（保持时间连续性）
    cleaned_df['weight'] = cleaned_df.groupby(['date', 'code'])['weight'].bfill()
    
    # 步骤3：生成分钟频数据后新增时间合并步骤
    cleaned_df = minute_frequency_df.copy().assign(  # 使用链式操作避免警告
        datetime=lambda x: pd.to_datetime(x['date'].astype(str) + ' ' + x['time'].astype(str))
    )
    
    # 步骤4：执行权重调整（考虑涨跌停和交易状态限制）
    # 重构调整器输入参数（确保保留datetime列）
    adjusted_weights = adjuster.adjust_weights_over_days()
    print(adjusted_weights)
    # 后处理步骤拆分时间列（直接从原始数据获取）
    output_df = cleaned_df[['datetime', 'code', 'weight']].copy()
    output_df = output_df.pivot_table(
        index='datetime',
        columns='code',
        values='weight'
    ).reset_index()

    output_df['date'] = output_df['datetime'].dt.date  # 正确拆分日期
    output_df['time'] = output_df['datetime'].dt.time
    output_df.drop('datetime', axis=1, inplace=True)

    # 步骤5：处理调整后的权重数据
    # 将调整后的numpy数组转换为DataFrame
    output_df = pd.DataFrame(adjusted_weights, columns=codes, index=dates)
    
    # 再次转换为长格式（日期、股票代码、调整后权重）
    long_format_df = output_df.reset_index().melt(
        id_vars='index',
        var_name='code',
        value_name='weight'
    ).rename(columns={'index': 'date'})

    # 步骤6：数据清洗与填充
    # 过滤零权重数据（减少存储空间）
    long_format_df = long_format_df[long_format_df['weight'] != 0]
    
    # 确保同一日期、同一股票的权重一致（取第一个有效值）
    long_format_df['weight'] = long_format_df.groupby(['date', 'code'])['weight'].transform('first')

    # 步骤7：生成最终分钟频数据并保存
    final_minute_df = generate_minute_frequency_data(long_format_df)
    final_minute_df = final_minute_df.drop_duplicates(['date', 'time', 'code'])
    
    # 按日期和时间排序
    final_minute_df.sort_values(by=['date', 'time'], inplace=True)
    
    # 输出到CSV文件（格式：date, time, code, weight）
    final_minute_df.to_csv('adjusted_weights_minute.csv', 
                          columns=['date', 'time', 'code', 'weight'],  # 新增列顺序指定
                          index=False)
    print(f"调整后的分钟频权重已保存到: adjusted_weights_minute.csv")
    
    return final_minute_df

if __name__ == "__main__":
    weight_list = adjust_weights_to_minute_frequency(
        source_type='csv',
        change_limit=0.05,
        data_source='data/test_daily_weight.csv',
        data_directory='D:\\Data'
    )
    print(weight_list)