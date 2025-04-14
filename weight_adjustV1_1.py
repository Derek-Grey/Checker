# %%
import os
import pandas as pd
import pymongo
import numpy as np
import time
import plotly.graph_objects as go
from plotly.offline import plot
import sys
from pathlib import Path  # 添加这行以导入Path类
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
OUTPUT_DIR = Path(__file__).parent / 'output'  
from urllib.parse import quote_plus
from pre_import.DictionaryDTType import D1_11_dtype, D1_11_numpy_dtype, D1_6_numpy_dtype, D1_3_numpy_dtype

# %%
def read_npq_file(file_path, dtype, columns):
    """读取NPQ文件并返回DataFrame"""
    npq_data = np.fromfile(file_path, dtype=dtype)
    quote = npq_data['quote']
    df = pd.DataFrame(quote)
    df = df.astype(str)
    df = df[columns]
    return df

class DataSource:
    def __init__(self, data_directory):
        self.data_directory = data_directory  # 修正为直接赋值路径字符串

    def get_limit_status(self, date, codes):
        """获取指定日期的股票涨跌停状态"""
        start_time = time.time()
        try:
            limit_status_data = []
            npq_file_path = Path(self.data_directory) / date.strftime('%Y-%m-%d') / "1" / "3.npq"
            if npq_file_path.exists():
                # 使用新的read_npq_file函数
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
        """获取指定日期的股票交易状态"""
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
        """判断是否可以调整权重"""
        if trade_status.get(code, 1) == 0:  # 停牌状态
            return False
        status = limit_status.get(code, 0)
        if status == 1 and weight_change > 0: return False
        if status == -1 and weight_change < 0: return False
        return True
# %%
class PortfolioWeightAdjuster:
    def __init__(self, weights_array, dates, codes, change_limit, data_directory):
        """初始化调整器"""
        self._start_time = time.time()
        self.weights = weights_array
        self.dates = pd.to_datetime(dates)
        self.codes = np.array(codes)
        self.change_limit = change_limit
        self.data_source = DataSource(data_directory) 
        print(f"初始化耗时: {time.time() - self._start_time:.2f}秒")

    def validate_weights_sum(self) -> bool:
        """验证权重和"""
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
        """调整权重"""
        _start = time.time()  # 记录开始时间
        n_dates, n_codes = self.weights.shape  # 获取日期和股票代码的数量
        adjusted_weights = np.zeros_like(self.weights)  # 初始化调整后的权重数组
        current_weights = self.weights[0].copy()  # 使用第一天的权重作为初始权重
    
        for day in range(n_dates):
            _loop_start = time.time()  # 记录每一天开始处理的时间
    
            # 获取当天的涨跌停状态和交易状态
            limit_status = self.data_source.get_limit_status(self.dates[day], self.codes)
            trade_status = self.data_source.get_trade_status(self.dates[day], self.codes)
    
            # 计算目标权重和权重变化
            target_weights = self.weights[day]
            weight_changes = target_weights - current_weights
    
            # 创建可调整的掩码数组
            can_adjust_mask = np.zeros(n_codes, dtype=bool)
            for i, code in enumerate(self.codes):
                # 判断每个股票是否可以调整权重
                can_adjust_mask[i] = self.data_source.can_adjust_weight(
                    code, weight_changes[i], limit_status, trade_status)
    
            # 限制权重变化在指定范围内
            weight_changes_limited = np.clip(weight_changes, -self.change_limit, self.change_limit)
            # 仅对可调整的股票应用权重变化
            current_weights[can_adjust_mask] += weight_changes_limited[can_adjust_mask]
    
            # 保存当日调整后的权重
            adjusted_weights[day] = current_weights.copy()
    
            # 每50天打印一次处理时间
            if day % 50 == 0:
                print(f"处理第 {day+1}/{n_dates} 天, 单次耗时: {time.time() - _loop_start:.2f}秒")
    
        # 打印总耗时和平均每天耗时
        print(f"\n权重调整总耗时: {time.time() - _start:.2f}秒")
        print(f"平均每天耗时: {(time.time() - _start)/n_dates:.2f}秒")
        return adjusted_weights

    def plot_adjusted_weight_sums(self, adjusted_weights):
        """绘制权重和变化图"""
        _start = time.time()
        try:
            adjusted_sums = np.sum(adjusted_weights, axis=1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.dates, y=adjusted_sums, mode='lines+markers', name='实际权重和'))
            fig.add_hline(y=1.0, line=dict(color='#E74C3C', dash='dash'), opacity=0.5, name='目标权重和')
            fig.update_layout(title={'text': '调整后权重和变化', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=20)},
                              xaxis_title='时间', yaxis_title='权重和', template='plotly_white', showlegend=True,
                              legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                              xaxis=dict(tickangle=30, tickformat='%Y-%m-%d'), hovermode='x unified')
            max_sum = max(adjusted_sums)
            min_sum = min(adjusted_sums)
            margin = (max_sum - min_sum) * 0.1
            fig.update_yaxes(range=[min_sum - margin, max_sum + margin])
            fig.show()
        except Exception as e:
            print(f"绘制图形时出错：{e}")
        finally:
            print(f"绘图耗时: {time.time() - _start:.2f}秒")

    @staticmethod
    def load_data(data_source, source_type):
        """通用数据加载接口"""
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
        """从CSV文件加载数据"""
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
# %%
def w_adjust(source_type, change_limit, data_source, data_directory):
    # 加载数据
    weights_array, dates, codes = PortfolioWeightAdjuster.load_data(data_source, source_type)
    adjuster = PortfolioWeightAdjuster(weights_array, dates, codes, change_limit, data_directory)
    
    # 验证权重和并调整权重
    if adjuster.validate_weights_sum():
        adjusted = adjuster.adjust_weights_over_days()
        adjuster.plot_adjusted_weight_sums(adjusted)
        
        # 转换为长表格式并过滤权重为0的行
        output_df = pd.DataFrame(adjusted, columns=codes, index=dates).reset_index()
        long_format_df = output_df.melt(id_vars='index', var_name='code', value_name='weight')
        long_format_df.rename(columns={'index': 'date'}, inplace=True)
        long_format_df = long_format_df[long_format_df['weight'] != 0]
        
        # 按时间序列排序并保存为CSV文件
        long_format_df.sort_values(by='date', inplace=True)
        long_format_df.to_csv('adjusted_weights.csv', index=False)
        print(f"调整后的权重已保存到: adjusted_weights.csv")
    
    return long_format_df

if __name__ == "__main__":
    weight_list = w_adjust(
        source_type='csv',
        change_limit=0.05,
        data_source='csv/test_daily_weight.csv',  
        data_directory='D:\\Data'  
    )
    print(weight_list)

    