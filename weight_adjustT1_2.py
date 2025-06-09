# %%
import os
import pandas as pd
import pymongo
import numpy as np
import time
import plotly.graph_objects as go
from plotly.offline import plot
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from urllib.parse import quote_plus
from db_client import get_client_U

client_u = get_client_U()  # 确保调用函数以获取MongoClient实例

# %%
class LimitPriceChecker:
    """检查股票涨跌停状态的类"""

    def get_limit_status(self, date, codes):
        """获取指定日期的股票涨跌停状态"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        date_str = date.strftime('%Y-%m-%d')
        codes_list = codes.tolist() if isinstance(codes, np.ndarray) else codes
        t_limit = client_u.basic_jq.jq_daily_price_none
        use_cols = {"_id": 0, "date": 1, "code": 1, "close": 1, "high_limit": 1, "low_limit": 1}
        df_limit = pd.DataFrame(t_limit.find({"date": date_str, "code": {"$in": codes_list}}, use_cols, batch_size=3000000))
        limit_status = {code: 0 for code in codes}  # 默认为非涨跌停状态
        if not df_limit.empty:
            closes = df_limit['close'].values
            high_limits = df_limit['high_limit'].values
            low_limits = df_limit['low_limit'].values
            limit_array = np.zeros(len(df_limit))
            limit_array = np.where(closes == high_limits, 1, limit_array)
            limit_array = np.where(closes == low_limits, -1, limit_array)
            df_limit['limit'] = limit_array.astype('int')
            limit_dict = dict(zip(df_limit['code'], df_limit['limit']))
            limit_status.update(limit_dict)
        return limit_status

    def get_trade_status(self, date, codes):
        """获取指定日期的股票交易状态"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        date_str = date.strftime('%Y-%m-%d')
        codes_list = codes.tolist() if isinstance(codes, np.ndarray) else codes
        t_info = client_u.basic_wind.w_basic_info
        use_cols = {"_id": 0, "date": 1, "code": 1, "trade_status": 1}
        df_info = pd.DataFrame(t_info.find({"date": date_str, "code": {"$in": codes_list}}, use_cols, batch_size=3000000))
        trade_status = {code: 1 for code in codes}  # 默认可交易
        if not df_info.empty:
            for _, row in df_info.iterrows():
                trade_status[row['code']] = row['trade_status']
        return trade_status

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
    def __init__(self, weights_array, dates, codes, change_limit=0.05):
        """初始化调整器"""
        self._start_time = time.time()
        self.weights = weights_array
        self.dates = pd.to_datetime(dates)
        self.codes = np.array(codes)
        self.change_limit = change_limit
        self.limit_checker = LimitPriceChecker()
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
            limit_status = self.limit_checker.get_limit_status(self.dates[day], self.codes)
            trade_status = self.limit_checker.get_trade_status(self.dates[day], self.codes)
    
            # 计算目标权重和权重变化
            target_weights = self.weights[day]
            weight_changes = target_weights - current_weights
    
            # 创建可调整的掩码数组
            can_adjust_mask = np.zeros(n_codes, dtype=bool)
            for i, code in enumerate(self.codes):
                # 判断每个股票是否可以调整权重
                can_adjust_mask[i] = self.limit_checker.can_adjust_weight(
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
def w_adjust(source_type, change_limit, data_source):
    # 加载数据
    weights_array, dates, codes = PortfolioWeightAdjuster.load_data(data_source, source_type)
    adjuster = PortfolioWeightAdjuster(weights_array, dates, codes, change_limit)
    
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
    result1 = w_adjust(
        source_type='csv',
        change_limit=0.05,
        data_source='csv/test_daily_weight.csv'  # 使用正斜杠
    )
    print(result1)

    