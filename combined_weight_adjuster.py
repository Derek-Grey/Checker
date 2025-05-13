import pandas as pd
import numpy as np
import time
from pathlib import Path

# 从 bb.py 中导入相关类和函数
from bb import PortfolioWeightAdjuster, DataSource

def generate_minute_frequency_data(df):
    # 创建时间范围
    morning_time = pd.date_range(start='09:30', end='11:30', freq='T').time
    afternoon_time = pd.date_range(start='13:00', end='15:00', freq='T').time
    valid_times = list(morning_time) + list(afternoon_time)
    
    # 为每个日期和代码生成完整的时间列
    expanded_df = pd.DataFrame()
    for date in df['date'].unique():
        for code in df['code'].unique():
            # 生成每个日期和代码的完整时间序列
            temp_df = pd.DataFrame({'date': date, 'code': code, 'time': valid_times})
            # 检查是否存在对应的权重
            weight_data = df[(df['date'] == date) & (df['code'] == code)]['weight']
            if not weight_data.empty:
                weight = weight_data.iloc[0]
                temp_df['weight'] = weight
                expanded_df = pd.concat([expanded_df, temp_df], ignore_index=True)
    
    return expanded_df

def adjust_weights_to_minute_frequency(source_type, change_limit, data_source, data_directory):
    # 加载数据
    weights_array, dates, codes = PortfolioWeightAdjuster.load_data(data_source, source_type)
    adjuster = PortfolioWeightAdjuster(weights_array, dates, codes, change_limit, data_directory)
    
    # 生成分钟频数据
    minute_frequency_df = generate_minute_frequency_data(pd.DataFrame({'date': dates, 'code': codes, 'weight': weights_array.flatten()}))
    
    # 验证权重和并调整权重
    if adjuster.validate_weights_sum():
        adjusted_weights = adjuster.adjust_weights_over_days()
        
        # 转换为长表格式并过滤权重为0的行
        output_df = pd.DataFrame(adjusted_weights, columns=codes, index=dates).reset_index()
        long_format_df = output_df.melt(id_vars='index', var_name='code', value_name='weight')
        long_format_df.rename(columns={'index': 'date'}, inplace=True)
        long_format_df = long_format_df[long_format_df['weight'] != 0]
        
        # 确保每个code在一天中weight相等
        long_format_df['weight'] = long_format_df.groupby(['date', 'code'])['weight'].transform('first')
        
        # 添加分钟时间列
        long_format_df = generate_minute_frequency_data(long_format_df)
        
        # 按时间序列排序并保存为CSV文件
        long_format_df.sort_values(by=['date', 'time'], inplace=True)
        long_format_df.to_csv('adjusted_weights_minute.csv', index=False)
        print(f"调整后的分钟频权重已保存到: adjusted_weights_minute.csv")
    
    return long_format_df

if __name__ == "__main__":
    weight_list = adjust_weights_to_minute_frequency(
        source_type='csv',
        change_limit=0.05,
        data_source='data/test_daily_weight.csv',  
        data_directory='D:\\Data'  
    )
    print(weight_list)