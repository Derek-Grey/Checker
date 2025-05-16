import pandas as pd

# 从CSV文件读取数据
df = pd.read_csv('csv/aa.csv')

# 创建时间序列，频率为1分钟，跳过11:30到13:00之间的时间
morning_times = pd.date_range(start='09:30:00', end='11:30:00', freq='1min')
afternoon_times = pd.date_range(start='13:00:00', end='15:00:00', freq='1min')
trading_times = morning_times.append(afternoon_times)

# 创建一个包含所有日期、代码和时间组合的DataFrame
unique_dates = df['date'].unique()
unique_codes = df['code'].unique()

# 使用笛卡尔积生成所有组合
full_index = pd.MultiIndex.from_product([unique_dates, unique_codes, trading_times], names=['date', 'code', 'time'])
full_df = pd.DataFrame(index=full_index).reset_index()

# 合并原始数据和完整的时间序列数据
merged_df = pd.merge(full_df, df, on=['date', 'code'], how='left')

# 向下填充weight列
merged_df['weight'] = merged_df['weight'].ffill()

# 格式化时间列以去掉日期部分
merged_df['time'] = merged_df['time'].dt.strftime('%H:%M:%S')

# 将结果保存为CSV文件
merged_df.to_csv('output.csv', index=False)

# 打印结果
print(merged_df)