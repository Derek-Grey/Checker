import pandas as pd

# 读取CSV文件
df = pd.read_csv('D:/Derek/Code/Checker/csv/test_minute_weight.csv')

# 创建时间范围
morning_time = pd.date_range(start='09:30', end='11:30', freq='T').time
afternoon_time = pd.date_range(start='13:00', end='15:00', freq='T').time
valid_times = list(morning_time) + list(afternoon_time)

# 插入时间列
df['time'] = [valid_times[i % len(valid_times)] for i in range(len(df))]

# 调整列顺序，将'time'列插入到第二列位置
cols = df.columns.tolist()
cols.insert(1, cols.pop(cols.index('time')))
df = df[cols]

# 向后填充其他列的数据
df.fillna(method='ffill', inplace=True)

# 确保在相同的date和code下，weight列的值相等
# 这里假设weight列的名称为'weight'
df['weight'] = df.groupby(['date', 'code'])['weight'].transform('first')

# 保存为新的CSV文件
df.to_csv('D:/Derek/Code/Checker/csv/test_minute_weight_minute.csv', index=False)