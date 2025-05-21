import pandas as pd
import numpy as np
import os
from datetime import datetime, time, timedelta
from collections import defaultdict

class WeightAdjuster:
    """权重调整类，将日频目标权重转换为分钟频调整权重"""
    
    def __init__(self, max_change_per_minute=0.01):
        """
        初始化权重调整器
        
        参数:
        max_change_per_minute: 每分钟最大权重变化限制
        """
        self.max_change_per_minute = max_change_per_minute
        
    def _generate_trading_minutes(self, date):
        """
        生成指定日期的交易分钟时间点
        
        参数:
        date: 日期对象
        
        返回:
        分钟时间点列表
        """
        minutes = []
        
        # 上午交易时段 09:30:00-11:30:00
        start_am = datetime.combine(date, time(9, 30, 0))
        end_am = datetime.combine(date, time(11, 30, 0))
        current = start_am
        while current <= end_am:
            minutes.append(current)
            current += timedelta(minutes=1)
        
        # 下午交易时段 13:00:00-15:00:00
        start_pm = datetime.combine(date, time(13, 0, 0))
        end_pm = datetime.combine(date, time(15, 0, 0))
        current = start_pm
        while current <= end_pm:
            minutes.append(current)
            current += timedelta(minutes=1)
            
        return minutes
    
    def adjust_weights(self, daily_weights_df):
        """
        将日频目标权重调整为分钟频权重
        
        参数:
        daily_weights_df: 包含date, code, weight三列的DataFrame
        
        返回:
        分钟频权重DataFrame，包含datetime, code, weight三列
        """
        # 验证输入数据
        required_columns = ['date', 'code', 'weight']
        for col in required_columns:
            if col not in daily_weights_df.columns:
                raise ValueError(f"输入数据缺少必要的列: {col}")
        
        # 确保日期格式正确
        daily_weights_df['date'] = pd.to_datetime(daily_weights_df['date']).dt.date
        
        # 初始化结果DataFrame
        minute_weights = []
        
        # 初始化权重完成时间统计
        completion_stats = []
        
        # 获取所有唯一日期
        unique_dates = sorted(daily_weights_df['date'].unique())
        
        # 对每个日期进行处理
        for i, date in enumerate(unique_dates):
            print(f"处理日期: {date}")
            
            # 获取当天的目标权重
            target_weights = daily_weights_df[daily_weights_df['date'] == date].copy()
            target_weights_dict = dict(zip(target_weights['code'], target_weights['weight']))
            
            # 获取前一天的最终权重作为当天的初始权重
            if i == 0:
                # 第一天的初始权重为0
                initial_weights_dict = {code: 0 for code in target_weights['code']}
            else:
                # 获取前一天的最后一个分钟的权重
                prev_date = unique_dates[i-1]
                prev_date_minutes = self._generate_trading_minutes(prev_date)
                prev_last_minute = prev_date_minutes[-1]
                
                # 从已生成的分钟权重中获取前一天最后一分钟的权重
                prev_last_weights = [w for w in minute_weights if w['datetime'] == prev_last_minute]
                initial_weights_dict = {w['code']: w['weight'] for w in prev_last_weights}
                
                # 确保当天所有股票代码都有初始权重
                for code in target_weights['code']:
                    if code not in initial_weights_dict:
                        initial_weights_dict[code] = 0
            
            # 生成当天的交易分钟
            trading_minutes = self._generate_trading_minutes(date)
            total_minutes = len(trading_minutes)
            
            # 每日股票完成时间统计
            daily_completion = {}
            
            # 对每个股票代码计算分钟权重
            for code in target_weights['code']:
                target_weight = target_weights_dict[code]
                initial_weight = initial_weights_dict.get(code, 0)
                
                # 计算每分钟需要调整的权重
                weight_diff = target_weight - initial_weight
                
                # 严格按照最大变化限制调整
                weight_change_per_minute = np.sign(weight_diff) * min(
                    self.max_change_per_minute,
                    abs(weight_diff) / 1  # 这里改为1，表示每分钟最多变化max_change_per_minute
                )
                
                # 计算每分钟的权重
                current_weight = initial_weight
                weight_completed = False
                completion_minute = None
                
                for idx, minute in enumerate(trading_minutes):
                    # 按照变化限制慢慢变化
                    current_weight += weight_change_per_minute
                    
                    # 确保不会超过目标权重
                    if weight_diff > 0:
                        current_weight = min(current_weight, target_weight)
                    else:
                        current_weight = max(current_weight, target_weight)
                    
                    minute_weights.append({
                        'datetime': minute,
                        'code': code,
                        'weight': current_weight
                    })
                    
                    # 检查是否达到目标权重
                    if not weight_completed and np.isclose(current_weight, target_weight, rtol=1e-5):
                        weight_completed = True
                        completion_minute = minute
                
                # 记录完成情况
                daily_completion[code] = {
                    'date': date,
                    'code': code,
                    'initial_weight': initial_weight,
                    'target_weight': target_weight,
                    'weight_diff': weight_diff,
                    'final_weight': current_weight,
                    'completed': weight_completed
                }
                
                if weight_completed:
                    daily_completion[code]['completion_time'] = completion_minute.time()
                    # 计算是上午还是下午完成的
                    if completion_minute.time() <= time(11, 30, 0):
                        daily_completion[code]['session'] = '上午'
                    else:
                        daily_completion[code]['session'] = '下午'
                    # 计算完成所需分钟数
                    minute_index = trading_minutes.index(completion_minute)
                    daily_completion[code]['minutes_taken'] = minute_index + 1
                else:
                    daily_completion[code]['completion_time'] = None
                    daily_completion[code]['session'] = '未完成'
                    daily_completion[code]['minutes_taken'] = total_minutes
                    daily_completion[code]['remaining_diff'] = target_weight - current_weight
            
            # 将每日完成情况添加到总统计中
            completion_stats.extend(daily_completion.values())
        
        # 转换为DataFrame
        minute_weights_df = pd.DataFrame(minute_weights)
        
        # 添加日期和时间列
        minute_weights_df['date'] = minute_weights_df['datetime'].dt.date
        minute_weights_df['time'] = minute_weights_df['datetime'].dt.time
        
        # 创建权重完成时间统计DataFrame
        completion_stats_df = pd.DataFrame(completion_stats)
        
        return minute_weights_df, completion_stats_df

def adjust(input_file, max_change):
    """
    主函数
    
    参数:
    input_file: 输入文件路径
    max_change: 每分钟最大权重变化限制
    """
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在!")
        return
    
    # 读取数据
    daily_weights = pd.read_csv(input_file)
    print(f"读取到 {len(daily_weights)} 条日频权重记录")
    
    # 获取最大权重变化限制
    try:
        max_change = float(max_change)
    except ValueError:
        print("输入无效，使用默认值0.001")
        max_change = 0.001
    
    # 创建权重调整器
    adjuster = WeightAdjuster(max_change_per_minute=max_change)
    
    # 调整权重
    minute_weights, completion_stats = adjuster.adjust_weights(daily_weights)
    
    # 确保output目录存在
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"minute_weights_{timestamp}.csv")
    minute_weights.to_csv(output_file, index=False)
    
    # 保存权重完成时间统计
    completion_stats_file = os.path.join(output_dir, f"weight_completion_stats_{timestamp}.csv")
    if not completion_stats.empty:
        # 确保completion_time列是字符串格式，避免NaN值导致的问题
        completion_stats['completion_time'] = completion_stats['completion_time'].astype(str)
        completion_stats.to_csv(completion_stats_file, index=False)
        print(f"权重完成时间统计已保存至 {completion_stats_file}")
    
    # 输出每日每股票完成时间统计表格
    print("\n每日股票权重完成时间统计表格:")
    
    # 按日期分组统计
    completion_by_date = defaultdict(list)
    for stat in completion_stats.to_dict('records'):
        completion_by_date[stat['date']].append(stat)
    
    # 输出每日统计表格
    for date, stats in sorted(completion_by_date.items()):
        print(f"\n日期: {date}")
        
        # 创建表格头部
        header = "| 股票代码 | 初始权重 | 目标权重 | 完成时间 | 用时(分钟) | 状态 |"
        separator = "| -------- | -------- | -------- | -------- | ---------- | ---- |"
        print(header)
        print(separator)
        
        # 按完成时间排序
        sorted_stats = sorted(stats, key=lambda x: (not x['completed'], 
                                                   x['minutes_taken'] if x['completed'] else float('inf')))
        
        # 输出每只股票的信息
        for stat in sorted_stats:
            code = stat['code']
            initial_weight = f"{stat['initial_weight']:.4f}"
            target_weight = f"{stat['target_weight']:.4f}"
            
            if stat['completed']:
                completion_time = str(stat['completion_time'])
                minutes_taken = str(stat['minutes_taken'])
                status = stat['session']
            else:
                completion_time = "未完成"
                minutes_taken = "-"
                status = f"剩余差异: {stat['remaining_diff']:.4f}"
            
            row = f"| {code} | {initial_weight} | {target_weight} | {completion_time} | {minutes_taken} | {status} |"
            print(row)
        
        # 输出统计信息
        completed = [s for s in stats if s['completed']]
        not_completed = [s for s in stats if not s['completed']]
        
        print(f"\n总结: 总股票数 {len(stats)}, 已完成 {len(completed)}, 未完成 {len(not_completed)}")
        
        if completed:
            am_completed = [s for s in completed if s['session'] == '上午']
            pm_completed = [s for s in completed if s['session'] == '下午']
            avg_minutes = sum(s['minutes_taken'] for s in completed) / len(completed)
            
            print(f"上午完成: {len(am_completed)}, 下午完成: {len(pm_completed)}, 平均用时: {avg_minutes:.2f}分钟")
    
    print(f"\n分钟频权重调整完成，共生成 {len(minute_weights)} 条记录")
    print(f"结果已保存至 {output_file}")
    
    return minute_weights, completion_stats

if __name__ == "__main__":
    # 使用原始字符串r前缀或双反斜杠来避免转义问题
    minute_weights, completion_stats = adjust(
        input_file=r"D:\Derek\Code\Checker\csv\mon3.csv",  # 使用r前缀
        max_change=0.000025
    )