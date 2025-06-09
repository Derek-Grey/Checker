'''
Created on 2025年05月22日
author: Derek
Data dictionary:
    date：日期
    code：股票代码
    initial_weight：初始权重
    target_weight：目标权重
    weight_diff：权重偏差
    final_weight：最终权重
    completed：是否完成
    is_sell：是否卖出
    completion_time：完成时间
    session：是否完成
    minutes_taken：完成时间点
    remaining_diff：剩余未完成部分
    operation_type：操作类型
    limit_up：是否涨停
    limit_down：是否跌停

'''

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
    
    def adjust_weights(self, daily_weights_df, limit_status_df=None):
        """
        将日频目标权重调整为分钟频权重
        
        参数:
        daily_weights_df: 包含date, code, weight三列的DataFrame
        limit_status_df: 可选，包含date, code, limit_up, limit_down的DataFrame，表示每日每只股票的涨跌停状态
        
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
        
        # 处理涨跌停状态数据
        if limit_status_df is not None:
            # 确保日期格式正确
            limit_status_df['date'] = pd.to_datetime(limit_status_df['date']).dt.date
            # 创建日期和股票代码的联合键
            limit_status_df['date_code'] = limit_status_df['date'].astype(str) + '_' + limit_status_df['code']
            # 创建涨跌停状态字典，方便查询
            limit_up_dict = dict(zip(limit_status_df['date_code'], limit_status_df['limit_up']))
            limit_down_dict = dict(zip(limit_status_df['date_code'], limit_status_df['limit_down']))
        else:
            # 如果没有提供涨跌停数据，创建空字典
            limit_up_dict = {}
            limit_down_dict = {}
        
        # 初始化结果DataFrame
        minute_weights = []
        
        # 初始化权重完成时间统计
        completion_stats = []
        
        # 获取所有唯一日期
        unique_dates = sorted(daily_weights_df['date'].unique())
        
        # 跟踪所有出现过的股票代码
        all_codes = set(daily_weights_df['code'].unique())
        
        # 对每个日期进行处理
        for i, date in enumerate(unique_dates):
            print(f"处理日期: {date}")
            
            # 获取当天的目标权重
            target_weights = daily_weights_df[daily_weights_df['date'] == date].copy()
            target_weights_dict = dict(zip(target_weights['code'], target_weights['weight']))
            
            # 获取前一天的最终权重作为当天的初始权重
            if i == 0:
                # 第一天的初始权重等于目标权重（即已持仓）
                initial_weights_dict = target_weights_dict.copy()
                # 确保所有出现过的股票代码都有初始权重
                for code in all_codes:
                    if code not in initial_weights_dict:
                        initial_weights_dict[code] = 0
            else:
                # 获取前一天的最后一个分钟的权重
                prev_date = unique_dates[i-1]
                prev_date_minutes = self._generate_trading_minutes(prev_date)
                prev_last_minute = prev_date_minutes[-1]
                
                # 从已生成的分钟权重中获取前一天最后一分钟的权重
                prev_last_weights = [w for w in minute_weights if w['datetime'] == prev_last_minute]
                initial_weights_dict = {w['code']: w['weight'] for w in prev_last_weights}
                
                # 确保所有出现过的股票代码都有初始权重
                for code in all_codes:
                    if code not in initial_weights_dict:
                        initial_weights_dict[code] = 0
            
            # 找出当天需要卖出的股票（前一天有权重但今天没有出现在目标权重中）
            sell_codes = []
            for code in all_codes:
                if code not in target_weights_dict and initial_weights_dict.get(code, 0) > 0:
                    sell_codes.append(code)
                    # 为需要卖出的股票添加目标权重为0的记录
                    target_weights_dict[code] = 0
            
            # 生成当天的交易分钟
            trading_minutes = self._generate_trading_minutes(date)
            total_minutes = len(trading_minutes)
            
            # 每日股票完成时间统计
            daily_completion = {}
            
            # 对每个股票代码计算分钟权重
            for code in list(target_weights_dict.keys()) + sell_codes:
                # 去重
                if code in daily_completion:
                    continue
                    
                target_weight = target_weights_dict.get(code, 0)
                initial_weight = initial_weights_dict.get(code, 0)
                
                # 计算每分钟需要调整的权重
                weight_diff = target_weight - initial_weight
                
                # 如果初始权重为0且目标权重也为0，则跳过
                if initial_weight == 0 and target_weight == 0:
                    continue
                
                # 检查股票是否涨停或跌停
                date_code_key = f"{date}_{code}"
                is_limit_up = limit_up_dict.get(date_code_key, False)
                is_limit_down = limit_down_dict.get(date_code_key, False)
                
                # 涨停股票不能增加权重，跌停股票不能减少权重
                if (is_limit_up and weight_diff > 0) or (is_limit_down and weight_diff < 0):
                    # 如果涨停且需要增加权重，或者跌停且需要减少权重，则无法调整
                    weight_change_per_minute = 0
                    can_adjust = False
                else:
                    # 正常情况下按照最大变化限制调整
                    weight_change_per_minute = np.sign(weight_diff) * min(self.max_change_per_minute, abs(weight_diff) / 1)
                    can_adjust = True
                
                # 计算每分钟的权重
                current_weight = initial_weight
                weight_completed = False
                completion_minute = None
                
                for idx, minute in enumerate(trading_minutes):
                    if can_adjust:
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
                is_sell = target_weight == 0 and initial_weight > 0
                
                daily_completion[code] = {
                    'date': date,
                    'code': code,
                    'initial_weight': initial_weight,
                    'target_weight': target_weight,
                    'weight_diff': weight_diff,
                    'final_weight': current_weight,
                    'completed': weight_completed,
                    'is_sell': is_sell,
                    'limit_up': is_limit_up,
                    'limit_down': is_limit_down
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
                    # 如果remaining_diff没有数据则填充为0
                    remaining_diff = target_weight - current_weight if target_weight is not None and current_weight is not None else 0
                    daily_completion[code]['remaining_diff'] = remaining_diff
                    
                # 计算完成比例
                if abs(weight_diff) > 0:
                    completion_ratio = (abs(weight_diff) - abs(target_weight - current_weight)) / abs(weight_diff) * 100
                    daily_completion[code]['completion_ratio'] = completion_ratio
                else:
                    daily_completion[code]['completion_ratio'] = 100
                
                # 如果是因为涨跌停导致无法完成，记录原因
                if is_limit_up and weight_diff > 0:
                    daily_completion[code]['fail_reason'] = '涨停无法买入'
                elif is_limit_down and weight_diff < 0:
                    daily_completion[code]['fail_reason'] = '跌停无法卖出'
            
            # 将每日完成情况添加到总统计中
            completion_stats.extend(daily_completion.values())
        
        # 转换为DataFrame
        minute_weights_df = pd.DataFrame(minute_weights)
        
        # 添加日期和时间列
        if not minute_weights_df.empty:
            minute_weights_df['date'] = minute_weights_df['datetime'].dt.date
            minute_weights_df['time'] = minute_weights_df['datetime'].dt.time
            
            # 按照datetime时间顺序排序
            minute_weights_df = minute_weights_df.sort_values(by='datetime')
        
        # 创建权重完成时间统计DataFrame
        completion_stats_df = pd.DataFrame(completion_stats)
        
        return minute_weights_df, completion_stats_df

def adjust(input_file, max_change, limit_status_file=None):
    """
    主函数
    
    参数:
    input_file: 输入文件路径
    max_change: 每分钟最大权重变化限制
    limit_status_file: 可选，涨跌停状态文件路径
    """
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在!")
        return
    
    # 读取数据
    daily_weights = pd.read_csv(input_file)
    print(f"读取到 {len(daily_weights)} 条日频权重记录")
    
    # 读取涨跌停状态数据（如果提供）
    limit_status_df = None
    if limit_status_file and os.path.exists(limit_status_file):
        limit_status_df = pd.read_csv(limit_status_file)
        print(f"读取到 {len(limit_status_df)} 条涨跌停状态记录")
    
    # 获取最大权重变化限制
    try:
        max_change = float(max_change)
    except ValueError:
        print("输入无效，使用默认值0.001")
        max_change = 0.001
    
    # 创建权重调整器
    adjuster = WeightAdjuster(max_change_per_minute=max_change)
    
    # 调整权重
    minute_weights, completion_stats = adjuster.adjust_weights(daily_weights, limit_status_df)
    
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
        # 添加操作类型列
        completion_stats['operation_type'] = completion_stats.apply(
            lambda x: "卖出" if x.get('is_sell', False) else 
                     ("买入" if x['initial_weight'] == 0 and x['target_weight'] > 0 else
                     ("加仓" if x['target_weight'] > x['initial_weight'] else
                     ("减仓" if x['target_weight'] < x['initial_weight'] else "持平"))), axis=1)
        
        # 确保completion_time列是字符串格式，避免NaN值导致的问题
        completion_stats['completion_time'] = completion_stats['completion_time'].astype(str)
        completion_stats.to_csv(completion_stats_file, index=False)
        print(f"权重完成时间统计已保存至 {completion_stats_file}")
    
    # 按日期分组统计
    completion_by_date = defaultdict(list)
    for stat in completion_stats.to_dict('records'):
        completion_by_date[stat['date']].append(stat)
    
    # 输出每日统计表格
    for date, stats in sorted(completion_by_date.items()):

        # 按完成时间排序
        sorted_stats = sorted(stats, key=lambda x: (not x['completed'], 
                                                   x['minutes_taken'] if x['completed'] else float('inf')))
        
        # 输出每只股票的信息
        for stat in sorted_stats:
            code = stat['code']
            initial_weight = f"{stat['initial_weight']:.4f}"
            target_weight = f"{stat['target_weight']:.4f}"
            
            # 确定操作类型
            if stat.get('is_sell', False):
                operation_type = "卖出"
            elif stat['initial_weight'] == 0 and stat['target_weight'] > 0:
                operation_type = "买入"
            elif stat['target_weight'] > stat['initial_weight']:
                operation_type = "加仓"
            elif stat['target_weight'] < stat['initial_weight']:
                operation_type = "减仓"
            else:
                operation_type = "持平"
            
            # 添加涨跌停状态
            limit_status = ""
            if stat.get('limit_up', False):
                limit_status = "涨停"
            elif stat.get('limit_down', False):
                limit_status = "跌停"
            
            if stat['completed']:
                completion_time = str(stat['completion_time'])
                minutes_taken = str(stat['minutes_taken'])
                status = stat['session']
            else:
                completion_time = "未完成"
                minutes_taken = "-"
                if 'fail_reason' in stat:
                    status = stat['fail_reason']
                else:
                    # 使用完成比例替代剩余差异
                    completion_ratio = stat.get('completion_ratio', 0)
                    status = f"已完成: {completion_ratio:.2f}%"
            
            row = f"| {code} | {initial_weight} | {target_weight} | {completion_time} | {minutes_taken} | {status} | {operation_type} | {limit_status} |"

        # 输出统计信息
        completed = [s for s in stats if s['completed']]
        not_completed = [s for s in stats if not s['completed']]
        sell_operations = [s for s in stats if s.get('is_sell', False)]
        limit_up_stocks = [s for s in stats if s.get('limit_up', False)]
        limit_down_stocks = [s for s in stats if s.get('limit_down', False)]
        
        print(f"{date} 总结: 总股票数 {len(stats)}, 已完成 {len(completed)}, 未完成 {len(not_completed)}, 卖出操作 {len(sell_operations)}, 涨停 {len(limit_up_stocks)}, 跌停 {len(limit_down_stocks)}")
        
        if completed:
            am_completed = [s for s in completed if s['session'] == '上午']
            pm_completed = [s for s in completed if s['session'] == '下午']
            avg_minutes = sum(s['minutes_taken'] for s in completed) / len(completed)
            
            print(f"上午完成: {len(am_completed)}, 下午完成: {len(pm_completed)}, 平均用时: {avg_minutes:.2f}分钟")
    
    print(f"分钟频权重调整完成，共生成 {len(minute_weights)} 条记录")
    print(f"结果已保存至 {output_file}")
    
    return minute_weights, completion_stats

if __name__ == "__main__":
    # 使用原始字符串r前缀或双反斜杠来避免转义问题
    minute_weights, completion_stats = adjust(
        input_file=r"D:\Derek\Code\Checker\csv\mon341.csv",  # 使用r前缀
        max_change=0.000001,
        limit_status_file=r"D:\Derek\Code\Checker\csv\limit_status.csv"  # 涨跌停状态文件
    )