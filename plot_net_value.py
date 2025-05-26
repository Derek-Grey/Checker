import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import os
import plotly.graph_objects as go
import logging
import plotly.colors as colors

# 设置中文字体
try:
    font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
except:
    font = FontProperties()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrategyPlotter:
    """策略绘图类"""
    def __init__(self, output_dir='output'):
        """初始化绘图类，创建输出目录"""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_net_value(self, df: pd.DataFrame, strategy_name: str):
        """绘制净值曲线图"""
        df = df.copy()
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        elif 'index' in df.columns and df['index'].dtype == 'datetime64[ns]':
            df.set_index('index', inplace=True)
            
        # 确保索引为DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        if len(df) > 0:
            start_date = df.index[0]
        else:
            logger.error("DataFrame为空")
            return

        # 检查必要的列是否存在
        if 'net_value' not in df.columns:
            logger.error("DataFrame 不包含 'net_value' 列。")
            return
            
        # 计算回撤
        self._calculate_drawdown(df)
        
        # 计算统计指标
        stats = self._calculate_statistics(df)
        
        # 绘制图表
        self._create_plot(df, strategy_name, start_date, stats)

    def _calculate_drawdown(self, df: pd.DataFrame):
        """计算最大回撤（直接使用已有净值列）"""
        if 'net_value' not in df.columns:
            raise ValueError("DataFrame必须包含net_value列")
            
        # 计算最大净值和回撤
        dates = df.index.unique().tolist()
        for date in dates:
            df.loc[date, 'max_net'] = df.loc[:date].net_value.max()
        df['back_net'] = df['net_value'] / df['max_net'] - 1
    
    def _calculate_statistics(self, df: pd.DataFrame):
        """计算统计指标（基于已有净值）"""
        if 'net_value' not in df.columns:
            raise ValueError("需要net_value列进行统计计算")
            
        s_ = df.iloc[-1]
        return {
            'annualized_return': format(s_.net_value ** (252 / df.shape[0]) - 1, '.2%'),
            'monthly_volatility': format(df.net_value.pct_change().std() * 21 ** 0.5, '.2%'),
            'end_date': s_.name
        }
    
    def _create_plot(self, df, strategy_name, start_date, stats):
        """创建图表"""
        # 创建净值和回撤的plotly图形对象
        g1 = go.Scatter(x=df.index.unique().tolist(), y=df['net_value'], name='策略净值')
        g2 = go.Scatter(x=df.index.unique().tolist(), y=df['back_net'] * 100, name='回撤', xaxis='x', yaxis='y2', mode="none",
                        fill="tozeroy")
        
        # 如果存在指数净值列，则添加到图表
        if 'index_net_value' in df.columns:
            g3 = go.Scatter(x=df.index.unique().tolist(), y=df['index_net_value'], name='日频净值')
            data = [g1, g2, g3]
        else:
            data = [g1, g2]

        # 修正后的图表配置
        fig = go.Figure(
            data=data,
            layout={
                'height': 1122,
                "title": f"{strategy_name}策略，<br>净值（左）& 回撤（右），<br>全期：{start_date} ~ {stats['end_date']}，<br>年化收益：{stats['annualized_return']}，月波动：{stats['monthly_volatility']}",
                "font": {"size": 22},
                "yaxis": {"title": "净值", "side": "left"},
                "yaxis2": {
                    "title": "最大回撤",
                    "side": "right",
                    "overlaying": "y",
                    "ticksuffix": "%",
                    "showgrid": False
                },
                "xaxis": {
                    "title": "交易日序列",
                    "type": "category",
                    "tickmode": "array",
                    "tickvals": df.index[::len(df)//10],
                    "ticktext": df.index.strftime('%Y-%m-%d')[::len(df)//10]
                },
                "legend": {"x": 0, "y": 1},
                "hovermode": "x unified"
            }
        )
        
        # 保存图表
        output_path = os.path.join(self.output_dir, f"{strategy_name}_net_value.html")
        fig.write_html(output_path)
        logger.info(f"图表已保存至: {output_path}")
        
        # 显示图表
        fig.show()
        
    def plot_multiple_net_values(self, file_paths, output_filename="net_value_comparison"):
        """绘制多个文件的净值对比图"""
        fig = go.Figure()
        
        # 记录所有策略的开始和结束日期
        all_start_dates = []
        all_end_dates = []
        
        # 定义策略名称映射
        strategy_name_map = {
            '20250526_100254': '分钟0.000001净值',
            '20250523_142247': '分钟0.0001净值',
            '20250523_141923': '分钟0.00001净值'
        }
        
        # 定义颜色映射，按照用户指定的顺序
        color_map = {
            '分钟0.00001净值': '#1f77b4',  # 蓝色
            '分钟0.0001净值': '#d62728',   # 红色
            '分钟0.000001净值': '#2ca02c'  # 绿色
        }
        
        # 处理每个文件
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                continue
                
            # 从文件名中提取标识
            file_name = os.path.basename(file_path)
            file_id = file_name.replace('daily_summary_', '').replace('.csv', '')
            
            # 获取对应的策略名称，如果没有映射则使用文件ID
            strategy_name = strategy_name_map.get(file_id, file_id)
            
            # 选择颜色
            color = color_map.get(strategy_name, '#ff7f0e')  # 默认使用橙色
            
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 确保date列是日期格式
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # 检查必要的列
                    if 'net_value' in df.columns:
                        # 添加净值曲线到图表
                        fig.add_trace(go.Scatter(
                            x=df['date'],
                            y=df['net_value'],
                            name=strategy_name,
                            mode='lines',
                            line=dict(color=color, width=3)  # 使用实线，加粗线宽
                        ))
                        
                        # 记录日期范围
                        all_start_dates.append(df['date'].min())
                        all_end_dates.append(df['date'].max())
                        
                        # 如果存在指数净值列，且是第一个文件，则添加到图表
                        if 'index_net_value' in df.columns and file_path == file_paths[0]:
                            fig.add_trace(go.Scatter(
                                x=df['date'],
                                y=df['index_net_value'],
                                name='日频净值',
                                mode='lines',
                                line=dict(color='black', width=3)  # 使用实线黑色，加粗线宽
                            ))
                    else:
                        logger.error(f"文件 {file_name} 中没有找到 'net_value' 列")
                else:
                    logger.error(f"文件 {file_name} 中没有找到 'date' 列")
            except Exception as e:
                logger.error(f"处理文件 {file_name} 时出错: {e}")
        
        # 设置图表布局
        if all_start_dates and all_end_dates:
            start_date = min(all_start_dates)
            end_date = max(all_end_dates)
            
            fig.update_layout(
                title=f"策略净值对比图<br>全期：{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}",
                font=dict(size=18),
                xaxis=dict(
                    title="日期",
                    tickformat="%Y-%m-%d",
                    gridcolor='rgba(220,220,220,0.5)'  # 浅灰色网格线
                ),
                yaxis=dict(
                    title="净值",
                    gridcolor='rgba(220,220,220,0.5)'  # 浅灰色网格线
                ),
                legend=dict(
                    x=0.01, 
                    y=0.99, 
                    bgcolor='rgba(255,255,255,0.8)',  # 半透明白色背景
                    bordercolor='rgba(200,200,200,0.7)',  # 浅灰色边框
                    borderwidth=1
                ),
                hovermode="x unified",
                height=900,  # 增加高度
                width=1600,  # 增加宽度
                autosize=True,  # 自动调整大小
                margin=dict(l=50, r=50, t=100, b=50),  # 调整边距
                template="plotly_white",  # 使用白色模板，更清晰
                plot_bgcolor='rgba(250,250,250,0.9)',  # 浅灰色背景
                paper_bgcolor='white'  # 白色纸张背景
            )
            
            # 保存图表
            output_path = os.path.join(self.output_dir, f"{output_filename}.html")
            fig.write_html(output_path, config={'responsive': True})  # 添加响应式支持
            logger.info(f"对比图表已保存至: {output_path}")
            
            # 显示图表
            fig.show(config={'responsive': True})
        else:
            logger.error("没有有效的数据可以绘制")

# 主函数
def main():
    # 文件路径
    file_paths = [
        r'D:\Derek\Code\Checker\output\daily_summary_20250523_142247.csv',
        r'D:\Derek\Code\Checker\output\daily_summary_20250523_141923.csv',
        r'D:\Derek\Code\Checker\output\daily_summary_20250526_100254.csv'
    ]
    
    # 创建绘图器实例
    plotter = StrategyPlotter(output_dir=r'D:\Derek\Code\Checker\output')
    
    # 绘制净值对比图
    plotter.plot_multiple_net_values(file_paths)


if __name__ == "__main__":
    main()