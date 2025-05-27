import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
st.set_page_config(layout="wide")

file_paths = [
    r'D:/Derek/Code/Checker/output/daily_summary_20250526_155533.csv',
    r'D:/Derek/Code/Checker/output/daily_summary_20250526_155755.csv',
    r'D:/Derek/Code/Checker/output/daily_summary_20250526_160000.csv'
]
strategy_name_map = {
    '20250526_155533': '分钟0.0001净值',
    '20250526_155755': '分钟0.00001净值',
    '20250526_160000': '分钟0.000001净值'
}
color_map = {
    '分钟0.0001净值': '#d62728',
    '分钟0.00001净值': '#1f77b4',
    '分钟0.000001净值': '#2ca02c'
}

fig = go.Figure()
all_start_dates = []
all_end_dates = []
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    file_id = file_name.replace('daily_summary_', '').replace('.csv', '')
    strategy_name = strategy_name_map.get(file_id, file_id)
    color = color_map.get(strategy_name, '#ff7f0e')
    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        if 'net_value' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['net_value'],
                name=strategy_name,
                mode='lines',
                line=dict(color=color, width=3)
            ))
            all_start_dates.append(df['date'].min())
            all_end_dates.append(df['date'].max())
            if 'index_net_value' in df.columns and file_path == file_paths[0]:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['index_net_value'],
                    name='日频净值',
                    mode='lines',
                    line=dict(color='black', width=3)
                ))

if all_start_dates and all_end_dates:
    start_date = min(all_start_dates)
    end_date = max(all_end_dates)
    fig.update_layout(
        title=f"策略净值对比图<br>全期：{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}",
        font=dict(size=18),
        xaxis=dict(title="日期", tickformat="%Y-%m-%d"),
        yaxis=dict(title="净值"),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(200,200,200,0.7)', borderwidth=1),
        hovermode="x unified",
        height=900,
        width=2000,  
        autosize=True,
        margin=dict(l=50, r=50, t=100, b=50),
        template="plotly_white",
        plot_bgcolor='rgba(250,250,250,0.9)',
        paper_bgcolor='white'
    )

st.title('策略净值对比图')
st.plotly_chart(fig)

st.header("权重完成度原始表格展示")
csv_paths = [
    r"D:\Derek\Code\Checker\output\weight_completion_stats_20250526_160244.csv",
    r"D:\Derek\Code\Checker\output\weight_completion_stats_20250526_160300.csv",
    r"D:\Derek\Code\Checker\output\weight_completion_stats_20250526_160317.csv"
]
table_names = ["分钟权重0.0001", "分钟权重0.00001", "分钟权重0.000001"]
table_colors = ["#ffeaea", "#eaf6ff", "#eaffea"]  
for path, name, color in zip(csv_paths, table_names, table_colors):
    st.subheader(f"表格：{name}")
    df = pd.read_csv(path)
    show_cols = [col for col in ['date','code','completed','completion_time','completion_ratio'] if col in df.columns]
    styled_df = df[show_cols].style.set_properties(**{'background-color': color})
    st.dataframe(styled_df, use_container_width=True)

#streamlit run d:/Derek/Code/Checker/streamlit_app.py