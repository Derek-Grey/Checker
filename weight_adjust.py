class PortfolioWeightAdjuster:
    def __init__(self, df, change_limit, data_directory):
        self.df = df.copy()
        self.change_limit = change_limit
        self.data_source = DataSource(data_directory)
        
        # 创建完整时间戳
        self.df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
        self.unique_datetimes = self.df['datetime'].unique()
        self.codes = self.df['code'].unique()

    def adjust_weights_over_time(self):
        adjusted_df = pd.DataFrame()
        
        # 按时间点顺序处理
        for dt in sorted(self.unique_datetimes):
            current_data = self.df[self.df['datetime'] == dt]
            
            # 获取市场状态
            limit_status = self.data_source.get_limit_status(dt.date(), current_data['code'])
            trade_status = self.data_source.get_trade_status(dt.date(), current_data['code'])
            
            # 计算权重调整
            current_data['weight_change'] = current_data['weight'] - current_data.groupby('code')['weight'].shift(1).fillna(0)
            
            # 应用调整限制
            current_data['adjustable'] = current_data.apply(lambda row: 
                self.data_source.can_adjust_weight(row['code'], row['weight_change'], limit_status, trade_status), axis=1)
            
            current_data['adjusted_weight'] = current_data.apply(lambda row:
                row['weight'] if row['adjustable'] else 
                np.clip(row['weight'], 
                        row['weight'] - self.change_limit, 
                        row['weight'] + self.change_limit), axis=1)
            
            adjusted_df = pd.concat([adjusted_df, current_data])
        
        return adjusted_df[['date', 'time', 'code', 'adjusted_weight']].rename(
            columns={'adjusted_weight': 'weight'})