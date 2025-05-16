from pre_import.DictionaryDTType import D1_11_dtype, D1_11_numpy_dtype

def read_npq_file(file_path):
    """读取NPQ文件并返回DataFrame"""
    npq_data = np.fromfile(file_path, dtype=D1_11_numpy_dtype)
    quote = npq_data['quote']
    df = pd.DataFrame(quote)  
    df = df.astype(str)
    df = df[['date', 'code', 'pct_chg']]
    return df

def get_returns_from_db(self, start_date, end_date, codes):
    """从数据库获取收益率数据"""
    start_time = time.time()
    try:
        returns_data = []
        for date in pd.date_range(start=start_date, end=end_date):
            npq_file_path = Path(self.data_directory) / date.strftime('%Y-%m-%d') / "2" / "Min.npq"
            if not npq_file_path.exists():
                continue
            df = read_npq_file(str(npq_file_path))
            daily_returns = df[df['code'].isin(codes)]
            for _, record in daily_returns.iterrows():
                returns_data.append({'date': record['date'], 'code': record['code'], 'return': float(record['pct_chg'])})
        returns = pd.DataFrame(returns_data)
    
        print(f"获取收益率数据总耗时: {time.time() - start_time:.2f}秒\n")
        return returns
    except Exception as e:
        raise Exception(f"从NPQ文件获取收益率数据时出错: {str(e)}")