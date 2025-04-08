import sys
sys.path.append('/home/sdq/Code/Test')  # 修改为个人的工作路径
from data.DictionaryDTType import D1_11_dtype, D1_11_numpy_dtype
from dsdd import cal_backtestV1_1

def main():
    data_directory = '/home/sdq/Code/Test/data'         #数据路径名
    frequency = 'daily'       #数据频率，日频为daily，分钟频为minute
    #股票表路径名，csv格式，第一列为日期，第二列为股票代码，第三列为权重（等权重的话可以没有第三列）    
    stock_path = '/home/sdq/Code/Test/csv/test_daily_weight.csv'      
    return_file = None        #日频为daily无需输入收益率表，分钟频暂时还需输入收益率表
    use_equal_weights = True      #是否使用等权，True为等权，False为不使用等权重
    
    result = cal_backtestV1_1.backtest(data_directory, frequency, stock_path, return_file, use_equal_weights)
    print("Result from main():", result)

if __name__ == "__main__":
    main()

