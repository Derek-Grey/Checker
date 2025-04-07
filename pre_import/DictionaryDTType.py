import numpy as np
D1_1_dtype = np.dtype([
    ('date', 'S64'),
    ('code', 'S64'),
    ('code_jq','S64'),
    ('display_name','S64'),
    ('name','S64'),
    ('start_date','S64'),
    ('end_date','S64'),
    ('type','S64'),
    # ('margin','i4'), 
    # ('trade_days','i4'),
    # ('is_st','i4')
   
], align=True)

D1_1_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote',D1_1_dtype),
], align=True)


D1_2_dtype = np.dtype([
    ('date', 'S64'),
    ('indus_code', 'i4'),
    ('name','S64'),
    ('start_date','S64'),
    ('stocks', object)
   
], align=True)

D1_2_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote',D1_2_dtype),
], align=True)


D1_3_dtype = np.dtype([
    ('date', 'S64'),
    ('code', 'S64'),
    ('code_jq','S64'),
    ('open','f4'),
    ('close', 'f4'),
    ('high','f4'),
    ('low', 'f4'),
    ('high_limit','f4'),
    ('low_limit', 'f4'),
    ('volume','u8'),
    ('money','f8'),
    # ('paused','i4'),
    # ('factor','f4'), 

   
], align=True)

D1_3_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote',D1_3_dtype),
], align=True)


D1_4_dtype = np.dtype([
    ('code_jq','S64'),
    ('code', 'S64'),
    ('date', 'S64'),
    ('capitalization','f8'),
    ('circulating_cap','f8'),
    ('market_cap','f8'),
    ('circulating_market_cap','f8'),
    ('turnover_ratio','f4'),
    ('pe_ratio', 'f4'),
    ('pe_ratio_lyr','f4'),
    ('pb_ratio', 'f4'),
    ('ps_ratio','f4'),
    ('pcf_ratio', 'f4'),

], align=True)

D1_4_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote',D1_4_dtype),
], align=True)


D1_5_dtype = np.dtype([
    ('date', 'S64'),
    ('code','S64'),
    ('code_w', 'S64'),
    ('mkt_cap_ard','f8'),
    ('total_shares','f8'),
    ('free_float_shares','f8'),
    ('or_ttm','f8'),
    ('gr_ttm','f8'),
    ('operatecashflow_ttm', 'f8'),

], align=True)

D1_5_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote',D1_5_dtype),
], align=True)


D1_6_dtype = np.dtype([
    ('date', 'S64'),
    ('code','S64'),
    ('code_w', 'S64'),
    ('ipo_date', 'S64'),
    ('riskwarning','i4'),
    ('trade_days','i4'),
    ('trade_status','i4'),
    # ('sec_status','S64'),
    ('industry_csrc12_n','S64'),
    ('industry_gics','S256'),


], align=True)

D1_6_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote',D1_6_dtype),
], align=True)



D1_7_dtype = np.dtype([
    ('date', 'S64'),
    ('code','S64'),
    ('pct_chg', 'S64'),#这里要注意，这个是个string，下面都是double
    ('volume', 'f8'),
    ('amt', 'f8'),
    ('pre_close', 'f8'),
    ('close', 'f8'),
   
], align=True)

D1_7_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote',D1_7_dtype),
], align=True)



D1_8_dtype = np.dtype([
    ('date', 'S64'),
    ('code','S64'),
    ('pct_chg', 'S64'),#这里要注意，这个是个string，下面都是double
    ('volume', 'f8'),
    ('amt', 'f8'),
    ('pre_close', 'f8'),
    ('close', 'f8'),
    ('oi', 'f8'),
    ('if_basis', 'f8'),
   
], align=True)

D1_8_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote',D1_8_dtype),
], align=True)


D1_9_dtype = np.dtype([
    ('date', 'S64'),
    ('code','S64'),
    ('pct_chg_settlement', 'S64'),#这里要注意，这个是个string，下面都是double
    ('volume', 'f8'),
    ('amt', 'f8'),
    ('pre_settle', 'f8'),
    ('settle', 'f8'),
    ('oi', 'f8'),
    ('if_basis', 'f8'),
   
], align=True)

D1_9_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote',D1_9_dtype),
], align=True)

D1_10_dtype = np.dtype([
    ('date', 'S64'),
    ('code','S64'),
    ('code_w','S64'),
    ('rptdate','S64'),
    ('monetary_cap','f8'),  #这里要注意，f8没有把握
    ('minority_int', 'i8'),
    ('tot_equity', 'i8'),  
    ('tot_assets', 'i8'),
    ('tot_liab', 'i8'),
    ('stm_issuingdate', 'S64'),
    ('interestexpense_ttm','f8'),
    ('gc_ttm2', 'f8'),
    ('or_ttm2', 'f8'),
    ('netprofit_ttm2', 'f8'),
    ('ebit2_ttm', 'f8')
 
], align=True)

D1_10_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote',D1_10_dtype),
], align=True)
# 连接mongodb数据库

D1_11_dtype = np.dtype([
    ('date', 'S64'),
    ('code', 'S64'),
    ('code_w','S64'),
    ('pct_chg', 'f8'),
    ('volume', 'f8'),
], align=True)

D1_11_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote', D1_11_dtype),
], align=True)


D1_12_dtype = np.dtype([
    ('date', 'S64'),
    ('code','S64'),
    ('code_w','S64'),
    ('rptdate','S64'),
    ('stm_issuingdate','S64'),
    ('stmnote_tax','f8'),  #这里要注意，f8没有把握
    ('stmnote_audit_category', 'f8')
 
], align=True)

D1_12_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote',D1_12_dtype),
], align=True)


D2_1_dtype = np.dtype([
    ('date', 'S64'),
    ('time', 'S64'),
    ('code', 'S64'),
    ('code_jq','S64'),
    ('open','f4'),
    ('close', 'f4'),
    ('low', 'f4'),
    ('high','f4'),
    ('volume','u8'),
    ('money','f8'),
    ('avg','f4'),
    ('pre_day_close','f4'),
    ('today_close','f4'),
    ('chg_pre','S64'),
    ('chg_close','S64'), 

], align=True)

D2_1_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote',D2_1_dtype),
], align=True)