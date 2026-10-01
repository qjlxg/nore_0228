import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

# 战法名称：多头起航回调战法
# 逻辑说明：
# 1. 趋势过滤：5, 10, 20, 60日均线多头排列（或收盘价在60日线上方），确保处于上升通道。
# 2. 价格过滤：5.0元 < 最新收盘价 < 20.0元，排除高价股和垃圾股。
# 3. 板块过滤：排除ST股，排除30开头的创业板（仅限沪深主板）。
# 4. 回调逻辑：当前价格较近期高点有一定回落，但仍受均线支撑（如在20日或30日线附近）。
# 5. 量能逻辑：回调过程中成交量萎缩，代表抛压减弱，主力未大规模撤离。

def analyze_stock(file_path):
    try:
        # 读取数据
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        if df.empty or len(df) < 60:
            return None
        
        # 基础信息获取
        code = str(df['股票代码'].iloc[-1]).zfill(6)
        
        # 1. 排除创业板(30开头)和ST（假设文件名或代码不含ST，基础过滤）
        if code.startswith('30'):
            return None
        
        last_row = df.iloc[-1]
        close_price = last_row['收盘']
        
        # 2. 价格区间过滤
        if not (5.0 <= close_price <= 20.0):
            return None
            
        # 3. 计算均线
        df['MA5'] = df['收盘'].rolling(5).mean()
        df['MA10'] = df['收盘'].rolling(10).mean()
        df['MA20'] = df['收盘'].rolling(20).mean()
        df['MA60'] = df['收盘'].rolling(60).mean()
        
        curr_ma5 = df['MA5'].iloc[-1]
        curr_ma10 = df['MA10'].iloc[-1]
        curr_ma20 = df['MA20'].iloc[-1]
        curr_ma60 = df['MA60'].iloc[-1]
        
        # 4. 战法条件筛选
        # 条件A：中长期趋势向上 (收盘价在60日线上方，且60日线走平或向上)
        cond_trend = close_price > curr_ma60 and curr_ma60 >= df['MA60'].iloc[-5]
        
        # 条件B：处于多头排列后的回调（收盘价靠近20日线，但未跌破30日线）
        # 模拟“天津普林”形态：缩量回踩
        cond_support = curr_ma20 * 0.98 <= close_price <= curr_ma20 * 1.05
        
        # 条件C：量能萎缩 (今日成交量小于过去5日平均成交量)
        avg_vol_5 = df['成交量'].iloc[-6:-1].mean()
        cond_vol = last_row['成交量'] < avg_vol_5
        
        if cond_trend and cond_support and cond_vol:
            return {
                'code': code,
                'close': close_price,
                'pct_chg': last_row['涨跌幅'],
                'turnover': last_row['换手率']
            }
    except Exception as e:
        return None
    return None

def main():
    # 获取所有csv文件
    files = glob.glob('stock_data/*.csv')
    
    # 并行处理提高速度
    results = []
    with ProcessPoolExecutor() as executor:
        for result in executor.map(analyze_stock, files):
            if result:
                results.append(result)
    
    if not results:
        print("今日无符合条件股票")
        return

    # 匹配名称
    try:
        names_df = pd.read_csv('stock_names.csv', sep='\t', dtype={'code': str})
        # 转换为字典方便查询
        name_dict = dict(zip(names_df['code'].str.zfill(6), names_df['name']))
    except:
        name_dict = {}

    final_data = []
    for item in results:
        item['name'] = name_dict.get(item['code'], "未知")
        final_data.append(item)

    # 输出结果
    output_df = pd.DataFrame(final_data)
    output_df = output_df[['code', 'name', 'close', 'pct_chg', 'turnover']]
    
    # 路径处理
    now = datetime.now()
    dir_path = now.strftime('%Y%m')
    os.makedirs(dir_path, exist_ok=True)
    file_name = f"trend_pullback_strategy_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    
    save_path = os.path.join(dir_path, file_name)
    output_df.to_csv(save_path, index=False, encoding='utf_8_sig')
    print(f"筛选完成，结果保存至: {save_path}")

if __name__ == '__main__':
    main()
