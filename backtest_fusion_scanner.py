import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import multiprocessing as mp
import pytz
import warnings

# 基础设置
warnings.filterwarnings('ignore')
np.seterr(invalid='ignore', divide='ignore')

CONF = {
    'MA_PERIOD': 60,        # 60日均线作为基准
    'DEVIATION_MIN': 20,    # D-8.0 对应偏差：偏离60日线至少 20%
    'VOL_UP': 4.0,          # V0.6 对应逻辑：成交量放大 4 倍
    'C_LIMIT': 1.06,        # C1.18 对应逻辑：收盘涨幅 >= 6%
    'RSI_MAX': 25,          # 极度超跌辅助：RSI < 25
    'DATA_DIR': "stock_data",
    'NAMES_FILE': "stock_names.csv",
    'OUT_DIR': "results"
}

def calculate_rsi(prices, n=6):
    if len(prices) < n+1: return np.array([50.0]*len(prices))
    deltas = np.diff(prices)
    up = np.where(deltas > 0, deltas, 0)
    down = np.where(deltas < 0, -deltas, 0)
    avg_up = pd.Series(up).ewm(com=n-1, adjust=False).mean()
    avg_down = pd.Series(down).ewm(com=n-1, adjust=False).mean()
    rs = avg_up / np.where(avg_down == 0, 1e-9, avg_down)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[50.0], rsi.values])

def process_single_file(f):
    """单文件扫描逻辑，供并行调用"""
    try:
        df = pd.read_csv(f)
        if len(df) < 65: return None
        
        # 提取基础信息
        last = df.iloc[-1]
        prev = df.iloc[-2]
        code = str(last['股票代码']).zfill(6)
        price = last['收盘']
        
        # --- 过滤逻辑 1: 板块、ST、价格 ---
        if code.startswith('30') or 'ST' in f.upper(): return None
        if not (CONF['MIN_PRICE'] <= price <= CONF['MAX_PRICE']): return None

        # --- 指标计算 ---
        close_arr = df['收盘'].values
        vol_arr = df['成交量'].values
        
        ma60 = pd.Series(close_arr).rolling(CONF['MA_PERIOD']).mean().values[-1]
        deviation = (ma60 - price) / price * 100
        
        vol_ma5 = pd.Series(vol_arr).shift(1).rolling(5).mean().values[-1]
        vol_ratio = last['成交量'] / (vol_ma5 if vol_ma5 > 0 else 1e-9)
        
        change = price / prev['收盘']
        rsi = calculate_rsi(close_arr)[-1]

        # --- 核心条件判断 (D-8.0 最优组合) ---
        if (deviation >= CONF['DEVIATION_MIN'] and 
            vol_ratio >= CONF['VOL_UP'] and 
            change >= CONF['C_LIMIT']):
            
            return {
                '代码': code,
                '价格': price,
                '偏离度': round(deviation, 2),
                '成交倍数': round(vol_ratio, 2),
                '今日涨幅%': round((change-1)*100, 2),
                'RSI': round(rsi, 2)
            }
    except:
        return None
    return None

def main():
    start_t = datetime.now()
    files = glob.glob(os.path.join(CONF['DATA_DIR'], "*.csv"))
    
    # 并行扫描
    print(f"🚀 启动并行引擎，扫描 {len(files)} 个文件...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_single_file, files)
    
    # 过滤空值并匹配名称
    hits = [r for r in results if r is not None]
    
    if hits:
        res_df = pd.DataFrame(hits)
        # 匹配名称
        if os.path.exists(CONF['NAMES_FILE']):
            names = pd.read_csv(CONF['NAMES_FILE'], dtype={'code': str})
            names['code'] = names['code'].str.zfill(6)
            name_map = names.set_index('code')['name'].to_dict()
            res_df.insert(1, '名称', res_df['代码'].map(name_map).fillna("未知"))
        
        # 保存结果
        os.makedirs(CONF['OUT_DIR'], exist_ok=True)
        beijing_t = datetime.now(pytz.timezone('Asia/Shanghai'))
        file_name = f"backtest_fusion_scanner_{beijing_t.strftime('%Y%m%d_%H%M%s')}.csv"
        out_path = os.path.join(CONF['OUT_DIR'], file_name)
        res_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 扫描成功！发现 {len(res_df)} 只符合条件的个股。")
        print(f"📄 结果已保存至: {out_path}")
        print(res_df.to_markdown(index=False))
    else:
        print("\n💡 今日未发现符合条件的个股。")
    
    print(f"⏱️ 总耗时: {datetime.now() - start_t}")

if __name__ == "__main__":
    main()
