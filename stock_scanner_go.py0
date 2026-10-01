import pandas as pd
import numpy as np
from datetime import datetime
import os
import pytz
import glob
from multiprocessing import Pool, cpu_count, Manager
import warnings

# 忽略计算中的无关警告
warnings.filterwarnings('ignore')

# ==================== 核心参数配置 (回测优化版) ====================
MIN_PRICE = 5.0              
MAX_AVG_TURNOVER_30 = 2.5    
MIN_VOLUME_RATIO = 0.5       # 点火需要一定的量比，调高至0.5
MAX_VOLUME_RATIO = 2.5       # 允许适度放量
MAX_TODAY_CHANGE = 2.5       
MIN_PROFIT_POTENTIAL = 15.0  # 距60日线空间

# --- 多周期共振阈值 ---
RSI6_MAX = 25                
RSI_WEEKLY_MAX = 35          # 周线超跌阈值
LOOKBACK_WINDOW = 250        # 回测打分参考最近一年数据
# =====================================================================

SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')
STOCK_DATA_DIR = 'stock_data'
NAME_MAP_FILE = 'stock_names.csv'

def calculate_indicators(df):
    """计算日线核心指标"""
    df = df.reset_index(drop=True)
    close = df['收盘']
    
    # 1. 均线
    df['ma5'] = close.rolling(5).mean()
    df['ma10'] = close.rolling(10).mean()
    df['ma60'] = close.rolling(60).mean()
    df['ma5_up'] = df['ma5'] >= df['ma5'].shift(1)
    
    # 2. RSI6
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    df['rsi6'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    
    # 3. KDJ
    low_9 = df['最低'].rolling(9).min()
    high_9 = df['最高'].rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    df['k_line'] = rsv.ewm(com=2).mean()
    df['d_line'] = df['k_line'].ewm(com=2).mean()
    df['kdj_gold'] = (df['k_line'] > df['d_line']) & (df['k_line'].shift(1) <= df['d_line'].shift(1))
    
    # 4. MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd_hist'] = (ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()) * 2
    df['macd_improving'] = df['macd_hist'] > df['macd_hist'].shift(1)

    # 5. 量能
    df['vol_ma5'] = df['成交量'].shift(1).rolling(5).mean()
    df['vol_ratio'] = df['成交量'] / df['vol_ma5']
    return df

def get_weekly_resonance(df_daily):
    """周线重采样及超跌判定"""
    df_temp = df_daily.copy()
    df_temp['日期'] = pd.to_datetime(df_temp['日期'])
    df_weekly = df_temp.resample('W', on='日期').agg({'收盘': 'last'}).dropna()
    
    delta = df_weekly['收盘'].diff()
    gain = delta.where(delta > 0, 0).rolling(6).mean()
    loss = abs(delta).rolling(6).mean()
    df_weekly['w_rsi6'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    return df_weekly.iloc[-1]['w_rsi6']

def evaluate_stock_power(df):
    """回测历史胜率打分 (最近12个月信号表现)"""
    if len(df) < LOOKBACK_WINDOW: return 50.0
    history = df.iloc[-LOOKBACK_WINDOW:].copy()
    # 定义历史上的信号点
    history['sig'] = (history['rsi6'] < 25) & (history['收盘'] > history['ma5'])
    sig_indices = history[history['sig']].index
    
    if len(sig_indices) < 2: return 50.0
    
    wins = 0
    total = 0
    for idx in sig_indices:
        if idx + 20 < len(df): # 以20日胜率为打分标准
            total += 1
            if df.iloc[idx + 20]['收盘'] > df.iloc[idx]['收盘']:
                wins += 1
    return (wins / total * 100) if total > 0 else 50.0

def process_stock(args):
    file_path, name_map, stats_dict = args
    stock_code = os.path.basename(file_path).split('.')[0]
    stock_name = name_map.get(stock_code, "未知")
    if "ST" in stock_name.upper(): return None

    try:
        df_raw = pd.read_csv(file_path)
        if len(df_raw) < 120: return None
        
        df = calculate_indicators(df_raw)
        latest = df.iloc[-1]
        
        # 基础过滤
        stats_dict['total'] += 1
        if latest['收盘'] < MIN_PRICE: return None
        
        potential = (latest['ma60'] - latest['收盘']) / latest['收盘'] * 100
        
        # 判定条件
        is_oversold_daily = latest['rsi6'] < RSI6_MAX
        is_ignition = latest['收盘'] > latest['ma5'] and latest['macd_improving'] and latest['vol_ratio'] > MIN_VOLUME_RATIO
        
        if is_oversold_daily and is_ignition:
            # 引入周线共振
            w_rsi6 = get_weekly_resonance(df_raw)
            is_weekly_resonance = w_rsi6 < RSI_WEEKLY_MAX
            
            # 引入历史胜率评分
            power_score = evaluate_stock_power(df)
            
            # 等级判定
            if is_weekly_resonance and power_score >= 55:
                grade = "SSS-战略共振"
            elif is_weekly_resonance:
                grade = "S-周线底启动"
            elif power_score >= 55:
                grade = "A-高胜率点火"
            else:
                grade = "B-普通超跌"
            
            return {
                '等级': grade,
                '代码': stock_code,
                '名称': stock_name,
                '现价': round(latest['收盘'], 2),
                '量比': round(latest['vol_ratio'], 2),
                '历史胜率': f"{round(power_score, 1)}%",
                '周线RSI': round(w_rsi6, 1),
                '距60日线': f"{round(potential, 1)}%",
                '今日涨跌': f"{round(latest['涨跌幅'], 1)}%"
            }
    except:
        return None
    return None

def main():
    now = datetime.now(SHANGHAI_TZ)
    print(f"🚀 多周期潜力等级扫描仪启动... ({now.strftime('%Y-%m-%d %H:%M')})")
    
    manager = Manager()
    stats_dict = manager.dict({'total': 0})
    
    name_map = {}
    if os.path.exists(NAME_MAP_FILE):
        n_df = pd.read_csv(NAME_MAP_FILE, dtype={'code': str})
        name_map = dict(zip(n_df['code'].str.zfill(6), n_df['name']))

    files = glob.glob(os.path.join(STOCK_DATA_DIR, '*.csv'))
    tasks = [(f, name_map, stats_dict) for f in files]

    with Pool(cpu_count()) as pool:
        results = pool.map(process_stock, tasks)

    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        df_res = pd.DataFrame(valid_results)
        # 排序：等级优先，其次是距60日线空间
        df_res = df_res.sort_values(by=['等级', '距60日线'], ascending=[True, False])
        
        print(f"\n🎯 选出潜力标的 ({len(valid_results)} 只):")
        print(df_res.to_string(index=False))
        
        os.makedirs("results", exist_ok=True)
        df_res.to_csv(f"results/精选潜力等级表_{now.strftime('%Y%m%d')}.csv", index=False, encoding='utf_8_sig')
    else:
        print("\n当前市场未发现满足“日周共振”的高潜力信号。")

if __name__ == "__main__":
    main()
