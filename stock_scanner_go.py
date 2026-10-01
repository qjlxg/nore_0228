import pandas as pd
from datetime import datetime
import os
import pytz
import glob
from multiprocessing import Pool, cpu_count, Manager
import numpy as np

# ==================== 2026“最优确权”冠军参数 (基于584万次特征寻优) ===================
MIN_PRICE = 5.0              # 股价门槛
MAX_AVG_TURNOVER_30 = 3.5    # 适当放宽换手率，以兼容1.2倍量比的活跃度

# --- 寻优得出的核心阈值 ---
RSI6_MAX = 30                # 冠军RSI阈值：平衡了力度与机会30
KDJ_K_MAX = 50               # 冠军K值：要求极度超卖20
MIN_PROFIT_POTENTIAL = 1    # 冠军空间：距60日线至少25%回扣空间
BEST_VOL_RATIO = 1.1         # 寻优显示1.2最优，实战设1.1触发点火

# --- 确权微调参数 ---
MAX_SLOPE = 0.05             # 斜率寻优结果为0，实战允许 ±0.05 的走平波动
# =====================================================================

SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')
STOCK_DATA_DIR = 'stock_data'
NAME_MAP_FILE = 'stock_names.csv' 

def calculate_indicators(df):
    df = df.reset_index(drop=True)
    df = df[df['收盘'] > 0].copy()
    close = df['收盘']
    
    # RSI6
    delta = close.diff()
    g = (delta.where(delta > 0, 0)).rolling(6).mean()
    l = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df['rsi6'] = 100 - (100 / (1 + (g / l.replace(0, np.nan))))
    
    # KDJ (9,3,3)
    l9, h9 = df['最低'].rolling(9).min(), df['最高'].rolling(9).max()
    rsv = (close - l9) / (h9 - l9).replace(0, np.nan) * 100
    df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    
    # MA & MACD
    df['ma5'], df['ma60'] = close.rolling(5).mean(), close.rolling(60).mean()
    df['ema12'], df['ema26'] = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df['macd_hist'] = (df['ema12'] - df['ema26'] - (df['ema12'] - df['ema26']).ewm(span=9, adjust=False).mean()) * 2
    df['macd_improving'] = df['macd_hist'] > df['macd_hist'].shift(1)
    
    # 量能与斜率 (寻优关键因子)
    df['vol_ma5'] = df['成交量'].rolling(5).mean().shift(1) # 基准量：前5日均量
    df['vol_ratio'] = df['成交量'] / df['vol_ma5'].replace(0, np.nan)
    df['slope'] = df['ma5'].diff() # 5日线斜率
    df['potential'] = ((df['ma60'] - close) / close.replace(0, np.nan)) * 100
    
    # 确权逻辑
    df['not_lowest'] = close > df['最低'].shift(1).rolling(2).min()
    df['above_ma5'] = close > df['ma5']
    
    return df

def process_single_stock(args):
    file_path, name_map, stats = args
    stock_code = os.path.basename(file_path).split('.')[0]
    try:
        df = pd.read_csv(file_path)
        if len(df) < 100: return None
        
        stats['total_scanned'] += 1
        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        # 1. 冠军门槛过滤 (RSI/空间/K值)
        if last['收盘'] < MIN_PRICE: stats['fail_price'] += 1; return None
        if last['rsi6'] > RSI6_MAX: stats['fail_rsi'] += 1; return None
        if last['potential'] < MIN_PROFIT_POTENTIAL: stats['fail_potential'] += 1; return None
        if last['kdj_k'] > KDJ_K_MAX: stats['fail_kdj'] += 1; return None

        # 2. 核心确权过滤 (寻优结果：斜率接近0 + 站上5日线)
        if not (last['above_ma5'] and last['not_lowest']):
            stats['fail_confirm'] += 1; return None
        if abs(last['slope']) > MAX_SLOPE: 
            stats['fail_slope'] += 1; return None

        # 3. 逻辑分类 (基于寻优出的量能与MACD特征)
        res_type = ""
        # 模式A：点火反弹 (量比 > 1.1 + MACD改善)
        if last['vol_ratio'] >= BEST_VOL_RATIO and last['macd_improving']:
            res_type = "A-点火反弹"
        # 模式B：极致企稳 (RSI < 25 + 斜率极小)
        elif last['rsi6'] < 25 and abs(last['slope']) < 0.02:
            res_type = "B-极致企稳"
        
        if res_type:
            return {
                '代码': stock_code,
                '名称': name_map.get(stock_code, "未知"),
                '类型': res_type,
                '现价': last['收盘'],
                '涨跌幅': f"{last['涨跌幅']}%",
                'RSI6': round(last['rsi6'], 1),
                'K值': round(last['kdj_k'], 1),
                '量比': round(last['vol_ratio'], 2),
                '反弹空间': f"{round(last['potential'], 1)}%",
                '更新时间': last['日期']
            }
    except Exception: pass
    return None

def main():
    now_sh = datetime.now(SHANGHAI_TZ)
    manager = Manager()
    stats = manager.dict({
        'total_scanned': 0, 'fail_price': 0, 'fail_rsi': 0, 
        'fail_potential': 0, 'fail_kdj': 0, 'fail_confirm': 0, 'fail_slope': 0
    })

    # 读取名称
    n_df = pd.read_csv(NAME_MAP_FILE, dtype={'code': str})
    name_map = dict(zip(n_df['code'].str.zfill(6), n_df['name']))

    file_list = glob.glob(os.path.join(STOCK_DATA_DIR, '*.csv'))
    tasks = [(f, name_map, stats) for f in file_list]

    with Pool(processes=cpu_count()) as pool:
        results = [r for r in pool.map(process_single_stock, tasks) if r is not None]

    # --- 输出诊断与结果 ---
    print("\n" + "="*60)
    print(f"📊 冠军版扫描报告 (寻优参数集成) | {now_sh.strftime('%Y-%m-%d')}")
    print("-" * 60)
    print(f"1. 扫描总数: {stats['total_scanned']} 只")
    print(f"2. 空间不足 (距60线<25%): {stats['fail_potential']} 只")
    print(f"3. 确权失败 (未站上5日线): {stats['fail_confirm']} 只")
    print(f"4. 减速失败 (5日线未走平): {stats['fail_slope']} 只")
    print(f"5. 最终入选: {len(results)} 只")
    print("="*60)

    if results:
        res_df = pd.DataFrame(results).sort_values(by=['类型', '反弹空间'], ascending=[True, False])
        print(res_df.to_string(index=False))
        res_df.to_csv(f"champion_pick_{now_sh.strftime('%m%d')}.csv", index=False, encoding='utf_8_sig')
        print(f"\n✅ 结果已保存至: champion_pick_{now_sh.strftime('%m%d')}.csv")
    else:
        print("💡 今日暂无符合“冠军寻优参数”的个股，建议继续观望。")

if __name__ == '__main__':
    main()
