import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

# --- 基于最优组合的选股参数 ---
PICK_CONFIG = {
    'drop_limit': -6.0,        # 跌幅超过 6% (寻找极致深蹲)
    'limit_up_days': 9,        # 9日内必须有过涨停 (确保有妖性)
    'cost_bias': 1.15,         # 股价高于 MA20 15% (强势股回调而非弱势股阴跌)
    'vol_max_ratio': 0.7,      # 成交量萎缩至昨日 70% 以下 (地量出现)
    'vol_min_ratio': 0.3       # 排除成交极度匮乏的僵尸股
}

def clean_cols(df):
    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
    return df

def run_picker():
    print(f"🚀 启动选股引擎 [模型: D-6.0_L9_C1.2_V0.7]...")
    
    # 1. 加载代码名称映射
    name_dict = {}
    if os.path.exists('stock_names.csv'):
        n_df = pd.read_csv('stock_names.csv', sep=None, engine='python', encoding='utf_8_sig')
        n_df = clean_cols(n_df)
        name_dict = dict(zip(n_df['code'].astype(str).str.zfill(6), n_df['name']))

    # 2. 遍历数据
    files = glob.glob("stock_data/*.csv")
    final_picks = []

    for f in files:
        try:
            df = pd.read_csv(f, sep=None, engine='python', encoding='utf_8_sig')
            df = clean_cols(df)
            if len(df) < 65: continue
            
            df = df.sort_values('日期').reset_index(drop=True)
            curr = df.iloc[-1]   # 最新交易日
            prev = df.iloc[-2]   # 上一交易日
            
            # 计算指标
            df['MA20'] = df['收盘'].rolling(20).mean()
            df['MA60'] = df['收盘'].rolling(60).mean()
            df['is_limit_up'] = df['涨跌幅'] >= 9.9
            
            # --- 核心过滤逻辑 ---
            # 1. 趋势过滤：MA20在MA60之上，且今日收盘在MA60之上
            trend_ok = (df['MA20'].iloc[-1] > df['MA60'].iloc[-1]) and (curr['收盘'] > df['MA60'].iloc[-1])
            # 2. 9日内有涨停
            has_limit_up = df['is_limit_up'].tail(PICK_CONFIG['limit_up_days']).max()
            # 3. 极致缩量
            vol_r = curr['成交量'] / prev['成交量']
            vol_ok = PICK_CONFIG['vol_min_ratio'] < vol_r < PICK_CONFIG['vol_max_ratio']
            # 4. 跌幅门槛
            drop_ok = curr['涨跌幅'] <= PICK_CONFIG['drop_limit']
            # 5. 偏离度门槛 (强势回调)
            bias_ok = (curr['收盘'] / df['MA20'].iloc[-1]) > PICK_CONFIG['cost_bias']

            if trend_ok and has_limit_up and vol_ok and drop_ok and bias_ok:
                code = str(curr['股票代码']).zfill(6)
                final_picks.append({
                    '日期': curr['日期'],
                    '代码': code,
                    '名称': name_dict.get(code, "未知"),
                    '今日跌幅': f"{curr['涨跌幅']}%",
                    '成交量比': f"{vol_r:.2f}",
                    '偏离MA20': f"{((curr['收盘']/df['MA20'].iloc[-1])-1)*100:.1f}%",
                    '换手率': curr['换手率']
                })
        except:
            continue

    # 3. 保存并打印结果
    if final_picks:
        res_df = pd.DataFrame(final_picks)
        now = datetime.now()
        out_dir = now.strftime('%Y/%m')
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"{out_dir}/Daily_Picks_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        res_df.to_csv(out_name, index=False, encoding='utf_8_sig')
        
        print(f"\n✅ 选股完成！今日共命中 {len(final_picks)} 只极致深蹲个股：")
        print("-" * 60)
        print(res_df[['代码', '名称', '今日跌幅', '成交量比', '偏离MA20']].to_string(index=False))
        print("-" * 60)
    else:
        print("\n今日扫描完毕，无符合条件的极速回调个股。")

if __name__ == "__main__":
    run_picker()
