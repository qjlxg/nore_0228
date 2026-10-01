import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime

# ==================== 实战黄金参数配置 ====================
CONF = {
    'MA_PERIOD': 60,        # 选股之魂：60日均线支撑
    'VOL_UP': 3.5,          # 爆发力：3.5倍放量 3.5 倍量才是真主力，1.5 倍噪音太多
    'VOL_REDUCE': 0.5,      # 洗盘标准：地量必须缩到0.5（固定）
    'SUPPORT_LVL': 0.382,   # 强势位：仅允许回踩38.2%
    'MAX_GAP': 3,           # 窗口期：单阳后1-6天内
    'MIN_PRICE': 4.0,       # 价格下限
    'MAX_PRICE': 90.0       # 价格上限（适度放宽到中价股）
}
# ========================================================

def run_golden_strategy():
    target_dir = os.path.join('results', 'Golden_Strategy')
    os.makedirs(target_dir, exist_ok=True)
    
    # --- 1. 加载代码映射 ---
    name_map = {}
    if os.path.exists('stock_names.csv'):
        for enc in ['utf-8-sig', 'gbk', 'utf-8']:
            try:
                names_df = pd.read_csv('stock_names.csv', dtype={'code': str}, encoding=enc)
                names_df['code'] = names_df['code'].str.zfill(6)
                name_map = dict(zip(names_df['code'], names_df['name']))
                break
            except: continue

    files = glob.glob('stock_data/*.csv')
    results = []
    
    print(f"🚀 开始扫描全市场信号... 采用 MA{CONF['MA_PERIOD']} 黄金组合")

    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            df = df[df['收盘'] > 0.01].copy()
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)
            
            # 计算 MA60 和 斜率
            df['ma'] = df['收盘'].rolling(window=CONF['MA_PERIOD']).mean()
            if len(df) < CONF['MA_PERIOD'] + 5: continue
            
            last_idx = len(df) - 1
            curr_row = df.iloc[last_idx]
            code = os.path.basename(f).split('.')[0]
            
            # 过滤 A: 趋势过滤 (价格在MA上方且MA向上)
            if curr_row['收盘'] < curr_row['ma']: continue
            if df['ma'].iloc[last_idx] <= df['ma'].iloc[last_idx-3]: continue # 斜率过滤

            # 过滤 B: 价格空间
            if not (CONF['MIN_PRICE'] <= curr_row['收盘'] <= CONF['MAX_PRICE']): continue

            # 回溯窗口寻找符合条件的单阳
            for gap in range(1, CONF['MAX_GAP'] + 1):
                y_idx = last_idx - gap
                if y_idx < 1: break
                
                row_yang = df.iloc[y_idx]
                prev_row = df.iloc[y_idx - 1]
                adj_pd = df.iloc[y_idx + 1 : last_idx + 1] # 包含今天在内的调整期
                
                # 1. 单阳品质：涨幅 >= 5% 且 无长上影
                body = row_yang['收盘'] - row_yang['开盘']
                if row_yang['涨跌幅'] < 5.0 or body <= 0: continue
                if (row_yang['最高'] - row_yang['收盘']) > body * 0.4: continue # 拒绝避雷针

                # 2. 量能博弈：阳线倍量，调整期极致缩量
                cond_vol_up = row_yang['成交量'] >= prev_row['成交量'] * CONF['VOL_UP']
                # 关键：今天或回调均量 达到地量标准
                cond_vol_down = adj_pd['成交量'].mean() <= row_yang['成交量'] * CONF['VOL_REDUCE']

                # 3. 支撑博弈：Fibonacci 0.382 支撑位
                # 支撑价 = 阳线开盘 + 实体 * (1 - 0.382)
                support_price = row_yang['开盘'] + body * (1 - CONF['SUPPORT_LVL'])
                cond_price = adj_pd['最低'].min() >= support_price

                if cond_vol_up and cond_vol_down and cond_price:
                    results.append({
                        '代码': code,
                        '名称': name_map.get(code, '未知'),
                        '调整天数': gap,
                        '现价': round(curr_row['收盘'], 2),
                        '单阳涨幅': f"{row_yang['涨跌幅']}%",
                        '地量比': round(adj_pd['成交量'].mean() / row_yang['成交量'], 2),
                        '止损参考': round(row_yang['最低'] * 0.99, 2), # 阳线最低价下方1%
                        '建议关注': '🔥极强' if gap <= 3 else '稳健'
                    })
                    break
        except: continue

    # --- 3. 结果输出 ---
    if results:
        res_df = pd.DataFrame(results).sort_values(by=['建议关注', '地量比'])
        output_file = f"Selection_{datetime.now().strftime('%Y%m%d')}.csv"
        res_df.to_csv(output_file, index=False, encoding='utf_8_sig')
        print(f"\n✅ 选股完成！今日黄金信号共: {len(results)} 只")
        print("-" * 60)
        print(res_df.to_string(index=False))
        print("-" * 60)
        print(f"💡 建议：优先关注‘地量比’更低（筹码更干）且处于‘极强’类型的标的。")
    else:
        print("💡 今日全市场未发现符合‘黄金参数组合’的标的，空仓也是一种战斗。")

if __name__ == '__main__':
    run_golden_strategy()
