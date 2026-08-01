# -*- coding: utf-8 -*-
# ================================================
# 大盤風險預警系統 V1.1 (SPY Downside Risk Radar)
# 目標: 未來 10 個交易日內最大回撤 > 3%
# 驗證: 捕捉率 / 假警報率 / 防禦效益回測
# V1.1 修正: 全面改用 f-string, 消除 % 格式化衝突
# ================================================
import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

HORIZON = 10
DD_THRESHOLD = 0.03
HIGH = 0.65
WATCH = 0.35
N_FOLDS = 5
CAPTURE_MIN = 0.60
EDGE_MIN = 0.10

st.set_page_config(page_title="Market Risk Radar", page_icon="🛡", layout="centered")
st.title("大盤風險預警系統 (SPY)")

st.sidebar.header("系統參數")
WARN = st.sidebar.slider("警戒閾值", 0.35, 0.70, 0.50, 0.01)
if st.sidebar.button("強制刷新數據"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data(ttl=3600, show_spinner="正在下載市場數據...")
def load_data():
    raw = None
    for attempt in range(3):
        try:
            d = yf.download(["SPY", "^VIX", "^VIX3M", "HYG", "IEF"],
                            start="2018-01-01", progress=False, threads=False)
            if d is not None and len(d) > 500 and not d["Close"]["SPY"].dropna().empty:
                raw = d
                break
        except Exception:
            pass
        time.sleep(5)
    if raw is None:
        return pd.DataFrame()

    df = pd.DataFrame(index=raw.index)
    df["SPY_Close"] = raw["Close"]["SPY"]
    df["SPY_Low"] = raw["Low"]["SPY"]
    df["VIX"] = raw["Close"]["^VIX"]
    df["VIX3M"] = raw["Close"]["^VIX3M"]
    df["HYG"] = raw["Close"]["HYG"]
    df["IEF"] = raw["Close"]["IEF"]
    df = df.ffill().dropna(subset=["SPY_Close"])

    df["VIX_Term"] = df["VIX"] / df["VIX3M"]
    df["VIX_MA10_Ratio"] = df["VIX"] / df["VIX"].rolling(10).mean()
    df["Credit_Mom"] = (df["HYG"] / df["IEF"]).pct_change(20)
    df["DD_From_60H"] = df["SPY_Close"] / df["SPY_Close"].rolling(60).max() - 1
    ret = df["SPY_Close"].pct_change()
    df["RV_Ratio"] = ret.rolling(20).std() / ret.rolling(60).std()

    delta = df["SPY_Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["SPY_RSI"] = 100 - (100 / (1 + gain / loss))

    fwd_low = df["SPY_Low"].iloc[::-1].rolling(HORIZON, min_periods=1).min().iloc[::-1].shift(-1)
    df["Fwd_DD"] = fwd_low / df["SPY_Close"] - 1
    df["Target"] = (df["Fwd_DD"] < -DD_THRESHOLD).astype(int)
    return df

FEATURES = ["VIX_Term", "VIX_MA10_Ratio", "Credit_Mom",
            "DD_From_60H", "RV_Ratio", "SPY_RSI"]

def make_model():
    return XGBClassifier(n_estimators=200, max_depth=4,
                         learning_rate=0.03, random_state=42,
                         eval_metric="logloss")

@st.cache_data(ttl=3600, show_spinner="正在執行 Walk-Forward 驗證...")
def run_walk_forward(df):
    feat_df = df.dropna(subset=FEATURES)
    train_df = feat_df.dropna(subset=["Fwd_DD"]).copy()
    n = len(train_df)
    test_size = n // (N_FOLDS + 1)
    probs = pd.Series(dtype=float)
    for i in range(N_FOLDS):
        ts = n - (N_FOLDS - i) * test_size
        te = ts + test_size
        tr_end = ts - HORIZON
        m = make_model()
        m.fit(train_df[FEATURES].iloc[:tr_end], train_df["Target"].iloc[:tr_end])
        p = m.predict_proba(train_df[FEATURES].iloc[ts:te])[:, 1]
        probs = pd.concat([probs, pd.Series(p, index=train_df.index[ts:te])])
    fm = make_model()
    fm.fit(train_df[FEATURES], train_df["Target"])
    latest = feat_df.iloc[[-1]]
    latest_prob = float(fm.predict_proba(latest[FEATURES])[0][1])
    imp = pd.Series(fm.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return probs, train_df, feat_df, latest_prob, imp

df = load_data()
if df.empty or len(df.dropna(subset=FEATURES)) < 300:
    st.cache_data.clear()
    st.error("市場數據下載失敗 (Yahoo 暫時限流)。請等 1-2 分鐘後重新整理頁面再試。")
    st.stop()

probs, train_df, feat_df, latest_prob, imp = run_walk_forward(df)

y_true = train_df.loc[probs.index, "Target"].values
p_val = probs.values
base_rate = train_df["Target"].mean()

alerts = p_val >= WARN
events = y_true == 1
n_alerts = int(alerts.sum())
n_events = int(events.sum())
hits = int((alerts & events).sum())
capture = hits / n_events if n_events else float("nan")
precision = hits / n_alerts if n_alerts else float("nan")
false_alarm = 1 - precision if n_alerts else float("nan")
edge = precision - base_rate if n_alerts else float("nan")
model_valid = (n_alerts >= 20) and (capture >= CAPTURE_MIN) and (edge >= EDGE_MIN)

latest_date = feat_df.index[-1].strftime("%Y-%m-%d")
latest_price = float(feat_df["SPY_Close"].iloc[-1])

tab1, tab2, tab3 = st.tabs(["🛡 風險燈號", "🧪 模型驗證", "📉 防禦回測"])

with tab1:
    st.caption(f"數據截至: {latest_date} | 預警目標: 未來 {HORIZON} 日內回撤超過 3 個百分點")
    c1, c2 = st.columns(2)
    c1.metric("SPY 收盤", f"${latest_price:.2f}")
    c2.metric("10日內回撤>3% 概率", f"{latest_prob*100:.1f}%")

    st.divider()
    if not model_valid:
        st.error("模型未通過驗證 - 燈號僅供參考, 不應據此減倉")
    if latest_prob >= HIGH:
        st.error("🔴 高危 - 建議減至最低倉位, 或以 SPY put 對沖")
    elif latest_prob >= WARN:
        st.warning("🟠 警戒 - 建議減倉 30-50%, 獲利部位優先止盈")
    elif latest_prob >= WATCH:
        st.info("🟡 留意 - 停止新開倉, 維持現有持倉")
    else:
        st.success("🟢 正常 - 維持持倉")
    st.caption(f"基準率 (任意一日, 未來 {HORIZON} 日內發生 3% 以上回撤的歷史比例): {base_rate*100:.1f}%")

    st.divider()
    st.subheader("當前特徵讀數")
    row = feat_df.iloc[-1]
    fx = pd.DataFrame({
        "特徵": ["VIX 期限結構 (VIX/VIX3M)", "VIX / MA10", "信用動能 (HYG/IEF 20日)",
                 "距 60 日高點", "波動率比 (20/60)", "RSI(14)"],
        "讀數": [f"{row['VIX_Term']:.3f}", f"{row['VIX_MA10_Ratio']:.3f}",
                 f"{row['Credit_Mom']*100:+.2f}%", f"{row['DD_From_60H']*100:.2f}%",
                 f"{row['RV_Ratio']:.2f}", f"{row['SPY_RSI']:.1f}"],
        "警訊方向": ["高於 1.0 為逆價差", "偏高", "轉負", "跌幅擴大", "高於 1.2 為擴張", "偏低"]
    })
    st.dataframe(fx, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("樣本外驗證 (Walk-Forward, 5 折)")
    v1, v2, v3 = st.columns(3)
    v1.metric("捕捉率", f"{capture*100:.1f}%" if n_alerts else "N/A")
    v2.metric("假警報率", f"{false_alarm*100:.1f}%" if n_alerts else "N/A")
    v3.metric("精確率邊際", f"{edge*100:+.1f}pp" if n_alerts else "N/A")
    st.caption("捕捉率 = 實際發生的回撤中, 事前亮警戒燈的比例 | "
               "假警報率 = 亮燈但未發生回撤的比例 | "
               "精確率邊際 = 亮燈時的命中率 減 基準率")

    m1, m2, m3 = st.columns(3)
    m1.metric("警報次數", n_alerts)
    m2.metric("實際回撤事件", n_events)
    m3.metric("基準率", f"{base_rate*100:.1f}%")

    if model_valid:
        st.success(f"模型有效性檢驗: PASS (捕捉率 ≥ {CAPTURE_MIN*100:.0f}%, "
                   f"邊際 ≥ +{EDGE_MIN*100:.0f}pp, 警報 ≥ 20 次)")
    else:
        fails = []
        if n_alerts < 20:
            fails.append(f"警報次數 {n_alerts} < 20")
        if not (capture >= CAPTURE_MIN):
            fails.append(f"捕捉率 {capture*100:.1f}% < {CAPTURE_MIN*100:.0f}%")
        if not (edge >= EDGE_MIN):
            fails.append(f"精確率邊際 {edge*100:+.1f}pp < +{EDGE_MIN*100:.0f}pp")
        st.error("模型有效性檢驗: FAIL - 不可用於實際減倉決策。未達標項目: " + " ; ".join(fails))

    st.divider()
    st.subheader("特徵重要性")
    fig0, ax0 = plt.subplots(figsize=(8, 3))
    ax0.barh(imp.index[::-1], imp.values[::-1])
    ax0.set_xlabel("Importance")
    st.pyplot(fig0)

    st.divider()
    st.subheader("樣本外概率走勢 vs 實際回撤事件")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(probs.index, p_val, lw=0.9, label="P(drawdown>3%)")
    ax1.fill_between(probs.index, 0, 1, where=events, alpha=0.18,
                     color="red", label="actual drawdown window")
    ax1.axhline(WARN, color="orange", ls="--", lw=1, label="warn level")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

with tab3:
    st.subheader("防禦效益回測 (樣本外期間)")
    st.caption(f"規則: 概率低於 {WARN*100:.0f} 滿倉 | {WARN*100:.0f} 至 {HIGH*100:.0f} 降至六成倉 "
               f"| 高於 {HIGH*100:.0f} 降至三成倉")
    pr = probs.sort_index()
    nxt = df["SPY_Close"].pct_change().shift(-1).reindex(pr.index)
    expo = np.where(pr.values < WARN, 1.0, np.where(pr.values < HIGH, 0.6, 0.3))
    strat = pd.Series(expo * nxt.values, index=pr.index).fillna(0)
    bh = nxt.fillna(0)

    eq_s = (1 + strat).cumprod()
    eq_b = (1 + bh).cumprod()
    dd_s = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min()
    dd_b = ((eq_b - eq_b.cummax()) / eq_b.cummax()).min()

    r1, r2, r3 = st.columns(3)
    r1.metric("防禦策略總回報", f"{(eq_s.iloc[-1]-1)*100:+.1f}%")
    r2.metric("買入持有總回報", f"{(eq_b.iloc[-1]-1)*100:+.1f}%")
    r3.metric("平均倉位", f"{expo.mean()*100:.0f}%")
    r4, r5, r6 = st.columns(3)
    r4.metric("防禦策略最大回撤", f"{dd_s*100:.1f}%")
    r5.metric("買入持有最大回撤", f"{dd_b*100:.1f}%")
    r6.metric("回撤改善", f"{(dd_s-dd_b)*100:+.1f}pp")

    ret_cost = (eq_s.iloc[-1] - eq_b.iloc[-1]) * 100
    dd_gain = (dd_s - dd_b) * 100
    if dd_gain > 0 and dd_gain > abs(ret_cost) * 0.5:
        st.success(f"防禦有效: 回撤改善 {dd_gain:.1f}pp, 回報代價 {ret_cost:.1f}pp")
    else:
        st.warning(f"防禦效益不明顯: 回撤改善 {dd_gain:.1f}pp, 回報代價 {ret_cost:.1f}pp")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(eq_b.index, eq_b.values, label="Buy & Hold", lw=1)
    ax2.plot(eq_s.index, eq_s.values, label="Risk-Managed", lw=1)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylabel("Equity (start=1.0)")
    st.pyplot(fig2)

st.divider()
st.caption("本系統僅供研究參考, 不構成投資建議。未通過驗證時不應據以調整倉位。")
