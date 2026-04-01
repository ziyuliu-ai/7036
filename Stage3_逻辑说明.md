# Stage III 逻辑说明（当前版本）

本文档整理当前 notebook 中 Stage III 的完整实现逻辑，覆盖：
- Stage III（周频版本）
- Stage III（当前主用：日频 + 未来90交易日累计收益 + 百分比样本阈值 + 暴露衰减扩展）

---

## 1. 目标

Stage III 的核心目标是：
1. 将 Stage II 生成的文档因子暴露（doc-level loadings）映射到股票-日期层面；
2. 与个股收益数据对齐，构建监督目标收益；
3. 逐日（或逐周）做横截面回归，估计因子收益（factor return）；
4. 统计因子显著性；
5. 基于因子暴露分组构造 long-short 组合并计算绩效；
6. 输出结果文件并可视化。

---

## 2. 输入与依赖

### 2.1 核心输入

- Stage II 输出：tf_loadings_matrix.csv
  - 字段：doc_id + 多个因子列（factor1, factor2, ...）
- 个股收益库：trading_data.duckdb 或 trades.duckdb
- 股票映射：HS300.csv（股票简称 -> 股票代码）
- Stage I 主题文件（用于经济含义表）：text_factors_stage1_clusters.csv

### 2.2 主要中间表

- ldf：Stage II loadings 原始表
- stock_factor_raw / stock_factor：聚合后的股票-日期因子暴露
- ret_df：股票-日期收益及目标收益
- merged：暴露与目标收益合并后的回归样本
- factor_ret_df：逐期估计的因子收益序列
- port_df：因子 long-short 组合收益序列

---

## 3. 逻辑总流程

1. 读取 loadings，解析 doc_id 得到 报告日期 + 股票简称。
2. 通过 HS300 映射为股票代码，得到股票-日期-因子暴露。
3. 从 DuckDB 自动识别收益表结构（长表优先，宽表回退）。
4. 生成目标收益 target_ret：
   - 周频版：下一期周收益（shift）
   - 当前主用版：未来90交易日累计收益
5. （当前主用版）对暴露做时间衰减扩展（extend + half-life）。
6. 合并暴露与目标收益，按日期进行横截面回归（Ridge）。
7. 先做预筛选（按 t 值选 Top N 因子），再做最终回归。
8. 输出因子收益、显著性统计。
9. 按因子暴露分位构造 long-short，输出组合收益与绩效。
10. 生成图表与因子经济含义表。

---

## 4. 关键步骤细化

### 4.1 文档暴露 -> 股票暴露

- 从 doc_id 解析：
  - stock_name（通常来自路径中的股票目录名）
  - date（文件名中的 8 位日期）
- 将同一股票同一日期的多篇文档因子暴露取均值，得到 stock-date 暴露。
- 周频版会将日度暴露聚到周频（按周末时间戳取 last）。
- 当前主用版保留日频，并允许后续做暴露扩展。

### 4.2 个股收益读取与清洗

- DuckDB 自动识别字段：date, stock, ret。
- 若收益是百分数口径（中位数绝对值 > 0.5），会除以100。
- 清洗后字段统一为：date, stock, ret。

### 4.3 目标收益构造

#### A) 周频版

- 先把日收益聚成周收益：
  $r_{i,w}=\prod_{d\in w}(1+r_{i,d})-1$
- 若 USE_FWD_RET=True：
  $target\_ret_{i,w}=r_{i,w+1}$

#### B) 当前主用版（日频 + 90D）

- 先计算对数收益：
  $\ell_{i,t}=\log(1+r_{i,t})$
- 未来90交易日累计收益：
  $target\_ret_{i,t}=\exp\left(\sum_{k=1}^{90}\ell_{i,t+k}\right)-1$

### 4.4 暴露扩展（当前主用版）

用于模拟研报观点的持续影响：

- 对每个原始暴露向后扩展 0 到 EXPOSURE_EXTEND_DAYS 天；
- 第 k 天权重为：
  $w_k=2^{-k/h}$，其中 $h=EXPOSURE_HALF_LIFE_DAYS$；
- 同一股票同一天的多来源扩展暴露求和。

可选：EXPOSURE_FILLNA_ZERO 控制缺失暴露是否填0。

### 4.5 合并与截面回归

每个日期 t，做截面回归：

$y_{i,t}=\alpha_t + X_{i,t}\beta_t + \varepsilon_{i,t}$

其中：
- $y_{i,t}=target\_ret_{i,t}$
- $X_{i,t}$ 为当日股票的因子暴露向量
- 模型：Ridge(alpha=RIDGE_ALPHA)

稳健性处理：
- 目标收益按日期做 winsorize（Y_WINSOR_Q）
- 低方差因子剔除（LOW_VAR_EPS）
- 截面 z-score（USE_XS_ZSCORE=True 时）
- 回归系数与截距截断到 [-BETA_CAP, BETA_CAP]

输出：每个日期一组因子收益 $\beta_{f,t}$，保存为 factor_returns_daily(_90d).csv。

### 4.6 因子预筛选

- 先在全因子上回归得到初始因子收益序列；
- 计算每个因子的时间序列 t 统计量；
- 选取 |t| 较高的前 N_ROBUST_FACTORS 个因子；
- 再用筛选后因子集合执行最终回归。

### 4.7 Long-Short 组合构建

对每个因子、每个日期：
1. 按该因子暴露排序；
2. 取底部 q 比例为 short，顶部 q 比例为 long；
3. 计算：
   - long_ret = 顶部分组 target_ret 均值
   - short_ret = 底部分组 target_ret 均值
   - ls_ret = long_ret - short_ret

注意：当前实现中 ls_ret 使用的是 target_ret 口径（在90D版本即未来90日累计收益口径）。

### 4.8 绩效统计

对每个因子的 ls_ret 序列计算：
- mean, std, sharpe
- cum_ret（通过累计净值）
- beta（相对等权市场 mkt_ret）
- alpha_ann

并输出：factor_portfolio_metrics(_90d).csv。

---

## 5. 输出文件

### 5.1 周频版本（Stage III）

- factor_returns_daily.csv
- factor_significance_summary.csv
- factor_portfolio_daily.csv
- factor_portfolio_metrics.csv
- factor_economic_meaning.csv

### 5.2 当前主用版本（日频 + 90D）

- factor_returns_daily_90d.csv
- factor_significance_summary_90d.csv
- factor_portfolio_daily_90d.csv
- factor_portfolio_metrics_90d.csv
- factor_economic_meaning_90d.csv

---

## 6. 关键参数（当前主用版）

- RETURN_HORIZON_DAYS = 90
- MIN_XS_SAMPLE_PCT = 0.25
- FALLBACK_MIN_XS_SAMPLE_PCT = 0.20
- N_ROBUST_FACTORS = 20
- TOP_Q = 0.2
- RIDGE_ALPHA = 5.0
- BETA_CAP = 0.20
- EXPOSURE_EXTEND_ENABLED = True
- EXPOSURE_EXTEND_DAYS = 90
- EXPOSURE_HALF_LIFE_DAYS = 63
- EXPOSURE_FILLNA_ZERO = False

---

## 7. 当前实现的解释注意点

1. 在90D版本中，target_ret 是未来90日累计收益，而非单日收益；
2. long-short 组合同样基于 target_ret 计算，若直接按日期连续复利，数值可能出现非常大的净值量级；
3. 这类结果更适合作为“预测区分度”度量，不应直接等价解释为可交易日频净值曲线。

---

## 8. 一句话总结

当前 Stage III 本质是：用文本因子暴露去解释个股未来收益（周频或90日累计口径），通过逐期横截面 Ridge 反推因子收益，再用分位多空检验因子经济有效性。