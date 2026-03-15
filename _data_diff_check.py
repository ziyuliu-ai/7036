import duckdb
import pandas as pd

con = duckdb.connect('trading_data.duckdb', read_only=True)

sql = r'''
WITH raw_norm AS (
    SELECT
        source_file,
        TRIM("股票代码")::VARCHAR AS stock_code,
        TRIM("名称")::VARCHAR AS stock_name,
        TRIM("所属行业")::VARCHAR AS industry,
        TRIM("地域")::VARCHAR AS region,
        TRIM("TS代码")::VARCHAR AS ts_code,
        CASE
            WHEN length(trim("上市日期")) = 8 THEN strptime(trim("上市日期"), '%Y%m%d')::DATE
            WHEN trim("上市日期") LIKE '____-__-__' THEN strptime(trim("上市日期"), '%Y-%m-%d')::DATE
            ELSE NULL
        END AS list_date,
        CASE
            WHEN length(trim("交易日期")) = 8 THEN strptime(trim("交易日期"), '%Y%m%d')::DATE
            WHEN trim("交易日期") LIKE '____-__-__' THEN strptime(trim("交易日期"), '%Y-%m-%d')::DATE
            ELSE NULL
        END AS trade_date,
        TRY_CAST("开盘价" AS DOUBLE) AS open,
        TRY_CAST("最高价" AS DOUBLE) AS high,
        TRY_CAST("最低价" AS DOUBLE) AS low,
        TRY_CAST("收盘价" AS DOUBLE) AS close,
        TRY_CAST("前收盘价" AS DOUBLE) AS prev_close,
        TRY_CAST("涨跌额" AS DOUBLE) AS change_amt,
        TRY_CAST("涨跌幅(%)" AS DOUBLE) AS change_pct,
        TRY_CAST("成交量(手)" AS DOUBLE) AS volume_lot,
        COALESCE(TRY_CAST("成交额(千元)" AS DOUBLE) * 1000.0, TRY_CAST("成交额(万元)" AS DOUBLE) * 10000.0) AS turnover_amount,
        TRY_CAST("换手率(%)" AS DOUBLE) AS turnover_pct,
        COALESCE(TRY_CAST("市盈率" AS DOUBLE), TRY_CAST("市盈率(TTM,亏损的PE为空)" AS DOUBLE)) AS pe,
        TRY_CAST("市净率" AS DOUBLE) AS pb,
        COALESCE(TRY_CAST("市销率" AS DOUBLE), TRY_CAST("市销率(TTM)" AS DOUBLE)) AS ps,
        TRY_CAST("股息率(%)" AS DOUBLE) AS dividend_yield_pct,
        TRY_CAST("流通市值(万元)" AS DOUBLE) * 10000.0 AS free_float_mkt_cap,
        TRY_CAST("总市值(万元)" AS DOUBLE) * 10000.0 AS total_mkt_cap
    FROM trading_data_raw
),
cmp AS (
    SELECT
        e.*, r.stock_name AS r_stock_name, r.industry AS r_industry, r.region AS r_region, r.ts_code AS r_ts_code,
        r.list_date AS r_list_date, r.trade_date AS r_trade_date,
        r.open AS r_open, r.high AS r_high, r.low AS r_low, r.close AS r_close, r.prev_close AS r_prev_close,
        r.change_amt AS r_change_amt, r.change_pct AS r_change_pct, r.volume_lot AS r_volume_lot,
        r.turnover_amount AS r_turnover_amount, r.turnover_pct AS r_turnover_pct,
        r.pe AS r_pe, r.pb AS r_pb, r.ps AS r_ps, r.dividend_yield_pct AS r_dividend_yield_pct,
        r.free_float_mkt_cap AS r_free_float_mkt_cap, r.total_mkt_cap AS r_total_mkt_cap
    FROM trading_data_en e
    LEFT JOIN raw_norm r
      ON e.source_file = r.source_file
     AND e.stock_code = r.stock_code
     AND ((e.trade_date = r.trade_date) OR (e.trade_date IS NULL AND r.trade_date IS NULL))
)
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN r_stock_name IS NULL AND stock_name IS NOT NULL THEN 1 ELSE 0 END) AS unmatched_rows,

    SUM(CASE WHEN COALESCE(stock_name,'') <> COALESCE(r_stock_name,'') THEN 1 ELSE 0 END) AS diff_stock_name,
    SUM(CASE WHEN COALESCE(industry,'') <> COALESCE(r_industry,'') THEN 1 ELSE 0 END) AS diff_industry,
    SUM(CASE WHEN COALESCE(region,'') <> COALESCE(r_region,'') THEN 1 ELSE 0 END) AS diff_region,
    SUM(CASE WHEN COALESCE(ts_code,'') <> COALESCE(r_ts_code,'') THEN 1 ELSE 0 END) AS diff_ts_code,
    SUM(CASE WHEN NOT ((list_date = r_list_date) OR (list_date IS NULL AND r_list_date IS NULL)) THEN 1 ELSE 0 END) AS diff_list_date,

    SUM(CASE WHEN NOT ((open = r_open) OR (open IS NULL AND r_open IS NULL)) THEN 1 ELSE 0 END) AS diff_open,
    MAX(ABS(open - r_open)) AS max_abs_diff_open,
    SUM(CASE WHEN NOT ((high = r_high) OR (high IS NULL AND r_high IS NULL)) THEN 1 ELSE 0 END) AS diff_high,
    MAX(ABS(high - r_high)) AS max_abs_diff_high,
    SUM(CASE WHEN NOT ((low = r_low) OR (low IS NULL AND r_low IS NULL)) THEN 1 ELSE 0 END) AS diff_low,
    MAX(ABS(low - r_low)) AS max_abs_diff_low,
    SUM(CASE WHEN NOT ((close = r_close) OR (close IS NULL AND r_close IS NULL)) THEN 1 ELSE 0 END) AS diff_close,
    MAX(ABS(close - r_close)) AS max_abs_diff_close,
    SUM(CASE WHEN NOT ((prev_close = r_prev_close) OR (prev_close IS NULL AND r_prev_close IS NULL)) THEN 1 ELSE 0 END) AS diff_prev_close,
    MAX(ABS(prev_close - r_prev_close)) AS max_abs_diff_prev_close,
    SUM(CASE WHEN NOT ((change_amt = r_change_amt) OR (change_amt IS NULL AND r_change_amt IS NULL)) THEN 1 ELSE 0 END) AS diff_change_amt,
    MAX(ABS(change_amt - r_change_amt)) AS max_abs_diff_change_amt,
    SUM(CASE WHEN NOT ((change_pct = r_change_pct) OR (change_pct IS NULL AND r_change_pct IS NULL)) THEN 1 ELSE 0 END) AS diff_change_pct,
    MAX(ABS(change_pct - r_change_pct)) AS max_abs_diff_change_pct,
    SUM(CASE WHEN NOT ((volume_lot = r_volume_lot) OR (volume_lot IS NULL AND r_volume_lot IS NULL)) THEN 1 ELSE 0 END) AS diff_volume_lot,
    MAX(ABS(volume_lot - r_volume_lot)) AS max_abs_diff_volume_lot,

    SUM(CASE WHEN NOT ((turnover_amount = r_turnover_amount) OR (turnover_amount IS NULL AND r_turnover_amount IS NULL)) THEN 1 ELSE 0 END) AS diff_turnover_amount,
    MAX(ABS(turnover_amount - r_turnover_amount)) AS max_abs_diff_turnover_amount,
    SUM(CASE WHEN NOT ((turnover_pct = r_turnover_pct) OR (turnover_pct IS NULL AND r_turnover_pct IS NULL)) THEN 1 ELSE 0 END) AS diff_turnover_pct,
    MAX(ABS(turnover_pct - r_turnover_pct)) AS max_abs_diff_turnover_pct,

    SUM(CASE WHEN NOT ((pe = r_pe) OR (pe IS NULL AND r_pe IS NULL)) THEN 1 ELSE 0 END) AS diff_pe,
    MAX(ABS(pe - r_pe)) AS max_abs_diff_pe,
    SUM(CASE WHEN NOT ((pb = r_pb) OR (pb IS NULL AND r_pb IS NULL)) THEN 1 ELSE 0 END) AS diff_pb,
    MAX(ABS(pb - r_pb)) AS max_abs_diff_pb,
    SUM(CASE WHEN NOT ((ps = r_ps) OR (ps IS NULL AND r_ps IS NULL)) THEN 1 ELSE 0 END) AS diff_ps,
    MAX(ABS(ps - r_ps)) AS max_abs_diff_ps,
    SUM(CASE WHEN NOT ((dividend_yield_pct = r_dividend_yield_pct) OR (dividend_yield_pct IS NULL AND r_dividend_yield_pct IS NULL)) THEN 1 ELSE 0 END) AS diff_dividend_yield_pct,
    MAX(ABS(dividend_yield_pct - r_dividend_yield_pct)) AS max_abs_diff_dividend_yield_pct,
    SUM(CASE WHEN NOT ((free_float_mkt_cap = r_free_float_mkt_cap) OR (free_float_mkt_cap IS NULL AND r_free_float_mkt_cap IS NULL)) THEN 1 ELSE 0 END) AS diff_free_float_mkt_cap,
    MAX(ABS(free_float_mkt_cap - r_free_float_mkt_cap)) AS max_abs_diff_free_float_mkt_cap,
    SUM(CASE WHEN NOT ((total_mkt_cap = r_total_mkt_cap) OR (total_mkt_cap IS NULL AND r_total_mkt_cap IS NULL)) THEN 1 ELSE 0 END) AS diff_total_mkt_cap,
    MAX(ABS(total_mkt_cap - r_total_mkt_cap)) AS max_abs_diff_total_mkt_cap
FROM cmp;
'''

res = con.execute(sql).fetchdf()
print(res.to_string(index=False))
con.close()
