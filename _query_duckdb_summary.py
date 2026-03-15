import duckdb
con = duckdb.connect('trading_data.duckdb', read_only=True)
q = """
SELECT
    COUNT(*) AS raw_rows,
    COUNT(DISTINCT 股票代码) AS stock_count,
    MIN(trade_date) AS min_trade_date,
    MAX(trade_date) AS max_trade_date,
    SUM(CASE WHEN trade_date IS NULL THEN 1 ELSE 0 END) AS null_trade_date_rows,
    SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) AS null_close_rows,
    SUM(CASE WHEN total_mkt_cap IS NULL THEN 1 ELSE 0 END) AS null_total_mkt_cap_rows
FROM trading_data_clean
"""
df = con.execute(q).fetchdf()
df.to_csv('trading_data_duckdb_summary.csv', index=False, encoding='utf-8-sig')
con.close()
print('ok')
