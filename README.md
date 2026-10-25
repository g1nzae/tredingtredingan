# TradingAgents-Lite

Pipeline multi-agent: Analis -> Debat Bullish/Bearish -> Trader -> Tim Risiko -> Manager -> Execution (paper).

## Jalankan
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python tradingagents_lite.py NVDA
```

Opsional: `AGENT_MODEL` (default claude-sonnet-5-5), `DEBATE_ROUNDS` (default 2).
Hasil eksekusi simulasi masuk ke `paper_trades.jsonl`.

## Catatan
- Eksekusi hanya simulasi, tidak ada order sungguhan.
- Social media (X, Reddit, EODHD) masih placeholder di `get_social()`; sambungkan API-nya kalau mau.
- Ini untuk eksperimen, bukan saran investasi.
