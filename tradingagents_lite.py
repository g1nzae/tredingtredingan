"""
TradingAgents-Lite: multi-agent LLM trading pipeline (paper trading only).

Alur (sesuai diagram):
  Analis (Market, Social, News, Fundamentals)
    -> Researcher Team (Bullish vs Bearish, debat N ronde)
    -> Trader (usulan transaksi)
    -> Risk Management Team (Aggressive, Neutral, Conservative)
    -> Manager (keputusan akhir)
    -> Execution (simulasi, dicatat ke paper_trades.jsonl)

Pakai:
    export ANTHROPIC_API_KEY=sk-ant-...
    python tradingagents_lite.py NVDA
"""
import datetime as dt
import json
import os
import re
import sys

import pandas as pd
import yfinance as yf
from anthropic import Anthropic

MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")
DEBATE_ROUNDS = int(os.getenv("DEBATE_ROUNDS", "2"))
client = Anthropic()


# --------------------------------------------------------------------------
# LLM helper
# --------------------------------------------------------------------------
def ask(system: str, user: str, max_tokens: int = 1200) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------
# Data layer (sumber data tiap analis)
# --------------------------------------------------------------------------
def get_market_data(ticker: str) -> str:
    df = yf.Ticker(ticker).history(period="1y")
    if df.empty:
        return "Data harga tidak tersedia."
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return (
        f"Harga terakhir: {close.iloc[-1]:.2f}\n"
        f"Perubahan 5 hari: {close.pct_change(5).iloc[-1]*100:.2f}%\n"
        f"Perubahan 3 bulan: {close.pct_change(63).iloc[-1]*100:.2f}%\n"
        f"SMA50: {close.rolling(50).mean().iloc[-1]:.2f} | "
        f"SMA200: {close.rolling(200).mean().iloc[-1]:.2f}\n"
        f"RSI14: {rsi.iloc[-1]:.1f}\n"
        f"MACD: {macd.iloc[-1]:.3f} vs signal {signal.iloc[-1]:.3f}\n"
        f"52w high/low: {close.max():.2f} / {close.min():.2f}\n"
        f"Volume rata-rata 20 hari: {df['Volume'].tail(20).mean():,.0f}"
    )


def get_news(ticker: str, limit: int = 10) -> str:
    items = yf.Ticker(ticker).news or []
    lines = []
    for it in items[:limit]:
        c = it.get("content", it)  # format yfinance beda-beda per versi
        title = c.get("title", "")
        summary = c.get("summary", "")
        lines.append(f"- {title}. {summary}"[:300])
    return "\n".join(lines) or "Tidak ada berita."


def get_social(ticker: str) -> str:
    # TODO: sambungkan ke Reddit (praw), X API, atau EODHD sentiment API.
    # Untuk awal, pakai berita sebagai proxy sentimen.
    return "(Belum ada API sosmed terpasang; proxy dari headline berita)\n" + get_news(ticker, 5)


def get_fundamentals(ticker: str) -> str:
    t = yf.Ticker(ticker)
    info = t.info or {}
    keys = [
        "longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE",
        "priceToBook", "profitMargins", "revenueGrowth", "earningsGrowth",
        "debtToEquity", "returnOnEquity", "freeCashflow", "dividendYield",
    ]
    profile = "\n".join(f"{k}: {info.get(k)}" for k in keys)
    try:
        ins = t.insider_transactions
        insider = ins.head(8).to_string() if ins is not None and not ins.empty else "-"
    except Exception:
        insider = "-"
    return f"[Company Profile & Financials]\n{profile}\n\n[Insider Transactions]\n{insider}"


# --------------------------------------------------------------------------
# Tahap 1: Analis
# --------------------------------------------------------------------------
ANALYSTS = {
    "Market Analyst": (get_market_data, "analisis teknikal harga dan indikator"),
    "Social Media Analyst": (get_social, "sentimen publik/sosial media"),
    "News Analyst": (get_news, "berita dan peristiwa makro/perusahaan"),
    "Fundamentals Analyst": (get_fundamentals, "kesehatan keuangan dan valuasi"),
}


def run_analysts(ticker: str) -> dict:
    reports = {}
    for name, (fetch, focus) in ANALYSTS.items():
        print(f"[{name}] bekerja...")
        data = fetch(ticker)
        reports[name] = ask(
            f"Kamu {name}. Fokus: {focus}. Ringkas poin-poin kunci secara objektif, "
            "maksimal 150 kata, sebut angka konkret.",
            f"Ticker: {ticker}\n\nData:\n{data}",
        )
    return reports


# --------------------------------------------------------------------------
# Tahap 2: Debat Bullish vs Bearish
# --------------------------------------------------------------------------
def run_debate(ticker: str, reports: dict) -> tuple[str, str, str]:
    context = "\n\n".join(f"## {k}\n{v}" for k, v in reports.items())
    bull_args, bear_args, transcript = "", "", ""
    for r in range(1, DEBATE_ROUNDS + 1):
        print(f"[Debat ronde {r}]")
        bull = ask(
            "Kamu Bullish Researcher. Bangun argumen terkuat untuk BELI. "
            "Bantah argumen bearish terakhir. Maks 150 kata.",
            f"Ticker {ticker}\n{context}\n\nTranskrip debat:\n{transcript or '(awal)'}",
        )
        transcript += f"\nBULLISH: {bull}\n"
        bear = ask(
            "Kamu Bearish Researcher. Bangun argumen terkuat untuk JUAL/HINDARI. "
            "Bantah argumen bullish terakhir. Maks 150 kata.",
            f"Ticker {ticker}\n{context}\n\nTranskrip debat:\n{transcript}",
        )
        transcript += f"BEARISH: {bear}\n"
        bull_args, bear_args = bull, bear
    return bull_args, bear_args, transcript


# --------------------------------------------------------------------------
# Tahap 3: Trader
# --------------------------------------------------------------------------
def run_trader(ticker: str, reports: dict, transcript: str) -> dict:
    print("[Trader] menyusun proposal...")
    out = ask(
        "Kamu Trader. Dari laporan analis dan debat, buat usulan transaksi. "
        'Balas HANYA JSON: {"action":"BUY|SELL|HOLD","size_pct":0-100 (persen '
        'dari modal),"stop_loss_pct":angka,"take_profit_pct":angka,"rationale":"..."}',
        f"Ticker {ticker}\n\nLaporan:\n{json.dumps(reports, ensure_ascii=False)}\n\nDebat:\n{transcript}",
    )
    return parse_json(out) or {"action": "HOLD", "size_pct": 0, "rationale": "gagal parse"}


# --------------------------------------------------------------------------
# Tahap 4: Tim Risiko
# --------------------------------------------------------------------------
RISK_PERSONAS = {
    "Aggressive": "Kamu risk analyst agresif: kejar potensi return tinggi, toleran volatilitas.",
    "Neutral": "Kamu risk analyst netral: seimbangkan return dan risiko.",
    "Conservative": "Kamu risk analyst konservatif: utamakan proteksi modal, curiga pada ukuran posisi besar.",
}


def run_risk_team(ticker: str, proposal: dict) -> dict:
    views = {}
    for name, persona in RISK_PERSONAS.items():
        print(f"[Risk - {name}] menilai...")
        views[name] = ask(
            persona + " Nilai usulan trader: setuju/ubah/tolak + alasan. Maks 100 kata.",
            f"Ticker {ticker}\nProposal: {json.dumps(proposal)}",
            max_tokens=600,
        )
    return views


# --------------------------------------------------------------------------
# Tahap 5: Manager + Eksekusi
# --------------------------------------------------------------------------
def run_manager(ticker: str, proposal: dict, risk_views: dict) -> dict:
    print("[Manager] memutuskan...")
    out = ask(
        "Kamu Portfolio Manager. Beri keputusan akhir dengan mempertimbangkan "
        "proposal trader dan 3 pandangan risiko. Balas HANYA JSON: "
        '{"approved":true|false,"action":"BUY|SELL|HOLD","size_pct":angka,'
        '"stop_loss_pct":angka,"take_profit_pct":angka,"reason":"..."}',
        f"Ticker {ticker}\nProposal: {json.dumps(proposal)}\n\n"
        f"Risk views:\n{json.dumps(risk_views, ensure_ascii=False)}",
    )
    return parse_json(out) or {"approved": False, "action": "HOLD", "reason": "gagal parse"}


def execute_paper(ticker: str, decision: dict):
    """Simulasi eksekusi. TIDAK mengirim order sungguhan."""
    price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    record = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "price": round(float(price), 2),
        **decision,
    }
    with open("paper_trades.jsonl", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


# --------------------------------------------------------------------------
def main(ticker: str):
    ticker = ticker.upper()
    reports = run_analysts(ticker)
    _, _, transcript = run_debate(ticker, reports)
    proposal = run_trader(ticker, reports, transcript)
    risk_views = run_risk_team(ticker, proposal)
    decision = run_manager(ticker, proposal, risk_views)

    print("\n=== PROPOSAL TRADER ===\n", json.dumps(proposal, indent=2, ensure_ascii=False))
    print("\n=== KEPUTUSAN MANAGER ===\n", json.dumps(decision, indent=2, ensure_ascii=False))

    if decision.get("approved") and decision.get("action") in ("BUY", "SELL"):
        rec = execute_paper(ticker, decision)
        print("\n[Execution] paper trade dicatat:", rec)
    else:
        print("\n[Execution] tidak ada transaksi.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
