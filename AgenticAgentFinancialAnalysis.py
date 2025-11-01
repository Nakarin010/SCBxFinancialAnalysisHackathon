# from langgraph.graph import StateGraph, START, END
import talib
import numpy as np
import pandas as pd
from typing import Union, Dict, Set, List, TypedDict, Annotated
from langchain_core.tools import tool
import yahooquery as yq
import requests
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType
import os 
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta
from datetime import date as _date
from dateutil import parser as dateparser

# Global reference date extracted from the current query.
# Tools will fall back to today's date if it is None.
CURRENT_REF_DATE: _date | None = None

# Helper to extract reference date from user query
def extract_reference_date(text: str) -> _date | None:
    """
    Parse an explicit or relative date from the question text.
    Returns a _date object or None if no reference date is found.
    """
    import re, datetime as dt

    # 1) Explicit dates (13/12/2017, 2017-12-13, Dec 13 2017, etc.)
    m = re.search(
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        r'|\d{4}-\d{2}-\d{2}'
        r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',
        text,
        flags=re.I,
    )
    if m:
        return dateparser.parse(m.group(0), dayfirst=True).date()

    # 2) Simple relative phrases
    low = text.lower()
    today = dt.date.today()
    if "yesterday" in low:
        return today - dt.timedelta(days=1)
    if "last month" in low:
        first_of_this_month = today.replace(day=1)
        return (first_of_this_month - dt.timedelta(days=1)).replace(day=1)

    # Extend with more heuristics as needed …
    return None
from fredapi import Fred   
import time

# Load API key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_KEY")
fred_api_key = os.getenv("FRED_API")
polygon_key = os.getenv("POLYGON")
finhub_key = os.getenv("finhub")
mistral_key = os.getenv("MISTRAL")
groqK = os.getenv("groq")

# 1. Load data
test_df = pd.read_csv("test.csv")

# 2. Prepare submission template
submission = pd.DataFrame({
    "id": test_df["id"],
    "prediction": [""] * len(test_df)
})

# Enhanced Financial Metrics Tool
@tool
def get_financial_metrics(ticker: str) -> Union[str, str]:
    """Fetches comprehensive financial ratios and metrics for a given ticker. Input should be just the ticker symbol."""
    time.sleep(2)
    import urllib.error
    try:
        ticker = ticker.strip().lstrip('$').upper()
        # Use the globally‑set reference date (defaults to today)
        as_of = CURRENT_REF_DATE or _date.today()
        # Retry stock.info on HTTP 429 with exponential backoff
        retries = 3
        delay = 1
        info = None
        for attempt in range(retries):
            try:
                # Initialize Ticker and fetch combined modules
                stock = yq.Ticker(ticker, formatted=False)
                modules = stock.all_modules
                ticker_modules = modules.get(ticker, {})
                info = {}
                # Merge relevant modules into info
                for module_name in ['summaryDetail', 'financialData', 'defaultKeyStatistics', 'price']:
                    module_data = ticker_modules.get(module_name)
                    if isinstance(module_data, dict):
                        info.update(module_data)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    return f"Error fetching metrics for {ticker}: {e}"
        if info is None:
            return f"Error fetching metrics for {ticker}: rate limited after {retries} attempts"

        # Get historical data for additional calculations
        # Optionally wrap this in similar backoff if desired
        hist = None
        for attempt in range(retries):
            try:
                hist_full = stock.history(period="max", interval="1d")
                hist = hist_full[hist_full.index.get_level_values(1) <= pd.Timestamp(as_of)].tail(252)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    hist = None
                    break
        if hist is None:
            hist = pd.DataFrame()
        
        metrics = {
            # Valuation ratios
            'pe_ratio': info.get('forwardPE'),
            'trailing_pe': info.get('trailingPE'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'enterprise_value': info.get('enterpriseValue'),
            'ev_to_revenue': info.get('enterpriseToRevenue'),
            'ev_to_ebitda': info.get('enterpriseToEbitda'),
            
            # Financial health
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'cash_per_share': info.get('totalCashPerShare'),
            
            # Profitability
            'profit_margins': info.get('profitMargins'),
            'gross_margins': info.get('grossMargins'),
            'operating_margins': info.get('operatingMargins'),
            'return_on_equity': info.get('returnOnEquity'),
            'return_on_assets': info.get('returnOnAssets'),
            
            # Growth
            'earnings_growth': info.get('earningsGrowth'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
            
            # Market metrics
            'beta': info.get('beta'),
            'market_cap': info.get('marketCap'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'float_shares': info.get('floatShares'),
            'short_ratio': info.get('shortRatio'),
            
            # Dividend info
            'dividend_yield': info.get('dividendYield'),
            'payout_ratio': info.get('payoutRatio'),
            
            # Price movement
            'current_price': info.get('currentPrice'),
            '52_week_high': info.get('fiftyTwoWeekHigh'),
            '52_week_low': info.get('fiftyTwoWeekLow'),
        }
        
        # Calculate additional metrics from historical data
        if not hist.empty:
            current_price = hist['close'].iloc[-1]
            price_30d_ago = hist['close'].iloc[-30] if len(hist) > 30 else hist['close'].iloc[0]
            metrics['price_change_30d'] = (current_price - price_30d_ago) / price_30d_ago
            metrics['volatility_30d'] = hist['close'].rolling(30).std().iloc[-1] / hist['close'].iloc[-1]
        
        # Filter out None values and format as string
        clean_metrics = {k: v for k, v in metrics.items() if v is not None}
        
        # Format as readable string
        result = f"Financial metrics for {ticker}:\n"
        for key, value in clean_metrics.items():
            if isinstance(value, float):
                result += f"• {key}: {value:.4f}\n"
            else:
                result += f"• {key}: {value}\n"
        
        return result
        
    except Exception as e:
        return f"Error fetching metrics for {ticker}: {str(e)}"

@tool
def get_market_news(ticker: str) -> Union[str, str]:
    """Fetches recent news for a given ticker with sentiment analysis. Input should be just the ticker symbol."""
    try:
            ticker = ticker.upper().strip().lstrip('$')
            news_items: List[Dict] = []

            # ---------- 1) yahooquery ----------
            try:
                stock = yq.Ticker(ticker)
                raw_news = stock.news() if callable(stock.news) else stock.news
                if isinstance(raw_news, dict):
                    raw_news = raw_news.get(ticker) or next(iter(raw_news.values()), [])
                if isinstance(raw_news, list):
                    news_items = raw_news[:5]
            except Exception:
                # swallow and fall through to other sources
                pass

            # ---------- 2) Finnhub fallback ----------
            if not news_items:
                if finhub_key:
                    url = f"https://finnhub.io/api/v1/news?symbol={ticker}&token={finhub_key}"
                    try:
                        resp = requests.get(url, timeout=5)
                        if resp.status_code == 200:
                            for art in resp.json()[:5]:
                                news_items.append({
                                    "title": art.get("headline", ""),
                                    "summary": art.get("summary", "")
                                })
                    except Exception:
                        pass  # ignore and continue to Polygon

            # ---------- 3) Polygon fallback ----------
            if not news_items:
                
                if polygon_key:
                    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=5&apiKey={polygon_key}"
                    try:
                        resp = requests.get(url, timeout=5)
                        if resp.status_code == 200:
                            poly_news = resp.json().get("results", [])
                            for art in poly_news:
                                news_items.append({
                                    "title": art.get("title", ""),
                                    "summary": art.get("description", "")
                                })
                    except Exception:
                        pass

            # Keep only dict items; drop strings or malformed entries
            news_items = [n for n in news_items if isinstance(n, dict)]

            # Trim news to the reference date window
            if CURRENT_REF_DATE is not None:
                ref_cutoff = datetime.combine(CURRENT_REF_DATE, datetime.min.time())
                news_items = [
                    n
                    for n in news_items
                    if not n.get("providerPublishTime")
                    or datetime.utcfromtimestamp(n["providerPublishTime"]) <= ref_cutoff
                ]

            if not news_items:
                return f"No recent news found for {ticker}"

            # ---------- Sentiment tagging ----------
            positive_words = ['growth', 'profit', 'beat', 'exceed', 'strong', 'bullish', 'upgrade', 'buy']
            negative_words = ['loss', 'decline', 'miss', 'weak', 'bearish', 'downgrade', 'sell', 'concern']

            formatted_news = []
            for art in news_items:
                if not isinstance(art, dict):
                    continue
                title = art.get('title', '')
                summary = art.get('summary', '')
                text = f"{title} {summary}".lower()

                pos_count = sum(1 for w in positive_words if w in text)
                neg_count = sum(1 for w in negative_words if w in text)
                sentiment = 'positive' if pos_count > neg_count else 'negative' if neg_count > pos_count else 'neutral'

                formatted_news.append(f"• {title} [{sentiment}]: {summary[:200]}...")

            return f"Recent news for {ticker}:\n" + "\n".join(formatted_news)

    except Exception as e:
            return f"Error fetching news for {ticker}: {str(e)}"

@tool
def get_economic_indicators(indicator_type: str) -> str:
    """
    Fetches key economic indicators from FRED API.
    indicator_type options: 'general', 'inflation', 'employment', 'rates', 'gdp'
    """
    
    if not fred_api_key:
        return "❌ FRED_API_KEY not set. Please add it to your .env"
    fred = Fred(api_key=fred_api_key)

    indicators_map = {
        "general": {
            'Federal Funds Rate': 'FEDFUNDS',
            '10-Year Treasury':    'GS10',
            'VIX':                  'VIXCLS',
            'Unemployment Rate':    'UNRATE',
            'CPI':                  'CPIAUCSL'
        },
        'inflation': {
            'CPI All Items':         'CPIAUCSL',
            'Core CPI':              'CPILFESL',
            'PCE':                   'PCEPI',
            'Core PCE':              'PCEPILFE',
            '5Y5Y Inflation Expect': 'T5YIE'
        },
        'employment': {
            'Unemployment Rate':         'UNRATE',
            'Labor Force Participation': 'CIVPART',
            'Nonfarm Payrolls':          'PAYEMS',
            'Initial Jobless Claims':    'ICSA',
            'Job Openings':              'JTSJOL'
        },
        'rates': {
            'Federal Funds Rate': 'FEDFUNDS',
            '2-Year Treasury':    'GS2',
            '10-Year Treasury':   'GS10',
            '30-Year Treasury':   'GS30',
            'Yield Curve (10Y-2Y)': None
        },
        'gdp': {
            'Real GDP':            'GDPC1',
            'GDP Growth Rate':     'GDPC1',
            'Personal Consumption':'PCEC',
            'Business Investment':'GPDI',
            'Government Spending':'GCE'
        }
    }

    sel = indicators_map.get(indicator_type.lower(), indicators_map['general'])
    lines = [f"Economic Indicators ({indicator_type}):"]

    for name, sid in sel.items():
        try:
            # yield curve calculation
            if sid is None:
                y10 = fred.get_series('GS10')
                y2  = fred.get_series('GS2')
                if y10.empty or y2.empty:
                    lines.append(f"• {name}: no data")
                else:
                    val = y10.iloc[-1] - y2.iloc[-1]
                    date = y10.index[-1].strftime('%Y-%m-%d')
                    lines.append(f"• {name}: {val:.2f}% (as of {date})")
                continue

            # fetch full series and pick last values
            data = fred.get_series(sid)
            if data.empty:
                lines.append(f"• {name}: no data")
                continue
            latest = data.iloc[-1]
            date   = data.index[-1].strftime('%Y-%m-%d')

            # inflation YoY
            if name in ['CPI', 'CPI All Items', 'Core CPI', 'PCE', 'Core PCE']:
                if len(data) >= 12:
                    year_data = data[-12:]
                    start = year_data.iloc[0]
                    end   = year_data.iloc[-1]
                    yoy   = (end / start - 1) * 100
                    lines.append(f"• {name}: {yoy:.1f}% YoY (as of {date})")
                else:
                    lines.append(f"• {name}: {latest:.1f} (as of {date})")

            # GDP quarterly growth
            elif name == 'GDP Growth Rate':
                if len(data) >= 2:
                    prev = data.iloc[-2]
                    gr   = (latest / prev - 1) * 100
                    lines.append(f"• {name}: {gr:.1f}% qtr (as of {date})")
                else:
                    lines.append(f"• {name}: {latest:.1f} (as of {date})")

            # rates, yields, VIX
            elif 'Rate' in name or 'Treasury' in name or 'VIX' in name:
                lines.append(f"• {name}: {latest:.2f}% (as of {date})")

            # other numeric indicators
            else:
                lines.append(f"• {name}: {latest:,.1f} (as of {date})")

        except Exception as e:
            lines.append(f"• {name}: error fetching ({e})")

    return "\n".join(lines)
    
    
@tool
def get_fed_policy_info() -> str:
    """
    Get Federal Reserve policy-related indicators with recent values and trend arrows.
    """
    if not fred_api_key:
        return "❌ FRED_API_KEY not set in .env"
    fred = Fred(api_key=fred_api_key)
    policy = {
        'Federal Funds Rate': 'FEDFUNDS',
        'Upper Target':         'DFEDTARU',
        'Lower Target':         'DFEDTARL',
        'Total Assets':         'WALCL',
        'Excess Reserves':      'EXCSRESNS'
    }
    out = ["Fed Policy Indicators:"]
    for name, sid in policy.items():
        try:
            data = fred.get_series(sid, limit=3)
            if data.empty:
                out.append(f"• {name}: no data")
                continue
            latest = data.iloc[-1]
            date   = data.index[-1].strftime("%Y-%m-%d")
            if len(data) >= 2:
                prev = data.iloc[-2]
                diff = latest - prev
                arrow = "↑" if diff>0 else "↓" if diff<0 else "→"
            else:
                arrow = ""
            # Format numbers
            val = f"{latest:.2f}%" if "Rate" in name else f"{int(latest):,}"
            out.append(f"• {name}: {val} {arrow} (as of {date})")
        except Exception as e:
            out.append(f"• {name}: error fetching ({e})")
    return "\n".join(out)

@tool
def get_technical_indicators(ticker: str) -> dict:
    """
    Fetches a set of common technical indicators for the past 1 year:
      • 14-day RSI
      • MACD (12/26)
      • Current BB %-position
      • 14-day ATR
    Returns a dict of latest values.
    """
    try:
        # 1. Sanitize ticker
        t = ticker.strip().lstrip('$').upper()
        stock = yq.Ticker(t)
        
        # 2. Get 1y history, date-aware
        hist_full = stock.history(period="max", interval="1d")
        as_of = CURRENT_REF_DATE or _date.today()
        hist = hist_full[hist_full.index.get_level_values(1) <= pd.Timestamp(as_of)].tail(252)
        if hist.empty or len(hist) < 30:
            return {"error": f"Not enough price data for {t}"}
        
        close  = hist["close"].values
        high   = hist["high"].values
        low    = hist["low"].values
        volume = hist["volume"].values
        
        # 3. Compute indicators
        sma20 = talib.SMA(close, timeperiod=20)      # example, not returned
        ema12 = talib.EMA(close, timeperiod=12)      # example, not returned
        rsi   = talib.RSI(close, timeperiod=14)
        macd, macd_sig, macd_hist = talib.MACD(close)
        bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        atr   = talib.ATR(high, low, close, timeperiod=14)
        
        latest = {
            "RSI_14":        float(rsi[-1]),
            "MACD":          float(macd[-1]),
            "MACD_signal":   float(macd_sig[-1]),
            "BB_pct_pos":    float((close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])),
            "ATR_14":        float(atr[-1]),
        }
        return latest

    except ImportError:
        return {"error": "talib not installed. Run `pip install ta-lib`"}
    except Exception as e:
        return {"error": str(e)}

# Enhanced prompt with more examples and better structure
ENHANCED_SHOTS = """
Examples of financial analysis questions and answers:


Q: ตอบคำถามด้วยตัวเลือก A, B, C, หรือ D เท่านั้น
GDP ลดลงเรียกว่าอะไร?
A) เงินฝืด B) เงินเฟ้อ C) GDP Deflator D) GNP
A: A

Q: If the Fed signals a 25 bp rate cut while inflation remains at 3%, which strategy makes most sense?
A) Long 2-year Treasuries B) Short high-beta tech stocks C) Buy USD/JPY D) Sell long-dated bonds
A: A

Q: ECB hints at possible negative rates. What's the optimal trade?
A) Short EUR/USD B) Long German 10-yr Bunds C) Buy French bank stocks D) Sell Italian BTPs
A: B

Q: A company has PE ratio of 15, debt-to-equity of 0.3, and ROE of 18%. The sector average PE is 20. What's the likely assessment?
A) Overvalued B) Undervalued C) Fairly valued D) Insufficient data
A: B


INCORRECT EXAMPLES:
Q: จากข้อมูลและทวีตที่ให้มา คุณสามารถคาดการณ์ได้หรือไม่ว่าราคาปิดของ $chk ในวันที่ 13/12/2017 จะขึ้นหรือลง โปรดระบุว่า Rise หรือ Fall
A: I am unable to predict the closing price of $CHK on December 13, 2017, as the provided data only extends up to December 12, 2017.

""".strip()

# Enhanced prompt template
enhanced_prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are an expert financial analyst with deep knowledge of:\n"
        "- Financial ratios and valuation metrics\n"
        "- Market dynamics and economic indicators\n"
        "- Corporate finance and investment analysis\n"
        "- Technical and fundamental analysis\n\n"
        "you can use the tools such as Technical, Fundamental, and Sentiment analysis to best estimate the answer about stock price movements.\n\n"

        "INSTRUCTIONS:\n"
        # "1. Analyze the question carefully\n"
        # "2. Use available tools if specific company data is needed\n"
        # "3. Apply financial theory and market knowledge\n"
        # "4. For multiple choice questions, respond with ONLY the letter (A, B, C, or D)\n"
        # "5. For Thai questions, respond in only Letter (A, B, C, or D) or Fall or Rise only\n\n"
        "REMEMBER: ONLY ANSWER WITH A, B, C, OR D. or Fall or Rise ONLY, DO NOT EXPLAIN YOUR ANSWER.\n\n"
        "ALSO NO response with 'insufficient data' or 'not enough information'.\n\n"
        
        "EXAMPLES:\n"
        f"{ENHANCED_SHOTS}\n\n"
        
        "CURRENT QUERY: {query}\n\n"
        "ANALYSIS AND ANSWER:"
    )
)

# Enhanced tools list
tools = [get_financial_metrics, get_market_news, get_economic_indicators,get_fed_policy_info,get_technical_indicators]

# LLM with better parameters
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct", 
    temperature=0.05,  # Lower temperature for more consistent answers
    api_key=groqK, 
    # base_url='https://api.opentyphoon.ai/v1',
    max_tokens= 2048,
    max_retries=3,
)

validator_llm = ChatMistralAI(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.0,  # Use deterministic responses for validation
    api_key=mistral_key,
    max_tokens=32,
    max_retries=3,
)

promptValidator = PromptTemplate(
    input_variables=["query", "candidate"],
    template=(
        "You are a financial expert. Validate the answer to the following question:\n"
        "Question: {query}\n"
        "Candidate Answer: {candidate}\n"
        "Is this answer valid? Reply with 'OK' if'A', 'B', 'C', 'D', 'RISE', or 'FALL' is the answer else needs correction."
    )
)

# Create enhanced agent with better compatibility
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # Better for complex tools
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate"
)

def extract_answer(response: str) -> str:
    """
    Extracts a clean answer: one of 'A', 'B', 'C', 'D', 'Rise', or 'Fall'.
    Prioritizes a letter at the start, then standalone letters, then keywords.
    """

    text = response.strip()
    upper = text.upper()

    # 1. Check for letter at start (e.g., "A. ...", "B) ...", "C ...")
    m_start = re.match(r'^[ \t]*([A-D])(?:[\.\)]\s*|$)', upper)
    if m_start:
        return m_start.group(1)

    # 2. Look for any standalone letter answer elsewhere
    letter_matches = re.findall(r'\b([A-D])\b', upper)
    if letter_matches:
        return letter_matches[-1]

    # 3. Check for keywords that are not rise or fall
    low = text.lower()
    if any(word in low for word in ['up', 'increase', 'rise', 'higher']):
        return 'Rise'
    if any(word in low for word in ['down', 'downward', 'decrease', 'fall', 'lower']):
        return 'Fall'
    
    if any(word in low for word in ['ขึ้น', 'เพิ่มขึ้น', 'สูงขึ้น']):
        return 'Rise'
    if any(word in low for word in ['ลง', 'ตก', 'ลดลง']):
        return 'Fall'

    # 4. Fallback: return stripped text
    return text

# validator llm 
def validate_answer(q,candidate):
    """
    Validates the answer using a simple LLM call.
    Returns True if the answer is valid, False otherwise.
    """
    try:
        resp = validator_llm.predict(promptValidator).strip()
        # Normalise
        resp_up = resp.upper().replace('.', '').replace(')', '')
        if resp_up == "OK":
            return candidate
        if resp_up in {"A","B","C","D","RISE","FALL"}:
            return resp_up
    except Exception as e:
        # If validator barfs, trust the original
        print("Validator error:", e)
    return candidate


# Enhanced inference loop with error handling
print("Starting prediction generation...")
# Open log file for writing process logs
log_file = open('process_log.txt', 'w', encoding='utf-8')
for idx, row in test_df.iterrows():
    query = row["query"]
    print(f"\nProcessing query {idx + 1}/{len(test_df)}: {query[:100]}...")
    log_file.write(f"Processing query {idx+1}/{len(test_df)}: {query}\n")
    
    # Set the global reference date for this query
    global CURRENT_REF_DATE
    CURRENT_REF_DATE = extract_reference_date(query)
    
    try:
        # Run the agent
        raw_answer = agent.run(query)
        
        # Extract clean answer
        clean_answer = extract_answer(raw_answer)
        #final_answer = validate_answer(query, clean_answer)
        # Log raw and clean answers
        log_file.write(f"Raw answer: {raw_answer}\nClean answer: {clean_answer}\n\n")
        log_file.flush()
        
        print(f"Raw answer: {raw_answer}")
        print(f"Clean answer: {clean_answer}")
        
        submission.at[idx, "prediction"] = clean_answer
        
    except Exception as e:
        log_file.write(f"Error processing query {idx+1}: {str(e)}\n")
        print(f"Error processing query '{query}': {str(e)}")
        # Try a simple LLM call without tools as fallback
        try:
            fallback_prompt = f"Answer this financial question with A, B, C, or D only:\n{query}\nAnswer:"
            fallback_response = llm.predict(fallback_prompt)
            clean_answer = extract_answer(fallback_response)
            submission.at[idx, "prediction"] = clean_answer
            print(f"Fallback answer: {clean_answer}")
            log_file.write(f"Fallback answer: {clean_answer}\n\n")
            log_file.flush()
        except:
            submission.at[idx, "prediction"] = "A"  # Default fallback
            print("Using default answer: A")

    # End of for loop

# Close the log file
log_file.close()

print("\nFinished generating predictions. Saving to submission.csv...")

# 4. Save results
submission.to_csv("submission.csv", index=False)
print("Submission saved successfully!")

# Optional: Print some statistics
print(f"\nPrediction distribution:")
print(submission['prediction'].value_counts())




