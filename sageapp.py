"""
MarketSage — Flask Backend  (Tier 2: Live Data)
================================================
New in Tier 2:
  • Alpha Vantage — live quote prices, candlestick OHLCV data
  • NewsAPI        — live financial headlines + sentiment tagging
  • /api/stocks        — real prices from Alpha Vantage (falls back to mock)
  • /api/chart/<sym>   — 30-day daily OHLCV for Chart.js candlesticks
  • /api/news          — live headlines from NewsAPI (falls back to mock)
  • /api/predict/<sym> — uses live price to anchor the prediction

Dependencies:
    pip install flask flask-cors flask-jwt-extended bcrypt flask-sqlalchemy requests
"""

import os, smtplib, random, string, time, requests
from datetime import timedelta, datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from groq import Groq

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


# ─────────────────────────────────────────
# ★  CONFIG — paste your real keys here  ★
# ─────────────────────────────────────────
GMAIL_ADDRESS      = "ankitabandal45@gmail.com"
GMAIL_APP_PASS     = "oligapnxwfwupayx"

# Free-forever keys — replace with yours:
#   Alpha Vantage → https://www.alphavantage.co/support/#api-key  (25 req/day free)
#   NewsAPI       → https://newsapi.org/register                  (100 req/day free)
ALPHAVANTAGE_KEY   = "YOUR_ALPHAVANTAGE_KEY"   # ← replace
NEWSAPI_KEY        = "YOUR_NEWSAPI_KEY"         # ← replace
# ─────────────────────────────────────────


load_dotenv()

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)
app.config["JWT_SECRET_KEY"]                 = os.getenv("JWT_SECRET_KEY", "marketsage-secret")
app.config["JWT_ACCESS_TOKEN_EXPIRES"]       = timedelta(hours=24)
app.config["SQLALCHEMY_DATABASE_URI"]        = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


CORS(app)
jwt = JWTManager(app)
db  = SQLAlchemy(app)

# ─────────────────────────────────────────
# DATABASE MODELS
# ─────────────────────────────────────────

class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120), default="")
    email         = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.LargeBinary, nullable=False)
    verified      = db.Column(db.Boolean, default=False)
    created_at    = db.Column(db.Float, default=time.time)
    holdings      = db.relationship("Holding", backref="user", lazy=True,
                                    cascade="all, delete-orphan")

class OTPStore(db.Model):
    __tablename__ = "otp_store"
    id      = db.Column(db.Integer, primary_key=True)
    email   = db.Column(db.String(200), unique=True, nullable=False)
    otp     = db.Column(db.String(10), nullable=False)
    purpose = db.Column(db.String(20), default="verify")
    expires = db.Column(db.Float, nullable=False)

class Holding(db.Model):
    __tablename__ = "holdings"
    id      = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    sym     = db.Column(db.String(20), nullable=False)
    qty     = db.Column(db.Float, nullable=False)
    avg     = db.Column(db.Float, nullable=False)

# Simple in-memory price cache — avoids burning Alpha Vantage quota
# { "AAPL": { "data": {...}, "ts": 1234567890 } }
_price_cache = {}
_chart_cache = {}
_news_cache  = {"data": None, "ts": 0}

CACHE_TTL_PRICE = 60    # seconds — refresh price every 60s
CACHE_TTL_CHART = 3600  # 1 hour for daily candles
CACHE_TTL_NEWS  = 300   # 5 minutes for news

with app.app_context():
    db.create_all()
    db_url = os.getenv("DATABASE_URL", "not set")
    print(f"✅  Database ready → {db_url}")
# ─────────────────────────────────────────
# MOCK FALLBACK DATA
# ─────────────────────────────────────────
MOCK_STOCKS = [
    {"sym": "AAPL",  "co": "Apple Inc.",      "price": 193.45, "chgp": +0.64, "vol": "68.2M",  "sig": "BUY"},
    {"sym": "TSLA",  "co": "Tesla, Inc.",      "price": 247.32, "chgp": +3.72, "vol": "112.5M", "sig": "BUY"},
    {"sym": "MSFT",  "co": "Microsoft Corp.",  "price": 417.20, "chgp": +0.19, "vol": "22.1M",  "sig": "HOLD"},
    {"sym": "NVDA",  "co": "NVIDIA Corp.",     "price": 875.50, "chgp": +2.59, "vol": "55.3M",  "sig": "BUY"},
    {"sym": "META",  "co": "Meta Platforms",   "price": 502.80, "chgp": -1.64, "vol": "18.7M",  "sig": "SELL"},
    {"sym": "AMZN",  "co": "Amazon.com",       "price": 185.20, "chgp": +0.22, "vol": "30.4M",  "sig": "HOLD"},
    {"sym": "GOOGL", "co": "Alphabet Inc.",    "price": 176.80, "chgp": -1.17, "vol": "24.6M",  "sig": "SELL"},
    {"sym": "NFLX",  "co": "Netflix Inc.",     "price": 628.40, "chgp": +2.31, "vol": "9.8M",   "sig": "BUY"},
]

COMPANY_NAMES = {
    "AAPL": "Apple Inc.", "TSLA": "Tesla, Inc.", "MSFT": "Microsoft Corp.",
    "NVDA": "NVIDIA Corp.", "META": "Meta Platforms", "AMZN": "Amazon.com",
    "GOOGL": "Alphabet Inc.", "NFLX": "Netflix Inc.",
}

MOCK_NEWS = [
    {"src": "Reuters",   "time": "10m ago", "sent": "BULLISH", "scl": "buy",
     "title": "NVIDIA surges on record data center demand", "blurb": "Analysts raise price targets after Q3 beat.", "url": "#"},
    {"src": "Bloomberg", "time": "34m ago", "sent": "BEARISH", "scl": "sell",
     "title": "Meta faces fresh antitrust scrutiny in EU", "blurb": "Regulators preparing a formal probe.", "url": "#"},
    {"src": "WSJ",       "time": "1h ago",  "sent": "NEUTRAL", "scl": "hold",
     "title": "Fed holds rates steady — two cuts possible", "blurb": "Cooling inflation data cited.", "url": "#"},
    {"src": "CNBC",      "time": "2h ago",  "sent": "BULLISH", "scl": "buy",
     "title": "Apple Vision Pro 2 enters mass production", "blurb": "Major component orders confirmed.", "url": "#"},
    {"src": "FT",        "time": "3h ago",  "sent": "BEARISH", "scl": "sell",
     "title": "AWS loses two enterprise contracts to Google Cloud", "blurb": "Gemini-powered tools drove decisions.", "url": "#"},
]

# ─────────────────────────────────────────
# HELPERS — AUTH
# ─────────────────────────────────────────

def generate_otp(length=6):
    return "".join(random.choices(string.digits, k=length))

def hash_password(plain: str) -> bytes:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt())

def check_password(plain: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed)

def send_otp_email(to_email: str, otp: str, purpose: str = "verify"):
    subject = "MarketSage — Verify your email" if purpose == "verify" \
              else "MarketSage — Password reset code"
    body_html = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;
                background:#0b1120;color:#e8eaf0;padding:32px;border-radius:12px;">
      <h2 style="color:#6ee7b7;margin-bottom:8px;">MarketSage</h2>
      <p style="color:#9ca3af;font-size:14px;">
        {'Verify your email address' if purpose == 'verify' else 'Reset your password'}
      </p>
      <div style="background:#111827;border-radius:8px;padding:24px;
                  text-align:center;margin:24px 0;">
        <p style="color:#9ca3af;font-size:13px;margin-bottom:12px;">Your one-time code</p>
        <div style="font-family:monospace;font-size:36px;font-weight:600;
                    letter-spacing:10px;color:#6ee7b7;">{otp}</div>
      </div>
      <p style="color:#6b7280;font-size:12px;">
        Expires in <strong>10 minutes</strong>. Do not share this code.
      </p>
    </div>"""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = to_email
    msg.attach(MIMEText(body_html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
            server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())
        print(f"📧  OTP email sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌  Email failed: {e}")
        print(f"    Console fallback OTP for {to_email}: {otp}")
        return False

def upsert_otp(email: str, otp: str, purpose: str):
    record = OTPStore.query.filter_by(email=email).first()
    if record:
        record.otp = otp; record.purpose = purpose; record.expires = time.time() + 600
    else:
        record = OTPStore(email=email, otp=otp, purpose=purpose, expires=time.time() + 600)
        db.session.add(record)
    db.session.commit()

# ─────────────────────────────────────────
# HELPERS — ALPHA VANTAGE
# ─────────────────────────────────────────

def _av_get(params: dict):
    """Make a request to Alpha Vantage. Returns JSON dict or None."""
    if ALPHAVANTAGE_KEY == "YOUR_ALPHAVANTAGE_KEY":
        return None   # key not set — skip live call
    try:
        params["apikey"] = ALPHAVANTAGE_KEY
        r = requests.get("https://www.alphavantage.co/query", params=params, timeout=8)
        data = r.json()
        if "Note" in data or "Information" in data:
            # Rate limit hit
            print("⚠️  Alpha Vantage rate limit:", data.get("Note") or data.get("Information"))
            return None
        return data
    except Exception as e:
        print(f"❌  Alpha Vantage error: {e}")
        return None

def _signal_from_change(chgp: float) -> str:
    if chgp > 1.5:  return "BUY"
    if chgp < -1.5: return "SELL"
    return "HOLD"

def fetch_live_quote(symbol: str) -> dict | None:
    """Fetch live quote from Alpha Vantage GLOBAL_QUOTE endpoint."""
    cached = _price_cache.get(symbol)
    if cached and time.time() - cached["ts"] < CACHE_TTL_PRICE:
        return cached["data"]

    data = _av_get({"function": "GLOBAL_QUOTE", "symbol": symbol})
    if not data or "Global Quote" not in data:
        return None
    q = data["Global Quote"]
    if not q.get("05. price"):
        return None
    try:
        price = float(q["05. price"])
        prev  = float(q["08. previous close"])
        chgp  = float(q["10. change percent"].replace("%", ""))
        vol   = int(q["06. volume"])
        # Format volume nicely
        if vol >= 1_000_000:
            vol_str = f"{vol/1_000_000:.1f}M"
        elif vol >= 1_000:
            vol_str = f"{vol/1_000:.0f}K"
        else:
            vol_str = str(vol)
        result = {
            "sym":  symbol,
            "co":   COMPANY_NAMES.get(symbol, symbol),
            "price": round(price, 2),
            "chgp":  round(chgp, 2),
            "vol":   vol_str,
            "sig":   _signal_from_change(chgp),
            "live":  True,
        }
        _price_cache[symbol] = {"data": result, "ts": time.time()}
        return result
    except (ValueError, KeyError) as e:
        print(f"❌  Parse error for {symbol}: {e}")
        return None

def fetch_daily_candles(symbol: str) -> list | None:
    """Fetch last 30 daily candles from Alpha Vantage TIME_SERIES_DAILY."""
    cached = _chart_cache.get(symbol)
    if cached and time.time() - cached["ts"] < CACHE_TTL_CHART:
        return cached["data"]

    data = _av_get({"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "compact"})
    if not data or "Time Series (Daily)" not in data:
        return None
    try:
        series = data["Time Series (Daily)"]
        candles = []
        for date_str in sorted(series.keys())[-30:]:
            d = series[date_str]
            candles.append({
                "t": date_str,
                "o": float(d["1. open"]),
                "h": float(d["2. high"]),
                "l": float(d["3. low"]),
                "c": float(d["4. close"]),
                "v": int(d["5. volume"]),
            })
        _chart_cache[symbol] = {"data": candles, "ts": time.time()}
        return candles
    except Exception as e:
        print(f"❌  Candle parse error for {symbol}: {e}")
        return None

def make_mock_candles(symbol: str) -> list:
    """Generate realistic-looking mock candle data for demo."""
    # Use the mock stock's price as anchor
    anchor = next((s["price"] for s in MOCK_STOCKS if s["sym"] == symbol), 200.0)
    candles = []
    price = anchor * 0.88
    import datetime as dt_mod
    today = dt_mod.date.today()
    for i in range(30):
        day = today - dt_mod.timedelta(days=30 - i)
        # Skip weekends
        if day.weekday() >= 5:
            continue
        open_  = round(price + random.uniform(-2, 2), 2)
        close  = round(open_ + random.uniform(-5, 5), 2)
        high   = round(max(open_, close) + random.uniform(0, 3), 2)
        low    = round(min(open_, close) - random.uniform(0, 3), 2)
        vol    = random.randint(10_000_000, 80_000_000)
        candles.append({"t": str(day), "o": open_, "h": high, "l": low, "c": close, "v": vol})
        price  = close
    return candles

# ─────────────────────────────────────────
# HELPERS — NEWSAPI
# ─────────────────────────────────────────

BULLISH_WORDS = {"surge", "soar", "beat", "record", "rally", "gain", "profit",
                 "growth", "rise", "upgrade", "bullish", "strong", "win", "boost"}
BEARISH_WORDS  = {"fall", "drop", "crash", "loss", "decline", "antitrust", "probe",
                  "fine", "bearish", "weak", "miss", "cut", "concern", "risk", "fear"}

def _sentiment(text: str) -> tuple[str, str]:
    words = set(text.lower().split())
    b = len(words & BULLISH_WORDS)
    s = len(words & BEARISH_WORDS)
    if b > s:   return "BULLISH", "buy"
    if s > b:   return "BEARISH", "sell"
    return "NEUTRAL", "hold"

def _time_ago(published: str) -> str:
    try:
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        mins = int(delta.total_seconds() // 60)
        if mins < 60:    return f"{mins}m ago"
        if mins < 1440:  return f"{mins // 60}h ago"
        return f"{mins // 1440}d ago"
    except Exception:
        return "recently"

def fetch_live_news() -> list | None:
    if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY":
        return None
    # Check cache
    if _news_cache["data"] and time.time() - _news_cache["ts"] < CACHE_TTL_NEWS:
        return _news_cache["data"]
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": "stock market OR earnings OR NYSE OR NASDAQ OR Fed OR S&P500",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 15,
                "apiKey": NEWSAPI_KEY,
            },
            timeout=8,
        )
        data = r.json()
        if data.get("status") != "ok":
            print("⚠️  NewsAPI error:", data.get("message"))
            return None
        articles = []
        for a in data.get("articles", [])[:12]:
            title  = a.get("title", "") or ""
            blurb  = a.get("description", "") or ""
            src    = (a.get("source") or {}).get("name", "News") or "News"
            url    = a.get("url", "#") or "#"
            pub    = a.get("publishedAt", "")
            sent, scl = _sentiment(title + " " + blurb)
            articles.append({
                "src": src[:20],
                "time": _time_ago(pub),
                "sent": sent,
                "scl":  scl,
                "title": title[:120],
                "blurb": blurb[:200],
                "url":   url,
            })
        _news_cache["data"] = articles
        _news_cache["ts"]   = time.time()
        return articles
    except Exception as e:
        print(f"❌  NewsAPI error: {e}")
        return None

# ─────────────────────────────────────────
# ROUTES — FRONTEND
# ─────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(TEMPLATES_DIR, "dashboard.html")

# ─────────────────────────────────────────
# ROUTES — AUTH  (unchanged from Tier 1)
# ─────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def register():
    data     = request.get_json()
    name     = data.get("name", "").strip()
    email    = data.get("email", "").lower().strip()
    password = data.get("password", "")
    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required."}), 400
    if len(password) < 8:
        return jsonify({"success": False, "message": "Password must be at least 8 characters."}), 400
    existing = User.query.filter_by(email=email).first()
    if existing and existing.verified:
        return jsonify({"success": False, "message": "Email already registered."}), 409
    if existing and not existing.verified:
        existing.name = name; existing.password_hash = hash_password(password)
        db.session.commit()
    else:
        db.session.add(User(name=name, email=email,
                            password_hash=hash_password(password), verified=False))
        db.session.commit()
    otp = generate_otp()
    upsert_otp(email, otp, "verify")
    send_otp_email(email, otp, "verify")
    return jsonify({"success": True, "message": "OTP sent to your email."})

@app.route("/api/verify-otp", methods=["POST"])
def verify_otp():
    data   = request.get_json()
    email  = data.get("email", "").lower().strip()
    otp_in = data.get("otp", "").strip()
    record = OTPStore.query.filter_by(email=email, purpose="verify").first()
    if not record:
        return jsonify({"success": False, "message": "No OTP found. Register first."}), 404
    if time.time() > record.expires:
        db.session.delete(record); db.session.commit()
        return jsonify({"success": False, "message": "OTP expired. Please re-register."}), 410
    if record.otp != otp_in:
        return jsonify({"success": False, "message": "Invalid OTP."}), 401
    user = User.query.filter_by(email=email).first()
    if user: user.verified = True
    db.session.delete(record); db.session.commit()
    token = create_access_token(identity=email)
    return jsonify({"success": True, "token": token, "message": "Account verified successfully."})

@app.route("/api/resend-otp", methods=["POST"])
def resend_otp():
    data  = request.get_json()
    email = data.get("email", "").lower().strip()
    if not User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "Email not found."}), 404
    otp = generate_otp()
    upsert_otp(email, otp, "verify")
    send_otp_email(email, otp, "verify")
    return jsonify({"success": True, "message": "New OTP sent."})

@app.route("/api/login", methods=["POST"])
def login():
    data     = request.get_json()
    email    = data.get("email", "").lower().strip()
    password = data.get("password", "")
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"success": False, "message": "Email not found."}), 404
    if not user.verified:
        return jsonify({"success": False, "message": "Account not verified. Check your email."}), 403
    if not check_password(password, user.password_hash):
        return jsonify({"success": False, "message": "Incorrect password."}), 401
    token = create_access_token(identity=email)
    return jsonify({"success": True, "token": token, "name": user.name})

@app.route("/api/me", methods=["GET"])
@jwt_required()
def me():
    email = get_jwt_identity()
    user  = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"email": user.email, "name": user.name})

@app.route("/api/forgot-password", methods=["POST"])
def forgot_password():
    data  = request.get_json()
    email = data.get("email", "").lower().strip()
    user  = User.query.filter_by(email=email).first()
    if user:
        otp = generate_otp()
        upsert_otp(email, otp, "reset")
        send_otp_email(email, otp, "reset")
    return jsonify({"success": True, "message": "If that email is registered, a reset code has been sent."})

@app.route("/api/reset-password", methods=["POST"])
def reset_password():
    data         = request.get_json()
    email        = data.get("email", "").lower().strip()
    otp_in       = data.get("otp", "").strip()
    new_password = data.get("new_password", "")
    if len(new_password) < 8:
        return jsonify({"success": False, "message": "Password must be at least 8 characters."}), 400
    record = OTPStore.query.filter_by(email=email, purpose="reset").first()
    if not record:
        return jsonify({"success": False, "message": "No reset code found."}), 404
    if time.time() > record.expires:
        db.session.delete(record); db.session.commit()
        return jsonify({"success": False, "message": "Code expired."}), 410
    if record.otp != otp_in:
        return jsonify({"success": False, "message": "Invalid code."}), 401
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"success": False, "message": "User not found."}), 404
    user.password_hash = hash_password(new_password)
    user.verified      = True
    db.session.delete(record); db.session.commit()
    token = create_access_token(identity=email)
    return jsonify({"success": True, "token": token, "message": "Password reset successfully."})

# ─────────────────────────────────────────
# ROUTES — MARKET DATA (Tier 2 upgraded)
# ─────────────────────────────────────────

SYMBOLS = ["AAPL", "TSLA", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "NFLX"]

@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    """
    Try Alpha Vantage for each symbol.
    Alpha Vantage free tier = 25 req/day → we batch-fetch only once per minute
    and serve cached data after that.
    Falls back gracefully to mock data per symbol if API unavailable.
    """
    results = []
    for sym in SYMBOLS:
        live = fetch_live_quote(sym)
        if live:
            results.append(live)
        else:
            # Use mock with slight random jitter so the UI feels live
            mock = next((s for s in MOCK_STOCKS if s["sym"] == sym), None)
            if mock:
                results.append({**mock, "price": round(mock["price"] + random.uniform(-0.5, 0.5), 2), "live": False})
    return jsonify({"stocks": results})

@app.route("/api/stocks/<symbol>", methods=["GET"])
def get_stock(symbol):
    sym  = symbol.upper()
    live = fetch_live_quote(sym)
    if live:
        return jsonify(live)
    mock = next((s for s in MOCK_STOCKS if s["sym"] == sym), None)
    if not mock:
        return jsonify({"error": "Symbol not found"}), 404
    return jsonify({**mock, "live": False})

@app.route("/api/chart/<symbol>", methods=["GET"])
def get_chart(symbol):
    """
    Returns 30 daily OHLCV candles for Chart.js candlestick chart.
    Falls back to mock candles if API key not set or quota hit.
    """
    sym     = symbol.upper()
    candles = fetch_daily_candles(sym)
    if not candles:
        candles = make_mock_candles(sym)
    return jsonify({"symbol": sym, "candles": candles, "live": bool(fetch_daily_candles.__name__)})

@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    preds = []
    for sym in ["AAPL", "MSFT", "META", "NVDA", "AMZN"]:
        live = fetch_live_quote(sym)
        price = live["price"] if live else next((s["price"] for s in MOCK_STOCKS if s["sym"] == sym), 200.0)
        chgp  = live["chgp"]  if live else random.uniform(-3, 3)
        sig   = live["sig"]   if live else _signal_from_change(chgp)
        # Simple target: +/- 5% based on signal
        mult  = 1.05 if sig == "BUY" else (0.95 if sig == "SELL" else 1.01)
        cond_map = {"BUY": "Price expected to rise", "HOLD": "Consolidation phase", "SELL": "Price expected to fall"}
        preds.append({
            "sym": sym, "price": round(price, 2),
            "target": round(price * mult, 2),
            "cond": cond_map[sig], "sig": sig,
        })
    return jsonify({"predictions": preds})

@app.route("/api/predict/<symbol>", methods=["GET"])
def predict_symbol(symbol):
    sym   = symbol.upper()
    live  = fetch_live_quote(sym)
    price = live["price"] if live else round(random.uniform(100, 1000), 2)
    chgp  = live["chgp"]  if live else random.uniform(-3, 3)
    sig   = live["sig"]   if live else _signal_from_change(chgp)
    mult  = 1.05 if sig == "BUY" else (0.95 if sig == "SELL" else 1.01)
    cond_map = {"BUY": "Price expected to rise", "HOLD": "Consolidation phase", "SELL": "Price expected to fall"}
    return jsonify({
        "sym": sym, "price": round(price, 2),
        "target": round(price * mult, 2),
        "cond": cond_map[sig], "sig": sig,
    })

@app.route("/api/news", methods=["GET"])
def get_news():
    """Return live NewsAPI headlines, falls back to mock."""
    live = fetch_live_news()
    if live:
        return jsonify({"news": live, "live": True})
    return jsonify({"news": MOCK_NEWS, "live": False})

# ─────────────────────────────────────────
# ROUTES — PORTFOLIO
# ─────────────────────────────────────────

@app.route("/api/portfolio", methods=["GET"])
@jwt_required()
def get_portfolio():
    email = get_jwt_identity()
    user  = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"holdings": []})
    return jsonify({"holdings": [{"sym": h.sym, "qty": h.qty, "avg": h.avg}
                                  for h in user.holdings]})

@app.route("/api/portfolio/add", methods=["POST"])
@jwt_required()
def add_holding():
    email = get_jwt_identity()
    data  = request.get_json()
    sym   = data.get("sym", "").upper()
    qty   = data.get("qty", 0)
    avg   = data.get("avg", 0)
    if not sym or qty <= 0 or avg <= 0:
        return jsonify({"success": False, "message": "Invalid data."}), 400
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"success": False, "message": "User not found."}), 404
    h = Holding.query.filter_by(user_id=user.id, sym=sym).first()
    if h:
        h.qty = qty; h.avg = avg
    else:
        db.session.add(Holding(user_id=user.id, sym=sym, qty=qty, avg=avg))
    db.session.commit()
    return jsonify({"success": True, "message": f"{sym} added."})

@app.route("/api/portfolio/remove/<sym>", methods=["DELETE"])
@jwt_required()
def remove_holding(sym):
    email = get_jwt_identity()
    user  = User.query.filter_by(email=email).first()
    if user:
        Holding.query.filter_by(user_id=user.id, sym=sym.upper()).delete()
        db.session.commit()
    return jsonify({"success": True, "message": f"{sym} removed."})

# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    av_status  = "configured" if ALPHAVANTAGE_KEY != "YOUR_ALPHAVANTAGE_KEY" else "demo (key not set)"
    na_status  = "configured" if NEWSAPI_KEY       != "YOUR_NEWSAPI_KEY"      else "demo (key not set)"
    return jsonify({
        "status":        "ok",
        "service":       "MarketSage API",
        "version":       "3.0.0-tier2",
        "alpha_vantage": av_status,
        "newsapi":       na_status,
    })


#---------------for tire 3 work  (# ROUTES — AI CHATBOT (Groq/Llama3)   ) -----------------------


groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"success": False, "message": "No message provided."}), 400
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are MarketSage AI, an expert stock market analyst and investment advisor. 
                    You provide clear, concise advice about stocks, investments, and market trends.
                    Always remind users that your advice is for educational purposes only and not financial advice.
                    Keep responses under 150 words and be direct and helpful."""
                },
                {"role": "user", "content": user_message}
            ],
            max_tokens=200,
        )
        reply = completion.choices[0].message.content
        return jsonify({"success": True, "reply": reply})
    except Exception as e:
        print(f"❌ Groq error: {e}")
        return jsonify({"success": False, "reply": "AI is temporarily unavailable."}), 500
    
    

# ----------------tire 3 second work (prediction)---------------------------

def predict_stock_price(symbol: str, days: int = 7):
    try:
        import yfinance as yf
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if df.empty or len(df) < 30:
            return None
        prices = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(prices)
        # Create sequences
        X, y = [], []
        lookback = 20
        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        # Train simple model
        model = LinearRegression()
        model.fit(X, y)
        # Predict next N days
        last_seq = scaled[-lookback:, 0].tolist()
        predictions = []
        for _ in range(days):
            inp = np.array(last_seq[-lookback:]).reshape(1, -1)
            pred = model.predict(inp)[0]
            predictions.append(pred)
            last_seq.append(pred)
        # Inverse transform
        pred_prices = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten().tolist()
        current_price = float(prices[-1][0])
        target_price  = round(pred_prices[-1], 2)
        change_pct    = round((target_price - current_price) / current_price * 100, 2)
        signal = "BUY" if change_pct > 1.5 else ("SELL" if change_pct < -1.5 else "HOLD")
        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "predicted_price": target_price,
            "change_pct": change_pct,
            "signal": signal,
            "forecast": [round(p, 2) for p in pred_prices],
            "days": days
        }
    except Exception as e:
        print(f"❌ Prediction error for {symbol}: {e}")
        return None

@app.route("/api/lstm/<symbol>", methods=["GET"])
def lstm_predict(symbol):
    sym = symbol.upper()
    result = predict_stock_price(sym)
    if not result:
        return jsonify({"error": "Could not generate prediction"}), 500
    return jsonify(result)    



#-----------------tire 3 third work (ROUTES — FINBERT SENTIMENT ANALYSIS)--------------------


_sentiment_pipeline = None

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        print("⏳ Loading FinBERT model...")
        _sentiment_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=False
        )
        print("✅ FinBERT loaded!")
    return _sentiment_pipeline

@app.route("/api/sentiment", methods=["POST"])
def analyze_sentiment():
    data = request.get_json()
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    try:
        pipe = get_sentiment_pipeline()
        results = []
        for text in texts[:10]:  # max 10 at a time
            out = pipe(text[:512])[0]
            results.append({
                "text": text[:100],
                "label": out["label"].upper(),
                "score": round(out["score"], 3)
            })
        return jsonify({"results": results})
    except Exception as e:
        print(f"❌ FinBERT error: {e}")
        return jsonify({"error": str(e)}), 500
    
    
    
# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 54)
    print("  MarketSage API Tier 2 — http://localhost:5000")
    print(f"  Alpha Vantage : {'✅ live' if ALPHAVANTAGE_KEY != 'YOUR_ALPHAVANTAGE_KEY' else '⚠️  key not set — using mock'}")
    print(f"  NewsAPI       : {'✅ live' if NEWSAPI_KEY != 'YOUR_NEWSAPI_KEY' else '⚠️  key not set — using mock'}")
    print("=" * 54)
    app.run(debug=True, port=5000, use_reloader=False)