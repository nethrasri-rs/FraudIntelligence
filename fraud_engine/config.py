import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class Config:
    # PayPal Configuration
    PAYPAL_MODE = os.getenv("PAYPAL_MODE", "sandbox")
    PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
    PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "")
    PAYPAL_BASE_URL = os.getenv("PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com")
    PAYPAL_PERSONAL_EMAIL = os.getenv("PAYPAL_PERSONAL_EMAIL", "")
    PAYPAL_BUSINESS_EMAIL = os.getenv("PAYPAL_BUSINESS_EMAIL", "")

    # Gemini AI
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # Slack Integration
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
    SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
    SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")

    # Server & Storage
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DATABASE_PATH = BASE_DIR / "fraud_intelligence.db"
    DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-fraud-intelligence-2026")
    WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "fi_whsec_99b78e2d40b3c6")

    # Decision Thresholds
    THRESHOLD_BLOCK = float(os.getenv("THRESHOLD_BLOCK", "0.75"))
    THRESHOLD_ALLOW = float(os.getenv("THRESHOLD_ALLOW", "0.30"))
    THRESHOLD_INSUFFICIENT_VISIBILITY = float(os.getenv("THRESHOLD_INSUFFICIENT_VISIBILITY", "0.45"))

    # Observing institution ID (bank core id)
    HOME_INSTITUTION_ID = "PAYPAL_HOME_BANK"

config = Config()
