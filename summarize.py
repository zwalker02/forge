import os, json, re, feedparser, requests, yaml
from datetime import datetime, timedelta
from dateutil import tz
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import List, Dict, Any

BASE = os.path.dirname(os.path.abspath(__file__))
TZ = tz.gettz("America/Phoenix")

# ---------- Load config & sources ----------
with open(os.path.join(BASE, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

with open(os.path.join(BASE, "sources.yaml"), "r", encoding="utf-8") as f:
    SOURCES = yaml.safe_load(f)

# ---------- OpenAI ----------
def llm_json(title: str, snippet: str, max_tokens=180) -> Dict[str, str]:
    """Ask the model for JSON: {'summary': '...', 'why': '...'}."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    system = open(os.path.join(BASE, "prompts", "system.txt"), encoding="utf-8").read()
    user = f"Title: {title}\nSource snippet: {snippet}\nRespond with ONLY the JSON object."

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        },
        timeout=90,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:300]}")

    text = resp.json()["choices"][0]["message"]["content"].strip()
    # Try strict JSON; if it comes back with code fences, strip them.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        data = json.loads(text)
        out = {
            "summary": str(data.get("summary", "")).strip(),
            "why": str(data.get("why", "")).strip(),
        }
        if not out["summary"] or not out["why"]:
            raise ValueError("missing fields")
        return out
    except Exception:
        # Loose fallback: try to split on "Why it matters:"
        why = ""
        m = re.search(r"(?i)why it matters\s*:\s*(.+)$", text)
        if m:
            why = "Why it matters: " + m.group(1).strip()
            text = re.sub(r"(?i)why it matters\s*:\s*.+$", "", text).strip()
        summary = text.strip()
        if not summary:
            summary = snippet[:280].strip()
        if not why:
            why = "Why it matters: see source for context."
        return {"summary": summary, "why": why}

# ---------- Fetch & prep ----------
def fetch_rss() -> List[Dict[str, Any]]:
    items = []
    for src in SOURCES.get("rss", []):
        d = feedparser.parse(src["url"])
        for e in d.entries:
            when = None
            if getattr(e, "published_parsed", None):
                when = datetime(*e.published_parsed[:6], tzinfo=tz.tzutc()).astimezone(TZ)
            items.append({
                "source": src["name"],
                "url": getattr(e, "link", ""),
                "title": getattr(e, "title", ""),
                "summary": re.sub(r"<.*?>", "", getattr(e, "summary", "") or "").strip(),
                "published": when,
                "type": "rss",
            })
    return items

def fetch_sec_current() -> List[Dict[str, Any]]:
    url = SOURCES.get("sec", {}).get("current_feed")
    if not url: return []
    d = feedparser.parse(url)
    out = []
    for e in d.entries:
        when = None
        if getattr(e, "updated_parsed", None):
            when = datetime(*e.updated_parsed[:6], tzinfo=tz.tzutc()).astimezone(TZ)
        out.append({
            "source": "SEC EDGAR",
            "url": getattr(e, "link", ""),
            "title": getattr(e, "title", ""),
            "summary": re.sub(r"<.*?>", "", getattr(e, "summary", "") or "").strip(),
            "published": when,
            "type": "sec",
        })
    return out

def rank_and_dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); out = []
    for it in items:
        key = (it.get("title","")[:140].lower(), it.get("source",""))
        if key in seen: continue
        seen.add(key); out.append(it)
    out.sort(key=lambda x: (x.get("published") or datetime.now(TZ)), reverse=True)
    return out

# ---------- Bucketing ----------
MACRO_KEYS = ["federal reserve", "fomc", "bureau of labor", "bls", "bureau of economic", "bea",
              "cpi", "inflation", "gdp", "unemployment", "payrolls", "retail sales", "ppi"]

def _classify_item(it):
    if it.get("type") == "sec": return "companies"
    blob = (it.get("source","") + " " + it.get("title","")).lower()
    if any(k in blob for k in MACRO_KEYS): return "macro"
    return "markets"

def _bucket(items):
    b = {"macro": [], "markets": [], "companies": []}
    for it in items:
        b[_classify_item(it)].append(it)
    for k in b:
        b[k].sort(key=lambda x: x.get("published") or datetime.now(TZ), reverse=True)
    return b

# ---------- Summarize per-article ----------
def summarize_sections(collected: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    per_section = CONFIG.get("display", {}).get("per_section", 3)
    buckets = _bucket(collected)
    results = {"macro": [], "markets": [], "companies": []}

    for sec in ["macro", "markets", "companies"]:
        for it in buckets[sec][:per_section]:
            title = it.get("title", "(no title)")
            snippet = it.get("summary", "") or title
            try:
                js = llm_json(title, snippet)
            except Exception as e:
                js = {
                    "summary": snippet[:300],
                    "why": f"Why it matters: source context unavailable ({e})."
                }
            results[sec].append({
                "title": title,
                "url": it.get("url", "#"),
                "summary": js["summary"],
                "why": js["why"],
                "published": it.get("published").strftime("%Y-%m-%d") if it.get("published") else ""
            })
    return results

# ---------- Render & email ----------
def render_email(sections, sources, period_label):
    env = Environment(loader=FileSystemLoader(os.path.join(BASE, "templates")),
                      autoescape=select_autoescape())
    tpl = env.get_template("email.html.j2")

    subject_prefix = CONFIG.get("email", {}).get("subject_prefix", "[Weekly Finance Brief]")
    subject = f"{subject_prefix} Week of {period_label}"
    intro = "This week’s top financial developments — title, summary, and why they matter."

    html = tpl.render(
        subject=subject,
        intro=intro,
        week_range=period_label,
        macro=sections["macro"],
        markets=sections["markets"],
        companies=sections["companies"],
    )
    return subject, html

def compute_period_label(period: str) -> str:
    today = datetime.now(TZ).date()
    if period == "week":
        start = today - timedelta(days=today.weekday()); end = start + timedelta(days=6)
        return f"{start.isoformat()} — {end.isoformat()}"
    return today.isoformat()

def send_email(subject, html):
    sg_key = os.environ.get("SENDGRID_API_KEY")
    if sg_key:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail, Email
        msg = Mail(
            from_email=(os.environ.get("SMTP_FROM") or "no-reply@example.com",
                        CONFIG.get("email",{}).get("from_name","Finance Brief Bot")),
            to_emails=CONFIG["recipients"]["to"],
            subject=subject,
            html_content=html,
        )
        cc = CONFIG["recipients"].get("cc", []); bcc = CONFIG["recipients"].get("bcc", [])
        if cc: msg.cc = cc
        if bcc: msg.bcc = bcc
        if CONFIG.get("email",{}).get("reply_to"):
            msg.reply_to = Email(CONFIG["email"]["reply_to"])
        SendGridAPIClient(sg_key).send(msg); return

    # SMTP fallback
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    host = os.environ.get("SMTP_HOST"); port = int(os.environ.get("SMTP_PORT","587"))
    user = os.environ.get("SMTP_USER"); pwd = os.environ.get("SMTP_PASS")
    from_addr = os.environ.get("SMTP_FROM") or "no-reply@example.com"
    if not (host and user and pwd):
        raise RuntimeError("No email transport configured. Set SENDGRID_API_KEY or SMTP_* env vars.")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject; msg["From"] = from_addr
    msg["To"] = ", ".join(CONFIG["recipients"]["to"])
    if CONFIG["recipients"].get("cc"): msg["Cc"] = ", ".join(CONFIG["recipients"]["cc"])
    msg.attach(MIMEText(html, "html", "utf-8"))
    with smtplib.SMTP(host, port) as s:
        s.starttls(); s.login(user, pwd); s.send_message(msg)

# ---------- Main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="week", choices=["day","week"])
    ap.add_argument("--send", action="store_true")
    args = ap.parse_args()

    collected = rank_and_dedupe(fetch_rss() + fetch_sec_current())
    sections = summarize_sections(collected)
    subject, html = render_email(sections, collected, compute_period_label(args.period))

    os.makedirs(os.path.join(BASE, "out"), exist_ok=True)
    out_html = os.path.join(BASE, "out", "latest.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print("Wrote", out_html)

    if args.send:
        send_email(subject, html)

if __name__ == "__main__":
    main()
