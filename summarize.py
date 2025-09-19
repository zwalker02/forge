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

# ---------- Relevance + Bucketing Keywords ----------
FINANCE_TERMS = [
    # macro/markets
    "inflation","cpi","ppi","gdp","payrolls","unemployment","retail sales","fomc","federal reserve",
    "interest rate","rates","yield","treasury","bond","mortgage","housing starts",
    "stocks","equities","index","s&p","nasdaq","dow","rally","selloff","volatility","vix",
    "oil","brent","wti","gasoline","gold","silver","copper","bitcoin","crypto","fx","currency","dollar",
    # company / corporate
    "earnings","results","eps","revenue","guidance","outlook","profit","loss","margin",
    "filing","10-k","10-q","8-k","sec","merger","acquisition","m&a","buyback","dividend","layoffs","spinoff","ipo"
]
FIN_RX = re.compile("|".join([re.escape(t) for t in FINANCE_TERMS]), re.I)

MACRO_KEYS = ["federal reserve","fomc","bureau of labor","bls","bureau of economic","bea",
              "cpi","inflation","gdp","unemployment","payrolls","retail sales","ppi",
              "mortgage","treasury","bond","yield"]
COMPANY_KEYS = [
    r"\bearnings?\b", r"\bresults?\b", r"\bEPS\b", r"\brevenue\b", r"\bguidance\b",
    r"\boutlook\b", r"\bprofit\b", r"\bloss\b", r"\b10-K\b", r"\b10-Q\b", r"\b8-K\b",
    r"\bfiling\b", r"\bmerger\b", r"\bacquisition\b", r"\bM&A\b", r"\bbuyback\b",
    r"\bdividend\b", r"\blayoffs?\b", r"\bspinoff\b", r"\bIPO\b", r"\bSEC\b"
]
COMPANY_RX = re.compile("|".join(COMPANY_KEYS), re.I)

# ---------- OpenAI ----------
def llm_json(title: str, snippet: str, max_tokens=260) -> Dict[str, str]:
    """Ask the model for JSON: {'summary': '...', 'why': '...'} with 2–3 sentence WHY."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    system = open(os.path.join(BASE, "prompts", "system.txt"), encoding="utf-8").read()
    user = f"Title: {title}\nSource snippet: {snippet}\nRespond with ONLY the JSON object."

    def _ask(msg_user: str):
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": msg_user},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=90,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:300]}")
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
        return text

    text = _ask(user)
    data = _coerce_json_with_fallback(text, snippet)
    # Ensure WHY has >= 2 sentences; if not, force a rewrite once
    if len(re.findall(r"[.!?](?:\s|$)", data["why"])) < 2:
        user2 = (f"{user}\n\nYour previous WHY was too short. "
                 "Rewrite ONLY the JSON with a WHY that is 2–3 sentences beginning with 'Why it matters:'.")
        text2 = _ask(user2)
        data = _coerce_json_with_fallback(text2, snippet)
    if not data["why"].lower().startswith("why it matters:"):
        data["why"] = "Why it matters: " + data["why"].lstrip()
    return data

def _coerce_json_with_fallback(text: str, snippet: str) -> Dict[str, str]:
    try:
        parsed = json.loads(text)
        out = {
            "summary": str(parsed.get("summary", "")).strip(),
            "why": str(parsed.get("why", "")).strip(),
        }
        if not out["summary"]:
            out["summary"] = snippet[:300].strip()
        if not out["why"]:
            out["why"] = "Why it matters: see source for context."
        return out
    except Exception:
        m = re.search(r"(?i)why it matters\s*:\s*(.+)$", text)
        why = "Why it matters: " + m.group(1).strip() if m else "Why it matters: see source for context."
        if m:
            text = re.sub(r"(?i)why it matters\s*:\s*.+$", "", text).strip()
        summary = text.strip() or snippet[:300].strip()
        return {"summary": summary, "why": why}

# ---------- Fetch & prep ----------
def _clean_html(s: str) -> str:
    return re.sub(r"<.*?>", "", s or "").strip()

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
                "summary": _clean_html(getattr(e, "summary", "")),
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
            "summary": _clean_html(getattr(e, "summary", "")),
            "published": when,
            "type": "sec",
        })
    return out

def rank_and_dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); out = []
    for it in items:
        key = (it.get("title","")[:160].lower(), it.get("source",""))
        if key in seen: continue
        seen.add(key); out.append(it)
    out.sort(key=lambda x: (x.get("published") or datetime.now(TZ)), reverse=True)
    return out

# ---------- Finance relevance + Bucketing ----------
def finance_relevance(it: Dict[str, Any]) -> int:
    """Return #matches of finance terms in title+summary; 0 means not finance-focused."""
    blob = f"{it.get('title','')} {it.get('summary','')}".lower()
    return len(FIN_RX.findall(blob))

def looks_company(it: Dict[str, Any]) -> bool:
    if it.get("type") == "sec":
        return True
    blob = f"{it.get('title','')} {it.get('summary','')}"
    return COMPANY_RX.search(blob) is not None

def classify_item(it: Dict[str, Any]) -> str:
    # filter out non-finance at the classifier level by requiring at least 1 finance term
    if finance_relevance(it) == 0:
        return "drop"
    if looks_company(it):
        return "companies"
    blob = (it.get("source","") + " " + it.get("title","")).lower()
    if any(k in blob for k in MACRO_KEYS):
        return "macro"
    return "markets"

def bucket(items: List[Dict[str, Any]]) -> Dict[str, list]:
    b = {"macro": [], "markets": [], "companies": []}
    for it in items:
        cat = classify_item(it)
        if cat == "drop":  # toss non-finance stories (e.g., geopolitics without a market angle)
            continue
        b[cat].append(it)
    for k in b:
        b[k].sort(key=lambda x: x.get("published") or datetime.now(TZ), reverse=True)
    return b

def backfill_companies(b: Dict[str, list], collected: List[Dict[str, Any]], per_section: int):
    if len(b["companies"]) >= per_section:
        return
    have = {id(x) for x in b["companies"]}
    for it in collected:
        if id(it) in have: continue
        if finance_relevance(it) == 0: continue
        if looks_company(it):
            b["companies"].append(it)
            have.add(id(it))
            if len(b["companies"]) >= per_section:
                break

# ---------- Summarize per-article ----------
def llm_json_safe(title: str, snippet: str) -> Dict[str, str]:
    try:
        return llm_json(title, snippet)
    except Exception as e:
        return {
            "summary": (snippet or title)[:300],
            "why": f"Why it matters: source context unavailable ({e})."
        }

def summarize_sections(collected: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    per_section = CONFIG.get("display", {}).get("per_section", 3)
    b = bucket(collected)
    backfill_companies(b, collected, per_section)

    results = {"macro": [], "markets": [], "companies": []}
    for sec in ["macro", "markets", "companies"]:
        for it in b[sec][:per_section]:
            title = it.get("title", "(no title)")
            snippet = it.get("summary", "") or title
            js = llm_json_safe(title, snippet)
            results[sec].append({
                "title": title,
                "url": it.get("url", "#"),
                "summary": js["summary"],
                "why": js["why"],
                "published": it.get("published").strftime("%Y-%m-%d") if it.get("published") else ""
            })
        # If still empty (rare), synthesize a placeholder so the section isn't blank
        if not results[sec]:
            results[sec].append({
                "title": "No finance-focused items detected",
                "url": "#",
                "summary": "We filtered out non-financial stories this cycle.",
                "why": "Why it matters: fewer finance items can occur in slow news windows; collection will resume next run.",
                "published": ""
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
