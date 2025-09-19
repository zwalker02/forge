import os, sys, json, feedparser, requests, yaml
from datetime import datetime, timedelta
from dateutil import tz
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import List, Dict, Any

BASE = os.path.dirname(os.path.abspath(__file__))

# --- Load config & timezone ---
with open(os.path.join(BASE, "config.yaml"), "r") as f:
    CONFIG = yaml.safe_load(f)

with open(os.path.join(BASE, "sources.yaml"), "r") as f:
    SOURCES = yaml.safe_load(f)

TZ = tz.gettz("America/Phoenix")

def llm_summarize(messages: List[Dict[str, str]], max_tokens=1000) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "gpt-4o-mini", "messages": messages, "temperature": 0.2, "max_tokens": max_tokens},
        timeout=90,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]

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
                "url": e.link,
                "title": e.title,
                "summary": getattr(e, "summary", ""),
                "published": when,
                "type": "rss",
            })
    return items

def fetch_sec_current() -> List[Dict[str, Any]]:
    d = feedparser.parse(SOURCES["sec"]["current_feed"])
    items = []
    for e in d.entries:
        when = None
        if getattr(e, "updated_parsed", None):
            when = datetime(*e.updated_parsed[:6], tzinfo=tz.tzutc()).astimezone(TZ)
        items.append({
            "source": "SEC EDGAR",
            "url": e.link,
            "title": e.title,
            "summary": getattr(e, "summary", ""),
            "published": when,
            "type": "sec",
        })
    return items

def rank_and_dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); out = []
    for it in items:
        key = (it["title"][:100].lower(), it["source"])
        if key in seen: continue
        seen.add(key); out.append(it)
    out.sort(key=lambda x: (x.get("published") or datetime.now(TZ)), reverse=True)
    return out

def build_messages(collected: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = open(os.path.join(BASE, "prompts", "system.txt")).read()
    lines = []
    for it in collected[:40]:
        when = it.get("published").strftime("%Y-%m-%d %H:%M") if it.get("published") else ""
        lines.append(f"- [{it['source']}] {it['title']} ({when}) -> {it['url']}")
    user = "Write the weekly brief with sections Macro, Markets, Companies.\n\nSOURCES:\n" + "\n".join(lines)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def summarize_sections(collected: List[Dict[str, Any]]):
    text = llm_summarize(build_messages(collected), max_tokens=1200)
    sections = {"macro": [], "markets": [], "companies": []}
    cur = None
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("## macro"): cur = "macro"; continue
        if low.startswith("## markets"): cur = "markets"; continue
        if low.startswith("## company") or low.startswith("## companies"): cur = "companies"; continue
        if cur and line.strip(): sections[cur].append(line)
    return sections

def render_email(sections, sources, period_label):
    env = Environment(loader=FileSystemLoader(os.path.join(BASE, "templates")), autoescape=select_autoescape())
    tpl = env.get_template("email.html.j2")
    subject_prefix = CONFIG.get("email", {}).get("subject_prefix", "[Weekly Finance Brief]")
    subject = f"{subject_prefix} Week of {period_label}"
    intro = "This week’s top financial developments — what happened and why it matters, with sources."
    html = tpl.render(
        subject=subject, intro=intro, week_range=period_label,
        macro=["<p>"+l+"</p>" for l in sections.get("macro", [])],
        markets=["<p>"+l+"</p>" for l in sections.get("markets", [])],
        companies=["<p>"+l+"</p>" for l in sections.get("companies", [])],
        sources=[{"name": it["source"], "url": it["url"], "note": it.get("title","")[:120]} for it in sources[:40]],
    )
    return subject, html

def compute_period_label(period: str) -> str:
    today = datetime.now(TZ).date()
    if period == "week":
        start = today - timedelta(days=today.weekday()); end = start + timedelta(days=6)
        return f"{start.isoformat()} — {end.isoformat()}"
    return today.isoformat()

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
        send_email(subject, html)  # optional later

def send_email(subject, html):
    # You can use SendGrid or SMTP — same env secrets the workflow uses.
    sg_key = os.environ.get("SENDGRID_API_KEY")
    if sg_key:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail, Email
        msg = Mail(
            from_email=(os.environ.get("SMTP_FROM") or "no-reply@example.com", CONFIG.get("email",{}).get("from_name","Finance Brief Bot")),
            to_emails=CONFIG["recipients"]["to"],
            subject=subject,
            html_content=html,
        )
        cc = CONFIG["recipients"].get("cc", []); bcc = CONFIG["recipients"].get("bcc", [])
        if cc: msg.cc = cc
        if bcc: msg.bcc = bcc
        if CONFIG.get("email",{}).get("reply_to"): msg.reply_to = Email(CONFIG["email"]["reply_to"])
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

if __name__ == "__main__":
    main()
