import os, sys, json, feedparser, requests, yaml, re
from datetime import datetime, timedelta
from dateutil import tz
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import List, Dict, Any

BASE = os.path.dirname(os.path.abspath(__file__))
TZ = tz.gettz("America/Phoenix")

# --- Load config & sources ---
with open(os.path.join(BASE, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

with open(os.path.join(BASE, "sources.yaml"), "r", encoding="utf-8") as f:
    SOURCES = yaml.safe_load(f)

# -------------------- OpenAI helper --------------------
def llm_summarize(messages: List[Dict[str, str]], max_tokens=900) -> str:
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

# -------------------- Data collection --------------------
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
                "summary": getattr(e, "summary", ""),
                "published": when,
                "type": "rss",
            })
    return items

def fetch_sec_current() -> List[Dict[str, Any]]:
    url = SOURCES.get("sec", {}).get("current_feed")
    if not url:
        return []
    d = feedparser.parse(url)
    items = []
    for e in d.entries:
        when = None
        if getattr(e, "updated_parsed", None):
            when = datetime(*e.updated_parsed[:6], tzinfo=tz.tzutc()).astimezone(TZ)
        items.append({
            "source": "SEC EDGAR",
            "url": getattr(e, "link", ""),
            "title": getattr(e, "title", ""),
            "summary": getattr(e, "summary", ""),
            "published": when,
            "type": "sec",
        })
    return items

def rank_and_dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); out = []
    for it in items:
        key = (it.get("title","")[:140].lower(), it.get("source",""))
        if key in seen: continue
        seen.add(key); out.append(it)
    out.sort(key=lambda x: (x.get("published") or datetime.now(TZ)), reverse=True)
    return out

# -------------------- Prompting & parsing --------------------
def build_messages(collected: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = open(os.path.join(BASE, "prompts", "system.txt"), encoding="utf-8").read()
    # compact sources list
    lines = []
    for it in collected[:40]:
        when = it.get("published").strftime("%Y-%m-%d %H:%M") if it.get("published") else ""
        lines.append(f"- [{it['source']}] {it['title']} ({when}) -> {it['url']}")
    sources_block = "\n".join(lines)

    user = f"""Write a concise weekly finance brief for a general audience.

Use EXACTLY this Markdown template (no extra prose):

## Macro
- <what happened + why it matters (1–2 sentences).>
- <2–3 bullets total>

## Markets
- <what happened + why it matters (1–2 sentences).>
- <2–3 bullets total>

## Companies
- <what happened + why it matters (1–2 sentences).>
- <2–3 bullets total>

Rules:
- Plain English. Include key figures/dates if present in sources.
- Only those three headings; each bullet MUST start with "- ".
- Do NOT include a "Sources" section — we will add it.
- Base EVERYTHING on these SOURCES (no fabrication):

SOURCES:
{sources_block}
"""
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]

def _parse_sections(md: str) -> Dict[str, list]:
    sections = {"macro": [], "markets": [], "companies": []}
    cur = None
    for line in md.splitlines():
        t = line.strip()
        low = t.lower()
        if low == "## macro": cur = "macro"; continue
        if low == "## markets": cur = "markets"; continue
        if low in ("## companies", "## company"): cur = "companies"; continue
        if cur and t.startswith("- "):
            sections[cur].append(t[2:].strip())
    return sections

def _headline_fallback(collected: List[Dict[str, Any]]) -> Dict[str, list]:
    def bucket(it):
        if it.get("type") == "sec": return "companies"
        src = (it.get("source") or "").lower()
        if any(k in src for k in ["federal reserve", "bureau of labor", "bls", "bureau of economic", "bea", "cpi", "inflation", "gdp", "unemployment"]):
            return "macro"
        return "markets"
    out = {"macro": [], "markets": [], "companies": []}
    for it in collected[:9]:
        out[bucket(it)].append(f"{it.get('title','(no title)')} — see source: {it.get('url','#')}")
    # Trim to 3 per section
    for k in out:
        out[k] = out[k][:3]
    if not any(out.values()):
        out["macro"].append("No major items captured this cycle.")
    return out

def summarize_sections(collected: List[Dict[str, Any]]) -> Dict[str, list]:
    # Pass 1
    try:
        text1 = llm_summarize(build_messages(collected), max_tokens=900)
        sec1 = _parse_sections(text1)
        if sum(len(v) for v in sec1.values()) >= 3:
            # hard-cap to 3 per section
            for k in sec1: sec1[k] = sec1[k][:3]
            return sec1
        # Pass 2 (retry with stronger instruction)
        messages = build_messages(collected)
        messages.append({"role":"user","content":
            "Your previous reply did not follow the exact template. "
            "Return ONLY the three headings with 2–3 bullets each. No extra text."})
        text2 = llm_summarize(messages, max_tokens=700)
        sec2 = _parse_sections(text2)
        if sum(len(v) for v in sec2.values()) >= 3:
            for k in sec2: sec2[k] = sec2[k][:3]
            return sec2
        print("LLM returned unparseable output twice; using fallback.")
        return _headline_fallback(collected)
    except Exception as e:
        print("LLM error; using fallback:", repr(e))
        return _headline_fallback(collected)

# -------------------- Pair bullets with sources --------------------
def _classify_item(it):
    if it.get("type") == "sec":
        return "companies"
    src = (it.get("source") or "").lower()
    if any(k in src for k in ["federal reserve", "fomc", "bureau of labor", "bls", "bureau of economic", "bea", "cpi", "inflation", "gdp", "unemployment"]):
        return "macro"
    return "markets"

def _bucket_collected(collected):
    buckets = {"macro": [], "markets": [], "companies": []}
    for it in collected:
        buckets[_classify_item(it)].append(it)
    for k in buckets:
        buckets[k].sort(key=lambda x: x.get("published") or datetime.now(TZ), reverse=True)
    return buckets

def _pair_bullets_with_sources(sections, collected, per_section=3):
    buckets = _bucket_collected(collected)
    paired = {}
    for sec in ["macro", "markets", "companies"]:
        bullets = sections.get(sec, [])[:per_section]
        srcs = buckets.get(sec, [])[:per_section]
        items = []
        for i, b in enumerate(bullets):
            src = srcs[i] if i < len(srcs) else None
            items.append({
                "text": b,
                "source_name": (src.get("source") if src else "Source"),
                "source_url": (src.get("url") if src else "#"),
            })
        paired[sec] = items
    return paired

# -------------------- Render & email --------------------
def render_email(sections, sources, period_label):
    env = Environment(loader=FileSystemLoader(os.path.join(BASE, "templates")),
                      autoescape=select_autoescape())
    tpl = env.get_template("email.html.j2")

    per_section = CONFIG.get("display", {}).get("per_section", 3)
    paired = _pair_bullets_with_sources(sections, sources, per_section=per_section)

    subject_prefix = CONFIG.get("email", {}).get("subject_prefix", "[Weekly Finance Brief]")
    subject = f"{subject_prefix} Week of {period_label}"
    intro = "This week’s top financial developments — what happened and why it matters, with sources."

    # Short “Sources” bibliography (limit to 12 or per_section*3)
    cap = max(12, per_section * 3)
    sources_list = [{"name": it["source"], "url": it["url"], "note": it.get("title","")[:120]} for it in sources[:cap]]

    html = tpl.render(
        subject=subject,
        intro=intro,
        week_range=period_label,
        macro=paired["macro"],
        markets=paired["markets"],
        companies=paired["companies"],
        sources=sources_list,
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

# -------------------- Main --------------------
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
