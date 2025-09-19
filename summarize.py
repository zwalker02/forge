import os, json, re, feedparser, requests, yaml
from datetime import datetime, timedelta
from dateutil import tz
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import List, Dict, Any
from urllib.parse import urlparse

BASE = os.path.dirname(os.path.abspath(__file__))
TZ = tz.gettz("America/Phoenix")

# ---------------------- Load config & sources ----------------------
with open(os.path.join(BASE, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

with open(os.path.join(BASE, "sources.yaml"), "r", encoding="utf-8") as f:
    SOURCES = yaml.safe_load(f)

def _cfg(block, key, default=None):
    node = CONFIG.get(block, {})
    return node.get(key, default) if isinstance(node, dict) else default

# Window (days) for recency filtering
WINDOW_DAYS = int(_cfg("window", "days", 7))

# Filters from config
MIN_REL = int(_cfg("filters", "min_relevance", 2))
NEG_TERMS = [t.lower() for t in _cfg("filters", "negative_terms", [])]
PATH_DROPS = [t.lower() for t in _cfg("filters", "drop_if_url_path_contains", [])]
ALLOWED_SOURCES = set(_cfg("filters", "allowed_sources", []) or [])

# ---------------------- Finance keywords & regex -------------------
FINANCE_TERMS = [
    # macro/markets
    "inflation","cpi","ppi","gdp","payrolls","unemployment","retail sales","fomc","federal reserve",
    "interest rate","rates","yield","treasury","bond","mortgage","housing starts","housing market",
    "stocks","equities","index","s&p","nasdaq","dow","rally","selloff","volatility","vix",
    "oil","brent","wti","gasoline","gold","silver","copper","bitcoin","crypto","fx","currency","dollar",
    # company/corporate
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

# ---------------------- OpenAI ----------------------
def llm_json(title: str, snippet: str, max_tokens=260) -> Dict[str, str]:
    """Return {'summary': 'two sentences', 'why': '2–3 sentences starting with Why it matters:'}."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    system = open(os.path.join(BASE, "prompts", "system.txt"), encoding="utf-8").read()
    base_user = f"Title: {title}\nSource snippet: {snippet}\nRespond with ONLY the JSON object."

    def _ask(msg_user: str) -> str:
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
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)

    text = _ask(base_user)
    data = _coerce_json_with_fallback(text, snippet)

    # Ensure WHY has >=2 sentences; if not, one retry
    if len(re.findall(r"[.!?](?:\s|$)", data["why"])) < 2:
        user2 = (f"{base_user}\n\nYour previous WHY was too short. "
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
            out["summary"] = (snippet or "")[:300].strip()
        if not out["why"]:
            out["why"] = "Why it matters: see source for context."
        return out
    except Exception:
        m = re.search(r"(?i)why it matters\s*:\s*(.+)$", text)
        why = "Why it matters: " + m.group(1).strip() if m else "Why it matters: see source for context."
        if m:
            text = re.sub(r"(?i)why it matters\s*:\s*.+$", "", text).strip()
        summary = text.strip() or (snippet or "")[:300].strip()
        return {"summary": summary, "why": why}

# ---------------------- Fetch & prep ----------------------
def _clean_html(s: str) -> str:
    return re.sub(r"<.*?>", "", s or "").strip()

def fetch_rss_list(feed_specs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    items = []
    for src in feed_specs or []:
        d = feedparser.parse(src["url"])
        for e in d.entries:
            when = None
            if getattr(e, "published_parsed", None):
                when = datetime(*e.published_parsed[:6], tzinfo=tz.tzutc()).astimezone(TZ)
            elif getattr(e, "updated_parsed", None):
                when = datetime(*e.updated_parsed[:6], tzinfo=tz.tzutc()).astimezone(TZ)
            items.append({
                "source": src.get("name") or "RSS",
                "url": getattr(e, "link", ""),
                "title": getattr(e, "title", ""),
                "summary": _clean_html(getattr(e, "summary", "")),
                "published": when,
                "type": "rss",
            })
    return items

def fetch_rss() -> List[Dict[str, Any]]:
    return fetch_rss_list(SOURCES.get("rss", []))

def fetch_sec_current() -> List[Dict[str, Any]]:
    url = SOURCES.get("sec", {}).get("current_feed")
    if not url: return []
    d = feedparser.parse(url)
    out = []
    for e in d.entries:
        when = None
        if getattr(e, "updated_parsed", None):
            when = datetime(*e.updated_parsed[:6], tzinfo=tz.tzutc()).astimezone(TZ)
        elif getattr(e, "published_parsed", None):
            when = datetime(*e.published_parsed[:6], tzinfo=tz.tzutc()).astimezone(TZ)
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
        key = (it.get("title","")[:180].lower(), it.get("source",""))
        if key in seen: continue
        seen.add(key); out.append(it)
    out.sort(key=lambda x: (x.get("published") or datetime.now(TZ)), reverse=True)
    return out

# ---------------------- Finance relevance & bucketing ----------------------
def finance_relevance(it: Dict[str, Any]) -> int:
    blob = f"{it.get('title','')} {it.get('summary','')}".lower()
    return len(FIN_RX.findall(blob))

def looks_company(it: Dict[str, Any]) -> bool:
    if it.get("type") == "sec":  # any SEC item is company-specific
        return True
    blob = f"{it.get('title','')} {it.get('summary','')}"
    return COMPANY_RX.search(blob) is not None

def is_finance_story(it: Dict[str, Any], min_rel: int = MIN_REL) -> bool:
    # Allowlist (optional)
    src_name = (it.get("source") or "").strip()
    if ALLOWED_SOURCES and src_name not in ALLOWED_SOURCES:
        return False

    rel = finance_relevance(it)
    if rel >= min_rel:
        return True  # clearly financial

    # Hard-drop geopolitics/conflict when not clearly financial
    text = f"{it.get('title','')} {it.get('summary','')}".lower()
    if any(term in text for term in NEG_TERMS):
        return False

    # Drop world/geo paths when not clearly financial
    try:
        path = (urlparse(it.get('url') or '').path or '').lower()
        if any(tok in path for tok in PATH_DROPS) and rel < max(3, min_rel + 1):
            return False
    except Exception:
        pass

    return False

def classify_item(it: Dict[str, Any], min_rel: int = MIN_REL) -> str:
    if not is_finance_story(it, min_rel=min_rel):
        return "drop"
    if looks_company(it):
        return "companies"
    blob = (it.get("source","") + " " + it.get("title","")).lower()
    if any(k in blob for k in MACRO_KEYS):
        return "macro"
    return "markets"

def bucket(items: List[Dict[str, Any]], min_rel: int = MIN_REL) -> Dict[str, list]:
    b = {"macro": [], "markets": [], "companies": []}
    for it in items:
        cat = classify_item(it, min_rel=min_rel)
        if cat == "drop":  # toss non-finance stories
            continue
        b[cat].append(it)
    for k in b:
        b[k].sort(key=lambda x: x.get("published") or datetime.now(TZ), reverse=True)
    return b

def backfill_companies(b: Dict[str, list], pool: List[Dict[str, Any]], per_section: int, min_rel: int):
    if len(b["companies"]) >= per_section:
        return
    have = {id(x) for x in b["companies"]}
    for it in pool:
        if id(it) in have: continue
        if not is_finance_story(it, min_rel=min_rel): continue
        if looks_company(it):
            b["companies"].append(it)
            have.add(id(it))
            if len(b["companies"]) >= per_section:
                break

# ---------------------- Recency filter (7-day window) ----------------------
def within_window(it: Dict[str, Any], days: int = WINDOW_DAYS) -> bool:
    """Keep items with a real published time within the last N days (America/Phoenix)."""
    when = it.get("published")
    if not isinstance(when, datetime):
        return False
    now = datetime.now(TZ)
    cutoff = now - timedelta(days=days)
    return when >= cutoff and when <= now + timedelta(hours=1)  # small skew tolerance

def apply_recency_window(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [it for it in items if within_window(it, WINDOW_DAYS)]

# ---------------------- Fallbacks ----------------------
def fetch_fallback_rss(section: str) -> List[Dict[str, Any]]:
    fb = SOURCES.get("fallback", {}) or {}
    feeds = []
    if section == "markets":
        feeds = fb.get("markets_rss", [])
    elif section == "companies":
        feeds = fb.get("companies_rss", [])
    return rank_and_dedupe(apply_recency_window(fetch_rss_list(feeds)))

def fetch_finnhub_market_news() -> List[Dict[str, Any]]:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key: return []
    try:
        url = f"https://finnhub.io/api/v1/news?category=general&token={key}"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        now = datetime.now(TZ)
        out = []
        for n in resp.json()[:50]:
            dt = n.get("datetime")
            when = datetime.fromtimestamp(dt, tz=tz.tzutc()).astimezone(TZ) if dt else None
            it = {
                "source": n.get("source") or "Finnhub",
                "url": n.get("url"),
                "title": n.get("headline") or "",
                "summary": n.get("summary") or "",
                "published": when,
                "type": "rss",
            }
            if within_window(it):
                out.append(it)
        return rank_and_dedupe(out)
    except Exception:
        return []

def fetch_finnhub_company_news() -> List[Dict[str, Any]]:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key: return []
    tickers = SOURCES.get("fortune500_tickers", [])[:15]  # cap to keep it light
    if not tickers: return []
    frm = (datetime.now(TZ) - timedelta(days=WINDOW_DAYS)).date().isoformat()
    to = datetime.now(TZ).date().isoformat()
    out = []
    for sym in tickers:
        try:
            url = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={frm}&to={to}&token={key}"
            resp = requests.get(url, timeout=15)
            if resp.status_code >= 400: continue
            for n in resp.json()[:10]:
                when = datetime.fromtimestamp(n.get("datetime", 0), tz=tz.tzutc()).astimezone(TZ) if n.get("datetime") else None
                out.append({
                    "source": n.get("source") or f"Finnhub {sym}",
                    "url": n.get("url"),
                    "title": n.get("headline") or f"{sym} update",
                    "summary": n.get("summary") or "",
                    "published": when,
                    "type": "rss",
                })
        except Exception:
            continue
    out = [it for it in out if within_window(it)]
    return rank_and_dedupe(out)

# ---------------------- Summarize per-article ----------------------
def llm_json_safe(title: str, snippet: str) -> Dict[str, str]:
    try:
        return llm_json(title, snippet)
    except Exception as e:
        return {
            "summary": (snippet or title)[:300],
            "why": f"Why it matters: source context unavailable ({e})."
        }

def summarize_sections(collected: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    # 1) Only recent items
    recent = apply_recency_window(collected)
    if not recent:
        recent = collected[:12]  # extreme edge case

    per_section = CONFIG.get("display", {}).get("per_section", 3)

    # 2) First pass (strict)
    b = bucket(recent, min_rel=MIN_REL)
    backfill_companies(b, recent, per_section, min_rel=MIN_REL)

    # 3) If Markets/Companies are still empty, try relaxed relevance (min_rel - 1 down to 1)
    relaxed_min = max(1, MIN_REL - 1)
    if not b["markets"]:
        b_relaxed = bucket(recent, min_rel=relaxed_min)
        if b_relaxed["markets"]:
            b["markets"] = b_relaxed["markets"]
    if len(b["companies"]) < per_section:
        backfill_companies(b, recent, per_section, min_rel=relaxed_min)

    # 4) RSS fallbacks if still empty
    if not b["markets"]:
        b["markets"] = fetch_fallback_rss("markets")[:per_section]
    if len(b["companies"]) < per_section:
        need = per_section - len(b["companies"])
        fb_comp = fetch_fallback_rss("companies")[:need]
        b["companies"].extend(fb_comp)

    # 5) Finnhub fallbacks (optional, only if API key present)
    if not b["markets"]:
        fin_mkt = fetch_finnhub_market_news()[:per_section]
        if fin_mkt:
            b["markets"] = fin_mkt
    if len(b["companies"]) < per_section:
        need = per_section - len(b["companies"])
        fin_co = fetch_finnhub_company_news()[:need]
        b["companies"].extend(fin_co)

    # 6) Final: never empty — if still empty, synthesize placeholders
    results = {"macro": [], "markets": [], "companies": []}
    for sec in ["macro", "markets", "companies"]:
        if not b[sec]:
            b[sec] = [{
                "title": f"No recent {sec} items detected",
                "url": "#",
                "summary": f"No {sec} stories passed filters within the {WINDOW_DAYS}-day window.",
                "published": None,
                "type": "rss",
            }]

        for it in b[sec][:per_section]:
            title = it.get("title", "(no title)")
            snippet = it.get("summary", "") or title
            js = llm_json_safe(title, snippet)
            results[sec].append({
                "title": title,
                "url": it.get("url", "#"),
                "summary": js["summary"],
                "why": js["why"],
                "published": it.get("published").strftime("%B %d, %Y") if it.get("published") else ""
            })

    return results

# ---------------------- Render & email ----------------------
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
        week_range=period_label,    # keep for older templates
        date_range=period_label,    # used by your current template
        macro=sections["macro"],
        markets=sections["markets"],
        companies=sections["companies"],
    )
    return subject, html

def compute_period_label(period: str) -> str:
    today = datetime.now(TZ).date()
    if period == "week":
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return f"{start.strftime('%B %d, %Y')} — {end.strftime('%B %d, %Y')}"
    return today.strftime("%B %d, %Y")

# ---------------------- Send (SendGrid) ----------------------
def send_email(subject, html):
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email

    sg_key = os.environ.get("SENDGRID_API_KEY")
    if not sg_key:
        raise RuntimeError("Missing SENDGRID_API_KEY in environment/secrets.")

    from_addr = os.environ.get("SMTP_FROM", "no-reply@example.com")
    from_name = CONFIG.get("email", {}).get("from_name", "Finance Brief Bot")

    msg = Mail(
        from_email=(from_addr, from_name),
        to_emails=CONFIG["recipients"]["to"],
        subject=subject,
        html_content=html,
    )

    cc = CONFIG["recipients"].get("cc", [])
    bcc = CONFIG["recipients"].get("bcc", [])
    if cc:
        msg.cc = cc
    if bcc:
        msg.bcc = bcc

    reply_to = CONFIG.get("email", {}).get("reply_to")
    if reply_to:
        msg.reply_to = Email(reply_to)

    SendGridAPIClient(sg_key).send(msg)

# ---------------------- Main ----------------------
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
