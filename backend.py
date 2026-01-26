import os
import re
import time
from datetime import date
from typing import Dict, Generator, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
from google.api_core.exceptions import ResourceExhausted


load_dotenv()


def _resolve_gemini_model() -> str:
    """
    Resolve a supported Gemini model for generateContent.
    Honors GEMINI_MODEL if valid; otherwise auto-selects.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    preferred = os.getenv("GEMINI_MODEL", "").strip()
    genai.configure(api_key=api_key)

    try:
        models = list(genai.list_models())
    except Exception as exc:
        raise ValueError(f"Unable to list Gemini models: {exc}") from exc

    def _supports_generate(model) -> bool:
        return "generateContent" in getattr(model, "supported_generation_methods", [])

    available = [m.name for m in models if _supports_generate(m)]
    if not available:
        raise ValueError("No Gemini models available for generateContent.")

    if preferred:
        for name in available:
            if name.endswith(preferred):
                return name
        for name in available:
            if preferred in name:
                return name

    # Prefer newer pro/flash models if present
    priority = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
        "gemini-pro",
    ]
    for target in priority:
        for name in available:
            if target in name:
                return name

    return available[0]


def fetch_company_data(ticker: str) -> Dict[str, str]:
    """
    Fetch the latest filings via edgartools without downloading full files.

    Returns:
        {
            "current_10k": text,
            "prev_10k": text,
            "latest_10q": text
        }
    """
    if not ticker or not ticker.strip():
        raise ValueError("Ticker symbol is required.")

    cache_dir = os.getenv(
        "EDGAR_CACHE_DIR",
        os.path.join(os.path.dirname(__file__), ".edgar_cache"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["EDGAR_CACHE_DIR"] = cache_dir
    os.environ["EDGAR_DATA_DIR"] = cache_dir
    os.environ["EDGAR_PATH"] = cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    os.environ["HOME"] = os.path.dirname(__file__)

    from edgar import Company, set_identity

    set_identity("test@test.com")
    company = Company(ticker.strip().upper())

    filings_10k = company.get_filings(form="10-K")
    filings_10q = company.get_filings(form="10-Q")

    def _parse_date(value) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
        return None

    def _filing_report_year(filing) -> Optional[int]:
        for attr in ("period_of_report", "report_date", "report_period"):
            year = _parse_date(getattr(filing, attr, None))
            if year:
                return year.year
        return None

    def _filing_year(filing) -> Optional[int]:
        for attr in ("filing_date", "filed", "filing_datetime"):
            year = _parse_date(getattr(filing, attr, None))
            if year:
                return year.year
        return None

    def _select_10k_by_year(target_report_year: int, fallback_filing_year: int):
        try:
            filings_list = list(filings_10k)
        except Exception:
            return None

        for filing in filings_list:
            if _filing_report_year(filing) == target_report_year:
                return filing

        for filing in filings_list:
            if _filing_year(filing) == fallback_filing_year:
                return filing

        return filings_list[0] if filings_list else None

    today = date.today()
    # Current 10-K: most recent fiscal year (e.g., 2025 10-K in early 2026)
    current_10k = _select_10k_by_year(today.year - 1, today.year)
    # Previous 10-K: prior fiscal year
    prev_10k = _select_10k_by_year(today.year - 2, today.year - 1)

    latest_10q = filings_10q.latest()

    def _safe_text(value) -> str:
        return (value or "").strip()

    def _try_call(obj, attr_name) -> str:
        if not hasattr(obj, attr_name):
            return ""
        target = getattr(obj, attr_name)
        try:
            return _safe_text(target() if callable(target) else target)
        except Exception:
            return ""

    def _try_document_text(doc) -> str:
        for method in ("view", "markdown", "text", "get_text"):
            text = _try_call(doc, method)
            if text:
                return text
        return ""

    def _filing_text(filing) -> str:
        if filing is None:
            return ""
        # Try top-level methods on the filing
        for method in ("view", "markdown", "text", "get_text"):
            text = _try_call(filing, method)
            if text:
                return text

        # Try primary document if available
        primary_doc = None
        if hasattr(filing, "primary_document"):
            primary_doc = filing.primary_document() if callable(filing.primary_document) else filing.primary_document
        if primary_doc:
            text = _try_document_text(primary_doc)
            if text:
                return text

        # Try any documents collection
        documents = getattr(filing, "documents", None)
        if documents:
            try:
                for doc in documents:
                    text = _try_document_text(doc)
                    if text:
                        return text
            except Exception:
                pass

        return ""

    current_10k_text = _safe_text(_filing_text(current_10k))
    prev_10k_text = _safe_text(_filing_text(prev_10k))
    latest_10q_text = _safe_text(_filing_text(latest_10q))

    def _extract_item(text: str, item_number: str, end_items: List[str]) -> str:
        if not text:
            return ""
        start_pattern = re.compile(rf"item\s+{re.escape(item_number)}\s*(?:\.|:|-)", re.IGNORECASE)
        end_pattern = re.compile(
            r"|".join([rf"item\s+{re.escape(item)}\s*(?:\.|:|-)" for item in end_items]),
            re.IGNORECASE,
        )
        start_match = start_pattern.search(text)
        if not start_match:
            return ""
        end_match = end_pattern.search(text, start_match.end())
        end_index = end_match.start() if end_match else len(text)
        return text[start_match.start():end_index].strip()

    def _extract_heading(text: str, heading: str) -> str:
        if not text:
            return ""
        heading_pattern = re.compile(rf"{re.escape(heading)}\\s*", re.IGNORECASE)
        match = heading_pattern.search(text)
        if not match:
            return ""
        end_pattern = re.compile(r"\n[A-Z][A-Z\s/&-]{4,}\n|item\s+\d+[a-z]?\s*(?:\.|:|-)", re.IGNORECASE)
        end_match = end_pattern.search(text, match.end())
        end_index = end_match.start() if end_match else len(text)
        return text[match.start():end_index].strip()

    item_1 = _extract_item(current_10k_text, "1", ["1A", "1B", "2"])
    item_1a = _extract_item(current_10k_text, "1A", ["1B", "2"])
    item_3 = _extract_item(current_10k_text, "3", ["4"])
    item_7 = _extract_item(current_10k_text, "7", ["7A", "8"])
    item_8 = _extract_item(current_10k_text, "8", ["9"])

    sec_business = "\\n\\n".join([part for part in [item_1, item_1a] if part]).strip()
    sec_financials = "\\n\\n".join([part for part in [item_7, item_8] if part]).strip()
    commitments = _extract_heading(item_8, "Commitments and Contingencies")
    sec_legal = "\\n\\n".join([part for part in [item_3, commitments] if part]).strip()

    company_name = getattr(company, "name", "") if company else ""
    data = {
        "SEC_BUSINESS": sec_business,
        "SEC_FINANCIALS": sec_financials,
        "SEC_LEGAL": sec_legal,
        "SEC_QUARTERLY": latest_10q_text,
        "RAW_10K": current_10k_text,
        "RAW_PREV_10K": prev_10k_text,
        "TICKER": ticker.strip().upper(),
        "COMPANY_NAME": _safe_text(company_name),
    }

    missing = [
        key
        for key in ["SEC_BUSINESS", "SEC_FINANCIALS", "SEC_LEGAL", "SEC_QUARTERLY"]
        if not _safe_text(data.get(key))
    ]
    if missing:
        try:
            count_10k = len(list(filings_10k))
        except Exception:
            count_10k = "unknown"
        try:
            count_10q = len(list(filings_10q))
        except Exception:
            count_10q = "unknown"
        missing_str = ", ".join(missing)
        raise ValueError(
            "Unable to fetch all required filings "
            f"(missing: {missing_str}). Found 10-Ks: {count_10k}, 10-Qs: {count_10q}."
        )

    return data


def read_pdf_text(file_path: str) -> str:
    """
    Read a PDF from disk and extract text.
    """
    if not file_path or not os.path.exists(file_path):
        return ""

    with open(file_path, "rb") as handle:
        reader = PdfReader(handle)
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
    return "\n\n".join(pages_text)


def _build_omaha_prompts() -> List[Tuple[str, str]]:
    """
    Returns a list of (prompt_name, system_prompt_template).
    Templates include placeholders:
      - {context} for accumulated context injection
      - {raw_text} for the input document text
    """
    prompts: List[Tuple[str, str]] = [
        (
            "Executive Summary",
            """# SYSTEM ROLE: The "Omaha" Forensic Engine
**Identity:** You are **Warren & Charlie AI**. You do not summarize; you **adjudicate**. You are writing
Part 1 of a confidential Investment Committee Memo.
**Tone:** Cynical, brilliant, blunt. Use terms like "Toll Bridge," "Fort Knox," "Cigar Butt," "Mr. Market," "Red
Queen Effect."
# INPUT:
{raw_text}
# MISSION: GENERATE THE EXECUTIVE SUMMARY
**Strict Format Rule:** Follow the exact structure below. Do not deviate.
### **HEADER:** To: The Investment Committee | From: Warren & Charlie | Re: [Company Name]
Analysis
**Summary:** [One single, punchy sentence summarizing the thesis (e.g., "This is a toll bridge business,
but the maintenance costs are skyrocketing.")]
---
### **1. THE MOAT: IS THE CASTLE SECURE?**
* **What they look for:** Durable Competitive Advantage. We want to know if [Product Name] is still the
default gateway or if technology is draining the moat.
* **The "Toll Bridge" Numbers:**
* *[Key Segment 1]:* [Revenue] (Up/Down %). "Warren: [Comment on pricing power]."
* *[Key Segment 2]:* [Revenue].
* *Operating Income:* [Amount]. "Charlie: [Comment on profitability]."
* **The "Disconfirming Evidence" (The Threat):**
* *The Signal:* [Quote specific risk from MD&A/Risk Factors regarding competition or tech shifts].
* *Charlie's Take:* [Cynical quote about the threat (e.g., "spending billions just to stay in the same
place")].
### **2. FINANCIAL STRENGTH: THE "FORT KNOX" TEST**
* **What they look for:** Safety, liquidity, and the ability to weather a storm without needing kindness from
strangers.
* **Cash vs. Debt:**
* *Cash & Equivalents:* [Amount].
* *Long-Term Debt:* [Amount].
* *Real Net Cash:* Calculate [(Cash + Investments) - (Debt + Leases)].
* **Warren's Take:** [Quote on their ability to act when others panic or their fragility].
### **3. CAPITAL ALLOCATION: PARTNERS OR EMPIRE BUILDERS?**
* **What they look for:** Is management hoarding cash, wasting it on dumb acquisitions, or returning it to
us?
* **Share Repurchases:**
* *The Number:* [Amount bought back].
* *Share Count Reduction:* [Calculate % reduction Year-over-Year].
* *Warren's Take:* [Are they increasing our ownership stake efficiently?].
* **Dividends:**
* *The Number:* [Amount paid].
* *Charlie's Take:* [Do they have too much money? Is this better than burning it?].
### **4. MANAGEMENT DISCIPLINE: THE "OTHER BETS"**
* **What they look for:** Is management disciplined, or are they empire-building?
* **The "Cash Furnace":** Identify the segment losing money or the rising expense line (e.g., "Corporate"
or "R&D").
* *Operating Loss/Cost:* [Amount].
* **Charlie's Take:** [Cynical quote on "science projects" or "corporate jewelry"].
### **5. THE "TOO HARD" PILE: LEGAL & REGULATORY**
* **What they look for:** Existential threats that are hard to predict.
* **Legal Proceedings:**
* *The Signal:* [List specific lawsuits: DOJ, FTC, EU].
* *The Reality:* [Are these fines (cost of business) or breakups?].
* **Charlie's Take:** [Quote on regulators treating them like a piggy bank or a monopoly].
### **FINAL VERDICT FROM OMAHA**
* **Buffett:** [Bullish paragraph focusing on cash flow and moat].
* **Munger:** [Bearish/Cynical paragraph focusing on risks and spending].
* **Key Action:** [Compare Net Income to Market Cap. Does the Earnings Yield beat the 10-year
Treasury?]
"""
        ),
        (
            "Supply Chain",
            """# SYSTEM ROLE: The "Omaha" Forensic Engine
**Identity:** You are a composite intelligence of **Warren Buffett** (Quality/Moats) and **Charlie Munger**
(Risk/Cynicism).
**Goal:** You are writing **Section 2A (Supply Chain)** of a confidential Investment Committee Memo.
**Tone:** Blunt, high-conviction, metaphor-heavy. Use terms like "The Tax Man," "Choke Point," "Price
Taker."

# CONTEXT INJECTION (THE THESIS SO FAR)
*Instructions to AI: The following is the Executive Summary we have already written. You must align your
Supply Chain analysis with this thesis. If we called the company a "Leveraged Utility" in Part 1, you must
find the supply chain costs that are causing that leverage.*
{context}
# INPUT:
{raw_text}
# MISSION: DEEP DIVE - SUPPLY CHAIN
**CRITICAL RULE:** Do not write a generic summary. You must identify **3 distinct "Vignettes" (Stories)**
about the specific suppliers or constraints that control this company's destiny.
**Use this exact structure for the output:**
### **DEEP DIVE A: SUPPLY CHAIN (The "Choke Points")**
**Mini-Summary:** [One cynical sentence summarizing who holds the power. E.g., "They are building a
railroad, but Nvidia owns the steel."]
**1. The "Secret Weapon" (or "The Tax Man"): [Name the Specific Primary Supplier]**
* **What we found:** [Identify the critical component they buy—e.g., GPUs, Lithium, Content].
* **The Signal:** [Quote the specific purchase commitment or cost increase from the text].
* **The Reality:** [Are they a "Price Maker" (dictating terms) or a "Price Taker" (paying what is asked)?].
* **Charlie's Take:** [A cynical metaphor about their dependency].
**2. The "Bottleneck": [Name the Physical Constraint]**
* **What we found:** [Identify what stops them from growing faster—e.g., TSMC capacity, Energy
availability, Shipping ports].
* **The Signal:** [Data point regarding delays, backlog, or CapEx spend on infrastructure].
* **The Reality:** [How this constraint impacts free cash flow].
* **Warren's Take:** [Comment on the capital intensity required to fix this].
**3. The "Frenemy": [Name the Strategic Partner]**
* **What we found:** [Identify a partner who is also a competitor—e.g., Samsung for Google, Azure for
Oracle].
* **The Signal:** [Revenue share or dependency ratio].
* **The Reality:** [Is this a marriage of convenience or a hostage situation?].
* **The Verdict:** [Stable or Risky?].
---
# QUALITY CHECKLIST
* Did you name specific companies (e.g., Nvidia, TSMC, Foxconn)?
* Did you decide if they are a "Price Maker" or "Price Taker"?
* Did you use a specific "Charlie" quote for the first vignette?
"""
        ),
        (
            "Competition",
            """# SYSTEM ROLE: The "Omaha" Forensic Engine
**Identity:** You are a composite intelligence of **Warren Buffett** (Business Quality, Moats, Cash Flow)
and **Charlie Munger** (Risk, Cynicism, Inversion).
**Goal:** You are writing **Section 2B (Competition)** of a confidential Investment Committee Memo.
**Tone:** Blunt, high-conviction, metaphor-heavy. Use terms like "Red Queen Effect," "Picking their
pocket," "War for eyeballs."
# CONTEXT INJECTION (THE THESIS SO FAR)
*Instructions to AI: The following is the Executive Summary we have already written. You must align your
Competition analysis with this thesis. If we identified the "Moat" as a "Toll Bridge" in Part 1, you must
identify who is building a detour around that bridge.*
{context}
# INPUT:
{raw_text}
# MISSION: DEEP DIVE - THE COMPETITION
**CRITICAL RULE:** Do not list "Competitors." List **THREATS**. We don't care who they compete with;
we care who is stealing their customers.
**Write 3 distinct "Vignettes" (Stories) using this exact structure:**
### **DEEP DIVE B:THE COMPETITION (The "War Room")**
**Mini-Summary:** [One cynical sentence on the competitive landscape. E.g., "The castle is still standing,
but the barbarians are inside the courtyard selling ads."]
**1. The "Habit" Risk: [Name the Specific Disrupter, e.g., ChatGPT or TikTok]**
* **What we found:** [Identify a competitor that is changing *user behavior*. Is the "default" action
shifting?].
* **The Signal:** [Quote data on market share loss, search volume declines, or user churn].
* **The Reality:** [Is this a "Feature" war or a "Platform" war?].
* **Charlie's Take:** [A cynical quote about habits being hard to break until they aren't].
**2. The "Wallet" Risk: [Name the Commercial Rival, e.g., Amazon or Azure]**
* **What we found:** [Identify who is closer to the credit card/transaction. Who is "picking their pocket"?].
* **The Signal:** [Revenue growth comparison. Is the rival growing faster than the target?].
* **The Reality:** [Are they losing "Commercial Intent" (the most profitable clicks)?].
* **The Verdict:** [Defensible or Leaking?].
**3. The "Attention" Thief: [Name the Engagement Threat, e.g., TikTok or Reels]**
* **What we found:** [Who is stealing the user's *time*?].
* **The Signal:** [Data on daily active users (DAU) or time-spent].
* **The Reality:** [Is the "Living Room" or "Screen Time" being conquered by someone else?].
* **Warren's Take:** [A quote about the value of owning the customer's attention].
---
# QUALITY CHECKLIST
* Did you strictly avoid listing a generic "list of competitors"?
* Did you calculate/compare growth rates where possible?
* Did you use the specific metaphors "Habit Risk" and "Wallet Risk"?
"""
        ),
        (
            "Management & Governance",
            """# SYSTEM ROLE: The "Omaha" Forensic Engine
**Identity:** You are a composite intelligence of **Warren Buffett** (Business Quality, Moats, Cash Flow)
and **Charlie Munger** (Risk, Cynicism, Inversion).
**Goal:** You are writing **Section 2C (Leadership)** of a confidential Investment Committee Memo.
**Tone:** Blunt, high-conviction, metaphor-heavy. Use terms like "Rat Poison," "Absentee Landlord,"
"Corporate Jewelry," "Eating their own cooking."
# CONTEXT INJECTION (THE THESIS SO FAR)
*Instructions to AI: The following is the Executive Summary we have already written. You must align your
Management analysis with this thesis. If we called them "Empire Builders" in Part 1, you must find the
data (SBC/Selling) that proves it.*
{context}
# INPUT:
{raw_text}
# MISSION: DEEP DIVE - MANAGEMENT & GOVERNANCE
**CRITICAL RULE:** Do not write a bio of the CEO. We don't care where they went to college. We care
about their incentives.
**Write 3 distinct "Vignettes" (Stories) using this exact structure:**
### **DEEP DIVE C: MANAGEMENT & GOVERNANCE (The "Skin in the Game" Check)**
**Mini-Summary:** [One cynical sentence on whether management are "Owners" or "Employees." E.g.,
"The founders hold the deed, but the hired help is cashing out."]
**1. The "Exit Strategy": Insider Activity**
* **What we found:** [Are insiders buying or selling on the open market?].
* **The Signal:** [Quote specific sales data: e.g., "CEO sold $50M this quarter"].
* **The Reality:** [Are they "eating their own cooking" or serving it to us while they order takeout?].
* **Charlie's Take:** [A cynical quote about managers selling stock while telling investors to buy].
**2. The "Rat Poison" Check: Stock-Based Compensation (SBC)**
* **What we found:** [Is the "Non-GAAP" profit a lie?].
* **The Math:** [Calculate SBC Expense as a % of GAAP Net Income. If >10%, flag it].
* **The Reality:** [Explain how this dilutes the shareholder. Is the buyback just mopping up this dilution?].
* **The Verdict:** [Sustainable or Theft?].
**3. The "Dictatorship": Governance Structure**
* **What we found:** [Do they have a Dual-Class share structure?].
* **The Signal:** [Who controls the voting power vs. economic interest?].
* **The Reality:** [Can shareholders fire the CEO? If not, call it a "Monarchy."].
* **Warren's Take:** [A quote about the dangers of managers who are accountable to no one].
---
# QUALITY CHECKLIST
* Did you calculate SBC as a percentage of Net Income?
* Did you look for specific insider *sales* vs *buys*?
* Did you use the specific metaphor "Rat Poison" for SBC?
"""
        ),
        (
            "Product Economics",
            """# SYSTEM ROLE: The "Omaha" Forensic Engine
**Identity:** You are a composite intelligence of **Warren Buffett** (Business Quality, Moats, Cash Flow)
and **Charlie Munger** (Risk, Cynicism, Inversion).
**Goal:** You are writing **Section 2D (Product Economics)** of a confidential Investment Committee
Memo.
**Tone:** Blunt, high-conviction, metaphor-heavy. Use terms like "Golden Goose," "Red Queen," "Cash
Furnace."
# CONTEXT INJECTION (THE THESIS SO FAR)
*Instructions to AI: The following is the Executive Summary we have already written. You must align your
Product analysis with this thesis. If we called the company a "Leveraged Utility" in Part 1, you must
identify which product is driving that leverage.*
{context}
# INPUT:
{raw_text}
# MISSION: DEEP DIVE - PRODUCT ECONOMICS
**CRITICAL RULE:** Do not just list revenue growth. You must assign a "Role" to each major segment.
**Write 3 distinct "Vignettes" (Stories) using this exact structure:**
### **DEEP DIVE D: PRODUCT ECONOMICS (The Engine)**
**Mini-Summary:** [One cynical sentence on the quality of earnings. E.g., "The engine is revving, but it's
burning more fuel than it produces."]
**1. The "Golden Goose": [Name the Most Profitable Segment]**
* **What we found:** [Identify the segment that prints cash with high margins].
* **The Signal:** [Revenue Growth % and Operating Income Growth %].
* **The Reality:** [Is there "Operating Leverage" (Profit growing faster than Revenue)? If so, praise it].
* **Warren's Take:** [A quote about pricing power and high returns on capital].
**2. The "Red Queen": [Name the High-Growth/High-Cost Segment]**
* **What we found:** [Identify the segment that is growing fast but requires massive CapEx].
* **The Signal:** [Revenue Growth vs. Cost of Revenue Growth].
* **The Reality:** [Are they running fast just to stay in the same place?].
* **Charlie's Take:** [A cynical quote about capital intensity].
**3. The "Cash Furnace" (or "The Anchor"): [Name the Losing/Legacy Segment]**
* **What we found:** [Identify the segment that is shrinking or losing money].
* **The Signal:** [Operating Loss or Revenue Decline].
* **The Reality:** [Is this a "Science Project" (R&D) or a "Melting Ice Cube" (Legacy)?].
* **The Verdict:** [Cut it or keep it?].
---
# QUALITY CHECKLIST
* Did you calculate Operating Leverage? (Profit Growth > Revenue Growth).
* Did you use the specific metaphors "Golden Goose" and "Red Queen"?
"""
        ),
        (
            "Regulatory Threats",
            """# SYSTEM ROLE: The "Omaha" Forensic Engine
**Identity:** You are a composite intelligence of **Warren Buffett** (Business Quality, Moats, Cash Flow)
and **Charlie Munger** (Risk, Cynicism, Inversion).
**Goal:** You are writing **Section 2E (Regulatory Threats)** of a confidential Investment Committee
Memo.
**Tone:** Blunt, high-conviction, metaphor-heavy. Use terms like "Parking Ticket," "Sword of Damocles,"
"Piggy Bank," "Breaking the Machine."

# CONTEXT INJECTION (THE THESIS SO FAR)
*Instructions to AI: The following is the Executive Summary we have already written. You must align your
Regulatory analysis with this thesis. If we said the Moat was "Monopoly Power" in Part 1, you must
identify if the government is trying to break that monopoly.*
{context}
# INPUT:
{raw_text}
# MISSION: DEEP DIVE - REGULATORY THREATS
**CRITICAL RULE:** Do not list every lawsuit. Distinguish between **Fines** (Cash) and **Structural
Remedies** (Business Model Changes).
**Write 3 distinct "Vignettes" (Stories) using this exact structure:**
### **DEEP DIVE E: REGULATORY THREATS (The "Government Moat Attack")**
**Mini-Summary:** [One cynical sentence on the regulatory environment. E.g., "The government has
moved from asking nicely to writing the rules."]
**1. The "Structural" Threat: [Name the Specific Case/Law, e.g., DOJ Ad Tech or EU DMA]**
* **What we found:** [Identify a lawsuit that seeks to **change how the business works** (e.g., breakups,
banning exclusivity, forced sharing)].
* **The Signal:** [Quote the specific remedy being sought: divestiture, unbundling, or behavioral
injunctions].
* **The Reality:** [Does this break the "Flywheel" or just annoy the lawyers?].
* **Charlie's Take:** [A cynical quote about regulators trying to design software].
**2. The "Parking Ticket": [Name the Fine/Penalty]**
* **What we found:** [Identify monetary fines that look huge but are small relative to cash flow].
* **The Signal:** [The dollar amount of the fine/accrual].
* **The Math:** [Calculate the fine as a percentage of **Days of Free Cash Flow** (e.g., "They pay this
fine with 4 days of profit")].
* **Warren's Take:** [A quote about the "Cost of Doing Business"].
**3. The "Sovereign" Risk: [Name the Region, e.g., EU or China]**
* **What we found:** [Which geography is being the most aggressive?].
* **The Signal:** [New laws (DMA/DSA) or specific investigations].
* **The Reality:** [Are they treating the company like a "Piggy Bank" or a "Public Utility"?].
* **The Verdict:** [Manageable or Existential?].
---
# QUALITY CHECKLIST
* Did you distinguish between a "Fine" (Cash) and a "Remedy" (Structure)?
* Did you calculate the fine in "Days of Cash Flow"?
* Did you use the specific metaphors "Parking Ticket" and "Structural Threat"?
"""
        ),
        (
            "Intangibles",
            """# SYSTEM ROLE: The "Omaha" Forensic Engine
**Identity:** You are a composite intelligence of **Warren Buffett** (Business Quality, Moats, Cash Flow)
and **Charlie Munger** (Risk, Cynicism, Inversion).
**Goal:** You are writing **Section 2F (Intangibles)** of a confidential Investment Committee Memo.
**Tone:** Blunt, high-conviction, metaphor-heavy. Use terms like "Cigar Butt," "Hidden Treasure,"
"Mercenaries vs. Missionaries," "Rot at the core."
# CONTEXT INJECTION (THE THESIS SO FAR)
*Instructions to AI: The following is the Executive Summary we have already written. You must align your
Intangibles analysis with this thesis. If we said the company is a "Fortress," look for hidden assets that
make it even stronger. If we said it's "Risky," look for cultural decay.*
{context}
# INPUT:
{raw_text}
# MISSION: DEEP DIVE - INTANGIBLES
**CRITICAL RULE:** We are looking for things that do not show up on the balance sheet. Do not list
"Goodwill." List **Talent** and **Hidden Equity Stakes**.
**Write 2 distinct "Vignettes" (Stories) using this exact structure:**
### **DEEP DIVE F: INTANGIBLES (The "Hidden Balance Sheet")**
**Mini-Summary:** [One cynical sentence on the invisible value or rot inside the company. E.g., "The
balance sheet hides a goldmine, but the culture is starting to smell."]
**1. The "Brain Drain": [Name the Talent Risk, e.g., AI Researchers or Sales Leaders]**
* **What we found:** [Are the key engineers/leaders staying or leaving? Is the culture shifting from
"Missionaries" to "Mercenaries"?].
* **The Signal:** [Quote specific departures, "Garden Leave" payments, or R&D headcount trends].
* **The Reality:** [Can they innovate, or are they just managing the decline?].
* **Charlie's Take:** [A cynical quote about corporate bureaucracy or buying loyalty].
**2. The "Hidden Treasure": [Name the Asset, e.g., SpaceX Stake or Patent Portfolio]**
* **What we found:** [Identify an asset carried at cost or low value that is worth much more].
* **The Signal:** [Look for "Non-marketable equity securities," "Equity Method Investments," or "strategic
partnerships"].
* **The Valuation:** [Estimate the *real* market value vs. the book value. E.g., "Carried at $1B, worth
$10B"].
* **Warren's Take:** [A quote about "Margin of Safety" or finding a "Rembrandt in the attic"].
---
# QUALITY CHECKLIST
* Did you ignore "Goodwill" and "Intangible Assets" (accounting lines) and focus on *real* intangibles
(People/Stakes)?
* Did you estimate a specific dollar value for the "Hidden Treasure"?
* Did you use the specific metaphors "Brain Drain" and "Hidden Treasure"?
"""
        ),
        (
            "Footnotes",
            """# SYSTEM ROLE: The "Omaha" Forensic Engine
**Identity:** You are a composite intelligence of **Warren Buffett** (Business Quality, Moats, Cash Flow)
and **Charlie Munger** (Risk, Cynicism, Inversion).
**Goal:** You are writing **Section 2G (Footnotes)** of a confidential Investment Committee Memo.
**Tone:** Suspicious, inverted thinking. Use terms like "Cockroaches," "Shadow Debt," "Bodies buried,"
"Dumpster Fire."
# CONTEXT INJECTION (THE THESIS SO FAR)
*Instructions to AI: The following is the Executive Summary we have already written. You must align your
Footnotes analysis with this thesis. If we said the Balance Sheet was a "Fortress," prove it isn't by finding
hidden liabilities.*
{context}
# INPUT:
{raw_text}
# MISSION: DEEP DIVE - THE FOOTNOTES
**CRITICAL RULE:** Do not read the glossy cover. Look for the shovel. Find the liabilities that are NOT
on the Balance Sheet.
**Write 3 distinct "Vignettes" (Stories) using this exact structure:**
### **DEEP DIVE G: THE FOOTNOTES (The "Cockroach" Hunt)**
**Mini-Summary:** [One cynical sentence on what you found in the fine print. E.g., "The balance sheet
looks clean until you read Note 7."]
**1. The "Shadow Debt": [Name the Liability, e.g., Purchase Commitments]**
* **What we found:** [Identify unconditional obligations to pay suppliers in the future].
* **The Signal:** [Quote the specific dollar amount of "Purchase Obligations" or "Commitments" from the
Notes].
* **The Reality:** [This is debt. If revenue stops, they still have to pay this. Add it to the debt pile].
* **Charlie's Take:** [A cynical quote about off-balance-sheet financing].
**2. The "Real Debt": Lease Obligations**
* **What we found:** [Companies pretend rent isn't debt. We disagree].
* **The Signal:** [Quote the total "Operating Lease Liabilities" (Current + Non-Current) from the Notes].
* **The Reality:** [Are these leases for "Data Centers" or "Offices"? If Data Centers, it's a fixed cost of
production].
* **Warren's Take:** [A quote about fixed costs eating cash flow].
**3. The "Dumpster Fire": [Name the Ugly Item, e.g., Corporate Activities or Pension Liabilities]**
* **What we found:** [Identify a segment or cost center that is burning cash without generating revenue].
* **The Signal:** [Operating loss of "Corporate" segment or "Unallocated" costs].
* **The Reality:** [Is management hiding expenses here?].
* **The Verdict:** [Clean up or cover up?].
---
# QUALITY CHECKLIST
* Did you find the specific dollar amount for "Purchase Commitments"?
* Did you add up the Lease Liabilities?
* Did you use the specific metaphors "Shadow Debt" and "Dumpster Fire"?
"""
        ),
        (
            "Customers & Sentiment",
            """Crucial Note: This is the hardest section to generate solely from a 10-K because 10-Ks rarely
mention "Gen Z sentiment" or "Trust scores." The Google report you liked used external
knowledge (e.g., "64% of Gen Z use TikTok").
To fix this, I have added a specific instruction in this prompt to "Infer Sentiment from Financials"
(e.g., using RPO as a proxy for trust) if external data isn't provided.
# CONTEXT FROM PHASE 1:
"Here is the thesis we established. Align your Customer analysis with this:"
{context}

# INPUT:
{raw_text}
*(Optional: Paste recent news/analyst reports here for better sentiment analysis)*
# MISSION: DEEP DIVE - CUSTOMERS & SENTIMENT
**CRITICAL RULE:** 10-Ks don't explicitly talk about "feelings." You must **infer** sentiment from
**Backlog (RPO)** and **Retention Rates**.
**Write 3 distinct "Vignettes" (Stories) using this exact structure:**
### **DEEP DIVE I: CUSTOMERS & SENTIMENT (The Franchise)**
**Mini-Summary:** [One cynical sentence. E.g., "The customers are locked in, but they aren't
happy about the rent hike."]
**1. The "Lock-In" (or "The Handcuffs"): [Name the Metric, e.g., RPO/Backlog]**
* **What we found:** [Are customers committing to long-term contracts?].
* **The Signal:** [Quote the "Remaining Performance Obligation" (RPO) number and growth %].
* **The Reality:** [Is this "Love" (Voluntary) or "Hostage Taking" (Contractual)?].
* **Charlie's Take:** [A cynical quote about "stickiness" vs. "trapped customers"].
**2. The "Trust" Crisis:**
* **What we found:** [Is the brand deteriorating?].
* **The Signal:** [Look for "Net Retention Rates" or comments on "Churn" in the MD&A. If
missing, look for lawsuits from customers].
* **The Reality:** [Are they losing the "Hearts and Minds" of the user base?].
* **The Verdict:** [Loyal or Looking for an Exit?].
**3. The "Wallet" Share:**
* **What we found:** [Are big customers spending *more* each year?].
* **The Signal:** [Look for "Revenue from top customers" or "ARPU" (Average Revenue Per User)
trends].
* **The Reality:** [Pricing power. Can they raise prices without losing volume?].
* **Warren's Take:** [A quote about the beauty of a product people *must* buy].
---
# QUALITY CHECKLIST
* Did you find the **RPO (Remaining Performance Obligation)** number? (This is the best proxy
for customer commitment).
* Did you assess if customers are "Voluntary" or "Trapped"?
"""
        ),
        (
            "YTD Review",
            """# SYSTEM ROLE: The "Omaha" Forensic Engine
**Identity:** You are a composite intelligence of **Warren Buffett** (Business Quality, Moats, Cash Flow)
and **Charlie Munger** (Risk, Cynicism, Inversion).
**Goal:** You are writing **Section 3 (YTD Review)** of a confidential Investment Committee Memo.
**Tone:** Urgent, reactionary, and forensic. Use terms like "Red Queen Race," "Sugar High," "Bleeding
Cash," "Broken Thesis."
# CONTEXT INJECTION (THE THESIS SO FAR)
*Instructions to AI: The following is the Executive Summary we established from the 10-K. You must now
"Stress Test" this thesis against the latest Quarterly numbers. If the thesis was "Fortress Balance Sheet,"
but they just spent all the cash, you must scream "BROKEN THESIS."*
{context}
# INPUT:
{raw_text}
# MISSION: DEEP DIVE - THE YEAR-TO-DATE REVIEW
**CRITICAL RULE:** Do not just summarize the quarter. **Compare the Trend.** Is the business getting
better or worse since the annual report?
**Write 4 distinct "Vignettes" (Stories) using this exact structure:**
### **SECTION 3: THE YTD REVIEW (The Trendline)**
**Mini-Summary:** [One cynical sentence on the direction of travel. E.g., "The revenue is accelerating,
but they are burning the furniture to stay warm."]
**1. The "Capex Beast": [Feeding the AI God]**
* **What we found:** [Is Capital Expenditure accelerating?].
* **The Signal:** [List CapEx by Quarter: Q1 vs Q2 vs Q3].
* **The Math:** [Calculate **Free Cash Flow** for the YTD period (Operating Cash Flow - CapEx). Is it
negative?].
* **The Verdict:** [Is this "Moat Widening" (Good) or "The Red Queen Race" (Spending just to survive)?].
* **Charlie's Take:** [A cynical quote about capital intensity].
**2. The "Growth" Check: [Accelerating or Stalling?]**
* **What we found:** [Is top-line growth speeding up?].
* **The Signal:** [Compare Revenue Growth % in the latest quarter vs. the full year average].
* **The Reality:** [Is the "Capex Beast" actually generating new sales, or is it a bridge to nowhere?].
* **Warren's Take:** [A quote about "Intrinsic Value" growth].
**3. The "Sugar High": [Non-Operating Gains/Losses]**
* **What we found:** [Did they engineer their Earnings Per Share (EPS) with one-time tricks?].
* **The Signal:** [Look for "Gain on sale of assets," "Investment Income," or "Tax Benefits"].
* **The Reality:** [Strip this out. What is the *real* operating earning power?].
* **The Verdict:** [Quality Earnings or Accounting Gimmicks?].
**4. FINAL VERDICT FROM OMAHA (Updated)**
* **The Shift:** [Has the thesis changed since the 10-K?].
* **Buffett's Final Vote:** Buy, Sell, or Hold? (Focus on the long-term compounder).
* **Munger's Final Vote:** Buy, Sell, or Hold? (Focus on the "Insanity" of the spending/valuation).
---
# QUALITY CHECKLIST
* Did you calculate **Free Cash Flow** (OCF - CapEx)?
* Did you strip out "One-Time Gains" (e.g., investment sales) to find true earnings?
* Did you explicitly compare the latest quarter to the annual trend?
"""
        ),
    ]
    return prompts


def _run_omaha_chain(
    source_texts: Dict[str, str],
    extra_instruction: str = "",
) -> Generator[str, None, str]:
    """
    Execute the sequential Omaha Chain using Gemini 1.5 Pro.
    Streams tokens as they are generated.
    Retries once if the model refuses ("I cannot answer").
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    model_name = _resolve_gemini_model()
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.7,
        google_api_key=api_key,
        streaming=True,
    )
    reduce_llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=api_key,
        streaming=False,
    )

    prompts = _build_omaha_prompts()
    sanitized_sources = {
        key: (value or "").strip()
        for key, value in (source_texts or {}).items()
    }
    sec_business = sanitized_sources.get("SEC_BUSINESS", "")
    sec_financials = sanitized_sources.get("SEC_FINANCIALS", "")
    sec_legal = sanitized_sources.get("SEC_LEGAL", "")
    sec_quarterly = sanitized_sources.get("SEC_QUARTERLY", "")
    raw_prev_10k = sanitized_sources.get("RAW_PREV_10K", "")
    company_name = sanitized_sources.get("COMPANY_NAME", "")
    ticker_symbol = sanitized_sources.get("TICKER", "")

    if not sec_business or not sec_financials or not sec_quarterly:
        raise ValueError("Missing required section text. Check EDGAR fetch results.")

    def _extract_subsection(text: str, heading: str) -> str:
        if not text:
            return ""
        pattern = re.compile(rf"{re.escape(heading)}\\s*", re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return ""
        end_pattern = re.compile(r"\n[A-Z][A-Z\s/&-]{4,}\n|item\s+\d+[a-z]?\s*(?:\.|:|-)", re.IGNORECASE)
        end_match = end_pattern.search(text, match.end())
        end_index = end_match.start() if end_match else len(text)
        return text[match.start():end_index].strip()

    def _extract_first_heading(text: str, headings: List[str]) -> str:
        for heading in headings:
            result = _extract_subsection(text, heading)
            if result:
                return result
        return ""

    def _is_rate_limit_error(exc: Exception) -> bool:
        if isinstance(exc, ResourceExhausted):
            return True
        message = str(exc).lower()
        return "429" in message or "resource_exhausted" in message or "rate limit" in message

    def _retry_backoff_seconds(attempt: int) -> int:
        base = int(os.getenv("RETRY_BACKOFF_SECONDS", "30"))
        return base * (2 ** attempt)

    def _chunk_text(text: str, max_chars: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start = end
        return chunks

    def _reduce_large_text(text: str, step_label: str) -> str:
        max_chars = int(os.getenv("MAX_SECTION_CHARS", "20000"))
        enable_reduce = os.getenv("ENABLE_REDUCE", "true").lower() in {"1", "true", "yes"}
        if not enable_reduce:
            return text
        chunks = _chunk_text(text, max_chars)
        if len(chunks) == 1:
            return text

        summaries = []
        for idx, chunk in enumerate(chunks, start=1):
            summary_prompt = (
                "Extract the most relevant facts, numbers, risks, and direct quotes for this analysis step.\n"
                f"Step: {step_label}\n"
                "Be terse. Preserve exact numbers and quoted text.\n\n"
                f"CHUNK {idx}/{len(chunks)}:\n{chunk}"
            )
            summaries.append(_invoke_with_retry(summary_prompt))
        return "\n\n".join([s for s in summaries if s]).strip()

    def _invoke_with_retry(prompt: str) -> str:
        attempts = int(os.getenv("RETRY_ATTEMPTS", "4"))
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                result = reduce_llm.invoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content="Begin."),
                    ]
                )
                return (result.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_rate_limit_error(exc) and attempt < attempts - 1:
                    time.sleep(_retry_backoff_seconds(attempt))
                    continue
                raise
        raise last_exc if last_exc else RuntimeError("Unknown LLM failure.")

    prompt_map = {name: template for name, template in prompts}
    step_order = [
        "Executive Summary",
        "Supply Chain",
        "Competition",
        "Management & Governance",
        "Product Economics",
        "Regulatory Threats",
        "Intangibles",
        "Footnotes",
        "Customers & Sentiment",
        "YTD Review",
    ]

    accumulated_context = ""
    final_output = ""
    thesis_context = ""

    for step_index, prompt_name in enumerate(step_order, start=1):
        system_template = prompt_map[prompt_name]
        yield f"\n\n## Step {step_index}: {prompt_name}\n\n"
        raw_text = ""
        if prompt_name == "Executive Summary":
            mdna_only = _extract_first_heading(
                sec_financials,
                ["Management’s Discussion and Analysis", "Management's Discussion and Analysis"],
            )
            raw_text = "\n\n".join([part for part in [sec_business, mdna_only or sec_financials] if part]).strip()
        elif prompt_name == "Supply Chain":
            raw_text = _extract_subsection(sec_business, "Manufacturing")
            raw_text = raw_text or _extract_subsection(sec_business, "Suppliers") or sec_business
        elif prompt_name == "Competition":
            competition = _extract_subsection(sec_business, "Competition")
            mdna = _extract_first_heading(
                sec_financials,
                ["Management’s Discussion and Analysis", "Management's Discussion and Analysis"],
            ) or sec_financials
            raw_text = "\n\n".join([part for part in [competition, mdna] if part]).strip()
        elif prompt_name == "Management & Governance":
            liquidity = _extract_first_heading(
                sec_financials,
                ["Liquidity and Capital Resources", "Liquidity and Capital Resource"],
            )
            raw_text = liquidity or sec_financials
        elif prompt_name == "Product Economics":
            segment = _extract_first_heading(
                sec_financials,
                ["Segment Reporting", "Segments", "Segment"],
            ) or sec_financials
            raw_text = (
                "Compare current 10-K revenue versus prior 10-K revenue.\n\n"
                + segment
                + ("\n\nPRIOR 10-K (RAW):\n" + raw_prev_10k if raw_prev_10k else "")
            )
        elif prompt_name == "Regulatory Threats":
            risk_factors = _extract_subsection(sec_business, "Risk Factors") or sec_business
            raw_text = "\n\n".join([part for part in [sec_legal, risk_factors] if part]).strip()
        elif prompt_name == "Intangibles":
            human = _extract_subsection(sec_business, "Human Capital")
            raw_text = human or sec_business
        elif prompt_name == "Footnotes":
            notes = _extract_first_heading(
                sec_financials,
                ["Notes to the Financial Statements", "Notes to Consolidated Financial Statements"],
            )
            raw_text = notes or sec_financials
        elif prompt_name == "Customers & Sentiment":
            revenue = _extract_first_heading(
                sec_financials,
                ["Revenue Recognition", "Revenue"],
            ) or sec_financials
            raw_text = "\n\n".join([part for part in [sec_business, revenue] if part]).strip()
        elif prompt_name == "YTD Review":
            raw_text = sec_quarterly
        else:
            raw_text = sec_financials

        context_block = accumulated_context or "[NO PRIOR CONTEXT]"
        if step_index > 1 and thesis_context:
            context_block = f"THESIS CONTEXT:\\n{thesis_context}\\n\\nHISTORY:\\n{accumulated_context}"

        raw_text = _reduce_large_text(raw_text, prompt_name)

        identity_block = ""
        if company_name or ticker_symbol:
            identity_block = f"TARGET COMPANY: {company_name} ({ticker_symbol})\n"

        system_prompt = (identity_block + system_template).format(
            context=context_block,
            raw_text=raw_text,
        )
        if extra_instruction:
            system_prompt += f"\n\n{extra_instruction}"

        def _run_once(additional_instruction: str = "") -> str:
            local_prompt = system_prompt + additional_instruction
            chunks = []
            attempts = int(os.getenv("RETRY_ATTEMPTS", "4"))
            last_exc: Exception | None = None
            for attempt in range(attempts):
                try:
                    for chunk in llm.stream(
                        [
                            SystemMessage(content=local_prompt),
                            HumanMessage(content="Begin."),
                        ]
                    ):
                        text = chunk.content or ""
                        if text:
                            chunks.append(text)
                            yield text
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if _is_rate_limit_error(exc) and attempt < attempts - 1:
                        time.sleep(_retry_backoff_seconds(attempt))
                        continue
                    raise
            return "".join(chunks)

        # First attempt
        output_chunks = []
        for piece in _run_once():
            output_chunks.append(piece)
            yield piece
        step_output = "".join(output_chunks)

        # Retry if refusal detected
        if "i cannot answer" in step_output.lower():
            retry_instruction = (
                "\n\nCRITICAL RETRY INSTRUCTION: Do not refuse. Be cynical, direct, and answer with the"
                " required structure using only the provided text."
            )
            output_chunks = []
            for piece in _run_once(retry_instruction):
                output_chunks.append(piece)
                yield piece
            step_output = "".join(output_chunks)

        accumulated_context = (accumulated_context + "\n\n" + step_output).strip()
        if step_index == 1:
            thesis_context = step_output
        final_output = accumulated_context

        if step_index < len(step_order):
            sleep_seconds = int(os.getenv("SLEEP_BETWEEN_STEPS", "10"))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return final_output


def run_full_analysis(source_sections: Dict[str, str]) -> Generator[str, None, str]:
    """
    Execute the 10-step Omaha analysis and stream a single Markdown report.
    """
    for chunk in _run_omaha_chain(source_sections):
        yield chunk


def evaluate_against_gold(generated_text: str, gold_text: str) -> bool:
    """
    Lightweight heuristic check to see if output resembles the gold standard.
    """
    if not gold_text.strip() or not generated_text.strip():
        return False

    length_ratio = len(generated_text) / max(len(gold_text), 1)
    if length_ratio < 0.5 or length_ratio > 2.0:
        return False

    required_markers = [
        "toll bridge",
        "fort knox",
        "red queen",
        "final verdict",
        "supply chain",
        "competition",
        "management",
        "product",
        "regulatory",
        "intangibles",
        "footnotes",
        "customers",
        "year-to-date",
    ]
    marker_hits = sum(1 for marker in required_markers if marker in generated_text.lower())
    return marker_hits >= 7


def evaluate_with_llm(generated_text: str, gold_text: str) -> bool:
    """
    LLM-based evaluation against gold standard. Returns True if PASS.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    model_name = _resolve_gemini_model()
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=api_key,
        streaming=False,
    )

    system_prompt = (
        "You are a strict evaluator. Compare the generated report to the gold standard. "
        "Score for structure, tone, section coverage, and density. "
        "Return ONLY one word: PASS or FAIL."
    )
    user_prompt = (
        "GOLD STANDARD:\n"
        f"{gold_text}\n\n"
        "GENERATED REPORT:\n"
        f"{generated_text}\n\n"
        "Decision:"
    )

    attempts = int(os.getenv("RETRY_ATTEMPTS", "4"))
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            result = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            verdict = (result.content or "").strip().upper()
            return verdict.startswith("PASS")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            message = str(exc).lower()
            if "429" in message or "resource_exhausted" in message:
                time.sleep(30 * (2 ** attempt))
                continue
            raise
    raise last_exc if last_exc else RuntimeError("LLM evaluation failed.")


def omaha_chain_with_eval(
    source_texts: Dict[str, str],
    gold_standard_text: str,
) -> Generator[str, None, str]:
    """
    Run the 10-step chain and retry once if output fails gold-standard check.
    """
    buffer = []
    for chunk in _run_omaha_chain(source_texts):
        buffer.append(chunk)
        yield chunk

    generated = "".join(buffer)
    passes = False
    if gold_standard_text.strip():
        try:
            passes = evaluate_with_llm(generated, gold_standard_text)
        except Exception:
            passes = evaluate_against_gold(generated, gold_standard_text)
    else:
        passes = evaluate_against_gold(generated, gold_standard_text)

    if not passes:
        yield "\n\n---\n\n**Retrying to match the gold standard tone and structure...**\n\n"
        retry_instruction = (
            "Match the gold-standard report's tone, density, and structure. Use the same section "
            "headings and crisp, cynical voice."
        )
        for chunk in _run_omaha_chain(source_texts, extra_instruction=retry_instruction):
            yield chunk


def run_full_analysis_with_eval(
    source_sections: Dict[str, str],
    gold_standard_text: str,
) -> Generator[str, None, str]:
    """
    Wrapper for the 10-step analysis with gold-standard evaluation.
    """
    for chunk in omaha_chain_with_eval(source_sections, gold_standard_text):
        yield chunk
