"""
crawl_blogs.py
--------------
Crawls security research blogs and Exploit-DB for real attack write-ups.
Uses Crawl4AI — auto-converts HTML → clean Markdown.
Covers: payload_example, attack_method, evidence_summary, real exploit context
Output: raw_blogs.json

Install: pip install crawl4ai && playwright install
"""

import asyncio
import json
from crawl4ai import AsyncWebCrawler

# ── Target URLs ────────────────────────────────────────────────────────────
# Add or remove URLs as needed. These are public, scraping-permitted sources.
TARGET_URLS = [
    # Exploit-DB individual exploit write-ups
    "https://www.exploit-db.com/exploits/51893",
    "https://www.exploit-db.com/exploits/51839",
    "https://www.exploit-db.com/exploits/51777",

    # OWASP WSTG testing guide sections
    "https://owasp.org/www-project-web-security-testing-guide/v42/4-Web_Application_Security_Testing/07-Input_Validation_Testing/05-Testing_for_SQL_Injection",
    "https://owasp.org/www-project-web-security-testing-guide/v42/4-Web_Application_Security_Testing/11-Client-side_Testing/01-Testing_for_DOM-based_Cross_Site_Scripting",

    # Vulhub write-ups (GitHub raw markdown)
    "https://raw.githubusercontent.com/vulhub/vulhub/master/log4j/CVE-2021-44228/README.md",
    "https://raw.githubusercontent.com/vulhub/vulhub/master/shiro/CVE-2016-4437/README.md",

    # Add more as needed:
    # "https://blog.qualys.com/vulnerabilities-threat-research/...",
    # "https://www.rapid7.com/blog/post/...",
]

def classify_blog_record(url: str, markdown: str) -> dict:
    """
    Tag each crawled page with metadata for dataset builder to use.
    """
    text = markdown.lower()

    # Rough classification
    if "exploit-db.com" in url:
        source_type = "exploit_writeup"
    elif "owasp.org" in url:
        source_type = "owasp_guide"
    elif "vulhub" in url or "github.com" in url:
        source_type = "vulhub_writeup"
    else:
        source_type = "research_blog"

    # Detect referenced CVEs
    import re
    cves = list(set(re.findall(r"CVE-\d{4}-\d+", markdown, re.IGNORECASE)))

    return {
        "url":         url,
        "source_type": source_type,
        "cves_mentioned": cves,
        "content":     markdown
    }

async def crawl_all(urls: list[str]) -> list[dict]:
    results = []
    async with AsyncWebCrawler(verbose=False) as crawler:
        for url in urls:
            try:
                result = await crawler.arun(url=url)
                if result.success and result.markdown:
                    record = classify_blog_record(url, result.markdown)
                    results.append(record)
                    print(f"  ✅ {url[:80]}")
                else:
                    print(f"  ❌ Failed: {url[:80]}")
            except Exception as e:
                print(f"  ❌ Error ({url[:60]}): {e}")
    return results

def run(out="data/raw_blogs.json"):
    print(f"Crawling {len(TARGET_URLS)} security sources...")
    data = asyncio.run(crawl_all(TARGET_URLS))
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ Saved {len(data)} pages → {out}")

if __name__ == "__main__":
    run()