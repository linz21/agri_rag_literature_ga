"""
PubMed paper ingestion via NCBI E-utilities API.

Uses the free E-utilities REST API (no API key required, though one is
recommended for higher rate limits): esearch to find paper IDs matching
search terms, then efetch to retrieve abstracts and metadata.

NOTE ON FULL TEXT: PubMed's API only reliably provides ABSTRACTS, not full
paper text, for the vast majority of papers (most publishers don't allow
full-text redistribution via PubMed). Full text is only available for the
subset of papers in PubMed Central (PMC) that are open-access. This project
uses abstracts as the retrieval corpus — see README Known Gaps for the
implications of this and how to extend to full text via PMC if desired.

Usage:
    python src/data/ingest.py
    python src/data/ingest.py --demo   # generates synthetic demo papers, no API needed
"""

import argparse
import json
import logging
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def esearch(term: str, base_url: str, email: str, api_key: str = None,
            retmax: int = 100) -> list[str]:
    """Search PubMed for a term, return a list of PubMed IDs (PMIDs)."""
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(f"{base_url}esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def efetch_abstracts(pmids: list[str], base_url: str, email: str,
                     api_key: str = None) -> list[dict]:
    """Fetch abstract + metadata for a batch of PubMed IDs."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(f"{base_url}efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = article.findtext(".//PMID")
            title = article.findtext(".//ArticleTitle") or ""

            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (part.text or "") for part in abstract_parts
            ).strip()

            journal = article.findtext(".//Journal/Title") or ""
            year = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate") or ""

            authors = []
            for author in article.findall(".//Author"):
                last = author.findtext("LastName")
                fore = author.findtext("ForeName")
                if last:
                    authors.append(f"{fore} {last}".strip() if fore else last)

            if not abstract:
                continue  # skip papers with no abstract — nothing to retrieve/embed

            papers.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "authors": authors,
            })
        except Exception as e:
            log.warning(f"Failed to parse an article entry: {e}")
            continue

    return papers


def fetch_all_papers(cfg: dict) -> list[dict]:
    """Run esearch + efetch across all configured search terms, deduplicated by PMID."""
    pubmed_cfg = cfg["pubmed"]
    base_url = pubmed_cfg["base_url"]
    email = pubmed_cfg["email"]
    api_key = pubmed_cfg.get("api_key")
    retmax = pubmed_cfg["retmax_per_term"]

    if email == "REPLACE_WITH_YOUR_EMAIL@example.com":
        raise ValueError(
            "Set a real email in configs/config.yaml under pubmed.email — "
            "NCBI's usage policy requires this to identify API traffic."
        )

    all_papers = {}
    rate_limit_delay = 0.34 if not api_key else 0.11  # 3/s vs 10/s per NCBI policy

    for term in pubmed_cfg["search_terms"]:
        log.info(f"Searching PubMed for: '{term}' ...")
        pmids = esearch(term, base_url, email, api_key, retmax)
        log.info(f"  Found {len(pmids)} PMIDs")
        time.sleep(rate_limit_delay)

        # Fetch in batches of 50 (NCBI recommends batching efetch calls)
        for i in range(0, len(pmids), 50):
            batch = pmids[i:i + 50]
            papers = efetch_abstracts(batch, base_url, email, api_key)
            for p in papers:
                all_papers[p["pmid"]] = p  # dedupe across search terms
            time.sleep(rate_limit_delay)

    log.info(f"Total unique papers fetched: {len(all_papers)}")
    return list(all_papers.values())


def make_demo_papers(n: int = 30) -> list[dict]:
    """Generate synthetic demo papers for pipeline development without hitting the real API."""
    topics = [
        ("Nitrogen fertilization timing and corn yield response",
         "This study examines the effect of split nitrogen application timing on corn grain yield "
         "across three growing seasons. Results indicate that side-dress application at V6 growth "
         "stage improved nitrogen use efficiency by 18% compared to single pre-plant application, "
         "with yield increases of 8-12 bushels per acre under drought-stressed conditions."),
        ("Remote sensing indices for early drought stress detection in maize",
         "Normalized difference vegetation index (NDVI) and chlorophyll fluorescence measurements "
         "were used to detect drought stress in maize 5-7 days before visible wilting symptoms "
         "appeared. Multispectral drone imagery achieved 84% classification accuracy for stress "
         "detection when combined with soil moisture sensor data."),
        ("Soil organic matter effects on corn root development under water limitation",
         "Field trials across variable soil organic matter content (1.2%-4.8%) showed that higher "
         "organic matter correlated with deeper root penetration and improved water extraction "
         "efficiency during reproductive growth stages, contributing to yield stability in dry years."),
    ]

    papers = []
    for i in range(n):
        title, abstract = topics[i % len(topics)]
        papers.append({
            "pmid": f"DEMO{1000+i}",
            "title": f"{title} (Study {i+1})",
            "abstract": abstract,
            "journal": "Journal of Agronomic Science (demo)",
            "year": str(2015 + (i % 10)),
            "authors": [f"Author{i}, A."],
        })
    return papers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo papers")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "papers.json"

    if args.demo:
        log.info("Running in DEMO mode — generating synthetic papers ...")
        papers = make_demo_papers(n=30)
    else:
        papers = fetch_all_papers(cfg)

    with open(out_path, "w") as f:
        json.dump(papers, f, indent=2)

    log.info(f"Saved {len(papers)} papers → {out_path}")


if __name__ == "__main__":
    main()
