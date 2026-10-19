"""Download arXiv abstracts for a fixed set of CS categories and date window.

Usage:
    python src/fetch_arxiv.py                      # full dataset (~2000 papers kept per category)
    python src/fetch_arxiv.py --per-category 50    # quick test
"""

import argparse
from pathlib import Path

import arxiv
import pandas as pd

# Distinct CS fields, so LDA topics can be checked against the real arXiv labels.
# cs.LG is left out on purpose: it overlaps with almost every other field.
CATEGORIES = ["cs.CL", "cs.CV", "cs.RO", "cs.CR", "cs.DB", "cs.NI"]
# Fixed window + ascending sort => the same query returns the same papers on every run.
DATE_WINDOW = "[202501010000 TO 202506302359]"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "arxiv_abstracts.csv"


def fetch_category(client, category, max_results):
    """Fetch papers whose primary category is `category` within DATE_WINDOW.

    Args:
        client (arxiv.Client): Client handling paging and the API rate limit.
        category (str): arXiv category, e.g. "cs.CL".
        max_results (int): Maximum number of papers requested from the API.

    Returns:
        (list[dict]): One record per paper with id, title, abstract, category and date.
    """
    search = arxiv.Search(
        query=f"cat:{category} AND submittedDate:{DATE_WINDOW}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )
    # Cross-listed papers also match `cat:`; keep only those whose main field is `category`.
    return [
        {
            "id": paper.get_short_id(),
            "title": paper.title,
            "abstract": paper.summary.replace("\n", " "),
            "category": paper.primary_category,
            "published": paper.published.date(),
        }
        for paper in client.results(search)
        if paper.primary_category == category
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ~1/3 of matches are cross-listed and dropped, so 3000 requested gives ~2000 kept.
    parser.add_argument("--per-category", type=int, default=3000, help="papers requested per category")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    client = arxiv.Client(page_size=500, delay_seconds=3.0, num_retries=5)
    records = []
    for category in CATEGORIES:
        papers = fetch_category(client, category, args.per_category)
        print(f"{category}: {len(papers)} papers")
        records.extend(papers)

    df = pd.DataFrame(records).drop_duplicates("id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} papers to {args.output}")


if __name__ == "__main__":
    main()
