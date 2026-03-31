from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval hit rate against /search."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/erp_eval.json",
        help="Path to JSON file with eval rows.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8001",
        help="Base URL for python_rag service.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="topK to send to /search and use for hit@k.",
    )
    parser.add_argument(
        "--report-dir",
        default="data/eval/reports",
        help="Directory where CSV/Markdown reports are written.",
    )
    return parser.parse_args()


def post_search(base_url: str, query: str, top_k: int) -> list[dict]:
    url = f"{base_url.rstrip('/')}/search"
    payload = json.dumps({"query": query, "topK": top_k}).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("results", [])


def contains_expected(result: dict, expected: str) -> bool:
    needle = expected.strip().lower()
    if not needle:
        return False
    fields = [
        str(result.get("source", "")),
        str(result.get("chunk_id", "")),
        str(result.get("policy_type", "")),
        str(result.get("department", "")),
        str(result.get("text_preview", "")),
        str(result.get("text", "")),
    ]
    haystack = " ".join(fields).lower()
    return needle in haystack


def write_reports(
    report_dir: Path,
    total: int,
    hit_at_1: int,
    hit_at_k: int,
    top_k: int,
    rows: list[dict],
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    csv_path = report_dir / f"retrieval-eval-{stamp}.csv"
    md_path = report_dir / f"retrieval-eval-{stamp}.md"

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "question",
                "expected_source_contains",
                "hit_at_1",
                f"hit_at_{top_k}",
                "top_source",
                "top_chunk_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    hit1_pct = 100.0 * hit_at_1 / total
    hitk_pct = 100.0 * hit_at_k / total
    miss_rows = [r for r in rows if r[f"hit_at_{top_k}"] == "false"]
    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"- Evaluated: **{total}**",
        f"- Hit@1: **{hit_at_1}/{total} ({hit1_pct:.1f}%)**",
        f"- Hit@{top_k}: **{hit_at_k}/{total} ({hitk_pct:.1f}%)**",
        "",
        "## Misses (top 10)",
        "",
    ]
    if miss_rows:
        for row in miss_rows[:10]:
            lines.append(f"- Q: {row['question']}")
            lines.append(f"  - expected contains: `{row['expected_source_contains']}`")
            lines.append(f"  - top source: `{row['top_source']}`")
    else:
        lines.append("- None")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def run(dataset_path: Path, base_url: str, top_k: int, report_dir: Path) -> int:
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        print("Dataset is empty or invalid JSON list.")
        return 1

    total = 0
    hit_at_1 = 0
    hit_at_k = 0
    failures: list[tuple[str, str, str]] = []
    report_rows: list[dict] = []

    for row in rows:
        question = str(row.get("question", "")).strip()
        expected = str(row.get("expected_source_contains", "")).strip()
        if not question or not expected:
            continue
        total += 1
        try:
            results = post_search(base_url=base_url, query=question, top_k=top_k)
        except urllib.error.URLError as exc:
            print(f"Request failed for question: {question}\n{exc}")
            return 1

        if results and contains_expected(results[0], expected):
            hit_at_1 += 1
        has_hit_k = any(contains_expected(r, expected) for r in results[:top_k])
        if has_hit_k:
            hit_at_k += 1
        else:
            top_source = str(results[0].get("source")) if results else "NO_RESULTS"
            failures.append((question, expected, top_source))
        report_rows.append(
            {
                "question": question,
                "expected_source_contains": expected,
                "hit_at_1": str(bool(results and contains_expected(results[0], expected))).lower(),
                f"hit_at_{top_k}": str(has_hit_k).lower(),
                "top_source": str(results[0].get("source")) if results else "NO_RESULTS",
                "top_chunk_id": str(results[0].get("chunk_id")) if results else "NO_RESULTS",
            }
        )

    if total == 0:
        print("No valid rows found. Each row needs question + expected_source_contains.")
        return 1

    print(f"Evaluated: {total}")
    print(f"Hit@1: {hit_at_1}/{total} ({(100.0 * hit_at_1 / total):.1f}%)")
    print(f"Hit@{top_k}: {hit_at_k}/{total} ({(100.0 * hit_at_k / total):.1f}%)")
    print("")
    print("Misses (first 10):")
    for question, expected, top_source in failures[:10]:
        print(f"- Q: {question}")
        print(f"  expected contains: {expected}")
        print(f"  top source: {top_source}")

    csv_path, md_path = write_reports(
        report_dir=report_dir,
        total=total,
        hit_at_1=hit_at_1,
        hit_at_k=hit_at_k,
        top_k=top_k,
        rows=report_rows,
    )
    print("")
    print(f"CSV report: {csv_path}")
    print(f"Markdown report: {md_path}")

    return 0


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1
    if args.top_k < 1:
        print("--top-k must be >= 1")
        return 1
    return run(
        dataset_path=dataset_path,
        base_url=args.base_url,
        top_k=args.top_k,
        report_dir=Path(args.report_dir),
    )


if __name__ == "__main__":
    sys.exit(main())
