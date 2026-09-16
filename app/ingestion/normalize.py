from __future__ import annotations

import argparse
import sys

from app.ingestion.loader import get_raw_post, load_raw_posts
from app.ingestion.normalizer import normalize_post, write_normalized_document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize WordPress fixtures into NormalizedDocument JSON.")
    parser.add_argument("--id", dest="post_id", help="WordPress post id")
    parser.add_argument(
        "--type",
        dest="content_type",
        choices=("article", "review", "provider"),
        help="Fixture content type",
    )
    parser.add_argument("--all", action="store_true", help="Normalize every fixture record")
    args = parser.parse_args(argv)

    if args.all:
        posts = load_raw_posts(args.content_type)
    elif args.post_id:
        posts = [get_raw_post(args.post_id, args.content_type)]
    else:
        parser.error("Provide --id or --all")

    for post in posts:
        document = normalize_post(post)
        path = write_normalized_document(post, document)
        print(f"{path}  ({len(document.sections)} sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
