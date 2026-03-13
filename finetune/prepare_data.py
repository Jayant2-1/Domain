"""
Prepare fine-tuning data — exports positive interactions from the
database into instruction-tuning JSONL format.

Usage:
    python -m finetune.prepare_data \
        --db-path data/mlml.db \
        --output finetune/data/train.jsonl \
        --min-feedback 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import aiosqlite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def export_data(
    db_path: str,
    output_path: str,
    min_feedback: int = 1,
    limit: int = 50000,
) -> int:
    """
    Query positive interactions and write instruction-tuning JSONL.

    Each line:
        {"instruction": "<question>", "context": "<rag_context>", "response": "<response>"}

    Returns the number of records exported.
    """
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT i.question, i.rag_context, i.response, t.name AS topic
            FROM interactions i
            JOIN topics t ON t.id = i.topic_id
            WHERE i.feedback >= ?
            ORDER BY i.created_at DESC
            LIMIT ?
            """,
            (min_feedback, limit),
        )
        rows = await cursor.fetchall()

    if not rows:
        logger.warning("No positive interactions found (feedback >= %d).", min_feedback)
        return 0

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            instruction = row["question"]
            context = row["rag_context"] or ""
            response = row["response"]

            record = {
                "instruction": instruction,
                "context": context,
                "response": response,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Exported %d records to %s", count, out)
    return count


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"finetune/data/train_{ts}.jsonl"

    parser = argparse.ArgumentParser(description="Export positive interactions for LoRA training")
    parser.add_argument("--db-path", default="data/mlml.db", help="SQLite database path")
    parser.add_argument("--output", default=default_output, help="Output JSONL path")
    parser.add_argument("--min-feedback", type=int, default=1)
    parser.add_argument("--limit", type=int, default=50000)
    args = parser.parse_args()

    count = asyncio.run(export_data(args.db_path, args.output, args.min_feedback, args.limit))
    if count == 0:
        logger.warning("No data exported — collect more positive interactions first.")


if __name__ == "__main__":
    main()
