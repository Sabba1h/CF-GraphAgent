import json
from pathlib import Path

import pyarrow.parquet as pq


# Convert the first rows of the Hugging Face parquet file to JSON.
ROW_LIMIT = 2000 #要转换的条数
BASE_DIR = Path(__file__).resolve().parent
SRC = BASE_DIR / "validation-00000-of-00001.parquet"
DST = BASE_DIR / "hotpotqa_fullwiki_validation_2000.json"


def read_first_rows(src: Path, row_limit: int) -> list[dict]:
    parquet_file = pq.ParquetFile(src)
    rows: list[dict] = []

    for batch in parquet_file.iter_batches(batch_size=row_limit):
        rows.extend(batch.to_pylist())
        if len(rows) >= row_limit:
            break

    return rows[:row_limit]


def main() -> None:
    records = read_first_rows(SRC, ROW_LIMIT)

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"wrote {len(records)} records to {DST}")


if __name__ == "__main__":
    main()
