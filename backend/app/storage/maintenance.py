"""One-off maintenance commands for the metadata database.

    python -m app.storage.maintenance fail-ownerless-jobs

fail-ownerless-jobs: jobs created before job owners were recorded can't be
checked for a dead worker, so an orphaned one would block re-uploads of its
document. Run this once after upgrading, after every process running the old
code has stopped; running it while an old worker is mid-ingestion would fail
that worker's job.
"""

from __future__ import annotations

import sys

from app.config import get_settings
from app.storage.repository import Repository


def main(argv: list[str]) -> int:
    if argv != ["fail-ownerless-jobs"]:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    failed = Repository(get_settings().db_path).fail_ownerless_active_jobs()
    print(f"marked {failed} ownerless job(s) failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
