"""Google Drive からデータをダウンロードする."""

import tarfile
from pathlib import Path

import click
import gdown


@click.command()
@click.argument("url", type=str)
def main(url: str) -> None:
    """
    `url` からデータをダウンロードし、`data` ディレクトリに保存する.

    Examples
    --------
    >>> python scripts/download.py google_drive_url

    """
    filename = gdown.download(url, quiet=False, fuzzy=True)
    tarfile.open(filename, "r:gz").extractall(
        path=Path("data"),
        filter="data",
    )
    Path(filename).unlink(missing_ok=False)


if __name__ == "__main__":
    main()
