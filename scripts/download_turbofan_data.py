import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


RECORD = "https://zenodo.org/api/records/20053482"
OUTPUT = Path("data/external/dlr")


def download(url, destination):
    with urllib.request.urlopen(url) as response:
        with destination.open("wb") as stream:
            shutil.copyfileobj(response, stream)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD) as response:
        record = json.load(response)
    for item in record["files"]:
        name = item["key"]
        target = OUTPUT / name
        download(item["links"]["content"], target)
        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target) as archive:
                archive.extractall(OUTPUT)


if __name__ == "__main__":
    main()
