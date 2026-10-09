"""Download official Slovnet/Navec packs once; inference is offline."""
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKS = {
    "slovnet_ner_news_v1.tar": "https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_ner_news_v1.tar",
    "navec_news_v1_1B_250K_300d_100q.tar": "https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar",
}


def main():
    directory = ROOT / "models"
    directory.mkdir(exist_ok=True)
    previous = json.loads((directory / "manifest.json").read_text()) if (directory / "manifest.json").exists() else {}
    rows = {}
    for name, url in PACKS.items():
        path = directory / name
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "Lab4NER/1.0 (educational use)"})
            temporary = path.with_suffix(".download")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                    temporary.replace(path)
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in previous and digest != previous[name]["sha256"]:
            raise ValueError(f"Изменился хеш сохранённого пакета: {name}")
        rows[name] = dict(source_url=url, bytes=path.stat().st_size, sha256=digest,
                          retrieved_at=previous.get(name, {}).get("retrieved_at", datetime.now(timezone.utc).isoformat()))
        print(name, path.stat().st_size, flush=True)
    (directory / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
