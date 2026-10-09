"""Download fixed Wikipedia introductions; no queries or gold labels are read."""
import hashlib
import json
import re
import urllib.parse
import urllib.request
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELECTION = {
    "person": ["Ломоносов, Михаил Васильевич", "Менделеев, Дмитрий Иванович", "Пушкин, Александр Сергеевич", "Королёв, Сергей Павлович", "Гагарин, Юрий Алексеевич", "Ковалевская, Софья Васильевна", "Лобачевский, Николай Иванович", "Чебышёв, Пафнутий Львович"],
    "organization": ["Московский государственный университет", "Санкт-Петербургский государственный университет", "Казанский федеральный университет", "Новосибирский государственный университет", "Российская академия наук", "Объединённый институт ядерных исследований", "Государственный Эрмитаж", "Русское географическое общество"],
    "location": ["Москва", "Санкт-Петербург", "Казань", "Новосибирск", "Нижний Новгород", "Дубна", "Переславль-Залесский", "Йошкар-Ола"],
}


def main():
    raw_dir = ROOT / "data/corpus/raw"
    text_dir = ROOT / "data/corpus/texts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for category, titles in SELECTION.items():
        for number, title in enumerate(titles, 1):
            doc_id = f"{category}_{number:02d}"
            raw_path = raw_dir / f"{doc_id}.json"
            if raw_path.exists():
                payload = json.loads(raw_path.read_text())
            else:
                params = dict(action="query", format="json", formatversion=2, redirects=1,
                              prop="extracts|info|revisions", exintro=1, explaintext=1,
                              inprop="url", rvprop="ids|timestamp", titles=title)
                url = "https://ru.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
                request = urllib.request.Request(url, headers={"User-Agent": "Lab4Corpus/1.0 (educational text indexing)"})
                for attempt in range(5):
                    try:
                        with urllib.request.urlopen(request, timeout=45) as response:
                            payload = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "response": json.load(response)}
                        break
                    except urllib.error.HTTPError as exc:
                        if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                            raise
                        time.sleep(min(30, 3 * 2 ** attempt))
                raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                time.sleep(1)
            page = payload["response"]["query"]["pages"][0]
            if page.get("missing"):
                raise ValueError(f"Missing page: {title}")
            text = page["extract"].replace("\u0301", "")
            text = text.replace("\u2010", "-").replace("\u2011", "-")
            text = text.replace("[…]", "")
            text = text.removeprefix("Дореволюционные даты приводятся по юлианскому календарю.").lstrip()
            # Remove pronunciation markup, retaining historical names after ';'.
            text = re.sub(r"\(МФА:[^;)]*\)", "", text)
            text = re.sub(r"\(МФА:[^;)]*;\s*", "(", text)
            text = re.sub(r"\[[^\]]*[ɐɑəʲˈ][^\]]*\]", "", text)
            text = re.sub(r" +([,.)])", r"\1", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
            if len(text) < 200:
                raise ValueError(f"Short introduction: {title}")
            text_path = text_dir / f"{doc_id}.txt"
            text_path.write_text(text)
            rev = page["revisions"][0]
            manifest.append(dict(id=doc_id, category=category, requested_title=title,
                title=page["title"], page_id=page["pageid"], source_url=page["fullurl"],
                revision_id=rev["revid"], revision_timestamp=rev["timestamp"],
                permanent_url=f"https://ru.wikipedia.org/w/index.php?oldid={rev['revid']}",
                retrieved_at=payload["retrieved_at"], included_sections=["Вводная часть до первого раздела"],
                license="CC BY-SA 4.0; see Wikimedia Terms of Use",
                attribution_url=page["fullurl"] + "?action=history",
                raw_file=str(raw_path.relative_to(ROOT)), text_file=str(text_path.relative_to(ROOT)),
                chars=len(text), words=len(re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*", text)),
                sha256=hashlib.sha256(text.encode()).hexdigest()))
            print(doc_id, page["title"], len(text), flush=True)
    assert len(manifest) == len({x["sha256"] for x in manifest}) == 24
    (ROOT / "data/corpus/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
