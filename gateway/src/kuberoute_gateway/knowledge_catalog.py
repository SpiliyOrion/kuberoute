"""Validate a local, explicit source catalog without changing RAG's hashed IDs."""

import argparse
import ipaddress
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from kuberoute_gateway.rag.ingestion import DocumentInputError, load_documents

MAX_CATALOG_BYTES = 262_144
MAX_CATALOG_SOURCES = 100
_ID = re.compile(r"[a-z][a-z0-9_]{2,63}\Z")
_PMID = re.compile(r"[0-9]{1,12}\Z")
_DOI = re.compile(r"10\.[0-9]{4,9}/[^\s]{1,180}\Z")


class CatalogError(ValueError):
    """A safe catalog validation error, without document paths or contents."""


@dataclass(frozen=True)
class CatalogSource:
    id: str
    document: Path = field(repr=False)


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CatalogError("Catalog contains duplicate JSON keys")
        result[key] = value
    return result


def _required_text(record: dict[str, object], key: str, maximum: int = 300) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise CatalogError("Catalog metadata is incomplete or invalid")
    return value


def _validate_source(record: object, root: Path) -> CatalogSource:
    if not isinstance(record, dict):
        raise CatalogError("Catalog source is invalid")
    allowed = {
        "id",
        "document",
        "title",
        "authors",
        "publication_year",
        "publication_date_note",
        "original_url",
        "doi",
        "pmid",
        "source_type",
        "topics",
        "verified_on",
    }
    if set(record) - allowed:
        raise CatalogError("Catalog source metadata is invalid")
    source_id = _required_text(record, "id", 64)
    if not _ID.fullmatch(source_id):
        raise CatalogError("Catalog source ID is invalid")
    relative_text = _required_text(record, "document", 240)
    relative = PurePosixPath(relative_text)
    if (
        "\\" in relative_text
        or ":" in relative_text
        or relative.is_absolute()
        or not relative.parts
        or any(part in (".", "..") for part in relative.parts)
        or relative.parts[0] != "sources"
        or relative.suffix not in (".md", ".txt")
    ):
        raise CatalogError("Catalog document path is invalid")
    document = root.joinpath(*relative.parts)
    try:
        canonical = document.resolve(strict=True)
        if not canonical.is_relative_to(root.resolve(strict=True)) or not canonical.is_file():
            raise CatalogError("Catalog document path is invalid")
    except (OSError, RuntimeError, ValueError):
        raise CatalogError("Catalog document is unavailable") from None

    _required_text(record, "title")
    _required_text(record, "authors")
    _required_text(record, "source_type", 64)
    if "publication_year" not in record or "doi" not in record or "pmid" not in record:
        raise CatalogError("Catalog metadata is incomplete or invalid")
    year = record.get("publication_year")
    if year is None:
        _required_text(record, "publication_date_note", 160)
    elif (
        isinstance(year, bool) or not isinstance(year, int) or not 1800 <= year <= date.today().year
    ):
        raise CatalogError("Catalog publication year is invalid")
    try:
        url = urlsplit(_required_text(record, "original_url", 500))
        host = url.hostname
        try:
            address = ipaddress.ip_address(host or "")
        except ValueError:
            address = None
        invalid_url = (
            url.scheme != "https"
            or not host
            or "." not in host
            or host == "localhost"
            or host.endswith(".local")
            or (address is not None and not address.is_global)
            or "@" in url.netloc
            or url.username
            or url.password
            or url.port is not None
            or url.query
            or url.fragment
        )
    except ValueError:
        invalid_url = True
    if invalid_url:
        raise CatalogError("Catalog source URL is invalid")
    doi = record.get("doi")
    if doi is not None and (not isinstance(doi, str) or not _DOI.fullmatch(doi)):
        raise CatalogError("Catalog DOI is invalid")
    pmid = record.get("pmid")
    if pmid is not None and (not isinstance(pmid, str) or not _PMID.fullmatch(pmid)):
        raise CatalogError("Catalog PMID is invalid")
    topics = record.get("topics")
    if (
        not isinstance(topics, list)
        or not 1 <= len(topics) <= 8
        or any(not isinstance(topic, str) or not _ID.fullmatch(topic) for topic in topics)
        or len(set(topics)) != len(topics)
    ):
        raise CatalogError("Catalog topic labels are invalid")
    try:
        date.fromisoformat(_required_text(record, "verified_on", 10))
    except ValueError:
        raise CatalogError("Catalog verification date is invalid") from None
    return CatalogSource(source_id, canonical)


def load_catalog(manifest: str | Path) -> tuple[CatalogSource, ...]:
    """Validate metadata and ingestion compatibility for every explicit source."""
    path = Path(manifest)
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_CATALOG_BYTES + 1)
    except OSError:
        raise CatalogError("Catalog is unavailable") from None
    if len(raw) > MAX_CATALOG_BYTES:
        raise CatalogError("Catalog exceeds the size limit")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CatalogError("Catalog JSON is invalid") from None
    if not isinstance(data, dict) or set(data) != {"schema_version", "collection_id", "sources"}:
        raise CatalogError("Catalog structure is invalid")
    if data["schema_version"] != 1 or isinstance(data["schema_version"], bool):
        raise CatalogError("Catalog schema version is unsupported")
    collection_id = data["collection_id"]
    if not isinstance(collection_id, str) or not _ID.fullmatch(collection_id):
        raise CatalogError("Catalog collection ID is invalid")
    records = data["sources"]
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_CATALOG_SOURCES:
        raise CatalogError("Catalog source count is invalid")
    sources = tuple(_validate_source(record, path.parent) for record in records)
    if len({source.id for source in sources}) != len(sources):
        raise CatalogError("Catalog contains duplicate source IDs")
    if len({source.document for source in sources}) != len(sources):
        raise CatalogError("Catalog contains duplicate documents")
    try:
        load_documents([source.document for source in sources])
    except DocumentInputError:
        raise CatalogError("Catalog contains an unreadable document") from None
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an explicit local knowledge catalog")
    parser.add_argument("command", choices=("validate", "paths"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        sources = load_catalog(args.manifest)
    except CatalogError as exc:
        parser.exit(1, f"{exc}\n")
    if args.command == "validate":
        print(f"Catalog valid: {len(sources)} sources")
    else:
        print(json.dumps([str(source.document) for source in sources]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
