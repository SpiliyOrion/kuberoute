"""RAG ingestion tests use only temporary, synthetic local files."""

from pathlib import Path

import pytest
from kuberoute_gateway.rag import (
    ChunkingConfig,
    Document,
    DocumentInputError,
    SourceMetadata,
    chunk_document,
    ingest_paths,
    load_document,
    load_documents,
    normalize_text,
)


def test_normalization_is_deterministic_and_preserves_internal_layout() -> None:
    raw = "\r\nCafe\u0301  \r\n  indented\t\rsecond\n\n"
    normalized = "Café\n  indented\nsecond"
    assert normalize_text(raw) == normalized
    assert normalize_text(normalized) == normalized


@pytest.mark.parametrize("extension", [".txt", ".md", ".TXT", ".MD"])
def test_load_explicit_utf8_file_with_stable_private_source_id(
    tmp_path: Path, extension: str
) -> None:
    path = tmp_path / f"private-filename{extension}"
    raw = "\ufeffFirst line\r\nSecond line  \r\n".encode("utf-8")
    path.write_bytes(raw)
    first = load_document(path)
    second = load_document(str(path))
    assert first == second
    assert first.text == "First line\nSecond line"
    assert first.metadata == SourceMetadata(extension.lower(), len(raw))
    assert len(first.source_id) == 64
    assert "private-filename" not in repr(first.metadata)
    assert "private-filename" not in first.source_id
    assert str(tmp_path) not in repr(first.metadata)


def test_chunking_prefers_word_boundaries_and_retains_ordered_overlap() -> None:
    document = Document(
        "synthetic-source", "alpha beta gamma delta epsilon", SourceMetadata(".txt", 30)
    )
    config = ChunkingConfig(size=14, overlap=5)
    chunks = chunk_document(document, config)
    assert [chunk.text for chunk in chunks] == [
        "alpha beta",
        "beta gamma",
        "gamma delta",
        "delta epsilon",
    ]
    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3]
    assert all(chunk.source_id == document.source_id for chunk in chunks)
    assert all(chunk.metadata == document.metadata for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(len(chunk.text) <= config.size for chunk in chunks)
    assert chunk_document(document, config) == chunks


def test_chunk_ids_change_when_content_settings_or_source_change() -> None:
    source = SourceMetadata(".txt", 20)
    document = Document("source-a", "one two three four five", source)
    original = chunk_document(document, ChunkingConfig(10, 3))
    assert original
    assert (
        chunk_document(Document("source-b", document.text, source), ChunkingConfig(10, 3))[
            0
        ].chunk_id
        != original[0].chunk_id
    )
    assert (
        chunk_document(
            Document("source-a", "one two new four five", source), ChunkingConfig(10, 3)
        )[0].chunk_id
        != original[0].chunk_id
    )
    assert chunk_document(document, ChunkingConfig(11, 3))[0].chunk_id != original[0].chunk_id


def test_long_word_uses_hard_boundaries_and_makes_progress() -> None:
    document = Document("source", "x" * 31, SourceMetadata(".txt", 31))
    chunks = chunk_document(document, ChunkingConfig(size=10, overlap=3))
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.text and len(chunk.text) <= 10 for chunk in chunks)
    assert chunks[-1].text.endswith("x")
    assert len(chunks) == 4


def test_ingest_paths_keeps_explicit_file_and_chunk_order(tmp_path: Path) -> None:
    first = tmp_path / "one.txt"
    second = tmp_path / "two.md"
    first.write_text("alpha beta gamma delta epsilon", encoding="utf-8")
    second.write_text("heading and body", encoding="utf-8")
    config = ChunkingConfig(14, 5)
    documents = load_documents([first, second])
    chunks = ingest_paths([first, second], config=config)
    assert chunks == tuple(
        chunk for document in documents for chunk in chunk_document(document, config)
    )
    assert [chunk.source_id for chunk in chunks[:4]] == [documents[0].source_id] * 4
    assert chunks[-1].source_id == documents[1].source_id
    assert documents[0].source_id != documents[1].source_id


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (10_001, 0), (10, -1), (10, 10), (10, 11), (True, 0), (10, False)],
)
def test_invalid_chunk_settings_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="Chunk (size|overlap)"):
        ChunkingConfig(size=size, overlap=overlap)


def test_default_settings_and_excessive_chunk_count() -> None:
    assert ChunkingConfig() == ChunkingConfig(size=800, overlap=120)
    document = Document("source", "x" * 10_002, SourceMetadata(".txt", 10_002))
    with pytest.raises(DocumentInputError, match="too many chunks"):
        chunk_document(document, ChunkingConfig(size=1, overlap=0))


@pytest.mark.parametrize("bad_limit", [0, -1, 10_485_761, True, 1.5])
def test_invalid_file_size_limit_fails_before_read(tmp_path: Path, bad_limit: object) -> None:
    source = tmp_path / "example.txt"
    source.write_text("synthetic", encoding="utf-8")
    with pytest.raises(DocumentInputError, match="File-size limit"):
        load_document(source, max_file_bytes=bad_limit)


def test_oversized_empty_blank_invalid_utf8_and_control_character_files(tmp_path: Path) -> None:
    samples = (
        ("oversized.txt", b"abcdef", 5, "size limit"),
        ("empty.txt", b"", 100, "empty"),
        ("blank.md", b" \r\n\t ", 100, "no text"),
        ("invalid.txt", b"\xff", 100, "UTF-8"),
        ("binary.txt", b"secret\x00text", 100, "control characters"),
    )
    for name, raw, limit, message in samples:
        source = tmp_path / name
        source.write_bytes(raw)
        with pytest.raises(DocumentInputError, match=message) as error:
            load_document(source, max_file_bytes=limit)
        assert name not in str(error.value)
        assert "secret" not in str(error.value)
        assert str(tmp_path) not in str(error.value)


def test_unsupported_extension_directory_missing_file_and_duplicate_are_safe(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "private.pdf"
    unsupported.write_text("secret sample", encoding="utf-8")
    directory = tmp_path / "private.md"
    directory.mkdir()
    missing = tmp_path / "missing-private.txt"
    for source in (unsupported, directory, missing):
        with pytest.raises(DocumentInputError) as error:
            load_document(source)
        assert source.name not in str(error.value)
        assert str(tmp_path) not in str(error.value)
    valid = tmp_path / "valid.txt"
    valid.write_text("synthetic", encoding="utf-8")
    with pytest.raises(DocumentInputError, match="Duplicate source"):
        load_documents([valid, valid])


@pytest.mark.parametrize("paths", [[], "example.txt", Path("example.txt"), [None], ["x.txt"] * 101])
def test_source_list_is_explicit_bounded_and_typed(paths: object) -> None:
    with pytest.raises(DocumentInputError):
        load_documents(paths)


def test_chunking_rejects_blank_manual_document_without_exposing_content() -> None:
    document = Document("source", " \n ", SourceMetadata(".txt", 4))
    with pytest.raises(DocumentInputError, match="no text"):
        chunk_document(document)


def test_chunking_rejects_invalid_unicode_without_exposing_content() -> None:
    document = Document("source", "private\ud800text", SourceMetadata(".txt", 12))
    with pytest.raises(DocumentInputError) as error:
        chunk_document(document)
    assert "private" not in str(error.value)


def test_document_and_chunk_representations_do_not_expose_text() -> None:
    document = Document("source", "sensitive example text", SourceMetadata(".txt", 22))
    chunk = chunk_document(document)[0]
    assert "sensitive" not in repr(document)
    assert "sensitive" not in str(document)
    assert "sensitive" not in repr(chunk)
    assert "sensitive" not in str(chunk)
