from __future__ import annotations

from proofmode.gemma_client import GemmaClient
from proofmode.services.document_service import extract_text, make_document


def test_text_document_is_extracted_locally() -> None:
    document = make_document("notes.md", b"Gradient descent follows a loss gradient.", "text/markdown")
    assert "Gradient descent" in document.text
    assert not document.is_image


def test_unknown_binary_is_not_misread_as_text() -> None:
    assert extract_text("archive.bin", b"\x00\xff", "application/octet-stream") == ""


def test_multimodal_image_part_is_local_data_url() -> None:
    part = GemmaClient.file_part(b"image bytes", "image/png")
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")

