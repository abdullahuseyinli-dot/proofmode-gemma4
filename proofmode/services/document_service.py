from __future__ import annotations

import io
import mimetypes
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from pypdf import PdfReader


@dataclass
class StudyDocument:
    name: str
    mime_type: str
    data: bytes
    text: str = ""

    @property
    def is_image(self) -> bool:
        return self.mime_type.startswith("image/")

    @property
    def is_audio(self) -> bool:
        return self.mime_type.startswith("audio/")


def infer_mime(name: str, provided: str | None = None) -> str:
    return provided or mimetypes.guess_type(name)[0] or "application/octet-stream"


def extract_text(name: str, data: bytes, mime_type: str | None = None) -> str:
    mime_type = infer_mime(name, mime_type)
    suffix = Path(name).suffix.lower()
    if mime_type == "application/pdf" or suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if suffix == ".docx" or mime_type.endswith("wordprocessingml.document"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
        return " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t")).strip()
    if mime_type.startswith("text/") or suffix in {".md", ".csv", ".json", ".py", ".tex"}:
        return data.decode("utf-8", errors="replace").strip()
    return ""


def make_document(name: str, data: bytes, mime_type: str | None = None) -> StudyDocument:
    resolved = infer_mime(name, mime_type)
    return StudyDocument(name=name, mime_type=resolved, data=data, text=extract_text(name, data, resolved))

