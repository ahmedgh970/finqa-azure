"""Mise en forme des chunks retrouvés en sources numérotées et citables."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

from ai103.retriever.core import Chunk


@dataclass(frozen=True)
class Source:
    """Chunk présenté au modèle sous le numéro `number` (à partir de 1)."""

    number: int
    chunk: Chunk

    @property
    def document(self) -> str:
        return PurePosixPath(unquote(urlparse(self.chunk.source).path)).stem

    @property
    def pages(self) -> str:
        start, end = self.chunk.page_start, self.chunk.page_end
        if start is None or end is None:
            return "unknown"
        return str(start) if start == end else f"{start}-{end}"


def number_sources(chunks: Sequence[Chunk]) -> list[Source]:
    # L'ordre du classement est conservé : le passage le plus pertinent porte le numéro 1.
    return [Source(number, chunk) for number, chunk in enumerate(chunks, 1)]


def _attribute(value: object) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _neutralize(content: str) -> str:
    # Un document ne doit pas pouvoir fermer sa balise et se faire passer pour une consigne.
    for tag in ("source", "documents"):
        content = content.replace(f"<{tag}", f"&lt;{tag}").replace(f"</{tag}", f"&lt;/{tag}")
    return content


def build_context(sources: Sequence[Source]) -> str:
    blocks = []
    for source in sources:
        chunk = source.chunk
        attributes = {
            "id": source.number,
            "company": chunk.company or "unknown",
            "fiscal_year": chunk.fiscal_year if chunk.fiscal_year is not None else "unknown",
            "document": source.document,
            "pages": source.pages,
        }
        header = " ".join(f'{name}="{_attribute(value)}"' for name, value in attributes.items())
        blocks.append(f"<source {header}>\n{_neutralize(chunk.content)}\n</source>")
    return "\n\n".join(blocks)
