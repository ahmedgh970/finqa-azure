"""Vérifier par le code ce que le modèle affirme : extraits cités et statut de confiance."""

import html
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from ai103.rag.context import Source
from ai103.rag.prompts import RagAnswer

MIN_QUOTE_CHARS = 8
INLINE = re.compile(r"\[(\d+)\]")
TAG = re.compile(r"<[^>]+>")
TRANSLATION = str.maketrans({"\u2013": "-", "\u2014": "-", "\u2212": "-"})
# On ne compare que les lettres, les chiffres et les symboles qui changent un sens financier.
NOISE = re.compile(r"[^0-9a-z.,%$()/-]")
ELLIPSIS = re.compile(r"\.\.\.|\u2026")


class CheckStatus(StrEnum):
    VERIFIED = "verified"
    QUOTE_NOT_FOUND = "quote_not_found"
    QUOTE_TOO_SHORT = "quote_too_short"
    UNKNOWN_SOURCE = "unknown_source"


class AnswerStatus(StrEnum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BLOCKED_BY_CONTENT_FILTER = "blocked_by_content_filter"


@dataclass(frozen=True)
class CitationCheck:
    source: int
    quote: str
    status: CheckStatus


def compact(text: str) -> str:
    """Forme comparable d'un texte : sans balises, blancs, casse ni caractères parasites.

    Les PDF contiennent des espaces insécables, puces et apostrophes typographiques, que le
    modèle recopie parfois en caractères de contrôle (« \\u2019 » devient « \\x19 »). Seuls
    les lettres, chiffres et symboles financiers comptent : un chiffre différent reste refusé.
    """
    text = html.unescape(TAG.sub(" ", text))
    text = unicodedata.normalize("NFKC", text).translate(TRANSLATION).casefold()
    return NOISE.sub("", text)


def quote_found(quote: str, content: str) -> bool:
    """Un extrait coupé par « … » est accepté si chaque morceau est présent, dans l'ordre."""
    position = 0
    for part in filter(None, (compact(piece) for piece in ELLIPSIS.split(quote))):
        position = content.find(part, position)
        if position < 0:
            return False
        position += len(part)
    return True


def check_citations(answer: RagAnswer, sources: Sequence[Source]) -> list[CitationCheck]:
    by_number = {source.number: compact(source.chunk.content) for source in sources}
    checks = []
    for citation in answer.citations:
        if citation.source not in by_number:
            status = CheckStatus.UNKNOWN_SOURCE
        elif len(compact(citation.quote)) < MIN_QUOTE_CHARS:
            status = CheckStatus.QUOTE_TOO_SHORT
        elif quote_found(citation.quote, by_number[citation.source]):
            status = CheckStatus.VERIFIED
        else:
            status = CheckStatus.QUOTE_NOT_FOUND
        checks.append(CitationCheck(citation.source, citation.quote, status))
    return checks


def answer_status(answer: RagAnswer, checks: Sequence[CitationCheck]) -> AnswerStatus:
    """Statut décidé par le code, jamais par le modèle."""
    if answer.insufficient_evidence:
        return AnswerStatus.INSUFFICIENT_EVIDENCE
    if not checks:
        return AnswerStatus.UNSUPPORTED
    verified = {check.source for check in checks if check.status is CheckStatus.VERIFIED}
    if not verified:
        return AnswerStatus.UNSUPPORTED
    # Un renvoi [n] dans le texte sans extrait vérifié pour n affaiblit la réponse.
    inline = {int(number) for number in INLINE.findall(answer.answer)}
    if inline - verified or any(check.status is not CheckStatus.VERIFIED for check in checks):
        return AnswerStatus.PARTIALLY_VERIFIED
    return AnswerStatus.VERIFIED
