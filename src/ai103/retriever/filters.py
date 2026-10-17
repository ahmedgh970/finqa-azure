"""Filtres de métadonnées extraits de la seule question, par règles déterministes."""

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# Noms courts et tickers usuels du catalogue, indépendants des réponses du golden set.
# Les tickers de moins de trois lettres sont exclus : trop ambigus dans un texte (BA, KO, VZ).
ALIASES: dict[str, tuple[str, ...]] = {
    "3M": ("MMM",),
    "AES Corporation": ("AES",),
    "AMD": ("Advanced Micro Devices",),
    "Activision Blizzard": ("Activision", "ATVI"),
    "Adobe": ("ADBE",),
    "Amazon": ("AMZN",),
    "Amcor": ("AMCR",),
    "American Express": ("AMEX", "AXP"),
    "American Water Works": ("American Water", "AWK"),
    "Best Buy": ("BBY",),
    "CVS Health": ("CVS",),
    "Corning": ("GLW",),
    "General Mills": ("GIS",),
    "JPMorgan": ("JP Morgan", "J.P. Morgan", "JPM"),
    "Johnson & Johnson": ("JnJ", "J&J", "JNJ"),
    "Kraft Heinz": ("KHC",),
    "Lockheed Martin": ("Lockheed", "LMT"),
    "MGM Resorts": ("MGM",),
    "Microsoft": ("MSFT",),
    "Netflix": ("NFLX",),
    "Nike": ("NKE",),
    "Paypal": ("PYPL",),
    "PepsiCo": ("Pepsi", "PEP"),
    "Pfizer": ("PFE",),
    "Ulta Beauty": ("Ulta",),
    "Walmart": ("Wal-Mart", "WMT"),
}
# Noms qui sont aussi des mots courants : correspondance sensible à la casse.
COMMON_WORDS = frozenset({"Block"})
YEAR = re.compile(r"(?<![\d$.,])((?:19|20)\d{2})(?!\d)")
SHORT_FISCAL_YEAR = re.compile(r"\bFY\s?'?(\d{2})(?!\d)", re.IGNORECASE)


class Strategy(StrEnum):
    NONE = "none"
    COMPANY = "company"
    COMPANY_YEAR = "company_year"
    COMPANY_YEAR_TOLERANT = "company_year_tolerant"


@dataclass(frozen=True)
class QueryFilters:
    company: str | None
    years: tuple[int, ...]
    matched_companies: tuple[str, ...]
    matched_terms: tuple[str, ...]


def _pattern(term: str) -> re.Pattern[str]:
    # Espaces et tirets interchangeables : « Coca Cola » correspond à « Coca-Cola ».
    body = r"[\s-]+".join(re.escape(part) for part in re.split(r"[\s-]+", term))
    sensitive = term in COMMON_WORDS or (term.isupper() and " " not in term)
    flags = 0 if sensitive else re.IGNORECASE
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", flags)


class FilterExtractor:
    def __init__(self, companies: Iterable[str], *, strict: bool = False) -> None:
        catalog = set(companies)
        if not catalog:
            raise ValueError("Catalogue d'entreprises vide.")
        # strict : chaque alias doit viser une entreprise du corpus (détecte les fautes de frappe).
        if strict and not set(ALIASES) <= catalog:
            raise ValueError("Alias d'une entreprise absente du corpus.")
        self.terms = [
            (company, term, _pattern(term))
            for company in sorted(catalog)
            for term in (company, *ALIASES.get(company, ()))
        ]

    @classmethod
    def from_metadata(cls, path: Path) -> "FilterExtractor":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return cls((row["company"] for row in rows), strict=True)

    def extract(self, question: str) -> QueryFilters:
        hits = [
            (company, term) for company, term, pattern in self.terms if pattern.search(question)
        ]
        companies = tuple(sorted({company for company, _ in hits}))
        years = {int(year) for year in YEAR.findall(question)}
        years |= {2000 + int(year) for year in SHORT_FISCAL_YEAR.findall(question)}
        return QueryFilters(
            # Plusieurs entreprises détectées : pas de filtre plutôt qu'un mauvais choix.
            company=companies[0] if len(companies) == 1 else None,
            years=tuple(sorted(years)),
            matched_companies=companies,
            matched_terms=tuple(sorted({term for _, term in hits})),
        )


def filter_years(filters: QueryFilters, strategy: Strategy) -> tuple[int, ...]:
    """Exercices admis ; vide signifie aucune contrainte d'année."""
    strategy = Strategy(strategy)
    if strategy is Strategy.COMPANY_YEAR:
        return filters.years
    if strategy is Strategy.COMPANY_YEAR_TOLERANT:
        # Un rapport de l'exercice N contient les comparatifs N-1 et les perspectives N+1.
        return tuple(sorted({y + delta for y in filters.years for delta in (-1, 0, 1)}))
    return ()


def odata_filter(filters: QueryFilters, strategy: Strategy) -> str | None:
    strategy = Strategy(strategy)
    if strategy is Strategy.NONE or filters.company is None:
        return None
    clause = "company eq '{}'".format(filters.company.replace("'", "''"))
    if years := filter_years(filters, strategy):
        # Les documents sans exercice unique (8-K) restent admissibles.
        alternatives = " or ".join(f"fiscal_year eq {year}" for year in years)
        clause += f" and ({alternatives} or fiscal_year eq null)"
    return clause


def question_filter(extractor: FilterExtractor, strategy: Strategy) -> Callable[[str], str | None]:
    """Fonction question -> filtre OData, à brancher sur le retriever."""
    strategy = Strategy(strategy)

    def build(question: str) -> str | None:
        return odata_filter(extractor.extract(question), strategy)

    return build
