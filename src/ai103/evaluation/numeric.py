"""Contrôle déterministe des réponses chiffrées, sans modèle de langage.

La réponse gold FinanceBench est un nombre (`$1577.00`, `1.9%`, `24.26`) pour 52 questions.
On cherche ce nombre dans le premier paragraphe de la réponse, où la consigne impose de
répondre directement, avec une tolérance d'arrondi et des changements d'échelle usuels.
"""

import re
from dataclasses import dataclass

GOLD = re.compile(r"^\s*\$?\s*(-?[\d,]*\.?\d+)\s*%?\s*$")
NUMBER = re.compile(r"\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?")
CITATION = re.compile(r"\[\d+\]")
YEAR = re.compile(r"^(19|20)\d{2}$")
# Millions/milliards, pourcentage/fraction : écarts d'unité fréquents mais non fautifs.
SCALES = (1, 1_000, 1 / 1_000, 100, 1 / 100)


@dataclass(frozen=True)
class GoldNumber:
    value: float
    decimals: int

    @property
    def tolerance(self) -> float:
        # Demi-unité de la dernière décimale gold, et au moins 1 % de la valeur.
        return max(0.5 * 10.0**-self.decimals, 0.01 * abs(self.value))


@dataclass(frozen=True)
class NumericCheck:
    applicable: bool
    matched: bool | None = None
    matched_value: float | None = None


def gold_number(answer: str) -> GoldNumber | None:
    match = GOLD.match(answer)
    if not match:
        return None
    text = match.group(1).replace(",", "")
    decimals = len(text.split(".")[1]) if "." in text else 0
    return GoldNumber(float(text), decimals)


def numbers(text: str) -> list[float]:
    """Montants du texte, hors renvois [n] et années (FY2018) : sources de faux positifs."""
    values = []
    for token in NUMBER.findall(CITATION.sub(" ", text)):
        cleaned = token.strip("()$ ").replace(",", "").replace("$", "").replace(" ", "")
        if YEAR.match(cleaned):
            continue
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return values


def first_paragraph(text: str) -> str:
    return re.split(r"\n\s*\n", text.strip(), maxsplit=1)[0]


def check(response: str, gold_answer: str) -> NumericCheck:
    """Le signe est ignoré : une dépense peut être écrite (1,577) ou 1,577 selon la source."""
    gold = gold_number(gold_answer)
    if gold is None:
        return NumericCheck(applicable=False)
    for value in numbers(first_paragraph(response)):
        for scale in SCALES:
            if abs(abs(value * scale) - abs(gold.value)) <= gold.tolerance:
                return NumericCheck(True, True, value)
    return NumericCheck(True, False)
