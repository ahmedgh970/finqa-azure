"""Outils de l'agent : catalogue des rapports, recherche documentaire et calcul sûr.

Les outils sont exécutés par notre code (function calling) : le modèle ne fait que demander
un appel avec des arguments JSON. Chaque résultat est renvoyé au modèle sous forme de texte JSON.
"""

import ast
import json
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai103.rag.context import Source
from ai103.retriever.core import Chunk, Mode, Retriever
from ai103.retriever.filters import FilterExtractor

DOC_TYPE = re.compile(r"(10K|10Q|8K|EARNINGS)")
MAX_RESULTS = 10
MAX_EXPRESSION = 300
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _strict(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    """Définition d'outil en mode strict : tous les champs requis, valeurs nulles explicites."""
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOL_DEFINITIONS = [
    _strict(
        "list_filings",
        "List the filings available in the corpus for a company: document name, type "
        "(10K, 10Q, 8K, EARNINGS) and fiscal year. Use it to know which reports exist.",
        {"company": {"type": "string", "description": "Company name or common short name."}},
    ),
    _strict(
        "search_filings",
        "Search passages of the filings. Write a focused query naming the line item or topic "
        "(for example 'balance sheet property plant and equipment net'). Filter by company and "
        "fiscal years when known. Each passage gets a source number to cite.",
        {
            "query": {"type": "string"},
            "company": {"type": ["string", "null"], "description": "Exact company or null."},
            "fiscal_years": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "description": "Fiscal years of the reports to search, or null for all.",
            },
            "k": {"type": "integer", "description": "Number of passages, 1 to 10."},
        },
    ),
    _strict(
        "calculate",
        "Evaluate an arithmetic expression exactly (+, -, *, /, **, parentheses, round, abs, "
        "min, max). Use it for every ratio, growth rate, margin or average.",
        {"expression": {"type": "string", "description": "For example (8738 + 8866) / 2"}},
    ),
]


@dataclass
class SourceRegistry:
    """Numérote chaque passage une seule fois pour toute la session de l'agent."""

    by_chunk: dict[str, int] = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)

    def register(self, chunk: Chunk) -> Source:
        if chunk.chunk_id not in self.by_chunk:
            self.by_chunk[chunk.chunk_id] = len(self.sources) + 1
            self.sources.append(Source(len(self.sources) + 1, chunk))
        return self.sources[self.by_chunk[chunk.chunk_id] - 1]


@dataclass(frozen=True)
class Filing:
    doc_name: str
    company: str
    fiscal_year: int | None

    @property
    def doc_type(self) -> str:
        match = DOC_TYPE.search(self.doc_name)
        return match.group(1) if match else "OTHER"


def load_filings(path: Path) -> list[Filing]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        Filing(row["doc_name"], row["company"], row["fiscal_year"])
        for row in map(json.loads, filter(None, lines))
    ]


_OPERATORS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {"round": round, "abs": abs, "min": min, "max": max}


def calculate(expression: str) -> float:
    """Arithmétique seulement : l'arbre syntaxique est parcouru, jamais exécuté avec eval()."""
    if len(expression) > MAX_EXPRESSION:
        raise ValueError("Expression trop longue.")

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exposant trop grand.")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](visit(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            return _FUNCTIONS[node.func.id](*(visit(arg) for arg in node.args))
        raise ValueError("Seuls les nombres et opérations arithmétiques sont autorisés.")

    # Séparateurs de milliers retirés (1,577 -> 1577) ; les virgules d'arguments restent.
    return float(visit(ast.parse(THOUSANDS.sub("", expression), mode="eval")))


class ToolBox:
    """Exécute les outils demandés par le modèle et garde les passages pour la vérification."""

    def __init__(
        self,
        retriever: Retriever,
        embed: Callable[[str], list[float]],
        filings: list[Filing],
        *,
        mode: Mode = Mode.HYBRID,
    ) -> None:
        self.retriever = retriever
        self.embed = embed
        self.filings = filings
        self.mode = mode
        self.extractor = FilterExtractor(filing.company for filing in filings)
        self.registry = SourceRegistry()

    def canonical_company(self, name: str | None) -> str | None:
        """Nom tel que stocké dans l'index ; un alias (JnJ, AMEX) est converti, sinon None."""
        if not name:
            return None
        known = {filing.company for filing in self.filings}
        if name in known:
            return name
        return self.extractor.extract(name).company

    def list_filings(self, company: str) -> dict[str, Any]:
        canonical = self.canonical_company(company)
        if canonical is None:
            companies = sorted({filing.company for filing in self.filings})
            return {"error": f"Unknown company '{company}'.", "known_companies": companies}
        return {
            "company": canonical,
            "filings": [
                {"document": f.doc_name, "type": f.doc_type, "fiscal_year": f.fiscal_year}
                for f in self.filings
                if f.company == canonical
            ],
        }

    def search_filings(
        self, query: str, company: str | None, fiscal_years: list[int] | None, k: int
    ) -> dict[str, Any]:
        if not query.strip():
            return {"error": "Empty query."}
        canonical = self.canonical_company(company)
        if company and canonical is None:
            return {"error": f"Unknown company '{company}'. Call list_filings first."}
        clauses = []
        if canonical:
            clauses.append("company eq '{}'".format(canonical.replace("'", "''")))
        if fiscal_years:
            years = " or ".join(f"fiscal_year eq {int(year)}" for year in fiscal_years[:6])
            # Les rapports sans exercice unique (8-K) restent visibles.
            clauses.append(f"({years} or fiscal_year eq null)")
        expression = " and ".join(clauses) or None
        chunks = self.retriever.retrieve(
            query,
            mode=self.mode,
            k=max(1, min(int(k), MAX_RESULTS)),
            vector=self.embed(query) if self.mode is not Mode.LEXICAL else None,
            filter_expression=expression,
        )
        passages = []
        for chunk in chunks:
            source = self.registry.register(chunk)
            passages.append(
                {
                    "source": source.number,
                    "document": source.document,
                    "pages": source.pages,
                    "fiscal_year": chunk.fiscal_year,
                    "text": chunk.content,
                }
            )
        return {"filter": expression, "passages": passages}

    def call(self, name: str, arguments: str) -> str:
        """Une erreur d'outil est renvoyée au modèle comme résultat, pas comme exception."""
        try:
            args = json.loads(arguments)
            if name == "list_filings":
                result: dict[str, Any] = self.list_filings(str(args["company"]))
            elif name == "search_filings":
                result = self.search_filings(
                    str(args["query"]), args.get("company"), args.get("fiscal_years"), args["k"]
                )
            elif name == "calculate":
                result = {"result": calculate(str(args["expression"]))}
            else:
                result = {"error": f"Unknown tool '{name}'."}
        except (ValueError, KeyError, TypeError, ZeroDivisionError, SyntaxError) as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, ensure_ascii=False)
