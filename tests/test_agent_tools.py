import json
from unittest.mock import Mock

import pytest

from ai103.agent.tools import TOOL_DEFINITIONS, Filing, ToolBox, calculate
from ai103.retriever.core import Chunk, Mode

FILINGS = [
    Filing("3M_2018_10K", "3M", 2018),
    Filing("3M_2023Q2_10Q", "3M", 2023),
    Filing("JOHNSON_JOHNSON_2022Q4_EARNINGS", "Johnson & Johnson", 2022),
    Filing("AMCOR_2022_8K_dated-2022-07-01", "Amcor", None),
]


def chunk(i, text="Property, plant and equipment, net 8,738"):
    return Chunk(f"c{i}", text, "https://s/financial-pdfs/3M_2018_10K.pdf", 1.0, "3M", 2018, 50, 50)


def toolbox():
    retriever = Mock()
    retriever.retrieve.return_value = [chunk(1), chunk(2)]
    return ToolBox(retriever, lambda q: [0.1] * 1024, FILINGS), retriever


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(8,738 + 8,866) / 2", 8802.0),
        ("round(1577 / 32765 * 100, 2)", 4.81),
        ("max(1, 2) - min(3, 4)", -1.0),
        ("-(1,373) + 2 ** 3", -1365.0),
    ],
)
def test_calculate_arithmetic(expression, expected):
    assert calculate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('x')",
        "a + 1",
        "2 ** 1000",
        "1 if 1 else 2",
        "x" * 400,
    ],
)
def test_calculate_refuses_anything_but_arithmetic(expression):
    with pytest.raises(ValueError):
        calculate(expression)


def test_strict_tool_definitions_require_every_property():
    for tool in TOOL_DEFINITIONS:
        parameters = tool["parameters"]
        assert tool["strict"] and parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])


def test_list_filings_resolves_aliases_and_unknown_companies():
    tools, _ = toolbox()
    listed = tools.list_filings("JnJ")
    assert listed["company"] == "Johnson & Johnson"
    assert listed["filings"][0]["type"] == "EARNINGS"
    unknown = tools.list_filings("Tesla")
    assert "error" in unknown and "3M" in unknown["known_companies"]
    assert tools.list_filings("Amcor")["filings"][0]["fiscal_year"] is None


def test_search_builds_filter_and_numbers_sources_once():
    tools, retriever = toolbox()
    first = tools.search_filings("net PP&E", "3M", [2018], 50)
    kwargs = retriever.retrieve.call_args.kwargs
    assert kwargs["mode"] is Mode.SEMANTIC_HYBRID and kwargs["k"] == 10 and kwargs["pool"] == 20
    assert first["filter"] == ("company eq '3M' and (fiscal_year eq 2018 or fiscal_year eq null)")
    assert [p["source"] for p in first["passages"]] == [1, 2]
    # Le même chunk retrouvé par une autre recherche garde son numéro.
    second = tools.search_filings("capex", None, None, 3)
    assert [p["source"] for p in second["passages"]] == [1, 2]
    assert second["filter"] is None and len(tools.registry.sources) == 2


def test_tool_errors_are_returned_to_the_model():
    tools, _ = toolbox()
    assert "error" in json.loads(tools.call("search_filings", "{not json"))
    assert "error" in json.loads(tools.call("delete_index", "{}"))
    assert "error" in json.loads(tools.call("calculate", '{"expression": "1 / 0"}'))
    assert "Unknown company" in tools.call(
        "search_filings", '{"query": "x", "company": "Tesla", "fiscal_years": null, "k": 3}'
    )
    assert json.loads(tools.call("calculate", '{"expression": "2 * 21"}')) == {"result": 42.0}
