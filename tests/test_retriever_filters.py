import pytest

from ai103.retriever.filter_evaluation import admissible, evaluate
from ai103.retriever.filters import (
    FilterExtractor,
    QueryFilters,
    Strategy,
    filter_years,
    odata_filter,
    question_filter,
)

CATALOG = [
    "3M",
    "Block",
    "Coca-Cola",
    "Johnson & Johnson",
    "JPMorgan",
    "PepsiCo",
    "AMD",
    "MGM Resorts",
]


def extractor():
    return FilterExtractor(CATALOG + sorted({"Ulta Beauty", "American Express", "Pfizer"}))


@pytest.mark.parametrize(
    ("question", "company"),
    [
        ("What is 3M's FY2018 capex?", "3M"),
        ("What is the FY2017 ROA for Coca Cola?", "Coca-Cola"),
        ("Are JnJ's FY2022 financials high growth?", "Johnson & Johnson"),
        ("Which of JPM's segments had the lowest revenue?", "JPMorgan"),
        ("Does AMEX have an improving margin?", "American Express"),
        ("Which region was best for MGM?", "MGM Resorts"),
        ("What is PepsiCo's dividend?", "PepsiCo"),
        ("What did Block report for Cash App?", "Block"),
    ],
)
def test_company_names_and_aliases(question, company):
    assert extractor().extract(question).company == company


def test_common_words_and_tickers_are_case_sensitive():
    filters = extractor().extract("Which block of shares did the amd board approve?")
    assert filters.company is None
    assert filters.matched_companies == ()


def test_partial_words_do_not_match():
    # « Pepsi » ne doit pas être détecté à l'intérieur d'un autre mot.
    assert extractor().extract("Is Pepsiology a word?").company is None


def test_several_companies_disable_company_filter():
    filters = extractor().extract("Compare 3M and PepsiCo in FY2022.")
    assert filters.company is None
    assert filters.matched_companies == ("3M", "PepsiCo")
    assert odata_filter(filters, Strategy.COMPANY) is None


def test_years_long_and_short_forms_without_amounts():
    filters = extractor().extract("3M FY2022 vs FY 2021 vs FY'20, revenue $2019 million, 12,2018")
    assert filters.years == (2020, 2021, 2022)


def test_odata_strategies_and_quote_escaping():
    filters = QueryFilters("McDonald's", (2021, 2022), ("McDonald's",), ("McDonald's",))
    assert odata_filter(filters, Strategy.NONE) is None
    assert odata_filter(filters, Strategy.COMPANY) == "company eq 'McDonald''s'"
    assert odata_filter(filters, Strategy.COMPANY_YEAR) == (
        "company eq 'McDonald''s' and "
        "(fiscal_year eq 2021 or fiscal_year eq 2022 or fiscal_year eq null)"
    )
    without_year = QueryFilters("3M", (), ("3M",), ("3M",))
    assert odata_filter(without_year, Strategy.COMPANY_YEAR) == "company eq '3M'"


def test_strict_catalog_rejects_alias_for_unknown_company():
    with pytest.raises(ValueError):
        FilterExtractor(["3M"], strict=True)
    with pytest.raises(ValueError):
        FilterExtractor([])


def test_admissibility_keeps_null_fiscal_years():
    filters = QueryFilters("3M", (2023,), ("3M",), ("3M",))
    assert admissible(Strategy.COMPANY_YEAR, filters, {"company": "3M", "fiscal_year": None})
    assert not admissible(Strategy.COMPANY_YEAR, filters, {"company": "3M", "fiscal_year": 2022})
    assert admissible(Strategy.COMPANY, filters, {"company": "3M", "fiscal_year": 2022})
    assert not admissible(Strategy.COMPANY, filters, {"company": "AMD", "fiscal_year": 2023})


def test_evaluation_counts_outcomes_and_exclusions():
    questions = [
        {"financebench_id": "q1", "question": "3M capex FY2023?", "doc_name": "3M_2022"},
        {"financebench_id": "q2", "question": "Who voted against?", "doc_name": "3M_2022"},
    ]
    metadata = {"3M_2022": {"doc_name": "3M_2022", "company": "3M", "fiscal_year": 2022}}
    result = evaluate(questions, metadata, extractor())
    assert result["company_outcomes"] == {"correct": 1, "none": 1}
    assert result["filtered_questions"] == {
        "none": 0,
        "company": 1,
        "company_year": 1,
        "company_year_tolerant": 1,
    }
    assert result["gold_document_excluded"] == {
        "none": 0,
        "company": 0,
        "company_year": 1,
        "company_year_tolerant": 0,
    }


def test_tolerant_strategy_widens_years_by_one():
    filters = QueryFilters("3M", (2022, 2023), ("3M",), ("3M",))
    assert filter_years(filters, Strategy.COMPANY_YEAR_TOLERANT) == (2021, 2022, 2023, 2024)
    assert filter_years(filters, Strategy.COMPANY) == ()
    assert odata_filter(filters, Strategy.COMPANY_YEAR_TOLERANT).count("fiscal_year eq") == 5
    forecast = QueryFilters("3M", (2023,), ("3M",), ("3M",))
    document = {"company": "3M", "fiscal_year": 2022}
    assert not admissible(Strategy.COMPANY_YEAR, forecast, document)
    assert admissible(Strategy.COMPANY_YEAR_TOLERANT, forecast, document)


def test_question_filter_builds_odata_from_text():
    build = question_filter(extractor(), Strategy.COMPANY)
    assert build("JnJ inventory turnover FY2022?") == "company eq 'Johnson & Johnson'"
    assert build("Who voted against the nominees?") is None
