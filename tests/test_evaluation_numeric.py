import pytest

from ai103.evaluation.numeric import check, gold_number, numbers


def test_gold_numbers_and_text_answers():
    assert gold_number("$1577.00").value == 1577 and gold_number("$1577.00").decimals == 2
    assert gold_number("1.9%").value == 1.9
    assert gold_number("-0.02").value == -0.02
    assert gold_number("No, the company is managing its CAPEX efficiently") is None
    assert gold_number("$8.70") is not None


def test_numbers_in_financial_text():
    text = "Capex was $ (1,577) million, i.e. 24.26x, margin 1.86% in FY2018."
    assert numbers(text) == [1577, 24.26, 1.86]


def test_citations_and_years_never_match():
    assert check("Insufficient evidence [1] for FY2018.", "1").matched is False
    assert check("See FY2018 and FY2019 filings [2].", "$2.02").matched is False


@pytest.mark.parametrize(
    ("response", "gold", "expected"),
    [
        ("3M's FY2018 capex was **USD $1,577 million** [1].", "$1577.00", True),
        ("Capex was $(1,577) million [1].", "$1577.00", True),
        ("Net PP&E was USD 8,738 million at year end [2].", "$8.70", True),
        ("Fixed asset turnover = **24.26x** [5].", "24.26", True),
        ("The 3-year average is 1.86% of revenue [1].", "1.9%", True),
        ("The ratio is 0.019 [3].", "1.9%", True),
        ("Capex was USD 1,373 million [1].", "$1577.00", False),
        ("Insufficient evidence to compute the ratio.", "24.26", False),
    ],
)
def test_check_matches_with_rounding_and_scale(response, gold, expected):
    result = check(response, gold)
    assert result.applicable and result.matched is expected


def test_only_first_paragraph_counts():
    # Le montant gold apparaît seulement dans les détails : la réponse directe est fausse.
    response = "Capex was USD 1,373 million [1].\n\nDetails: 2018 1,577; 2017 1,373."
    assert check(response, "$1577.00").matched is False


def test_text_gold_is_not_applicable():
    assert not check("Yes, dividends are stable [1].", "Yes, they increase every year").applicable
