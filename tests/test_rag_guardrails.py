import pytest

from ai103.rag.context import number_sources
from ai103.rag.guardrails import AnswerStatus, CheckStatus, answer_status, check_citations, compact
from ai103.rag.prompts import Citation, RagAnswer
from ai103.retriever.core import Chunk

NATIVE = "Purchases of property, plant and equipment (PP&E)  $ (1,577)  $ (1,373)"
LAYOUT = "<table><tr><td>Purchases of PP&amp;E</td><td>(1,577)</td></tr></table>"


def sources():
    return number_sources(
        [
            Chunk("n", NATIVE, "https://s/c/3M.pdf", 1.0),
            Chunk("l", LAYOUT, "https://s/c/3M.pdf", 0.9),
        ]
    )


def answer(citations, text="Capex was 1,577 [1].", insufficient=False):
    return RagAnswer(
        answer=text,
        citations=[Citation(source=n, quote=q) for n, q in citations],
        insufficient_evidence=insufficient,
    )


def statuses(citations):
    return [c.status for c in check_citations(answer(citations), sources())]


def test_quote_matching_ignores_spacing_case_html_and_typography():
    assert compact("Purchases  of\nPP&E") == compact("purchases of pp&e")
    assert compact("“PP&E” – 2018") == compact('"PP&E" - 2018')
    assert statuses([(1, "plant and equipment (PP&E) $ (1,577)")]) == [CheckStatus.VERIFIED]
    # Le tableau HTML Layout est comparé sans ses balises ni entités.
    assert statuses([(2, "Purchases of PP&E (1,577)")]) == [CheckStatus.VERIFIED]


@pytest.mark.parametrize(
    ("citation", "expected"),
    [
        ((1, "Purchases of PP&E $ (1,578)"), CheckStatus.QUOTE_NOT_FOUND),
        ((1, "(1,577)"), CheckStatus.QUOTE_TOO_SHORT),
        ((9, "Purchases of property"), CheckStatus.UNKNOWN_SOURCE),
        ((2, "plant and equipment (PP&E)"), CheckStatus.QUOTE_NOT_FOUND),
    ],
)
def test_wrong_short_unknown_or_misattributed_quotes(citation, expected):
    assert statuses([citation]) == [expected]


def status_of(citations, **kwargs):
    a = answer(citations, **kwargs)
    return answer_status(a, check_citations(a, sources()))


def test_status_is_decided_by_code():
    good = (1, "Purchases of property, plant")
    bad = (2, "Revenue was 10 billion")
    assert status_of([good]) is AnswerStatus.VERIFIED
    assert status_of([good, bad], text="1,577 [1] and [2]") is AnswerStatus.PARTIALLY_VERIFIED
    assert status_of([bad], text="[2]") is AnswerStatus.UNSUPPORTED
    assert status_of([], text="Capex was 1,577.") is AnswerStatus.UNSUPPORTED
    # Un renvoi [2] dans le texte sans extrait vérifié pour la source 2.
    assert status_of([good], text="1,577 [1], see also [2]") is AnswerStatus.PARTIALLY_VERIFIED
    assert status_of([], insufficient=True) is AnswerStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize(
    ("chunk", "quote"),
    [
        # Cas réels de C1 : le modèle recopie l'espace insécable en \x00, l'apostrophe en \x19.
        ("Item\xa08.01 Other Events.", "Item\x008.01 Other Events."),
        ("Net cash provided by operating activities 3,565\xa0 3,521", "activities 3,565\x00 3,521"),
        ("the Former Issuer’s 3.625% Notes", "the Former Issuer\x19s 3.625% Notes"),
    ],
)
def test_pdf_and_model_character_artifacts_are_ignored(chunk, quote):
    sources = number_sources([Chunk("x", chunk, "https://s/c/X.pdf", 1.0)])
    assert check_citations(answer([(1, quote)]), sources)[0].status is CheckStatus.VERIFIED


def test_ellipsis_pieces_must_appear_in_order():
    chunk = "In re Opiate Litigation is pending in Ohio and includes hundreds of cases."
    sources = number_sources([Chunk("x", chunk, "https://s/c/X.pdf", 1.0)])

    def status(quote):
        return check_citations(answer([(1, quote)]), sources)[0].status

    assert status("In re Opiate Litigation... includes hundreds of cases") is CheckStatus.VERIFIED
    assert status("includes hundreds of cases… In re Opiate Litigation") is (
        CheckStatus.QUOTE_NOT_FOUND
    )


def test_digits_and_reformatted_tables_are_still_refused():
    chunk = "Adjusted EPS (Diluted) $10.70 – $10.80 $10.75 5.5% – 6.5%"
    sources = number_sources([Chunk("x", chunk, "https://s/c/X.pdf", 1.0)])
    assert (
        check_citations(answer([(1, "Adjusted EPS (Diluted) $10.70 - $10.80")]), sources)[0].status
        is CheckStatus.VERIFIED
    )
    for quote in ("Adjusted EPS (Diluted) $10.71", "Adjusted EPS (Diluted) / $10.70 / $10.80"):
        result = check_citations(answer([(1, quote)]), sources)[0].status
        assert result is CheckStatus.QUOTE_NOT_FOUND


def test_artifact_adding_a_digit_stays_refused():
    # Limite acceptée : la puce recopiée « \x199 » ajoute un chiffre ; ignorer des chiffres
    # rendrait le contrôle financier dangereux, donc ce faux négatif est conservé.
    chunk = "● $54 million increase due to new stores"
    sources = number_sources([Chunk("x", chunk, "https://s/c/X.pdf", 1.0)])
    quote = "\x199 $54 million increase due to new"
    assert check_citations(answer([(1, quote)]), sources)[0].status is CheckStatus.QUOTE_NOT_FOUND
