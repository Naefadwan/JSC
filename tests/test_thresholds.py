"""Tests for the ownership and customer threshold checks."""

from jsc_review.documents.cer_measurement import MeasurementStore, PageMeasurement
from jsc_review.review.findings import Requirement, review_document


def _requirement(requirement_id: str) -> Requirement:
    return Requirement(
        requirement_id=requirement_id,
        section="بند 4",
        text_ar="كبار مالكي الأسهم",
        text_en="Major shareholders",
        evaluation_method="deterministic",
    )


def _measurement_store(
    document_id: str, page: int, *, cer: float, engine: str = "indra"
) -> MeasurementStore:
    return MeasurementStore(
        threshold=0.05,
        measurements=(
            PageMeasurement(
                document_id=document_id,
                page=page,
                engine=engine,
                cer=cer,
                wer=None,
                reference_chars=100,
                hypothesis_chars=100,
                verified_evidence_eligible=cer <= 0.05,
                measured_at="2026-10-25T00:00:00+00:00",
            ),
        ),
    )


def test_major_holder_at_or_above_five_percent_is_met() -> None:
    page = "كبار المساهمين نسبة الملكية %28 عدد الأسهم 23,294,614"

    finding = review_document(
        "report",
        _requirement("DIS-010"),
        pages={4: page},
        ocr_pages={},
    )

    assert finding.result.status == "Met"
    assert finding.result.human_review == "Not Required"
    assert finding.result.evidence is not None
    assert finding.result.evidence.verified is True
    assert finding.result.evidence.page == 4
    assert "5%" in finding.result.reason or "28%" in finding.result.reason


def test_share_count_is_required_for_the_holder_count_rule() -> None:
    page = "كبار المساهمين نسبة الملكية %28"

    named = review_document("report", _requirement("DIS-010"), {4: page}, {})
    counted = review_document("report", _requirement("DIS-011"), {4: page}, {})

    assert named.result.status == "Met"
    assert counted.result.status == "Needs Review"
    assert counted.result.evidence is not None
    assert "عدد الأسهم" in counted.result.reason


def test_comma_decimal_and_grouped_shares_are_read() -> None:
    page = "كبار مالكي الأسهم %49,11 22,304,872 %7,68 3,490,166"

    finding = review_document("report", _requirement("DIS-011"), {2: page}, {})

    assert finding.result.status == "Met"
    assert "49.11%" in finding.result.reason


def test_holding_below_five_percent_stays_needs_review() -> None:
    page = "كبار المساهمين نسبة الملكية %4 1,200,000"

    finding = review_document("report", _requirement("DIS-010"), {3: page}, {})

    assert finding.result.status == "Needs Review"
    assert finding.result.evidence is not None


def test_owner_disclosure_found_only_by_ocr_cannot_be_met() -> None:
    finding = review_document(
        "report",
        _requirement("DIS-010"),
        pages={2: ""},
        ocr_pages={2: "كبار المساهمين %28 23,294,614"},
    )

    assert finding.result.status == "Needs Review"
    assert finding.result.evidence is not None
    assert finding.result.evidence.verified is False


def test_owner_disclosure_from_ocr_is_promoted_when_cer_is_measured_low() -> None:
    store = _measurement_store("report", 2, cer=0.03)

    finding = review_document(
        "report",
        _requirement("DIS-010"),
        pages={2: ""},
        ocr_pages={2: "كبار المساهمين %28 23,294,614"},
        cer_store=store,
    )

    assert finding.result.status == "Met"
    assert finding.result.evidence is not None
    assert finding.result.evidence.verified is True
    assert "3.00" in finding.result.reason or "3,00" in finding.result.reason


def test_owner_disclosure_from_ocr_stays_review_when_cer_too_high() -> None:
    store = _measurement_store("report", 2, cer=0.4)

    finding = review_document(
        "report",
        _requirement("DIS-010"),
        pages={2: ""},
        ocr_pages={2: "كبار المساهمين %28 23,294,614"},
        cer_store=store,
    )

    assert finding.result.status == "Needs Review"
    assert finding.result.evidence is not None
    assert finding.result.evidence.verified is False


def test_measurement_for_a_different_page_does_not_promote_ocr() -> None:
    store = _measurement_store("report", 99, cer=0.01)

    finding = review_document(
        "report",
        _requirement("DIS-010"),
        pages={2: ""},
        ocr_pages={2: "كبار المساهمين %28 23,294,614"},
        cer_store=store,
    )

    assert finding.result.status == "Needs Review"


def test_customer_dependence_header_is_met() -> None:
    # DIS-017 is now heading-based. Any recognizable customer- or supplier-
    # concentration disclosure counts as Met regardless of the exact
    # percentage, which rescues the common "no customers above the 10%
    # threshold" wording that the old percentage-only path missed.
    page = "درجة الاعتماد على موردين محددين تبلغ 12% من إجمالي المشتريات."

    finding = review_document("report", _requirement("DIS-017"), {6: page}, {})

    assert finding.result.status == "Met"
    assert finding.result.evidence is not None
    assert finding.result.evidence.verified is True
    # The reason now cites the matched heading variant rather than the
    # percentage; the evidence excerpt still shows the surrounding text.
    assert "درجة الاعتماد على موردين" in finding.result.reason


def test_customer_dependence_without_disclosure_phrase_stays_needs_review() -> None:
    # A page that mentions suppliers in passing (without the disclosure
    # heading or the "no major concentration" phrasing) still falls to
    # the human reviewer.
    page = "تعاونت الشركة مع عدة موردين خلال العام الماضي بشكل عام."

    finding = review_document("report", _requirement("DIS-017"), {6: page}, {})

    assert finding.result.status == "Needs Review"


def test_share_count_on_the_line_above_a_percentage_is_read() -> None:
    page = "يملكون ما نسبته\n132,010,000\n%66.005"

    finding = review_document("report", _requirement("DIS-011"), {75: page}, {})

    assert finding.result.status == "Met"
    page = "يملكون ما نسبته\n132,010,000\n%66.005"

    finding = review_document("report", _requirement("DIS-010"), {75: page}, {})

    assert finding.result.status == "Met"
    assert "66.005%" in finding.result.reason


def test_customer_percentage_is_not_taken_from_the_shareholder_table() -> None:
    page = "كبار مالكي الأسهم %28 23,294,614\nالاعتماد على الموردين يبلغ 4% من المشتريات."

    owner = review_document("report", _requirement("DIS-010"), {1: page}, {})
    customer = review_document("report", _requirement("DIS-017"), {1: page}, {})

    assert owner.result.status == "Met"
    assert customer.result.status == "Needs Review"


def test_shareholding_percentage_wording_is_recognized() -> None:
    page = "أسماء المالكين ونسبة مساهمتهم في رأس المال %10 30,060,000"

    named = review_document("report", _requirement("DIS-010"), {74: page}, {})
    counted = review_document("report", _requirement("DIS-011"), {74: page}, {})

    assert named.result.status == "Met"
    assert counted.result.status == "Met"
    assert named.result.evidence is not None
    assert named.result.evidence.page == 74


def test_missing_threshold_disclosure_has_no_evidence() -> None:
    finding = review_document(
        "report",
        _requirement("DIS-017"),
        pages={1: "بيان الأنشطة الرئيسية للشركة."},
        ocr_pages={},
    )

    assert finding.result.status == "Needs Review"
    assert finding.result.evidence is None
