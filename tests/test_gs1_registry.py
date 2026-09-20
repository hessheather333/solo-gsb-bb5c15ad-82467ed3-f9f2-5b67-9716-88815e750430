"""Tests for the declarative, immutable GS1 AI registry and recogniser kernel.

These tests exercise ``app.gs1`` directly (no HTTP):

* the registry is built once at startup from declarative exact/family rows
  and rejects ambiguous or incomplete tables;
* :func:`recognize_key` classifies whole identifier keys for both carriers;
* the element-string carrier and the Digital Link carrier agree on AI
  recognition, value validation and error codes.
"""

import pytest

from app.gs1 import (
    AI_SPECS,
    AIKeyStatus,
    AISpec,
    DIGITS,
    ExactAIDecl,
    FamilyAIDecl,
    REGISTRY,
    RegistryDefinitionError,
    ValueField,
    _check_date,
    recognize_key,
    build_registry,
    parse,
)

GS = "\x1d"
GTIN_OK = "09501101530003"
DL_BASE = "https://id.gs1.org"


# ------------------------------------------------------------------ registry


def test_registry_contains_exact_and_family_concrete_ais():
    assert set(REGISTRY.exact) == {"01", "10", "17", "21"}
    assert set(REGISTRY.families) == {"310", "320"}
    # 4 exact AIs + 10 precision members per family (3100..3109, 3200..3209)
    assert len(REGISTRY.specs) == 4 + 2 * 10
    for prefix in ("310", "320"):
        for n in range(10):
            spec = REGISTRY.resolve(f"{prefix}{n}")
            assert isinstance(spec, AISpec)
            assert spec.ai_length == 4
            assert spec.decimals == n
            assert spec.fixed_length is True
            assert spec.max_length == 6
            assert spec.charset is DIGITS
            assert spec.family is REGISTRY.family_at(prefix)
            assert spec.unit == ("kg" if prefix == "310" else "lb")


def test_registry_specs_are_complete_for_every_ai():
    for ai, spec in REGISTRY.specs.items():
        assert spec.ai == ai
        assert spec.name
        assert spec.charset_rule
        assert spec.min_length >= 1
        assert spec.max_length >= spec.min_length
        assert callable(spec.semantics)
        assert isinstance(spec.charset, frozenset) and spec.charset
        assert spec.ai_length == len(ai)


def test_registry_and_legacy_aliases_are_immutable():
    import dataclasses

    with pytest.raises(TypeError):
        REGISTRY.specs["99"] = None  # type: ignore[index]
    with pytest.raises(TypeError):
        REGISTRY.families["310"] = None  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        REGISTRY.specs["01"].name = "X"  # type: ignore[misc]
    with pytest.raises(TypeError):
        AI_SPECS["99"] = None  # type: ignore[index]


def test_specs_are_frozen_dataclasses():
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        REGISTRY.resolve("3103").decimals = 9  # type: ignore[misc]


# ------------------------------------------------------- startup validation


def _exact(ai="99", name="X", **field_kwargs):
    field_kwargs.setdefault("min_length", 1)
    field_kwargs.setdefault("max_length", 1)
    field_kwargs.setdefault("fixed_length", True)
    field_kwargs.setdefault("charset", DIGITS)
    field_kwargs.setdefault("charset_rule", "digits 0-9")
    return ExactAIDecl(ai=ai, name=name, field=ValueField(**field_kwargs))


def _family(prefix="990", name="F", unit="kg"):
    return FamilyAIDecl(
        prefix=prefix,
        name=name,
        field=ValueField(6, 6, True, DIGITS, "digits 0-9"),
        unit=unit,
    )


def test_build_rejects_non_digit_exact_ai():
    with pytest.raises(RegistryDefinitionError):
        build_registry((_exact(ai="9A"),), ())


def test_build_rejects_duplicate_exact_ai():
    with pytest.raises(RegistryDefinitionError):
        build_registry((_exact(ai="01"), _exact(ai="01")), ())


def test_build_rejects_duplicate_family_prefix():
    with pytest.raises(RegistryDefinitionError):
        build_registry((), (_family(prefix="310"), _family(prefix="310")))


def test_build_rejects_family_prefix_shadowing_another_family():
    with pytest.raises(RegistryDefinitionError):
        build_registry((), (_family(prefix="31"), _family(prefix="310")))


def test_build_rejects_family_prefix_colliding_with_exact_ai():
    with pytest.raises(RegistryDefinitionError):
        build_registry((_exact(ai="310"),), (_family(prefix="310"),))


def test_build_rejects_bad_length_bounds():
    with pytest.raises(RegistryDefinitionError):
        build_registry((_exact(min_length=5, max_length=2),), ())


def test_build_rejects_fixed_length_with_unequal_bounds():
    with pytest.raises(RegistryDefinitionError):
        build_registry((_exact(min_length=1, max_length=3, fixed_length=True),), ())


def test_build_rejects_empty_charset():
    with pytest.raises(RegistryDefinitionError):
        build_registry((_exact(charset=frozenset()),), ())


def test_build_rejects_non_callable_semantics():
    decl = ExactAIDecl(
        ai="99",
        name="X",
        field=ValueField(1, 1, True, DIGITS, "digits 0-9"),
        semantics="not callable",  # type: ignore[arg-type]
    )
    with pytest.raises(RegistryDefinitionError):
        build_registry((decl,), ())


def test_build_rejects_family_with_broken_semantics_factory():
    bad_family = FamilyAIDecl(
        prefix="990",
        name="F",
        field=ValueField(6, 6, True, DIGITS, "digits 0-9"),
        semantics_factory=lambda fam, n: (_ for _ in ()).throw(RuntimeError("boom")),
        unit="kg",
    )
    with pytest.raises(RegistryDefinitionError):
        build_registry((), (bad_family,))


def test_build_rejects_family_with_non_callable_member_semantics():
    bad_family = FamilyAIDecl(
        prefix="990",
        name="F",
        field=ValueField(6, 6, True, DIGITS, "digits 0-9"),
        semantics_factory=lambda fam, n: "not callable",
        unit="kg",
    )
    with pytest.raises(RegistryDefinitionError):
        build_registry((), (bad_family,))


def test_build_rejects_missing_name():
    with pytest.raises(RegistryDefinitionError):
        build_registry((_exact(name=""),), ())


def test_extra_family_with_distinct_prefix_builds():
    registry = build_registry(
        (),
        (
            _family(prefix="990", unit="kg"),
            _family(prefix="980", unit="lb"),
        ),
    )
    assert registry.resolve("9907").unit == "kg"
    assert registry.resolve("9807").unit == "lb"
    assert registry.resolve("9907").decimals == 7


# ---------------------------------------------------------------- recogniser


def test_recognize_exact_ais():
    for ai in ("01", "10", "17", "21"):
        result = recognize_key(ai)
        assert result.status is AIKeyStatus.MATCHED
        assert result.spec.ai == ai


def test_recognize_family_members_all_precisions():
    for prefix in ("310", "320"):
        for n in range(10):
            result = recognize_key(f"{prefix}{n}")
            assert result.status is AIKeyStatus.MATCHED
            assert result.spec.decimals == n


def test_recognize_missing_parameter():
    result = recognize_key("310")
    assert result.status is AIKeyStatus.MISSING_PARAMETER
    assert result.family.prefix == "310"
    assert recognize_key("320").status is AIKeyStatus.MISSING_PARAMETER


def test_recognize_bad_parameter():
    result = recognize_key("310A")
    assert result.status is AIKeyStatus.BAD_PARAMETER
    assert result.family.prefix == "310"
    assert result.parameter == "A"
    assert recognize_key("320-").status is AIKeyStatus.BAD_PARAMETER


def test_recognize_unknown_numeric_ai():
    for key in ("99", "31", "00", "999", "1234", "3110"):
        assert recognize_key(key).status is AIKeyStatus.UNSUPPORTED, key


def test_recognize_non_numeric_key():
    for key in ("", "gtin", "31A", "1A", "abc"):
        assert recognize_key(key).status is AIKeyStatus.INVALID_KEY, key


# ------------------------------------------------------- dual-carrier parity


def _codes(elements):
    return [(e.ai, e.name, e.value, e.unit, e.decimal_value) for e in elements]


def test_element_string_and_digital_link_agree_on_happy_path():
    element_string = f"01{GTIN_OK}1725013110ABC123{GS}21SN0001"
    digital_link = f"{DL_BASE}/01/{GTIN_OK}/17/250131?10=ABC123&21=SN0001"
    assert _codes(parse(element_string)) == _codes(parse(digital_link))


def test_element_string_and_digital_link_agree_on_weights():
    # Weight fields self-delimit by length: no separator is allowed (or
    # needed) between the 6-digit fixed values.
    element_string = f"01{GTIN_OK}31030025133202002513"
    digital_link = f"{DL_BASE}/01/{GTIN_OK}/3103/002513?3202=002513"
    es = _codes(parse(element_string))
    dl = _codes(parse(digital_link))
    assert es == dl
    weight_es = {(ai, unit, value) for ai, name, value, unit, dv in es if unit}
    weight_dl = {(ai, unit, value) for ai, name, value, unit, dv in dl if unit}
    assert weight_es == {("3103", "kg", "002513"), ("3202", "lb", "002513")}
    assert weight_es == weight_dl


@pytest.mark.parametrize(
    "element_string,digital_link",
    [
        # bad GTIN check digit
        (
            "0109501101530004",
            f"{DL_BASE}/01/09501101530004",
        ),
        # invalid date
        (
            "17250230",
            f"{DL_BASE}/17/250230",
        ),
        # weight value too short
        (
            "310300251",
            f"{DL_BASE}/3103/00251",
        ),
        # weight value with illegal character
        (
            "31030025A3",
            f"{DL_BASE}/3103/0025A3",
        ),
        # family prefix without its parameter
        (
            "310",
            f"{DL_BASE}/310/002513",
        ),
        # family prefix with an illegal parameter
        (
            "310A002513",
            f"{DL_BASE}/310A/002513",
        ),
        # unknown AI
        (
            "99ABC",
            f"{DL_BASE}/99/abc",
        ),
        # unknown 4-character numeric AI / family lookalike
        (
            "3110000000",
            f"{DL_BASE}/3110/000000",
        ),
        # empty variable field
        (
            f"10{GS}21ABC",
            f"{DL_BASE}?10=",
        ),
    ],
)
def test_element_string_and_digital_link_agree_on_error_code(element_string, digital_link):
    from app.gs1 import GS1ParseError

    def code(data):
        with pytest.raises(GS1ParseError) as exc_info:
            parse(data)
        return exc_info.value.code

    assert code(element_string) == code(digital_link)


def test_weight_value_validation_is_shared_for_kg_and_lb_all_precisions():
    from decimal import Decimal

    from app.gs1 import decode_weight

    for prefix, unit in (("310", "kg"), ("320", "lb")):
        for n in range(10):
            ai = f"{prefix}{n}"
            element = parse(f"{ai}000001")[0]
            assert element.ai == ai
            assert element.unit == unit
            assert Decimal(str(element.decimal_value)) == Decimal(
                str(float(decode_weight("000001", n)))
            )


def test_check_date_helper_remains_available_for_other_modules():
    # app.recall imports the private date validator directly.
    _check_date("250200", list(range(6)))
    with pytest.raises(Exception):
        _check_date("251301", list(range(6)))
