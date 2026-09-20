"""GS1 element-string and Digital Link URI parser.

Supports AI 01 (GTIN-14), AI 10 (batch/lot), AI 17 (expiry date),
AI 21 (serial number) and the variable-metric weight family
AI 310n (net weight, kilograms) and AI 320n (net weight, pounds),
where ``n`` (0-9) is the number of implied decimal places and the
field is a fixed 6-digit number.

The supported vocabulary lives in one immutable, declarative registry
(:data:`REGISTRY`). Every entry is either:

* an :class:`ExactAIDecl` -- a concrete, fixed-length identifier (01, 10,
  17, 21), or
* a :class:`FamilyAIDecl` -- an identifier prefix (``310`` / ``320``)
  together with exactly one *parameter* character ``n`` (a digit 0-9
  selecting the implied decimal precision); the concrete identifiers are
  ``3100``..``3109`` and ``3200``..``3209``.

The registry is built and validated once at import time
(:func:`build_registry`): declarations must carry complete lengths and
character sets, a callable semantic validator, concrete identifiers must
not collide, and no family prefix may shadow another prefix or an exact
AI -- the concrete vocabulary is therefore unambiguous and longest-match
resolution always has a unique answer.

Two input forms are accepted by :func:`parse`:

1. Raw GS1 element strings as emitted by scanners. Fixed-length fields
   (01, 17, 310n, 320n) are delimited implicitly by their length;
   variable-length fields (10, 21) must be terminated by the FNC1
   separator (ASCII 29, GS) unless they are the last field in the
   string. Identifiers are recognised by longest match (the
   four-character 310n/320n family takes precedence over any
   two-character AI); a weight-family prefix with a missing or
   non-numeric parameter is reported as INVALID_PRECISION rather than
   being mistaken for a shorter AI.

2. GS1 Digital Link URIs (``http://`` / ``https://``). AI/value pairs
   are read from the URI path (``/AI/value/AI/value``) first and then
   from query parameters (``?AI=value&AI=value``). The URI carrier
   recognises its keys with the same registry recogniser
   (:func:`recognize_key`) -- whole-key match with family-prefix
   awareness -- and both carriers dispatch to the same value validation
   (charset, length and per-AI semantic checks). Percent-encoded bytes
   are decoded as UTF-8 before validation.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Callable, FrozenSet, Mapping, Optional, Tuple

GS = "\x1d"  # ASCII 29 <GS>, transmitted as FNC1 by scanners

DIGITS = frozenset("0123456789")
HEX_DIGITS = frozenset("0123456789ABCDEFabcdef")

URI_SCHEMES = ("https://", "http://")

# GS1 AI encodable character set 82 (GS1 General Specifications, fig. 7.11-1)
AI82 = frozenset(
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "!\"%&'()*+,-./:;<=>?_"
)

PARAMETER_DIGITS: Tuple[str, ...] = tuple("0123456789")


SUPPORTED_AIS = (
    "01, 10, 17, 21 and the weight family 310n/320n "
    "(n = 0-9 implied decimal places, fixed 6 digits)"
)


# ============================================================ error / elements


class GS1ParseError(Exception):
    """Parse/validation failure carrying a byte offset and the expected rule."""

    def __init__(self, code, message, offset, expected, found=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.offset = offset
        self.expected = expected
        self.found = found

    def to_dict(self):
        detail = {
            "error": self.code,
            "message": self.message,
            "offset": self.offset,
            "expected": self.expected,
        }
        if self.found is not None:
            detail["found"] = self.found
        return detail


@dataclass(frozen=True)
class Element:
    ai: str
    name: str
    value: str
    offset: int  # byte offset of the AI within the input
    decimal_value: Optional[float] = None  # 310n/320n only: value with implied decimal point
    unit: Optional[str] = None  # 310n/320n only: "kg" or "lb"


def _byte_offset(data: str, index: int) -> int:
    """Byte offset (UTF-8) of the character at ``index``."""
    return len(data[:index].encode("utf-8"))


# ============================================================ semantic handlers


SemanticHandler = Callable[[str, "AISpec", list], Optional["WeightSemantics"]]
"""Validate a charset/length-checked value and describe its result.

``offsets[i]`` is the raw byte offset of ``value[i]`` so semantic errors
can be positioned precisely. The handler raises :class:`GS1ParseError` on
failure and, for the weight family, returns :class:`WeightSemantics`
carrying the measurement unit; it returns ``None`` for AIs without a
decoded value.
"""


@dataclass(frozen=True)
class WeightSemantics:
    """Result-side description of a decoded 310n/320n weight value."""

    unit: str


def _check_gtin(gtin, offsets):
    total = 0
    for i, ch in enumerate(reversed(gtin[:-1])):
        total += int(ch) * (3 if i % 2 == 0 else 1)
    expected = (10 - total % 10) % 10
    if expected != int(gtin[-1]):
        raise GS1ParseError(
            "INVALID_CHECK_DIGIT",
            f"GTIN check digit mismatch: expected {expected}, found {gtin[-1]}",
            offset=offsets[-1],
            expected=f"Mod-10 check digit {expected}",
            found=gtin[-1],
        )


def _check_date(yymmdd, offsets):
    yy = int(yymmdd[0:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    year = 2000 + yy
    if not 1 <= mm <= 12:
        raise GS1ParseError(
            "INVALID_DATE",
            f"Invalid expiry month {mm:02d} in AI 17 date '{yymmdd}'",
            offset=offsets[2],
            expected="month 01-12",
            found=yymmdd[2:4],
        )
    last_day = calendar.monthrange(year, mm)[1]
    if dd == 0:
        return  # GS1: day "00" denotes the last day of the month
    if dd > last_day:
        raise GS1ParseError(
            "INVALID_DATE",
            f"Invalid expiry day {dd:02d} for {year}-{mm:02d} (max {last_day})",
            offset=offsets[4],
            expected=f"day 01-{last_day} for {year}-{mm:02d} (00 = last day of month)",
            found=yymmdd[4:6],
        )


def decode_weight(value: str, decimals: int) -> Decimal:
    """Interpret the 6 fixed digits with ``decimals`` implied places exactly.

    The GS1 variable-metric rule is ``int(digits) / 10**n``; when n > 6
    the missing high-order fractional digits are implied zero, e.g.
    ``3109 000001`` -> Decimal('0.000000001').  The result is built from
    the integer digits and a power of ten, so no binary float is ever
    involved.
    """
    return Decimal(int(value)).scaleb(-decimals)


def _noop_semantics(value: str, spec: "AISpec", offsets: list) -> None:
    """Default semantic handler: charset/length checks are sufficient."""
    return None


def _gtin_semantics(value: str, spec: "AISpec", offsets: list) -> None:
    _check_gtin(value, offsets)
    return None


def _date_semantics(value: str, spec: "AISpec", offsets: list) -> None:
    _check_date(value, offsets)
    return None


def _weight_semantics(fam: "FamilyAIDecl", parameter: int) -> SemanticHandler:
    """Build the semantic handler of one 310n/320n concrete AI."""

    unit = fam.unit

    def handler(value: str, spec: "AISpec", offsets: list) -> WeightSemantics:
        # The value is already known to be six digits; the exact Decimal
        # decoding happens on demand via decode_weight(); the handler only
        # attaches the unit carried by every weight element.
        return WeightSemantics(unit=unit)

    return handler


# ============================================================ the registry


@dataclass(frozen=True)
class ValueField:
    """Declarative shape of an AI's data field."""

    min_length: int
    max_length: int
    fixed_length: bool
    charset: FrozenSet[str]
    charset_rule: str


@dataclass(frozen=True)
class ExactAIDecl:
    """An AI with a concrete, literal identifier (e.g. ``"01"``)."""

    ai: str
    name: str
    field: ValueField
    semantics: SemanticHandler = _noop_semantics


@dataclass(frozen=True)
class FamilyAIDecl:
    """An identifier family: a prefix plus one parameter character.

    Concrete identifiers are ``prefix + parameter`` for every parameter in
    :data:`PARAMETER_DIGITS` (``310`` + ``0``..``9`` -> ``3100``..``3109``);
    the parameter value is the number of implied decimal places and is
    exposed on each resolved spec as ``decimals``.
    """

    prefix: str
    name: str
    field: ValueField
    semantics_factory: Callable[["FamilyAIDecl", int], SemanticHandler] = _weight_semantics
    unit: Optional[str] = None  # measurement unit emitted with the element

    @property
    def identifier_length(self) -> int:
        return len(self.prefix) + 1

    def expected_pattern(self) -> str:
        """Human-readable concrete form, e.g. ``"310n"``."""
        return f"{self.prefix}n"


@dataclass(frozen=True)
class AISpec:
    """Resolved, immutable view of one concrete identifier used by the parser."""

    ai: str
    name: str
    min_length: int
    max_length: int
    fixed_length: bool
    charset: FrozenSet[str]
    charset_rule: str
    ai_length: int
    semantics: SemanticHandler = _noop_semantics
    decimals: Optional[int] = None  # implied decimal places (310n/320n only)
    unit: Optional[str] = None  # measurement unit emitted alongside decimal_value
    family: Optional[FamilyAIDecl] = None  # set for 310n/320n members


class RegistryDefinitionError(Exception):
    """The declarative AI table is malformed (raised once, at import time)."""


# Declarations, in canonical (first-seen) order. The table is the single
# source of truth: adding an AI means adding one row here -- nothing in the
# recogniser or in either parser is hard-coded to a particular AI.
_EXACT_DECLARATIONS: Tuple[ExactAIDecl, ...] = (
    ExactAIDecl(
        ai="01",
        name="GTIN",
        field=ValueField(14, 14, True, DIGITS, "digits 0-9"),
        semantics=_gtin_semantics,
    ),
    ExactAIDecl(
        ai="17",
        name="EXPIRY",
        field=ValueField(6, 6, True, DIGITS, "digits 0-9"),
        semantics=_date_semantics,
    ),
    ExactAIDecl(
        ai="10",
        name="BATCH/LOT",
        field=ValueField(1, 20, False, AI82, "GS1 AI character set 82"),
    ),
    ExactAIDecl(
        ai="21",
        name="SERIAL",
        field=ValueField(1, 20, False, AI82, "GS1 AI character set 82"),
    ),
)

_FAMILY_DECLARATIONS: Tuple[FamilyAIDecl, ...] = (
    FamilyAIDecl(
        prefix="310",
        name="NET WEIGHT (KG)",
        field=ValueField(6, 6, True, DIGITS, "digits 0-9"),
        semantics_factory=_weight_semantics,
        unit="kg",
    ),
    FamilyAIDecl(
        prefix="320",
        name="NET WEIGHT (LB)",
        field=ValueField(6, 6, True, DIGITS, "digits 0-9"),
        semantics_factory=_weight_semantics,
        unit="lb",
    ),
)


@dataclass(frozen=True)
class AIRegistry:
    """Immutable, validated registry of exact and parameterised family AIs."""

    specs: Mapping[str, AISpec] = field(default_factory=dict)
    exact: Mapping[str, ExactAIDecl] = field(default_factory=dict)
    families: Mapping[str, FamilyAIDecl] = field(default_factory=dict)
    exact_order: Tuple[str, ...] = ()
    family_order: Tuple[str, ...] = ()

    def resolve(self, ai: str) -> Optional[AISpec]:
        """Resolve a complete concrete identifier (longest, unambiguous)."""
        return self.specs.get(ai)

    def family_at(self, prefix: str) -> Optional[FamilyAIDecl]:
        return self.families.get(prefix)

    @property
    def min_identifier_length(self) -> int:
        lengths = [len(ai) for ai in self.exact]
        lengths.extend(len(prefix) + 1 for prefix in self.families)
        return min(lengths, default=0)

    def family_prefixes_longest_first(self) -> Tuple[str, ...]:
        return tuple(
            sorted(self.family_order, key=lambda p: (-len(p), self.family_order.index(p)))
        )


def _validate_value_field(owner: str, fld: ValueField) -> None:
    if not isinstance(fld, ValueField):
        raise RegistryDefinitionError(f"{owner}: field must be a ValueField")
    if (
        not isinstance(fld.min_length, int)
        or not isinstance(fld.max_length, int)
        or fld.min_length <= 0
        or fld.max_length < fld.min_length
    ):
        raise RegistryDefinitionError(f"{owner}: invalid length bounds")
    if fld.fixed_length and fld.min_length != fld.max_length:
        raise RegistryDefinitionError(
            f"{owner}: fixed_length field must have min_length == max_length"
        )
    if not isinstance(fld.charset, frozenset) or not fld.charset:
        raise RegistryDefinitionError(f"{owner}: charset must be a non-empty frozenset")
    if not isinstance(fld.charset_rule, str) or not fld.charset_rule:
        raise RegistryDefinitionError(f"{owner}: charset_rule must be a non-empty string")


def build_registry(
    exact_declarations: Tuple[ExactAIDecl, ...] = _EXACT_DECLARATIONS,
    family_declarations: Tuple[FamilyAIDecl, ...] = _FAMILY_DECLARATIONS,
) -> AIRegistry:
    """Build and validate the immutable registry from declarative rows.

    Validates, once at startup:

    * each row has a well-formed all-digit identifier/prefix, a complete
      length and character-set description and a callable semantic handler;
    * exact identifiers and family prefixes are unique and all digits;
    * no family prefix shadows (equals or is a prefix of) another family
      prefix, and no exact identifier collides with a family prefix -- the
      vocabulary is therefore unambiguous under longest-match resolution;
    * the expanded concrete identifiers (family prefix + every parameter
      digit) are all distinct and complete.
    """
    exact: dict[str, ExactAIDecl] = {}
    families: dict[str, FamilyAIDecl] = {}
    exact_order: list[str] = []
    family_order: list[str] = []
    specs: dict[str, AISpec] = {}

    for decl in exact_declarations:
        if not isinstance(decl, ExactAIDecl):
            raise RegistryDefinitionError(f"{decl!r}: not an ExactAIDecl")
        owner = f"exact AI {decl.ai!r}"
        if not decl.ai or not all(c in DIGITS for c in decl.ai):
            raise RegistryDefinitionError(f"{owner}: identifier must be non-empty digits")
        if not isinstance(decl.name, str) or not decl.name:
            raise RegistryDefinitionError(f"{owner}: name is required")
        _validate_value_field(owner, decl.field)
        if not callable(decl.semantics):
            raise RegistryDefinitionError(f"{owner}: semantic handler must be callable")
        if decl.ai in exact:
            raise RegistryDefinitionError(f"duplicate exact AI declaration: {decl.ai!r}")
        exact[decl.ai] = decl
        exact_order.append(decl.ai)
        specs[decl.ai] = AISpec(
            ai=decl.ai,
            name=decl.name,
            min_length=decl.field.min_length,
            max_length=decl.field.max_length,
            fixed_length=decl.field.fixed_length,
            charset=decl.field.charset,
            charset_rule=decl.field.charset_rule,
            ai_length=len(decl.ai),
            semantics=decl.semantics,
        )

    for fam in family_declarations:
        if not isinstance(fam, FamilyAIDecl):
            raise RegistryDefinitionError(f"{fam!r}: not a FamilyAIDecl")
        owner = f"family AI {fam.prefix!r}"
        if not fam.prefix or not all(c in DIGITS for c in fam.prefix):
            raise RegistryDefinitionError(f"{owner}: prefix must be non-empty digits")
        if not isinstance(fam.name, str) or not fam.name:
            raise RegistryDefinitionError(f"{owner}: name is required")
        _validate_value_field(owner, fam.field)
        if not callable(fam.semantics_factory):
            raise RegistryDefinitionError(f"{owner}: semantics_factory must be callable")
        if fam.prefix in families:
            raise RegistryDefinitionError(
                f"duplicate family AI prefix declaration: {fam.prefix!r}"
            )
        families[fam.prefix] = fam
        family_order.append(fam.prefix)

    # Cross-declaration ambiguity checks. A concrete identifier must have a
    # unique producer, and longest-match resolution must never need to break
    # a tie between a family and another prefix or an exact AI.
    for prefix, fam in families.items():
        if prefix in exact:
            raise RegistryDefinitionError(
                f"family prefix {prefix!r} collides with exact AI {prefix!r}"
            )
        for other in families:
            if other != prefix and (
                other.startswith(prefix) or prefix.startswith(other)
            ):
                raise RegistryDefinitionError(
                    f"family prefixes {prefix!r} and {other!r} shadow each other; "
                    "the identifier vocabulary would be ambiguous"
                )
        for n, parameter in enumerate(PARAMETER_DIGITS):
            ai = f"{prefix}{parameter}"
            if ai in specs:
                raise RegistryDefinitionError(
                    f"concrete AI {ai!r} produced by family {prefix!r} is already "
                    "declared elsewhere"
                )
            try:
                semantics = fam.semantics_factory(fam, n)
            except Exception as exc:  # a broken table must fail at startup
                raise RegistryDefinitionError(
                    f"family {prefix!r}: semantics factory failed for parameter {parameter!r}"
                ) from exc
            if not callable(semantics):
                raise RegistryDefinitionError(
                    f"family {prefix!r}: semantics for parameter {parameter!r} is not callable"
                )
            specs[ai] = AISpec(
                ai=ai,
                name=fam.name,
                min_length=fam.field.min_length,
                max_length=fam.field.max_length,
                fixed_length=fam.field.fixed_length,
                charset=fam.field.charset,
                charset_rule=fam.field.charset_rule,
                ai_length=fam.identifier_length,
                semantics=semantics,
                decimals=n,
                unit=fam.unit,
                family=fam,
            )

    return AIRegistry(
        specs=MappingProxyType(specs),
        exact=MappingProxyType(exact),
        families=MappingProxyType(families),
        exact_order=tuple(exact_order),
        family_order=tuple(family_order),
    )


REGISTRY: AIRegistry = build_registry()

# Backwards-compatible introspection aliases (read-only views).
AI_SPECS: Mapping[str, AISpec] = REGISTRY.specs
WEIGHT_AI_FAMILIES: Mapping[str, tuple] = MappingProxyType(
    {
        prefix: (fam.name, fam.unit)
        for prefix, fam in REGISTRY.families.items()
    }
)


# ============================================================ the recogniser


class AIKeyStatus(Enum):
    """Outcome of recognising a complete identifier key."""

    MATCHED = "matched"
    INVALID_KEY = "invalid_key"  # not a numeric identifier
    MISSING_PARAMETER = "missing_parameter"  # family prefix without its parameter
    BAD_PARAMETER = "bad_parameter"  # family parameter outside 0-9
    UNSUPPORTED = "unsupported"  # well-formed numeric identifier, not in the registry


@dataclass(frozen=True)
class AIKeyRecognition:
    status: AIKeyStatus
    spec: Optional[AISpec] = None
    family: Optional[FamilyAIDecl] = None
    parameter: Optional[str] = None  # offending parameter when BAD_PARAMETER


def recognize_key(key: str) -> AIKeyRecognition:
    """Recognise a *complete* identifier token against the registry.

    Shared by both carriers: Digital Link keys are whole path segments /
    query keys, and the element-string parser feeds the candidate token it
    has read. Recognition is registry-driven:

    1. a family prefix without its parameter -> MISSING_PARAMETER;
    2. a family prefix whose parameter is not a digit -> BAD_PARAMETER;
    3. the longest concrete identifier present in the registry -> MATCHED;
    4. any other non-empty all-digit token -> UNSUPPORTED;
    5. an empty / non-numeric token -> INVALID_KEY.

    The family rules (1-2) are deliberately evaluated *before* the generic
    all-digit rule: "310" and "310x" are precision errors, never an
    unsupported identifier or a mistaken shorter AI.
    """
    # Family-prefix rules first: "310" (parameter missing) and "310x"
    # (parameter not a digit) are precision errors even though the token is
    # not fully numeric or fully present.
    for prefix in REGISTRY.family_prefixes_longest_first():
        if key.startswith(prefix):
            family = REGISTRY.family_at(prefix)
            if len(key) == len(prefix):
                return AIKeyRecognition(
                    status=AIKeyStatus.MISSING_PARAMETER, family=family
                )
            parameter = key[len(prefix)]
            if parameter not in DIGITS:
                return AIKeyRecognition(
                    status=AIKeyStatus.BAD_PARAMETER,
                    family=family,
                    parameter=parameter,
                )
            break

    if not key or not all(c in DIGITS for c in key):
        return AIKeyRecognition(status=AIKeyStatus.INVALID_KEY)

    # Longest concrete identifier wins (310n/320n over any shorter AI).
    spec = REGISTRY.resolve(key)
    if spec is not None:
        return AIKeyRecognition(status=AIKeyStatus.MATCHED, spec=spec)
    return AIKeyRecognition(status=AIKeyStatus.UNSUPPORTED)


# ============================================================ entry point


def parse(data: str) -> list[Element]:
    """Parse a GS1 element string or Digital Link URI into ordered elements."""
    if not data:
        raise GS1ParseError(
            "EMPTY_INPUT",
            "Input is empty",
            offset=0,
            expected=f"at least one element string starting with AI {SUPPORTED_AIS}",
        )
    if data.startswith(URI_SCHEMES):
        return _parse_digital_link(data)
    return _parse_element_string(data)


# ----------------------------------------------------------------- element strings


def _parse_element_string(data: str) -> list[Element]:
    elements = []
    pos = 0
    while pos < len(data):
        ai_offset = _byte_offset(data, pos)
        if data[pos] == GS:
            raise GS1ParseError(
                "UNEXPECTED_SEPARATOR",
                "Unexpected separator (ASCII 29); expected an application identifier",
                offset=ai_offset,
                expected=f"numeric AI, one of: {SUPPORTED_AIS}",
                found="<GS>",
            )
        spec, pos = _match_element_string_ai(data, pos, ai_offset)
        value_pos = pos

        if spec.fixed_length:
            value, pos = _read_fixed(data, pos, spec)
        else:
            value, pos = _read_variable(data, pos, spec)

        # UTF-8 byte offset of every value character in the raw input
        offsets = [_byte_offset(data, value_pos + i) for i in range(len(value))]
        decimal_value, unit = _validate_value(spec, value, offsets)

        elements.append(
            Element(
                ai=spec.ai,
                name=spec.name,
                value=value,
                offset=ai_offset,
                decimal_value=decimal_value,
                unit=unit,
            )
        )
    return elements


def _match_element_string_ai(data: str, pos: int, ai_offset: int):
    """Recognise the AI at ``pos`` by longest match.

    Returns ``(spec, new_pos)`` or raises :class:`GS1ParseError`. The
    four-character 310n/320n identifiers take precedence over the
    two-character set; a malformed family prefix (missing or non-digit
    parameter) is reported as INVALID_PRECISION rather than being
    mistaken for a 2-character AI.
    """
    min_len = REGISTRY.min_identifier_length
    head = data[pos : pos + min_len]
    if len(head) < min_len or not all(c in DIGITS for c in head):
        raise GS1ParseError(
            "INVALID_AI",
            f"Expected a numeric application identifier, found {head!r}",
            offset=ai_offset,
            expected=f"numeric AI, one of: {SUPPORTED_AIS}",
            found=head,
        )

    # The scanner stream has no delimiter in front of the AI, so first detect
    # a family prefix whose parameter is physically absent (end of input) or
    # replaced by the FNC1 separator -- those cases carry stream-specific
    # messages the whole-key recogniser cannot produce.
    family_prefix = None
    for prefix in REGISTRY.family_prefixes_longest_first():
        if data.startswith(prefix, pos):
            family_prefix = prefix
            break
    if family_prefix is not None:
        fam = REGISTRY.family_at(family_prefix)
        parameter_pos = pos + len(family_prefix)
        if parameter_pos >= len(data):
            raise GS1ParseError(
                "INVALID_PRECISION",
                f"AI family {family_prefix} requires a precision digit (0-9) but the input ended",
                offset=_byte_offset(data, len(data)),
                expected=f"precision digit 0-9 for AI {fam.expected_pattern()}",
                found="end of input",
            )
        if data[parameter_pos] == GS:
            raise GS1ParseError(
                "INVALID_PRECISION",
                f"AI family {family_prefix} requires a precision digit (0-9) but a "
                "separator (ASCII 29) was found",
                offset=_byte_offset(data, parameter_pos),
                expected=f"precision digit 0-9 for AI {fam.expected_pattern()}",
                found="<GS>",
            )

    # Build the longest candidate window -- concrete identifier lengths when
    # the stream begins with one, otherwise the prefix-plus-parameter window
    # (so "310A" reaches the recogniser intact instead of being truncated to
    # the 2-character "31") -- and classify it with the shared recogniser.
    longest = min_len
    for concrete in REGISTRY.specs.values():
        if data.startswith(concrete.ai, pos):
            longest = max(longest, concrete.ai_length)
    if family_prefix is not None:
        longest = max(longest, len(family_prefix) + 1)
    recognition = recognize_key(data[pos : pos + longest])

    if recognition.status is AIKeyStatus.MATCHED:
        spec = recognition.spec
        return spec, pos + spec.ai_length
    if recognition.status is AIKeyStatus.BAD_PARAMETER:
        fam = recognition.family
        parameter = recognition.parameter
        parameter_pos = pos + len(fam.prefix)
        raise GS1ParseError(
            "INVALID_PRECISION",
            f"AI family {fam.prefix} requires a precision digit 0-9 as its 4th "
            f"character, found {parameter!r}",
            offset=_byte_offset(data, parameter_pos),
            expected=f"precision digit 0-9 for AI {fam.expected_pattern()}",
            found=parameter,
        )
    # INVALID_KEY is already excluded by the head check; MISSING_PARAMETER is
    # handled by the stream-specific branches above. Anything remaining is a
    # well-formed numeric identifier the registry does not know, reported on
    # the shortest declared identifier window (the historical 2-char AI).
    ai = data[pos : pos + min_len]
    raise GS1ParseError(
        "UNSUPPORTED_AI",
        f"Unsupported application identifier '{ai}'",
        offset=ai_offset,
        expected=f"one of: {SUPPORTED_AIS}",
        found=ai,
    )


def _read_fixed(data, pos, spec):
    field = data[pos : pos + spec.max_length]
    sep = field.find(GS)
    if sep != -1:
        raise GS1ParseError(
            "FIELD_TOO_SHORT",
            f"AI {spec.ai} ({spec.name}) requires exactly {spec.max_length} characters "
            f"but a separator was found after {sep}",
            offset=_byte_offset(data, pos + sep),
            expected=f"{spec.max_length} characters for AI {spec.ai}",
            found="<GS>",
        )
    if len(field) < spec.max_length:
        raise GS1ParseError(
            "FIELD_TOO_SHORT",
            f"AI {spec.ai} ({spec.name}) requires exactly {spec.max_length} characters "
            f"but the input ended after {len(field)}",
            offset=_byte_offset(data, len(data)),
            expected=f"{spec.max_length} characters for AI {spec.ai}",
            found=f"{len(field)} character(s)",
        )
    return field, pos + spec.max_length


def _read_variable(data, pos, spec):
    end = data.find(GS, pos)
    terminated = end != -1
    if not terminated:
        end = len(data)
    field = data[pos:end]
    if len(field) < spec.min_length:
        raise GS1ParseError(
            "FIELD_EMPTY",
            f"AI {spec.ai} ({spec.name}) requires at least {spec.min_length} character",
            offset=_byte_offset(data, pos),
            expected=f"{spec.min_length}-{spec.max_length} characters for AI {spec.ai}",
            found="empty field",
        )
    if len(field) > spec.max_length:
        raise GS1ParseError(
            "FIELD_TOO_LONG",
            f"AI {spec.ai} ({spec.name}) allows at most {spec.max_length} characters "
            f"but got {len(field)} before {'the separator' if terminated else 'end of input'}",
            offset=_byte_offset(data, pos + spec.max_length),
            expected=f"separator (ASCII 29) or end of input within {spec.max_length} characters",
        )
    return field, end + 1 if terminated else end


# ------------------------------------------------------------ shared value checks


def _check_charset(spec, field, offsets):
    """``offsets[i]`` is the raw byte offset of ``field[i]``."""
    for i, ch in enumerate(field):
        if ch not in spec.charset:
            raise GS1ParseError(
                "INVALID_CHARACTER",
                f"Character {ch!r} (U+{ord(ch):04X}) is not allowed in AI {spec.ai} ({spec.name})",
                offset=offsets[i],
                expected=spec.charset_rule,
                found=ch,
            )


def _validate_value(spec: AISpec, value: str, offsets):
    """Run the charset and per-AI semantic checks shared by both carriers.

    Length checks are carrier-specific (a scanner stream ends a fixed field
    at its length or at a separator, while a URI segment ends at the next
    slash / ampersand) and are performed before calling this function.

    Returns ``(decimal_value, unit)``, both ``None`` unless the AI belongs
    to the 310n/320n weight family.
    """
    _check_charset(spec, value, offsets)
    result = spec.semantics(value, spec, offsets)
    if isinstance(result, WeightSemantics):
        return float(decode_weight(value, spec.decimals)), result.unit
    return None, None


# ------------------------------------------------------------ Digital Link URIs


def _parse_digital_link(data: str) -> list[Element]:
    authority_start = data.index("://") + 3  # works for http:// and https://

    # Locate path / query / fragment boundaries directly on the raw URI so
    # that every error can carry an accurate byte offset.
    path_start = len(data)
    for marker in ("/", "?", "#"):
        j = data.find(marker, authority_start)
        if j != -1:
            path_start = min(path_start, j)
    fragment_start = data.find("#", path_start)
    if fragment_start == -1:
        fragment_start = len(data)
    query_mark = data.find("?", path_start, fragment_start)
    if query_mark == -1:
        path_end = fragment_start
        query_start = query_end = -1
    else:
        path_end = query_mark
        query_start = query_mark + 1
        query_end = fragment_start

    # ---- path: /AI/value/AI/value (a single trailing slash is tolerated) ----
    segments = _split_path(data, path_start, path_end)
    if len(segments) % 2:
        odd_start = segments[-1][0]
        odd_raw = data[segments[-1][0] : segments[-1][1]]
        raise GS1ParseError(
            "UNPAIRED_PATH",
            "URI path must contain AI/value pairs but ends with a value missing "
            f"its AI or an AI missing its value: {odd_raw!r}",
            offset=_byte_offset(data, odd_start),
            expected="even number of path segments: /AI/value[/AI/value]...",
            found=odd_raw,
        )

    # ---- query: AI=value&AI=value (query order follows path order) ----
    pairs = _split_query(data, query_start, query_end)

    if not segments and not pairs:
        raise GS1ParseError(
            "EMPTY_INPUT",
            "URI contains no AI/value pairs in its path or query string",
            offset=0,
            expected=f"at least one path segment or query parameter with AI {SUPPORTED_AIS}",
        )

    elements = []
    seen: dict[str, tuple[int, str]] = {}

    def consume(ai_span, value_span, location):
        ai_start, ai_end = ai_span
        val_start, val_end = value_span
        ai, ai_offsets = _percent_decode(data, ai_start, ai_end)
        ai_byte = ai_offsets[0] if ai_offsets else _byte_offset(data, ai_start)

        # In a URI the AI is the whole path segment / query key, so unlike
        # element strings there is no fixed 2-character window to read; the
        # shared registry recogniser classifies the complete key.
        if not ai:
            raw_ai = data[ai_start:ai_end]
            raise GS1ParseError(
                "INVALID_AI",
                f"Expected a numeric application identifier in the {location}, "
                "found an empty segment",
                offset=ai_byte,
                expected=f"numeric AI, one of: {SUPPORTED_AIS}",
                found=raw_ai,
            )

        recognition = recognize_key(ai)
        if recognition.status is AIKeyStatus.MISSING_PARAMETER:
            fam = recognition.family
            raise GS1ParseError(
                "INVALID_PRECISION",
                f"AI family {fam.prefix} requires a precision digit (0-9) as "
                f"its 4th character in the {location}",
                offset=_byte_offset(data, ai_end),
                expected=f"4-character AI {fam.expected_pattern()} (precision digit 0-9)",
                found=ai,
            )
        if recognition.status is AIKeyStatus.BAD_PARAMETER:
            fam = recognition.family
            parameter = recognition.parameter
            raise GS1ParseError(
                "INVALID_PRECISION",
                f"AI family {fam.prefix} requires a precision digit 0-9 as its "
                f"4th character, found {parameter!r}",
                offset=ai_offsets[len(fam.prefix)],
                expected=f"precision digit 0-9 for AI {fam.expected_pattern()}",
                found=parameter,
            )
        if recognition.status is AIKeyStatus.INVALID_KEY:
            raw_ai = data[ai_start:ai_end]
            raise GS1ParseError(
                "INVALID_AI",
                f"Expected a numeric application identifier in the {location}, "
                f"found {ai!r}",
                offset=ai_byte,
                expected=f"numeric AI, one of: {SUPPORTED_AIS}",
                found=raw_ai,
            )
        if recognition.status is AIKeyStatus.UNSUPPORTED:
            raise GS1ParseError(
                "UNSUPPORTED_AI",
                f"Unsupported application identifier '{ai}' in the {location}",
                offset=ai_byte,
                expected=f"one of: {SUPPORTED_AIS}",
                found=ai,
            )
        spec = recognition.spec
        if ai in seen:
            first_offset, first_location = seen[ai]
            if first_location == location:
                message = (
                    f"Application identifier '{ai}' is repeated in the {location}; "
                    "each AI may appear only once"
                )
            else:
                message = (
                    f"Application identifier '{ai}' appears in both the "
                    f"{first_location} (byte {first_offset}) and the {location}; "
                    "an AI may not be given in both places"
                )
            raise GS1ParseError(
                "DUPLICATE_AI",
                message,
                offset=ai_byte,
                expected=f"AI '{ai}' to appear at most once",
                found=ai,
            )
        seen[ai] = (ai_byte, location)

        value, offsets = _percent_decode(data, val_start, val_end)
        decimal_value, unit = _validate_uri_value(
            spec, value, offsets, _byte_offset(data, val_end)
        )
        elements.append(
            Element(
                ai=spec.ai,
                name=spec.name,
                value=value,
                offset=ai_byte,
                decimal_value=decimal_value,
                unit=unit,
            )
        )

    for k in range(0, len(segments), 2):
        consume(segments[k], segments[k + 1], "URI path")
    for key_span, value_span in pairs:
        consume(key_span, value_span, "query string")

    return elements


def _split_path(data, start, end):
    """Return ``[(seg_start, seg_end), ...]`` for raw path segments."""
    segments = []
    i = start
    if i < end and data[i] == "/":
        i += 1
    while i < end:
        slash = data.find("/", i, end)
        if slash == -1:
            slash = end
        segments.append((i, slash))
        i = slash + 1
    # A trailing slash yields one empty segment anchored at ``end``; ignore it
    # so that "/01/x/" and "/01/x" are equivalent.
    if segments and segments[-1] == (end, end):
        segments.pop()
    return segments


def _split_query(data, start, end):
    """Return ``[(key_span, value_span), ...]`` for raw ``key=value`` pairs."""
    if start == -1:
        return []
    pairs = []
    i = start
    while i < end:
        amp = data.find("&", i, end)
        if amp == -1:
            amp = end
        equals = data.find("=", i, amp)
        if equals == -1:
            # Treat "AI" without '=' as AI with an empty value; the field
            # validation reports the empty-field error.
            pairs.append(((i, amp), (amp, amp)))
        else:
            pairs.append(((i, equals), (equals + 1, amp)))
        i = amp + 1
    return pairs


def _percent_decode(data: str, start: int, end: int) -> tuple[str, list[int]]:
    """Percent-decode ``data[start:end]`` as UTF-8.

    Returns the decoded text and, for every decoded character, the raw
    UTF-8 byte offset of the first byte it came from.
    """
    raw = bytearray()
    origin = []  # character index in ``data`` where each decoded byte began
    i = start
    while i < end:
        ch = data[i]
        if ch == "%":
            hex_ok = (
                i + 2 < end
                and data[i + 1] in HEX_DIGITS
                and data[i + 2] in HEX_DIGITS
            )
            if not hex_ok:
                raise GS1ParseError(
                    "INVALID_PERCENT_ENCODING",
                    f"Invalid percent-encoding at byte {_byte_offset(data, i)}: "
                    f"expected %XX but found {data[i:end]!r}",
                    offset=_byte_offset(data, i),
                    expected="percent-encoded UTF-8 byte of the form %XX",
                    found=data[i : min(end, i + 3)],
                )
            raw.append(int(data[i + 1 : i + 3], 16))
            origin.append(i)
            i += 3
        else:
            for b in ch.encode("utf-8"):
                raw.append(b)
                origin.append(i)
            i += 1

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        bad = origin[exc.start]
        raise GS1ParseError(
            "INVALID_PERCENT_ENCODING",
            "Percent-encoded bytes are not valid UTF-8: "
            f"0x{raw[exc.start]:02X} at byte {_byte_offset(data, bad)}",
            offset=_byte_offset(data, bad),
            expected="valid UTF-8 percent-encoded bytes",
            found=f"0x{raw[exc.start]:02X}",
        ) from None

    offsets = []
    b = 0
    for ch in text:
        offsets.append(_byte_offset(data, origin[b]))
        b += len(ch.encode("utf-8"))
    return text, offsets


def _validate_uri_value(spec, value, offsets, end_offset):
    """Length checks for a percent-decoded URI value, then shared validation.

    Returns ``(decimal_value, unit)`` (both ``None`` outside the 310n/320n
    weight family).
    """
    if spec.fixed_length:
        if len(value) < spec.max_length:
            raise GS1ParseError(
                "FIELD_TOO_SHORT",
                f"AI {spec.ai} ({spec.name}) requires exactly {spec.max_length} characters "
                f"but found {len(value)} after percent-decoding",
                offset=end_offset,
                expected=f"{spec.max_length} characters for AI {spec.ai}",
                found=f"{len(value)} character(s)",
            )
        if len(value) > spec.max_length:
            raise GS1ParseError(
                "FIELD_TOO_LONG",
                f"AI {spec.ai} ({spec.name}) requires exactly {spec.max_length} characters "
                f"but found {len(value)} after percent-decoding",
                offset=offsets[spec.max_length],
                expected=f"{spec.max_length} characters for AI {spec.ai}",
                found=f"{len(value)} characters",
            )
    else:
        if len(value) < spec.min_length:
            raise GS1ParseError(
                "FIELD_EMPTY",
                f"AI {spec.ai} ({spec.name}) requires at least {spec.min_length} character",
                offset=end_offset,
                expected=f"{spec.min_length}-{spec.max_length} characters for AI {spec.ai}",
                found="empty field",
            )
        if len(value) > spec.max_length:
            raise GS1ParseError(
                "FIELD_TOO_LONG",
                f"AI {spec.ai} ({spec.name}) allows at most {spec.max_length} characters "
                f"but found {len(value)} after percent-decoding",
                offset=offsets[spec.max_length],
                expected=f"at most {spec.max_length} characters for AI {spec.ai}",
                found=f"{len(value)} characters",
            )

    return _validate_value(spec, value, offsets)
