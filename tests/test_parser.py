import pytest
from src.parser import _parse_contract_type, detect_contract_type

def test_parse_contract_type_pattern_c():
    """Tests if Pattern C (Company - Contract Type) parses correctly."""
    title = "MetLife, Inc. - Remarketing Agreement"
    result = _parse_contract_type(title)
    assert result == "Remarketing Agreement"

def test_detect_contract_type_normalization():
    """Tests if raw titles are mapped to the correct normalized category."""
    # Should map "non-solicit" to the broader "Non-Compete Agreement" category
    raw_title = "VIVINT_SOLAR_INC_non-solicit_and_nda"
    result = detect_contract_type(raw_title)
    assert result == "Non-Compete Agreement"

def test_detect_contract_type_fallback():
    """Tests if an unknown contract type defaults to 'Other'."""
    raw_title = "Random_Corporate_Document_01"
    result = detect_contract_type(raw_title)
    assert result == "Other"