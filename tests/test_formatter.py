"""
Tests for PatternFormatter: token sequences → human-readable crochet patterns.
"""
import pytest
from alphaknit.formatter import PatternFormatter


@pytest.fixture
def fmt():
    return PatternFormatter()


class TestFormatTokens:

    def test_empty_returns_empty_message(self, fmt):
        assert fmt.format_tokens([]) == "Empty pattern."

    def test_magic_ring_only(self, fmt):
        result = fmt.format_tokens(["mr_6"])
        assert "R1" in result
        assert "mr_6" in result

    def test_mr6_plus_sc_row(self, fmt):
        tokens = ["mr_6"] + ["sc"] * 6
        result = fmt.format_tokens(tokens)
        assert "R1" in result
        assert "R2" in result
        # Row 1 stitch count = 6, Row 2 stitch count = 6
        assert "(6)" in result

    def test_expansion_row_stitch_count(self, fmt):
        # mr_6 + inc x 6 → R1: 6 stitches, R2: 12 stitches
        tokens = ["mr_6"] + ["inc"] * 6
        result = fmt.format_tokens(tokens)
        assert "(6)" in result   # R1
        assert "(12)" in result  # R2

    def test_contraction_row_stitch_count(self, fmt):
        # mr_6 + inc*6 + dec*6 → R1: 6, R2: 12, R3: 6
        tokens = ["mr_6"] + ["inc"] * 6 + ["dec"] * 6
        result = fmt.format_tokens(tokens)
        assert "(6)" in result
        assert "(12)" in result

    def test_format_ids_strips_special_tokens(self, fmt):
        from alphaknit.config import SOS_ID, EOS_ID, PAD_ID, VOCAB, ID_TO_TOKEN
        mr6_id = VOCAB.get("mr_6", None)
        sc_id = VOCAB.get("sc", None)
        if mr6_id is None or sc_id is None:
            pytest.skip("Token IDs not in vocabulary")
        ids = [SOS_ID, mr6_id, sc_id, sc_id, EOS_ID, PAD_ID]
        result = fmt.format_ids(ids)
        assert "R1" in result

    def test_compress_row_single_stitch_type(self, fmt):
        # All sc → "sc x 6"
        result = fmt._compress_row(["sc"] * 6)
        assert "sc x 6" in result

    def test_compress_row_single_token(self, fmt):
        result = fmt._compress_row(["inc"])
        assert result == "inc"

    def test_compress_row_empty(self, fmt):
        assert fmt._compress_row([]) == ""

    def test_multi_row_labels_incremental(self, fmt):
        tokens = ["mr_6"] + ["inc"] * 6 + ["sc", "inc"] * 6
        result = fmt.format_tokens(tokens)
        assert "R1" in result
        assert "R2" in result
        assert "R3" in result
