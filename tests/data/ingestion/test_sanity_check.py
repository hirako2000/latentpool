from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from latentpool.data.ingestion.sanity_check import check_status


@patch("latentpool.data.ingestion.sanity_check.IngestionCoordinator")
@patch("latentpool.data.ingestion.sanity_check.Path.glob")
def test_check_status_full_flow(
    mock_glob: MagicMock,
    mock_coord_class: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ MEV and Normal categorization."""
    mock_coord_inst = mock_coord_class.return_value
    # First call: Arbitrage CSV hashes, Second call: Sandwich CSV hashes
    mock_coord_inst.extract_hashes.side_effect = [
        ["0xabc"],  # Arbitrage
        ["0xdef"],  # Sandwich
    ]

    # 1 Arb, 1 Normal (0x999 is not in CSVs)
    file_arb = MagicMock(spec=Path)
    file_arb.stem = "0xabc"

    file_norm = MagicMock(spec=Path)
    file_norm.stem = "0x999"

    mock_glob.return_value = [file_arb, file_norm]

    check_status()

    captured = capsys.readouterr().out

    assert "Total files on disk: 2" in captured
    assert "Arbitrage: 1" in captured
    assert "Sandwich:  0" in captured
    assert "Normal:    1" in captured

    assert "Arbitrage: 1/1 (100.0%)" in captured
    assert "Sandwich : 0/1 (0.0%)" in captured


@patch("latentpool.data.ingestion.sanity_check.IngestionCoordinator")
@patch("latentpool.data.ingestion.sanity_check.Path.glob")
def test_check_status_imbalance_warning(
    mock_glob: MagicMock,
    mock_coord_class: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tests that the warning triggers when Normal < MEV."""
    mock_coord_inst = mock_coord_class.return_value
    mock_coord_inst.extract_hashes.side_effect = [["0x1"], ["0x2"]]

    f1 = MagicMock(spec=Path)
    f1.stem = "0x1"
    f2 = MagicMock(spec=Path)
    f2.stem = "0x2"
    mock_glob.return_value = [f1, f2]  # 2 MEV, 0 Normal

    check_status()
    captured = capsys.readouterr().out

    assert "Warning: Dataset is imbalanced" in captured


@patch("latentpool.data.ingestion.sanity_check.IngestionCoordinator")
@patch("latentpool.data.ingestion.sanity_check.Path.glob")
def test_check_status_empty_state(
    mock_glob: MagicMock,
    mock_coord_class: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tests behavior when no files or hashes exist."""
    mock_glob.return_value = []
    mock_coord_inst = mock_coord_class.return_value
    mock_coord_inst.extract_hashes.return_value = []

    check_status()
    captured = capsys.readouterr().out

    assert "Total files on disk: 0" in captured
    assert "Normal:    0" in captured


@patch("latentpool.data.ingestion.sanity_check.IngestionCoordinator")
def test_check_status_exception(mock_coord_class: MagicMock) -> None:
    """Tests that exceptions propagate correctly."""
    mock_coord_inst = mock_coord_class.return_value
    mock_coord_inst.extract_hashes.side_effect = Exception("System Failure")

    with pytest.raises(Exception, match="System Failure"):
        check_status()
