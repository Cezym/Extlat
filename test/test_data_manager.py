import pytest

from src.data_manager import TransactionLoader


@pytest.fixture
def loader():
    return TransactionLoader()


def test_load_valid_file(tmp_path, loader):
    """Normal file - integer tokens only."""
    data = "1 2\n3\n4 5\n\n6"
    file_path = tmp_path / "data.txt"
    file_path.write_text(data)

    expected = [{1, 2}, {3}, {4, 5}, {6}]
    assert loader.load(file_path) == expected


def test_load_nonexistent_file_raises(loader):
    """Missing file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        loader.load("non_existent.txt")


def test_load_invalid_tokens(tmp_path, loader):
    """A line that contains a non‑integer token must raise ValueError."""
    data = "1 two\n3"
    file_path = tmp_path / "bad.txt"
    file_path.write_text(data)
    with pytest.raises(ValueError) as exc:
        loader.load(file_path)
    assert "Non‑integer token" in str(exc.value)


def test_to_vertical_basic(loader):
    """Check that the vertical format maps items to transaction IDs correctly."""
    dataset = [{1, 2}, {3}, {2}]
    vertical = loader.to_vertical(dataset)

    expected = {1: {0}, 2: {0, 2}, 3: {1}}
    assert vertical == expected


def test_to_vertical_empty(loader):
    """Empty dataset -> empty mapping."""
    assert loader.to_vertical([]) == {}
