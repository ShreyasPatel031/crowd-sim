from simulator.agent import validate_public_url
import pytest


def test_rejects_local_urls():
    with pytest.raises(ValueError):
        validate_public_url("http://localhost:3000")
    with pytest.raises(ValueError):
        validate_public_url("file:///etc/passwd")


def test_accepts_https_and_adds_scheme():
    assert validate_public_url("https://books.toscrape.com/").startswith("https://")
    assert validate_public_url("books.toscrape.com") == "https://books.toscrape.com"
