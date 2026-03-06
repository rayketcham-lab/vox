"""Tests for contacts / address book."""


import pytest

from vox.contacts import (
    _fuzzy_score,
    add_contact,
    list_all,
    lookup,
    lookup_group,
    remove_contact,
    resolve_email,
)


@pytest.fixture(autouse=True)
def _temp_contacts(tmp_path, monkeypatch):
    monkeypatch.setattr("vox.contacts._CONTACTS_FILE", tmp_path / "contacts.json")


def test_add_and_list():
    add_contact("John Doe", email="john@example.com", phone="555-1234")
    contacts = list_all()
    assert len(contacts) == 1
    assert contacts[0]["name"] == "John Doe"
    assert contacts[0]["email"] == "john@example.com"


def test_add_multiple():
    add_contact("Alice", email="alice@test.com")
    add_contact("Bob", email="bob@test.com")
    assert len(list_all()) == 2


def test_auto_increment_id():
    c1 = add_contact("First")
    c2 = add_contact("Second")
    assert c2["id"] == c1["id"] + 1


def test_lookup_exact_name():
    add_contact("John Smith", email="john@test.com")
    add_contact("Jane Doe", email="jane@test.com")
    results = lookup("John")
    assert len(results) >= 1
    assert results[0]["name"] == "John Smith"


def test_lookup_fuzzy():
    add_contact("Michael Johnson", email="mike@test.com")
    # Default threshold 0.5 catches Mike→Michael (score ~0.55)
    results = lookup("Mike")
    assert len(results) >= 1


def test_lookup_no_match():
    add_contact("Alice", email="alice@test.com")
    results = lookup("Zorro", threshold=0.7)
    assert len(results) == 0


def test_lookup_group():
    add_contact("Alice", email="alice@test.com", tags=["team"])
    add_contact("Bob", email="bob@test.com", tags=["team"])
    add_contact("Charlie", email="charlie@test.com", tags=["family"])
    group = lookup_group("team")
    assert len(group) == 2
    names = {c["name"] for c in group}
    assert names == {"Alice", "Bob"}


def test_resolve_email_by_name():
    add_contact("John Doe", email="john@example.com")
    emails = resolve_email("John")
    assert emails == ["john@example.com"]


def test_resolve_email_by_group():
    add_contact("Alice", email="alice@test.com", tags=["team"])
    add_contact("Bob", email="bob@test.com", tags=["team"])
    emails = resolve_email("team")
    assert set(emails) == {"alice@test.com", "bob@test.com"}


def test_resolve_email_no_match():
    emails = resolve_email("nobody")
    assert emails == []


def test_remove_contact():
    c = add_contact("To Delete", email="del@test.com")
    assert remove_contact(c["id"]) is True
    assert len(list_all()) == 0


def test_remove_nonexistent():
    assert remove_contact(999) is False


def test_fuzzy_score_exact():
    assert _fuzzy_score("john", "John") > 0.9


def test_fuzzy_score_partial():
    # Mike → Michael has ~0.55 similarity (enough for default threshold)
    assert _fuzzy_score("mike", "Michael") > 0.5


def test_fuzzy_score_unrelated():
    assert _fuzzy_score("xyz", "Alice") < 0.5


def test_empty_list():
    assert list_all() == []
