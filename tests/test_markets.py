import pytest
from fakes import event

from smcal.markets import Excluded, is_match_event, parse_event, settlement_rule


def test_parse_event_reads_tokens_kickoff_and_resolution() -> None:
    m = parse_event(event(winner="A"))
    assert (m.home, m.away) == ("Spain", "Argentina")
    assert m.kickoff.isoformat() == "2026-07-19T19:00:00+00:00"
    assert m.tokens == {
        "H": "fifwc-spa-arg-2026-07-19-H",
        "D": "fifwc-spa-arg-2026-07-19-D",
        "A": "fifwc-spa-arg-2026-07-19-A",
    }
    assert m.resolved == "A"
    assert m.settles_on == "90min"


def test_market_order_in_the_event_does_not_matter() -> None:
    e = event(winner="H")
    e["markets"] = e["markets"][::-1]
    assert parse_event(e).tokens["H"].endswith("-H")


def test_rule_not_stated_is_excluded_not_guessed() -> None:
    with pytest.raises(Excluded, match="settlement rule"):
        settlement_rule("Resolves to the team that advances to the next round.")


def test_wrapped_description_still_matches() -> None:
    text = "outcome within the first 90 minutes of regular\n play plus stoppage time."
    assert settlement_rule(text) == "90min"


def test_two_yes_resolutions_are_excluded() -> None:
    e = event(winner="H")
    e["markets"][1]["outcomePrices"] = '["1", "0"]'
    with pytest.raises(Excluded, match="2 markets resolved Yes"):
        parse_event(e)


def test_unresolved_market_is_excluded() -> None:
    e = event(winner="H")
    e["markets"][0]["outcomePrices"] = '["0.6", "0.4"]'
    with pytest.raises(Excluded, match="not resolved"):
        parse_event(e)


def test_side_events_are_not_match_events() -> None:
    assert is_match_event({"slug": "fifwc-esp-arg-2026-07-19"})
    assert not is_match_event({"slug": "fifwc-esp-arg-2026-07-19-exact-score"})
