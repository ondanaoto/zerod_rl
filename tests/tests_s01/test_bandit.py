from s01_bandit.bandit.fix import FixBandit


def test_bandit():
    bandit = FixBandit(rates=[0.1, 0.5, 0.9])
    assert bandit.play(0) <= 1
    assert bandit.play(1) <= 1
    assert bandit.play(2) <= 1
    assert bandit.play(0) >= 0
    assert bandit.play(1) >= 0
    assert bandit.play(2) >= 0
