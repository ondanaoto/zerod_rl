from c01_grid_world.env import GridWorld


def test_render():
    env = GridWorld()

    env.render()
    r, s, d = env.step(0)  # UP
    assert s == (1, 0)
    assert r == 0.0
    assert not d

    env.render()
    r, s, d = env.step(3)  # RIGHT (failed)
    assert s == (1, 0)
    assert r == 0.0
    assert not d

    env.render()
    r, s, d = env.step(1)  # DOWN
    assert s == (2, 0)
    assert r == 0.0
    assert not d

    env.render()
    r, s, d = env.step(2)  # LEFT (failed)
    assert s == (2, 0)
    assert r == 0.0
    assert not d

    env.render()
    r, s, d = env.step(3)  # RIGHT
    assert s == (2, 1)
    assert r == 0.0
    assert not d
    env.render()


def test_bomb():
    env = GridWorld()
    env.state = (1, 2)
    env.render()
    r, s, d = env.step(3)  # RIGHT
    assert s == (1, 3)
    assert r == -1.0
    assert not d


def test_goal():
    env = GridWorld()
    env.state = (0, 2)
    env.render()
    r, s, d = env.step(3)  # RIGHT
    assert s == (0, 3)
    assert r == 1.0
    assert d
