from c01_grid_world.env import Action, GridWorld, State


def test_render():
    env = GridWorld()

    env.render()
    r, s, d = env.step(Action.UP)
    assert s.row, s.col == (1, 0)
    assert r == 0.0
    assert not d

    env.render()
    r, s, d = env.step(Action.RIGHT)  # RIGHT (failed)
    assert s.row, s.col == (1, 0)
    assert r == 0.0
    assert not d

    env.render()
    r, s, d = env.step(Action.DOWN)  # DOWN
    assert s.row, s.col == (2, 0)
    assert r == 0.0
    assert not d

    env.render()
    r, s, d = env.step(Action.LEFT)  # LEFT (failed)
    assert s.row, s.col == (2, 0)
    assert r == 0.0
    assert not d

    env.render()
    r, s, d = env.step(Action.RIGHT)
    assert s.row, s.col == (2, 1)
    assert r == 0.0
    assert not d
    env.render()


def test_bomb():
    env = GridWorld()
    env.state = State(row=1, col=2)
    env.render()
    r, s, d = env.step(Action.RIGHT)
    assert s.row, s.col == (1, 3)
    assert r == -1.0
    assert not d


def test_goal():
    env = GridWorld()
    env.state = State(row=0, col=2)
    env.render()
    r, s, d = env.step(Action.RIGHT)
    assert s.row == 0
    assert s.col == 3
    assert r == 1.0
    assert d
