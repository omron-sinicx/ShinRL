import os

from shinrl import Scalars


def test_scalars(tmpdir):
    scalars = Scalars()
    for i in range(3):
        scalars.add("test1", i, i)
        scalars.add("test2", i, 10 * i)
    assert scalars["test1"] == {"x": [0, 1, 2], "y": [0, 1, 2]}
    assert scalars["test2"] == {"x": [0, 1, 2], "y": [0, 10, 20]}

    # save & load
    path = tmpdir.mkdir("tmp")
    path = os.path.join(path, "tmp.csv")
    scalars.dump_xy("test1", path)
    assert "test1" not in scalars
    scalars.load_xy("test1", path)
    assert scalars["test1"] == {"x": [0, 1, 2], "y": [0, 1, 2]}
