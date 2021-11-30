import itertools

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

NOOP = np.array([[-0.1, 0.1], [-0.1, -0.1], [0.1, -0.1], [0.1, 0.1]])
UP = np.array([[0, 0], [0.5, 0.5], [-0.5, 0.5]])
LEFT = np.array([[0, 0], [-0.5, 0.5], [-0.5, -0.5]])
RIGHT = np.array([[0, 0], [0.5, 0.5], [0.5, -0.5]])
DOWN = np.array([[0, 0], [0.5, -0.5], [-0.5, -0.5]])

ROTATION = [0, 0, 0, 90, 270]

TXT_OFFSET_VAL = 0.3
TXT_CENTERING = np.array([-0.08, -0.05])
TXT_NOOP = np.array([0.0, 0]) + TXT_CENTERING
TXT_UP = np.array([0, TXT_OFFSET_VAL]) + TXT_CENTERING
TXT_LEFT = np.array([-TXT_OFFSET_VAL, 0]) + TXT_CENTERING
TXT_RIGHT = np.array([TXT_OFFSET_VAL, 0]) + TXT_CENTERING
TXT_DOWN = np.array([0, -TXT_OFFSET_VAL]) + TXT_CENTERING

# Action scheme
ACT_OFFSETS = [
    [NOOP, TXT_NOOP],
    [UP, TXT_UP],
    [DOWN, TXT_DOWN],
    [LEFT, TXT_LEFT],
    [RIGHT, TXT_RIGHT],
]

# supports 5 actions: up, down, left, right, and noop (circle in middle)


class TabularQValuePlotter(object):
    def __init__(self, w, h, num_action=5, invert_y=True, text_values=True):
        self.w = w
        self.h = h
        self.num_action = num_action
        self.text_values = text_values
        assert num_action == 5
        self.invert_y = invert_y
        self.data = np.zeros((w, h, num_action))

    def set_value(self, x, y, action, cost):
        self.data[x, y, action] = cost

    def make_plot(self, title=None, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()

        # scale the color since abs value of LAVA is too large
        normalized_values = np.where(self.data < 0, self.data * 0.01, self.data)
        normalized_values = normalized_values - np.min(normalized_values)
        normalized_values = normalized_values / np.max(normalized_values)

        cmap = cm.get_cmap("RdYlBu")

        for x, y in itertools.product(range(self.w), range(self.h)):
            if self.invert_y:
                y = self.h - y - 1
            xy = np.array([x, y])
            xy3 = np.expand_dims(xy, axis=0)

            for a in range(self.num_action - 1, -1, -1):
                val = normalized_values[x, y, a]
                og_val = self.data[x, y, a]
                patch_offset, txt_offset = ACT_OFFSETS[a]
                rotation = ROTATION[a]
                if self.text_values:
                    xy_text = xy + txt_offset
                    ax.text(
                        xy_text[0],
                        xy_text[1],
                        "%.1f" % og_val,
                        size="xx-small",
                        rotation=rotation,
                    )
                color = cmap(val)
                ax.add_patch(Polygon(xy3 + patch_offset, True, color=color))

        ax.set_xticks(np.arange(-1, self.w + 1, 1))
        ax.set_yticks(np.arange(-1, self.h + 1, 1))
        ax.grid()
        if title is not None:
            ax.set_title(title, fontsize=23)

    def show(self):
        plt.show()


def plot_SA(gs, q_values, title=None, ax=None):
    import itertools

    plotter = TabularQValuePlotter(gs.width, gs.height, text_values=True)
    for i, (x, y, a) in enumerate(
        itertools.product(range(gs.width), range(gs.height), range(5))
    ):
        plotter.set_value(x, gs.height - y - 1, a, q_values[gs.xy_to_idx((x, y)), a])
    plotter.make_plot(title=title, ax=ax)
