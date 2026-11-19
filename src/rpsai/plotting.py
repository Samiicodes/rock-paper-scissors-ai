'''Charts for the evaluation report.'''

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.text import Annotation
from matplotlib.ticker import FixedLocator, PercentFormatter
from matplotlib.typing import RcKeyType

from rpsai.config import GESTURE_CLASSES
from rpsai.gesture_pipeline import GestureResults
from rpsai.predictor_pipeline import EXPERT_LABELS, PredictorResults

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
# Reference strategies and baselines stay neutral so that the AI under test stands out
NEUTRAL = '#a8a69f'
NEUTRAL_LIGHT = '#d9d8d2'
# Categorical slots in fixed order; the first three stay distinguishable for colour-blind readers
CATEGORICAL = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')
SEQUENTIAL_BLUE = ('#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b')
HIGHLIGHT = CATEGORICAL[0]
CHANCE = 1 / 3
DPI = 200

# Rock, paper, scissors, other, and asked again
DECISION_COLOURS = (CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2], NEUTRAL, NEUTRAL_LIGHT)
DECISION_NAMES = ('Rock', 'Paper', 'Scissors', 'Not a move', 'Asked again')
READER_COLOURS = (NEUTRAL, CATEGORICAL[1], HIGHLIGHT)

_STYLE: dict[RcKeyType, Any] = {
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.titlelocation': 'left',
    'axes.titlecolor': INK,
    'axes.labelcolor': INK_SECONDARY,
    'axes.edgecolor': AXIS,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': AXIS,
    'ytick.color': AXIS,
    'xtick.labelcolor': INK_SECONDARY,
    'ytick.labelcolor': INK_SECONDARY,
    'grid.color': GRID,
    'grid.linewidth': 0.6,
    'grid.linestyle': '-',
    'legend.frameon': False,
    'legend.labelcolor': INK_SECONDARY,
    'lines.linewidth': 1.8,
    'lines.solid_capstyle': 'round',
    'lines.solid_joinstyle': 'round',
}


def plot_strategies(results: PredictorResults, path: Path) -> Path:
    '''Win rate of every AI strategy against the test players, with 95% intervals.'''
    strategies = results.strategies
    names = [summary.name for summary in strategies]
    values = np.array([summary.win_rate.value for summary in strategies])
    errors = np.array(
        [
            [s.win_rate.value - s.win_rate.low for s in strategies],
            [s.win_rate.high - s.win_rate.value for s in strategies],
        ]
    )
    colours = [HIGHLIGHT if index == len(names) - 1 else NEUTRAL for index in range(len(names))]
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 0.36 * len(names) + 1.0))
        positions = np.arange(len(names))[::-1]
        axis.barh(positions, values, height=0.62, color=colours)
        axis.errorbar(
            values,
            positions,
            xerr=errors,
            fmt='none',
            ecolor=INK_SECONDARY,
            elinewidth=0.8,
            capsize=2,
        )
        for position, value, high in zip(positions, values, values + errors[1], strict=True):
            axis.text(
                high + 0.006, position, f'{100 * value:.1f}%', va='center', color=INK, fontsize=8
            )
        _chance_line(axis, vertical=True)
        axis.set_yticks(positions, names)
        axis.tick_params(axis='y', length=0)
        axis.set_xlim(0, max(0.5, float((values + errors[1]).max()) + 0.05))
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.grid(axis='x')
        axis.set_axisbelow(True)
        axis.set_xlabel('Rounds won by the AI against players it had not seen')
        figure.tight_layout()
        return _save(figure, path)


def plot_learning(results: PredictorResults, path: Path, block: int = 10) -> Path:
    '''Win rate in each block of rounds: the combined AI against the best single predictor.'''
    combined = results.strategies[-1]
    singles = [s for s in results.strategies if s.name in EXPERT_LABELS.values()]
    best_single = max(singles, key=lambda s: s.win_rate.value)
    shown = [(combined, HIGHLIGHT), (best_single, CATEGORICAL[1])]
    blocks = len(combined.win_rate_by_block)
    labels = [
        f'{start + 1}-{min(start + block, results.rounds)}'
        for start in range(0, results.rounds, block)
    ]
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 3.4))
        positions = np.arange(blocks)
        end_labels = []
        for summary, colour in shown:
            axis.plot(
                positions,
                summary.win_rate_by_block,
                color=colour,
                marker='o',
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
            )
            end_labels.append(
                axis.annotate(
                    summary.name,
                    (positions[-1], summary.win_rate_by_block[-1]),
                    xytext=(8, 0),
                    textcoords='offset points',
                    va='center',
                    color=INK_SECONDARY,
                    fontsize=8,
                )
            )
        _chance_line(axis, vertical=False)
        axis.set_xticks(positions, labels)
        axis.set_xlim(-0.2, blocks - 1 + 1.6)
        axis.set_ylim(0.25, 0.55)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.grid(axis='y')
        axis.set_axisbelow(True)
        axis.set_xlabel('Rounds')
        axis.set_ylabel('Rounds won by the AI')
        figure.tight_layout()
        _separate_labels(figure, axis, end_labels)
        return _save(figure, path)


def plot_expert_weights(results: PredictorResults, path: Path) -> Path:
    '''How the combiner shares its trust between the predictors as a game goes on.'''
    weights = np.array(results.weights_by_round).T
    rounds = np.arange(1, weights.shape[1] + 1)
    names = [EXPERT_LABELS[name] for name in results.expert_names]
    # The no-pattern predictor is a reference, so it takes the neutral colour
    patterned = [name for name in results.expert_names if name != 'uniform']
    palette = dict(zip(patterned, CATEGORICAL[: len(patterned)], strict=True))
    colours = [palette.get(name, NEUTRAL) for name in results.expert_names]
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 3.6))
        axis.stackplot(
            rounds, weights, colors=colours, labels=names, edgecolor=SURFACE, linewidth=0.6
        )
        axis.set_xlim(1, rounds[-1])
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_xlabel('Round')
        axis.set_ylabel('Share of the combined forecast')
        axis.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8, reverse=True)
        figure.tight_layout()
        return _save(figure, path)


def plot_readers(results: GestureResults, path: Path) -> Path:
    '''The three ways of reading hands: moves read correctly and non-moves played as moves.'''
    readers = results.readers
    panels = (
        ('Moves read correctly', [r.metrics.moves_read_correctly for r in readers], (0.0, 1.0)),
        (
            'Non-moves played as a move',
            [r.metrics.others_played_as_moves for r in readers],
            (0.0, 0.45),
        ),
    )
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.1), sharey=True, layout='constrained')
        positions = np.arange(len(readers))[::-1]
        for axis, (title, values, limits) in zip(axes, panels, strict=True):
            axis.barh(positions, values, height=0.6, color=READER_COLOURS)
            for position, value in zip(positions, values, strict=True):
                axis.text(
                    value + (limits[1] - limits[0]) * 0.02,
                    position,
                    f'{100 * value:.1f}%',
                    va='center',
                    color=INK,
                    fontsize=8,
                )
            axis.set_title(title)
            axis.set_xlim(*limits)
            axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            axis.grid(axis='x')
            axis.set_axisbelow(True)
            axis.tick_params(axis='y', length=0)
        axes[0].set_yticks(positions, [reader.name for reader in readers])
        return _save(figure, path)


def plot_stretch(results: GestureResults, path: Path) -> Path:
    '''Accuracy on test hands stretched vertically, standing in for unknown image shapes.'''
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 3.2))
        for reader, colour in zip(results.readers, READER_COLOURS, strict=True):
            axis.plot(
                results.stretches,
                reader.stretch_accuracy,
                color=colour,
                marker='o',
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
            )
            axis.annotate(
                reader.name,
                (results.stretches[-1], reader.stretch_accuracy[-1]),
                xytext=(8, 0),
                textcoords='offset points',
                va='center',
                color=INK_SECONDARY,
                fontsize=8,
            )
        axis.set_xscale('log')
        axis.xaxis.set_major_locator(FixedLocator(results.stretches))
        axis.set_xticks(results.stretches, [f'{value:g}' for value in results.stretches])
        axis.minorticks_off()
        axis.set_xlim(results.stretches[0] * 0.9, results.stretches[-1] * 1.9)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.grid(axis='y')
        axis.set_axisbelow(True)
        axis.set_xlabel('Vertical stretch applied to the test hands')
        axis.set_ylabel('Accuracy')
        figure.tight_layout()
        return _save(figure, path)


def plot_labels(results: GestureResults, path: Path) -> Path:
    '''What the game's reader makes of each HaGRID gesture, grouped by the class it belongs to.'''
    readings = results.labels
    names = [f'{r.label.replace("_", " ")}' for r in readings]
    shares = np.array([r.shares for r in readings])
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 0.26 * len(readings) + 1.2))
        positions = np.arange(len(readings))[::-1]
        left = np.zeros(len(readings))
        for column, (colour, name) in enumerate(zip(DECISION_COLOURS, DECISION_NAMES, strict=True)):
            axis.barh(
                positions,
                shares[:, column],
                left=left,
                height=0.72,
                color=colour,
                label=name,
                edgecolor=SURFACE,
                linewidth=0.6,
            )
            left += shares[:, column]
        tick_labels = [
            f'{name}  ({reading.gesture_class})'
            for name, reading in zip(names, readings, strict=True)
        ]
        axis.set_yticks(positions, tick_labels)
        axis.tick_params(axis='y', length=0)
        axis.set_xlim(0, 1)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_xlabel('Share of test hands')
        axis.legend(
            loc='lower center', bbox_to_anchor=(0.5, 1.0), ncols=len(DECISION_NAMES), fontsize=8
        )
        for spine in ('left', 'bottom'):
            axis.spines[spine].set_visible(False)
        figure.tight_layout()
        return _save(figure, path)


def plot_confusion(confusion: Sequence[Sequence[int]], path: Path) -> Path:
    '''True class against the reader's most likely class, as row percentages.'''
    counts = np.array(confusion, dtype=float)
    percentages = 100 * counts / counts.sum(axis=1, keepdims=True)
    names = [name.capitalize() if name != 'other' else 'Not a move' for name in GESTURE_CLASSES]
    colour_map = LinearSegmentedColormap.from_list('sequential_blue', ['#ffffff', *SEQUENTIAL_BLUE])
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(4.4, 3.8))
        axis.imshow(percentages, cmap=colour_map, vmin=0, vmax=100)
        for row in range(len(names)):
            for column in range(len(names)):
                value = percentages[row, column]
                if value >= 0.05:
                    axis.text(
                        column,
                        row,
                        f'{value:.1f}',
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='#ffffff' if value >= 50 else INK,
                    )
        axis.set_xticks(range(len(names)), names)
        axis.set_yticks(range(len(names)), names)
        axis.tick_params(length=0)
        axis.set_xlabel('Read as')
        axis.set_ylabel('Shown')
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.tight_layout()
        return _save(figure, path)


def _separate_labels(
    figure: Figure, axis: Axes, labels: Sequence[Annotation], gap: float = 11.0
) -> None:
    '''Move end labels apart vertically so that they are at least gap points apart.

    Call it after the layout is final, because it works in display coordinates.
    '''
    figure.canvas.draw()
    points_per_pixel = 72 / figure.dpi
    anchored = [
        (axis.transData.transform(label.xy)[1] * points_per_pixel, label) for label in labels
    ]
    previous = -np.inf
    for anchor, label in sorted(anchored, key=lambda item: item[0]):
        placed = max(anchor, previous + gap)
        label.xyann = (label.xyann[0], placed - anchor)
        previous = placed


def _chance_line(axis: Axes, vertical: bool) -> None:
    '''A thin reference line at one third, the win rate of random play.'''
    if vertical:
        axis.axvline(CHANCE, color=INK_MUTED, linewidth=0.9, linestyle=(0, (3, 3)))
        axis.annotate(
            'Chance',
            (CHANCE, 1.0),
            xycoords=('data', 'axes fraction'),
            xytext=(3, -2),
            textcoords='offset points',
            va='top',
            color=INK_MUTED,
            fontsize=8,
        )
    else:
        axis.axhline(CHANCE, color=INK_MUTED, linewidth=0.9, linestyle=(0, (3, 3)))
        axis.annotate(
            'Chance',
            (0.0, CHANCE),
            xycoords=('axes fraction', 'data'),
            xytext=(2, 3),
            textcoords='offset points',
            color=INK_MUTED,
            fontsize=8,
        )


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return path
