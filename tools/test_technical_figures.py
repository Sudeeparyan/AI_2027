"""Check the visible topology of repaired teaching diagrams.

Run with .venv/Scripts/python.exe tools/test_technical_figures.py.
The tests inspect Matplotlib artists before saving, with no model downloads.
They check reachable blocks and residual junctions, not a coordinate snapshot.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from collections import defaultdict, deque
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "curriculum" / "figures"))

import _style  # noqa: E402,F401 - configures the noninteractive backend
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402


def point(xy):
    return tuple(round(float(value), 7) for value in xy)


def on_segment(p, a, b, tolerance=1e-6):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return None
    fraction = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2
    error = abs((p[0] - a[0]) * dy - (p[1] - a[1]) * dx)
    return fraction if -tolerance <= fraction <= 1 + tolerance and error <= tolerance else None


class Diagram:
    """Directed graph derived from box boundaries, arrows and routed lines."""

    def __init__(self, ax, dashed=None):
        self.ax = ax
        self.shapes = [artist for artist in ax.patches if isinstance(artist, (FancyBboxPatch, Circle))]
        self.labels = {}
        for index, shape in enumerate(self.shapes):
            if isinstance(shape, Circle):
                centre = shape.center
            else:
                centre = (shape.get_x() + shape.get_width() / 2, shape.get_y() + shape.get_height() / 2)
            texts = [label for label in ax.texts if point(label.get_position()) == point(centre)]
            self.labels[index] = " ".join(texts[-1].get_text().split()) if texts else ""
        segments = []
        for artist in ax.texts:
            if isinstance(artist, Annotation) and artist.arrow_patch is not None:
                self._require_data_coordinates(artist)
                if dashed is not None and ((artist.arrowprops or {}).get("linestyle") == "--") != dashed:
                    continue
                segments.append((point(artist.xyann), point(artist.xy)))
        for line in ax.lines:
            if line.get_linestyle() in ("None", "", " "):
                continue
            if dashed is not None and (line.get_linestyle() == "--") != dashed:
                continue
            xs, ys = line.get_data()
            route = [point(xy) for xy in zip(xs, ys)]
            segments.extend(zip(route, route[1:]))
        points = {p for segment in segments for p in segment}
        self.edges = defaultdict(set)
        for start, finish in segments:
            stops = sorted((fraction, p) for p in points if (fraction := on_segment(p, start, finish)) is not None)
            for (_, a), (_, b) in zip(stops, stops[1:]):
                self.edges[self.node(a)].add(self.node(b))

    @staticmethod
    def _require_data_coordinates(annotation):
        if annotation.xycoords != "data" or annotation.anncoords != "data":
            raise AssertionError("Diagram check needs data-coordinate arrows")

    def node(self, p):
        tolerance = 1e-5
        for index, shape in enumerate(self.shapes):
            if isinstance(shape, Circle):
                x, y = shape.center
                inside = (p[0] - x) ** 2 + (p[1] - y) ** 2 <= (shape.radius + tolerance) ** 2
            else:
                inside = (shape.get_x() - tolerance <= p[0] <= shape.get_x() + shape.get_width() + tolerance
                          and shape.get_y() - tolerance <= p[1] <= shape.get_y() + shape.get_height() + tolerance)
            if inside:
                return ("shape", index)
        return ("point", p)

    def named(self, text):
        matches = self.named_all(text)
        if len(matches) != 1:
            raise AssertionError(f"Expected one block labelled {text!r}, found {len(matches)}")
        return matches[0]

    def named_all(self, text):
        return [("shape", index) for index, label in self.labels.items() if label == text]

    def external_input(self, text, target):
        """Find the input arrow adjacent to an authored external-data label."""
        labels = [t for t in self.ax.texts if " ".join(t.get_text().split()) == text]
        if len(labels) != 1:
            raise AssertionError(f"Expected one external input label {text!r}")
        renderer = self.ax.figure.canvas.get_renderer()
        bounds = labels[0].get_window_extent(renderer)
        candidates = []
        for arrow in self.ax.texts:
            if isinstance(arrow, Annotation) and arrow.arrow_patch is not None:
                self._require_data_coordinates(arrow)
                start = point(arrow.xyann)
                if self.node(point(arrow.xy)) != target or self.node(start)[0] != "point":
                    continue
                x, y = self.ax.transData.transform(start)
                # The label sits just outside the incoming arrow, not on an arbitrary node.
                dx = max(bounds.x0 - x, 0, x - bounds.x1)
                dy = max(bounds.y0 - y, 0, y - bounds.y1)
                if max(dx, dy) < 2 * bounds.height:
                    candidates.append(self.node(start))
        if len(candidates) != 1:
            raise AssertionError(f"Expected one labelled input arrow {text!r}, found {len(candidates)}")
        return candidates[0]

    def reachable(self, start, target, avoiding=()):
        queue, seen = deque([start]), set()
        forbidden = set(avoiding)
        while queue:
            current = queue.popleft()
            if current in forbidden or current in seen:
                continue
            if current == target:
                return True
            seen.add(current)
            queue.extend(self.edges[current])
        return False


def capture_figures(module, name, folder):
    """Keep real save dispatch and the font normalization used by scene exports."""
    captured = []

    def raster(fig, *_args, **_kwargs):
        captured.append(fig)

    def editable(fig, *_args, **_kwargs):
        for label in [*fig.texts, *(t for ax in fig.axes for t in
                       [*ax.texts, ax.title, ax._left_title, ax._right_title])]:
            label.set_fontfamily("Arial")
        captured.append(fig)

    with ExitStack() as stack:
        # Week wrappers choose raster for charts and editable scenes for schematics.
        # Patching the wrapper itself would silently skip its Arial export behavior.
        if hasattr(module, "save_raster"):
            stack.enter_context(patch.object(module, "save_raster", raster))
        elif hasattr(module, "save"):
            stack.enter_context(patch.object(module, "save", raster))
        if hasattr(module, "save_editable_scene"):
            stack.enter_context(patch.object(module, "save_editable_scene", editable))
        module.FIGURES[name](folder / f"{name}.png")
    return captured


class TechnicalFiguresTests(unittest.TestCase):
    def draw(self, week, figure):
        module = importlib.import_module(f"week_{week:02d}")
        with tempfile.TemporaryDirectory(prefix="figure-test-") as folder:
            captured = capture_figures(module, figure, Path(folder))
        self.assertEqual(len(captured), 1)
        captured[0].canvas.draw()
        self.addCleanup(plt.close, captured[0])
        return captured[0]

    def test_unet_has_complete_main_and_skip_paths(self):
        ax = self.draw(4, "unet_dit").axes[0]
        diagram = Diagram(ax)
        labels = ["Encode 28 × 28", "Shrink 14 × 14", "Bottleneck 7 × 7",
                  "Expand + join 14 × 14", "Expand + join 28 × 28", "Predict noise 28 × 28"]
        nodes = list(map(diagram.named, labels))
        start, bottleneck, finish = nodes[0], nodes[2], nodes[-1]
        for a, b in zip(nodes, nodes[1:]):
            self.assertTrue(diagram.reachable(a, b), "Every multiscale operation must connect in order")
        self.assertTrue(diagram.reachable(start, bottleneck), "Encoder must reach bottleneck")
        self.assertTrue(diagram.reachable(bottleneck, finish), "Decoder must reach noise output")
        self.assertTrue(diagram.reachable(nodes[0], nodes[4], avoiding=[bottleneck]), "28×28 skip must reach its matching join")
        self.assertTrue(diagram.reachable(nodes[1], nodes[3], avoiding=[bottleneck]), "14×14 skip must reach its matching join")

    def test_dit_connects_all_blocks_and_conditioning(self):
        diagram = Diagram(self.draw(4, "unet_dit").axes[0])
        blocks = [diagram.named(label) for label in ["Embed patches", "Transformer blocks", "Project + reshape"]]
        for a, b in zip(blocks, blocks[1:]):
            self.assertTrue(diagram.reachable(a, b), "DiT embedding, transformer stack and reshape must connect in order")
        conditioning = diagram.external_input("time + condition", blocks[1])
        self.assertIn(blocks[1], diagram.edges[conditioning], "The transformer stack needs its separate conditioning input")

    def test_transformer_residuals_rejoin_visible_add_nodes(self):
        diagram = Diagram(self.draw(5, "block").axes[0])
        adds = diagram.named_all("Add")
        self.assertEqual(len(adds), 2, "Both residual paths need an addition junction")
        adds.sort(key=lambda node: diagram.shapes[node[1]].get_x())
        norms = sorted(diagram.named_all("Normalise"), key=lambda node: diagram.shapes[node[1]].get_x())
        self.assertEqual(len(norms), 2, "Both pre-norm branches need their own normalization")
        start, attention, mlp = map(diagram.named, ["Input x", "Attend", "MLP"])
        self.assertTrue(diagram.reachable(start, attention), "Input must enter the attention branch")
        self.assertTrue(diagram.reachable(start, adds[0], avoiding=[norms[0], attention]), "First residual must bypass normalization and attention")
        self.assertTrue(diagram.reachable(attention, adds[0]), "Attention output must enter the first add")
        self.assertTrue(diagram.reachable(adds[0], mlp), "First addition must enter the MLP branch")
        self.assertTrue(diagram.reachable(adds[0], adds[1], avoiding=[norms[1], mlp]), "Second residual must bypass normalization and MLP")
        self.assertTrue(diagram.reachable(mlp, adds[1]), "MLP output must enter the second add")

    def test_agent_validates_records_and_repeats_observations(self):
        ax = self.draw(12, "agent_loop").axes[0]
        diagram = Diagram(ax, dashed=False)
        names = ["Read", "Propose", "Parse", "Decide", "Execute", "Record"]
        nodes = [diagram.named(name) for name in names]
        for a, b in zip(nodes, nodes[1:]):
            self.assertTrue(diagram.reachable(a, b), f"Missing agent transition: {a} → {b}")
        self.assertFalse(diagram.reachable(nodes[1], nodes[4], avoiding=[nodes[2]]), "Execution must not bypass parsing")
        self.assertFalse(diagram.reachable(nodes[1], nodes[4], avoiding=[nodes[3]]), "Execution must not bypass the decision")
        self.assertTrue(diagram.reachable(nodes[3], diagram.named("Return answer"), avoiding=[nodes[4], nodes[5]]), "Final-answer branch must skip tool execution")
        self.assertIn("Execute validates; refusals skip the tool.", [t.get_text() for t in ax.texts])
        feedback = Diagram(ax, dashed=True)
        self.assertTrue(feedback.reachable(nodes[5], nodes[1]), "A dashed observation route must connect Record to Propose; the legend alone is insufficient")

    def test_repaired_box_labels_fit(self):
        for week, name in [(1, "families"), (4, "unet_dit"), (5, "block"), (5, "multihead"), (8, "decision"), (9, "architectures"), (12, "agent_loop")]:
            fig = self.draw(week, name)
            renderer = fig.canvas.get_renderer()
            checked = 0
            for ax in fig.axes:
                for shape in (item for item in ax.patches if isinstance(item, FancyBboxPatch)):
                    centre = point((shape.get_x() + shape.get_width() / 2, shape.get_y() + shape.get_height() / 2))
                    labels = [label for label in ax.texts if point(label.get_position()) == centre]
                    if not labels:
                        continue
                    if not labels[-1].get_text().strip():
                        continue
                    checked += 1
                    bounds = labels[-1].get_window_extent(renderer)
                    lo, hi = ax.transData.transform([(shape.get_x(), shape.get_y()),
                                                    (shape.get_x() + shape.get_width(), shape.get_y() + shape.get_height())])
                    self.assertLessEqual(bounds.width, hi[0] - lo[0] - 3, f"{week}/{name}: text exceeds box width")
                    self.assertLessEqual(bounds.height, hi[1] - lo[1] - 3, f"{week}/{name}: text exceeds box height")
            self.assertGreater(checked, 0, f"{week}/{name}: no actual block labels were checked")

    def test_fusion_tokens_all_reach_joint_transformer(self):
        diagram = Diagram(self.draw(9, "architectures").axes[0])
        image, text = map(diagram.named, ["Image tokens", "Text tokens"])
        joint = diagram.named("Joint encoder")
        self.assertTrue(diagram.reachable(image, joint, avoiding=[text]), "Image input must independently enter the joint encoder")
        self.assertTrue(diagram.reachable(text, joint, avoiding=[image]), "Text input must independently enter the joint encoder")
        self.assertFalse(diagram.reachable(image, text), "Fusion must not draw image data flowing through text input")
        self.assertFalse(diagram.reachable(text, image), "Fusion must not draw text data flowing through image input")

    def test_dual_encoders_both_feed_comparison_independently(self):
        diagram = Diagram(self.draw(9, "architectures").axes[0])
        image = diagram.named("Image")
        text = diagram.named("Text")
        comparison = diagram.named("Compare")
        self.assertTrue(diagram.reachable(image, comparison, avoiding=[text]), "Image vector must enter comparison")
        self.assertTrue(diagram.reachable(text, comparison, avoiding=[image]), "Text vector must enter comparison")
        self.assertFalse(diagram.reachable(image, text), "Dual encoders do not process one another's input")
        self.assertFalse(diagram.reachable(text, image), "Dual encoders do not process one another's input")

    def test_adaptation_decision_branches_are_reachable(self):
        ax = self.draw(8, "decision").axes[0]
        diagram = Diagram(ax)
        start = diagram.named("Prompt + test set")
        meets_bar = diagram.named("Meets the bar?")
        knowledge = diagram.named("Missing facts?")
        behaviour = diagram.named("Behaviour or cost gap?")
        self.assertTrue(diagram.reachable(start, meets_bar))
        self.assertTrue(diagram.reachable(meets_bar, diagram.named("Ship + monitor"), avoiding=[knowledge]))
        self.assertTrue(diagram.reachable(meets_bar, knowledge))
        self.assertTrue(diagram.reachable(knowledge, diagram.named("Retrieval (week 12)"), avoiding=[behaviour]))
        self.assertTrue(diagram.reachable(knowledge, behaviour))
        self.assertTrue(diagram.reachable(behaviour, diagram.named("LoRA fine-tune")))
        self.assertTrue(diagram.reachable(behaviour, diagram.named("Review task and data")))
        labels = [text.get_text() for text in ax.texts]
        self.assertGreaterEqual(labels.count("yes"), 3)
        self.assertGreaterEqual(labels.count("no"), 3)

    def test_every_drawn_figure_passes_layout_lint(self):
        import lint_figures
        with tempfile.TemporaryDirectory(prefix="figure-lint-") as folder:
            for week in range(1, 13):
                module = importlib.import_module(f"week_{week:02d}")
                for name in module.FIGURES:
                    for fig in capture_figures(module, name, Path(folder)):
                        issues = lint_figures.lint_figure(fig)
                        plt.close(fig)
                        with self.subTest(week=week, figure=name):
                            self.assertEqual(issues, [], f"week {week} {name}")

    def test_family_box_borders_are_not_clipped(self):
        fig = self.draw(1, "families")
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            for shape in (item for item in ax.patches if isinstance(item, FancyBboxPatch)):
                bounds = shape.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, ax.bbox.x0, "Left box border falls outside its axes")
                self.assertLessEqual(bounds.x1, ax.bbox.x1, "Right box border falls outside its axes")


if __name__ == "__main__":
    unittest.main()
