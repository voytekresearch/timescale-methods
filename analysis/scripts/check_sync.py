#!/usr/bin/env python
"""Check that no constant is written down twice.

``dvc status`` reports staleness: which stage has to rerun because a file it
depends on changed.  It cannot report the other failure, which is two
constants in two files that have to agree and no longer do.  That one is
silent -- both stages rerun, both succeed, and the figure filters a result
table for a condition that is no longer in it.

This script reads the module-level constants of the stage scripts and the
figure scripts without importing them, and reports any name a figure defines
that a stage also defines.  That pairing is the defect, whether or not the two
values agree today, because the stage decides what goes into the result table
and the figure has to read it back on the same terms; the fix is for the
figure to import the constant from the stage.

Two stages sharing a name is not a defect: each stage owns its own settings,
and ``N_REPS`` is legitimately 200 in one and 20 in another.

    python scripts/check_sync.py        # exit 1 if any constant is duplicated

EXPECTED lists the names exempted from the rule, and is empty.
"""
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE / 'figures'

#: Names a figure may share with a stage, with the reason.  Empty is the
#: goal: a figure that needs a stage's constant imports it, and a figure
#: constant that only looks like a stage's is named apart from it.
EXPECTED = {}


def constants(path):
    """Module-level ``NAME = <literal>`` assignments, without importing."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.isupper()):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass          # computed, not a literal; nothing to drift
    return out


def main():
    stages = sorted(HERE.glob('run_0*.py'))
    figures = sorted(FIGURES.glob('fig_*.py'))
    owned = {}
    for path in stages:
        for name, value in constants(path).items():
            owned.setdefault(name, []).append((path, value))

    problems = []
    for path in figures:
        for name, value in constants(path).items():
            if name not in owned or name in EXPECTED:
                continue
            for stage, stage_value in owned[name]:
                agree = 'same value' if stage_value == value else 'DIFFERENT VALUES'
                problems.append(
                    f'{name}: {path.relative_to(HERE.parent)}={value!r} and '
                    f'{stage.relative_to(HERE.parent)}={stage_value!r} ({agree})')

    if problems:
        print('A figure restates a constant that a stage defines.  Import it '
              'from the stage instead:\n', file=sys.stderr)
        for line in problems:
            print(f'  {line}', file=sys.stderr)
        print('\nIf the duplicate is deliberate, add the name to EXPECTED in '
              f'{Path(__file__).name} with the reason.', file=sys.stderr)
        return 1

    print(f'{len(stages)} stage scripts, {len(figures)} figure scripts: '
          'no figure restates a stage constant.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
