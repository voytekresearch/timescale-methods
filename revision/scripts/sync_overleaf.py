"""Copy the manuscript figures into the Overleaf project directory.

The list of figures is read out of the manuscript itself rather than kept in
a second list here.  A hand-maintained list is how the two directories drifted
apart before: a figure was regenerated under one name, referenced under
another, and nothing complained.

Run with --check to report staleness without copying, or with --force to copy
figures the staleness test flags.  The test compares modification times, so it
cannot tell whether a library change was relevant to a given figure: adding an
unused helper trips every figure at once.  --force exists for that case and
names what it overrode, so the decision is visible rather than silent.
"""

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'figures'
TEX = ROOT.parent / 'overleaf' / 'main_claude.tex'
DST = ROOT.parent / 'overleaf' / 'revision' / 'figures'

#: The shared library every figure is drawn with.  A figure older than this
#: was made by different code, whichever script drew it.
LIBRARY = sorted((ROOT / 'tsmodes').glob('*.py'))


def referenced():
    """Figure names the manuscript actually includes, in order."""
    tex = TEX.read_text()
    names = re.findall(r'\\includesvg\[[^\]]*\]\{revision/figures/([^}]+)\}', tex)
    return list(dict.fromkeys(names))


def drawn_by():
    """Map each figure name to the script that writes it."""
    out = {}
    for script in sorted((ROOT / 'scripts').glob('*.py')):
        if script.name == 'sync_overleaf.py':
            continue
        for name in re.findall(r"P\.save\(fig, FIG / '([^']+)'\)",
                               script.read_text()):
            out[name] = script
    return out


def source_mtime(name, scripts):
    """When the code that produces this figure last changed."""
    paths = list(LIBRARY)
    if name in scripts:
        paths.append(scripts[name])
    return max(p.stat().st_mtime for p in paths)


def main():
    check_only = '--check' in sys.argv
    force = '--force' in sys.argv
    scripts = drawn_by()
    DST.mkdir(parents=True, exist_ok=True)
    names = referenced()
    stale, missing = [], []

    for name in names:
        src = SRC / name
        if not src.exists():
            missing.append(name)
            print(f'  MISSING   {name}')
            continue
        if src.stat().st_mtime < source_mtime(name, scripts):
            stale.append(name)
            script = scripts.get(name)
            where = script.name if script else 'the library'
            if force and not check_only:
                shutil.copy2(src, DST / name)
                print(f'  FORCED    {name}  (stale vs {where}, copied anyway)')
            else:
                print(f'  STALE     {name}  (older than {where})')
        elif not check_only:
            shutil.copy2(src, DST / name)
            print(f'  copied    {name}')
        else:
            print(f'  ok        {name}')

    if not check_only:
        for path in DST.glob('*'):
            if path.name not in names:
                path.unlink()
                print(f'  removed   {path.name} (not referenced by the paper)')
        for path in sorted(SRC.glob('*.svg')):
            if path.name not in names:
                print(f'  note: {path.name} is not referenced by the paper')

    if missing or (stale and not force):
        print(f'\n{len(missing)} missing, {len(stale)} stale -- '
              'regenerate before compiling.')
        return 1
    if stale:
        print(f'\n{len(stale)} figures copied despite being older than the '
              'code; verify the change did not affect them.')
        return 0
    print(f'\n{len(names)} figures up to date.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
