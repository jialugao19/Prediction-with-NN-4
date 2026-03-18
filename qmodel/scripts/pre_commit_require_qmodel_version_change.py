"""Fail commit when `qmodel/` changes without `qmodel/version.py` change."""

import subprocess


def main() -> None:
    """Enforce that staged `qmodel/` changes include `qmodel/version.py`."""

    # Collect staged file paths for this commit.
    staged_paths_text = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"], text=True
    )
    staged_paths = [p for p in staged_paths_text.splitlines() if p]

    # Detect whether any staged path lives under `qmodel/`.
    has_qmodel_change = any(p.startswith("qmodel/") for p in staged_paths)

    # Require `qmodel/version.py` to be staged when `qmodel/` changes.
    has_version_change = "qmodel/version.py" in staged_paths
    if has_qmodel_change and not has_version_change:
        raise SystemExit(
            "qmodel/ has staged changes, but qmodel/version.py is not staged; "
            "bump the version and stage qmodel/version.py."
        )


if __name__ == "__main__":
    main()
