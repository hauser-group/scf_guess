import importlib.resources

from glob import iglob


def paths() -> list[str]:
    base = importlib.resources.files(__package__) / "geometries"
    return list(iglob(f"{base}/**/*.xyz", recursive=True))
