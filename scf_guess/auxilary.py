import inspect
import psi4

from contextlib import contextmanager
from filecache import filecache


@contextmanager
def clean_context():
    with psi4.driver.p4util.hold_options_state():
        try:
            psi4.core.clean_options()
            psi4.core.clean()

            yield
        finally:
            psi4.core.clean_options()
            psi4.core.clean()


class Cacheable:
    def __init__(self, instance):
        self._class = instance.__class__
        self._instance = instance

    @property
    def instance(self):
        return self._instance

    def __getattr__(self, attribute):
        return getattr(self._instance, attribute)

    def __getstate__(self):
        if issubclass(self._class, psi4.core.Molecule):
            return self._class, (self._instance.to_string("xyz+"), self._instance.name())
        elif issubclass(self._class, psi4.core.Wavefunction):
            return self._class, self._instance.to_file()
        elif issubclass(self._class, psi4.core.Matrix):
            return self._class, self._instance.to_serial()
        else:
            return self._class, self._instance

    def __setstate__(self, state):
        self._class, serialized = state

        if issubclass(self._class, psi4.core.Molecule):
            self._instance = psi4.core.Molecule.from_string(serialized[0], name=serialized[1], dtype="xyz+")
        elif issubclass(self._class, psi4.core.Wavefunction):
            self._instance = psi4.core.Wavefunction.from_file(serialized)
        elif issubclass(self._class, psi4.core.Matrix):
            self._instance = psi4.core.Matrix.from_serial(serialized)
        else:
            self._instance = serialized


def file_cache(expires: int = 60 * 60 * 24 * 365):
    def decorator(function):
        signature = inspect.signature(function)

        @filecache(seconds_of_validity=expires)
        def caller(cacheables):
            originals = {key: cacheable.instance for key, cacheable in cacheables.items()}
            return Cacheable(function(**originals))

        def wrapper(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()

            cacheables = {key: Cacheable(value) for key, value in arguments.arguments.items()}
            return caller(cacheables).instance

        return wrapper

    return decorator
