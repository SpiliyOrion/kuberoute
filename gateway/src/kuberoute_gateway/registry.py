"""In-memory mapping from public model IDs to inference backends."""

from kuberoute_gateway.backends import ChatBackend


class DuplicateModelError(ValueError):
    """Raised when a model ID has already been registered."""


class UnknownModelError(LookupError):
    """Raised when a requested model ID is absent."""


class ModelRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, ChatBackend] = {}

    def register(self, model_id: str, backend: ChatBackend) -> None:
        if model_id in self._backends:
            raise DuplicateModelError(f"Model already registered: {model_id}")
        self._backends[model_id] = backend

    def get(self, model_id: str) -> ChatBackend:
        try:
            return self._backends[model_id]
        except KeyError as exc:
            raise UnknownModelError(f"Unknown model: {model_id}") from exc

    def model_ids(self) -> tuple[str, ...]:
        """Return registered IDs in registration order."""
        return tuple(self._backends)
