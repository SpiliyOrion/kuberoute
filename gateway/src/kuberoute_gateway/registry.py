"""In-memory mapping from public model IDs to inference backends."""

from kuberoute_gateway.backends import ChatBackend


class DuplicateModelError(ValueError):
    """Raised when a model ID has already been registered."""


class UnknownModelError(LookupError):
    """Raised when a requested model ID is absent."""


class ReservedModelError(ValueError):
    """Raised when a model ID is reserved for gateway behavior."""


class ModelRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, ChatBackend] = {}
        self._auto_eligible: set[str] = set()

    def register(self, model_id: str, backend: ChatBackend, *, auto_eligible: bool = False) -> None:
        if model_id == "auto":
            raise ReservedModelError("Model ID is reserved: auto")
        if model_id in self._backends:
            raise DuplicateModelError(f"Model already registered: {model_id}")
        self._backends[model_id] = backend
        if auto_eligible:
            self._auto_eligible.add(model_id)

    def get(self, model_id: str) -> ChatBackend:
        try:
            return self._backends[model_id]
        except KeyError as exc:
            raise UnknownModelError(f"Unknown model: {model_id}") from exc

    def model_ids(self) -> tuple[str, ...]:
        """Return registered IDs in registration order."""
        return tuple(self._backends)

    def auto_eligible_model_ids(self) -> tuple[str, ...]:
        """Return explicitly eligible IDs in registration order."""
        return tuple(model_id for model_id in self._backends if model_id in self._auto_eligible)
