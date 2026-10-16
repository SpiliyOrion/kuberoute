"""Deterministic local backends for exercising gateway routing."""

from kuberoute_gateway.backends import ModelOutput, NormalizedChatRequest


class MockSmallBackend:
    async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
        user_messages = [message.content for message in request.messages if message.role == "user"]
        prompt = user_messages[-1] if user_messages else ""
        return ModelOutput(content=f"mock-small: {prompt}")


class MockLargeBackend:
    async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
        user_messages = [message.content for message in request.messages if message.role == "user"]
        prompt = user_messages[-1] if user_messages else ""
        return ModelOutput(
            content=f"mock-large: received {len(request.messages)} messages; {prompt}"
        )
