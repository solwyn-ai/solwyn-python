"""Solwyn (sync) and AsyncSolwyn (async) client wrappers.

Drop-in wrappers for openai.OpenAI / anthropic.Anthropic that add
budget enforcement, circuit breaking, and metadata reporting.

Usage::

    from openai import OpenAI
    from solwyn import Solwyn

    client = Solwyn(
        OpenAI(),
        api_key="sk_solwyn_...",
        project_id="proj_abc12345",
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

from solwyn._base import _SolwynBase
from solwyn._privacy import estimate_content_length, estimate_tokens_from_length
from solwyn._proxies import (
    _AsyncChatProxy,
    _AsyncMessagesProxy,
    _AsyncModelsProxy,
    _SyncChatProxy,
    _SyncMessagesProxy,
    _SyncModelsProxy,
)
from solwyn._token_details import TokenDetails
from solwyn._types import CallStatus, ProviderName
from solwyn.budget import (
    DEFAULT_COST_PER_TOKEN,
    AsyncBudgetEnforcer,
    BudgetEnforcer,
)
from solwyn.config import SolwynConfig
from solwyn.exceptions import BudgetExceededError
from solwyn.providers import get_adapter_for_client
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter
from solwyn.stream import AsyncStreamWrapper, SyncStreamWrapper

logger = logging.getLogger(__name__)


def _detect_provider(client: Any) -> ProviderName:
    """Auto-detect the LLM provider from the client instance.

    Delegates to the provider adapter registry for consistent detection.
    """
    try:
        adapter = get_adapter_for_client(client)
        return ProviderName(adapter.name)
    except ValueError as err:
        raise ValueError(
            f"Cannot auto-detect provider for client type {type(client).__name__}. "
            f"Supported: openai.OpenAI, anthropic.Anthropic, "
            f"google.generativeai.GenerativeModel"
        ) from err


class Solwyn(_SolwynBase):
    """Synchronous Solwyn client wrapper.

    Wraps an OpenAI or Anthropic client with budget enforcement,
    circuit breaking, and metadata reporting.

    Usage::

        from openai import OpenAI
        from solwyn import Solwyn

        client = Solwyn(
            OpenAI(),
            api_key="sk_solwyn_...",
            project_id="proj_abc12345",
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        client.close()
    """

    def __init__(
        self,
        client: Any,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        **config_kwargs: Any,
    ) -> None:
        # Detect provider and store adapter for usage extraction
        self._adapter = get_adapter_for_client(client)
        self._detected_provider = ProviderName(self._adapter.name)
        self._client = client

        # Build config — SolwynConfig._load_from_env fills missing
        # values from SOLWYN_API_KEY / SOLWYN_PROJECT_ID env vars.
        cfg_kwargs: dict[str, Any] = {"primary_provider": self._detected_provider, **config_kwargs}
        if api_key is not None:
            cfg_kwargs["api_key"] = api_key
        if project_id is not None:
            cfg_kwargs["project_id"] = project_id
        config = SolwynConfig(**cfg_kwargs)
        super().__init__(config)

        # Budget enforcer
        self._budget = BudgetEnforcer(
            project_id=config.project_id,
            api_url=config.api_url,
            api_key=config.api_key,
            budget_mode=config.budget_mode,
            fail_open=config.fail_open,
            cache_ttl=config.budget_check_cache_ttl,
        )

        # Metadata reporter
        self._reporter = MetadataReporter(
            config.api_url,
            config.api_key,
            batch_size=config.reporter_batch_size,
            flush_interval=config.reporter_flush_interval,
            max_queue_size=config.reporter_max_queue_size,
            max_in_flight=config.reporter_max_in_flight,
        )

    @functools.cached_property
    def chat(self) -> _SyncChatProxy:
        """Return a proxy that intercepts chat.completions.create().

        Cached: provider is fixed at construction so this is safe to create once.
        """
        return _SyncChatProxy(self)

    @functools.cached_property
    def messages(self) -> Any:
        """Anthropic-compatible: client.messages.create() goes through interception.

        Cached: _detected_provider is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._detected_provider == ProviderName.ANTHROPIC:
            return _SyncMessagesProxy(self)
        return self._client.messages

    @functools.cached_property
    def models(self) -> Any:
        """Google-compatible: client.models.generate_content() goes through interception.

        Cached: _detected_provider is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._detected_provider == ProviderName.GOOGLE:
            return _SyncModelsProxy(self)
        return self._client.models

    def _intercepted_call(self, *, _force_stream: bool = False, **kwargs: Any) -> Any:
        """Core interception logic for LLM calls.

        Steps:
        1. Estimate input tokens
        2. Check budget
        3. Select provider (circuit breaker)
        4. Prepare kwargs (inject stream_options if streaming)
        5. Call underlying client
        6a. Non-streaming: extract usage, confirm budget, report metadata
        6b. Streaming: return wrapped stream that does 6a on exhaustion
        """
        model = kwargs.get("model", "unknown")
        is_streaming = kwargs.get("stream", False) or _force_stream
        is_failover = False

        # 1. Estimate input tokens from input text (length-only; never materializes joined string)
        char_count = estimate_content_length(kwargs)
        estimated_input_tokens = (
            estimate_tokens_from_length(char_count, provider=self._detected_provider.value)
            if char_count
            else 0
        )

        # 2. Check budget
        budget_result = self._budget.check_budget(
            estimated_input_tokens=estimated_input_tokens,
            model=model,
            provider=self._detected_provider.value,
        )

        if not budget_result.allowed:
            # Report estimated tokens so the API keeps an accurate running total
            # even for calls that were blocked by hard-deny.
            try:
                event = self._build_metadata_event(
                    project_id=self._config.project_id,
                    model=model,
                    provider=self._detected_provider.value,
                    input_tokens=estimated_input_tokens,
                    output_tokens=0,
                    token_details=None,
                    latency_ms=0.0,
                    status=CallStatus.BUDGET_DENIED,
                    is_failover=False,
                )
                self._reporter.report(event)
            except Exception:
                logger.warning("Failed to report budget_denied metadata event", exc_info=True)

            raise BudgetExceededError(
                f"Budget exceeded for project {self._config.project_id}",
                budget_limit=budget_result.budget_limit,
                current_usage=budget_result.current_usage,
                estimated_cost=estimated_input_tokens * DEFAULT_COST_PER_TOKEN,
                budget_period="unknown",
                mode=budget_result.mode.value,
            )

        # 3. Select provider via circuit breaker
        selected_provider = self._select_provider()
        is_failover = selected_provider != self._detected_provider.value

        # 4. Prepare kwargs for streaming if needed
        if is_streaming:
            kwargs = self._adapter.prepare_streaming(kwargs)

        # 5. Call underlying client
        start_time = time.monotonic()
        try:
            if self._detected_provider == ProviderName.OPENAI:
                response = self._client.chat.completions.create(**kwargs)
            elif self._detected_provider == ProviderName.ANTHROPIC:
                response = self._client.messages.create(**kwargs)
            elif _force_stream:
                # _force_stream is only passed by _SyncModelsProxy.generate_content_stream(),
                # which is only constructed for Google clients.
                if self._detected_provider != ProviderName.GOOGLE:
                    raise RuntimeError(
                        f"_force_stream is Google-only but provider is {self._detected_provider}"
                    )
                response = self._client.models.generate_content_stream(**kwargs)
            else:
                response = self._client.models.generate_content(**kwargs)
        except Exception:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            cb = self._get_circuit_breaker(selected_provider)
            cb.record_failure()

            event = self._build_metadata_event(
                project_id=self._config.project_id,
                model=model,
                provider=selected_provider,
                input_tokens=0,
                output_tokens=0,
                token_details=None,
                latency_ms=elapsed_ms,
                status=CallStatus.ERROR,
                is_failover=is_failover,
            )
            self._reporter.report(event)
            raise

        # 6. Streaming vs non-streaming post-processing
        if is_streaming:
            accumulator = self._adapter.create_stream_accumulator()

            def on_complete(token_details: TokenDetails, elapsed_ms: float) -> None:
                cb = self._get_circuit_breaker(selected_provider)
                cb.record_success()
                if budget_result.reservation_id:
                    confirm = self._budget.build_confirm_request(
                        reservation_id=budget_result.reservation_id,
                        model=model,
                        token_details=token_details,
                    )
                    self._reporter.report_confirm(confirm)
                event = self._build_metadata_event(
                    project_id=self._config.project_id,
                    model=model,
                    provider=selected_provider,
                    input_tokens=token_details.input_tokens,
                    output_tokens=token_details.output_tokens,
                    token_details=token_details,
                    latency_ms=elapsed_ms,
                    status=CallStatus.SUCCESS,
                    is_failover=is_failover,
                )
                self._reporter.report(event)

            def on_error(exc: Exception) -> None:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                cb = self._get_circuit_breaker(selected_provider)
                cb.record_failure()
                event = self._build_metadata_event(
                    project_id=self._config.project_id,
                    model=model,
                    provider=selected_provider,
                    input_tokens=0,
                    output_tokens=0,
                    token_details=None,
                    latency_ms=elapsed_ms,
                    status=CallStatus.ERROR,
                    is_failover=is_failover,
                )
                self._reporter.report(event)

            return SyncStreamWrapper(response, accumulator, on_complete, on_error)

        # Non-streaming: existing flow
        token_details = self._adapter.extract_usage(response)
        elapsed_ms = (time.monotonic() - start_time) * 1000
        cb = self._get_circuit_breaker(selected_provider)
        cb.record_success()

        if budget_result.reservation_id:
            self._budget.confirm_cost(budget_result.reservation_id, model, token_details)

        event = self._build_metadata_event(
            project_id=self._config.project_id,
            model=model,
            provider=selected_provider,
            input_tokens=token_details.input_tokens,
            output_tokens=token_details.output_tokens,
            token_details=token_details,
            latency_ms=elapsed_ms,
            status=CallStatus.SUCCESS,
            is_failover=is_failover,
        )
        self._reporter.report(event)

        return response

    def close(self) -> None:
        """Shut down the reporter and close HTTP clients."""
        self._reporter.close()
        self._budget.close()

    def __enter__(self) -> Solwyn:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        """Pass through non-intercepted attributes to the underlying client."""
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


class AsyncSolwyn(_SolwynBase):
    """Asynchronous Solwyn client wrapper.

    Same API and behaviour as Solwyn, but async.

    Usage::

        from openai import AsyncOpenAI
        from solwyn import AsyncSolwyn

        async with AsyncSolwyn(
            AsyncOpenAI(),
            api_key="sk_solwyn_...",
            project_id="proj_abc12345",
        ) as client:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )
    """

    def __init__(
        self,
        client: Any,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        **config_kwargs: Any,
    ) -> None:
        # Detect provider and store adapter for usage extraction
        self._adapter = get_adapter_for_client(client)
        self._detected_provider = ProviderName(self._adapter.name)
        self._client = client

        cfg_kwargs: dict[str, Any] = {"primary_provider": self._detected_provider, **config_kwargs}
        if api_key is not None:
            cfg_kwargs["api_key"] = api_key
        if project_id is not None:
            cfg_kwargs["project_id"] = project_id
        config = SolwynConfig(**cfg_kwargs)
        super().__init__(config)

        self._budget = AsyncBudgetEnforcer(
            project_id=config.project_id,
            api_url=config.api_url,
            api_key=config.api_key,
            budget_mode=config.budget_mode,
            fail_open=config.fail_open,
            cache_ttl=config.budget_check_cache_ttl,
        )

        self._reporter = AsyncMetadataReporter(
            config.api_url,
            config.api_key,
            batch_size=config.reporter_batch_size,
            flush_interval=config.reporter_flush_interval,
            max_queue_size=config.reporter_max_queue_size,
            max_in_flight=config.reporter_max_in_flight,
        )

    @functools.cached_property
    def chat(self) -> _AsyncChatProxy:
        """Return an async proxy that intercepts chat.completions.create().

        Cached: provider is fixed at construction so this is safe to create once.
        """
        return _AsyncChatProxy(self)

    @functools.cached_property
    def messages(self) -> Any:
        """Anthropic-compatible: client.messages.create() goes through interception.

        Cached: _detected_provider is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._detected_provider == ProviderName.ANTHROPIC:
            return _AsyncMessagesProxy(self)
        return self._client.messages

    @functools.cached_property
    def models(self) -> Any:
        """Google-compatible: client.models.generate_content() goes through interception.

        Cached: _detected_provider is fixed at construction, so the conditional
        result is stable for the lifetime of this client instance.
        """
        if self._detected_provider == ProviderName.GOOGLE:
            return _AsyncModelsProxy(self)
        return self._client.models

    async def _intercepted_call(self, *, _force_stream: bool = False, **kwargs: Any) -> Any:
        """Async core interception logic. See Solwyn._intercepted_call."""
        model = kwargs.get("model", "unknown")
        is_streaming = kwargs.get("stream", False) or _force_stream
        is_failover = False

        char_count = estimate_content_length(kwargs)
        estimated_input_tokens = (
            estimate_tokens_from_length(char_count, provider=self._detected_provider.value)
            if char_count
            else 0
        )

        budget_result = await self._budget.check_budget(
            estimated_input_tokens=estimated_input_tokens,
            model=model,
            provider=self._detected_provider.value,
        )

        if not budget_result.allowed:
            # Report estimated tokens so the API keeps an accurate running total
            # even for calls that were blocked by hard-deny.
            try:
                event = self._build_metadata_event(
                    project_id=self._config.project_id,
                    model=model,
                    provider=self._detected_provider.value,
                    input_tokens=estimated_input_tokens,
                    output_tokens=0,
                    token_details=None,
                    latency_ms=0.0,
                    status=CallStatus.BUDGET_DENIED,
                    is_failover=False,
                )
                self._reporter.report(event)
            except Exception:
                logger.warning("Failed to report budget_denied metadata event", exc_info=True)

            raise BudgetExceededError(
                f"Budget exceeded for project {self._config.project_id}",
                budget_limit=budget_result.budget_limit,
                current_usage=budget_result.current_usage,
                estimated_cost=estimated_input_tokens * DEFAULT_COST_PER_TOKEN,
                budget_period="unknown",
                mode=budget_result.mode.value,
            )

        selected_provider = self._select_provider()
        is_failover = selected_provider != self._detected_provider.value

        if is_streaming:
            kwargs = self._adapter.prepare_streaming(kwargs)

        start_time = time.monotonic()
        try:
            if self._detected_provider == ProviderName.OPENAI:
                response = await self._client.chat.completions.create(**kwargs)
            elif self._detected_provider == ProviderName.ANTHROPIC:
                response = await self._client.messages.create(**kwargs)
            elif _force_stream:
                # _force_stream is only passed by _AsyncModelsProxy.generate_content_stream(),
                # which is only constructed for Google clients.
                if self._detected_provider != ProviderName.GOOGLE:
                    raise RuntimeError(
                        f"_force_stream is Google-only but provider is {self._detected_provider}"
                    )
                response = await self._client.models.generate_content_stream(**kwargs)
            else:
                response = await self._client.models.generate_content(**kwargs)
        except Exception:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            cb = self._get_circuit_breaker(selected_provider)
            cb.record_failure()

            event = self._build_metadata_event(
                project_id=self._config.project_id,
                model=model,
                provider=selected_provider,
                input_tokens=0,
                output_tokens=0,
                token_details=None,
                latency_ms=elapsed_ms,
                status=CallStatus.ERROR,
                is_failover=is_failover,
            )
            self._reporter.report(event)
            raise

        if is_streaming:
            accumulator = self._adapter.create_stream_accumulator()

            async def on_complete(token_details: TokenDetails, elapsed_ms: float) -> None:
                cb = self._get_circuit_breaker(selected_provider)
                cb.record_success()
                if budget_result.reservation_id:
                    await self._budget.confirm_cost(
                        budget_result.reservation_id, model, token_details
                    )
                event = self._build_metadata_event(
                    project_id=self._config.project_id,
                    model=model,
                    provider=selected_provider,
                    input_tokens=token_details.input_tokens,
                    output_tokens=token_details.output_tokens,
                    token_details=token_details,
                    latency_ms=elapsed_ms,
                    status=CallStatus.SUCCESS,
                    is_failover=is_failover,
                )
                self._reporter.report(event)

            async def on_error(exc: Exception) -> None:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                cb = self._get_circuit_breaker(selected_provider)
                cb.record_failure()
                event = self._build_metadata_event(
                    project_id=self._config.project_id,
                    model=model,
                    provider=selected_provider,
                    input_tokens=0,
                    output_tokens=0,
                    token_details=None,
                    latency_ms=elapsed_ms,
                    status=CallStatus.ERROR,
                    is_failover=is_failover,
                )
                self._reporter.report(event)

            return AsyncStreamWrapper(response, accumulator, on_complete, on_error)

        token_details = self._adapter.extract_usage(response)
        elapsed_ms = (time.monotonic() - start_time) * 1000
        cb = self._get_circuit_breaker(selected_provider)
        cb.record_success()

        if budget_result.reservation_id:
            await self._budget.confirm_cost(budget_result.reservation_id, model, token_details)

        event = self._build_metadata_event(
            project_id=self._config.project_id,
            model=model,
            provider=selected_provider,
            input_tokens=token_details.input_tokens,
            output_tokens=token_details.output_tokens,
            token_details=token_details,
            latency_ms=elapsed_ms,
            status=CallStatus.SUCCESS,
            is_failover=is_failover,
        )
        self._reporter.report(event)

        return response

    async def close(self) -> None:
        """Shut down the reporter and close HTTP clients."""
        await self._reporter.close()
        await self._budget.close()

    async def __aenter__(self) -> AsyncSolwyn:
        self._reporter.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def __getattr__(self, name: str) -> Any:
        """Pass through non-intercepted attributes to the underlying client."""
        return getattr(self._client, name)
