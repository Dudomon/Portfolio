"""RFC 9457 problem details for every error the API emits."""

from http import HTTPStatus
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ledgerflow_api.domain.lifecycle import InvalidTransitionError

PROBLEM_JSON = "application/problem+json"
_PROBLEM_TYPE_PREFIX = "urn:ledgerflow:problem:"

logger = structlog.get_logger(__name__)


class ProblemError(Exception):
    """Base for errors that map to a problem+json response.

    Subclasses pin ``status`` and ``slug``; ``slug`` becomes the problem ``type`` URN, which is
    what clients should branch on (titles are for humans and may change).
    """

    status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR
    slug: str = "internal-error"

    def __init__(
        self,
        detail: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        extensions: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail or self.status.phrase)
        self.detail = detail
        self.headers = headers or {}
        self.extensions = extensions or {}


class BadRequestError(ProblemError):
    status = HTTPStatus.BAD_REQUEST
    slug = "bad-request"


class UnauthorizedError(ProblemError):
    status = HTTPStatus.UNAUTHORIZED
    slug = "unauthorized"

    def __init__(self, detail: str = "Missing or invalid credentials.") -> None:
        super().__init__(detail, headers={"WWW-Authenticate": "Bearer"})


class ForbiddenError(ProblemError):
    status = HTTPStatus.FORBIDDEN
    slug = "forbidden"


class NotFoundError(ProblemError):
    status = HTTPStatus.NOT_FOUND
    slug = "not-found"


class ConflictError(ProblemError):
    status = HTTPStatus.CONFLICT
    slug = "conflict"


class PreconditionFailedError(ProblemError):
    status = HTTPStatus.PRECONDITION_FAILED
    slug = "precondition-failed"


class PreconditionRequiredError(ProblemError):
    status = HTTPStatus.PRECONDITION_REQUIRED
    slug = "precondition-required"


class PayloadTooLargeError(ProblemError):
    status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    slug = "payload-too-large"


class UnsupportedMediaTypeError(ProblemError):
    status = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    slug = "unsupported-media-type"


class UnprocessableError(ProblemError):
    status = HTTPStatus.UNPROCESSABLE_ENTITY
    slug = "validation-error"


def problem_response(
    request: Request,
    *,
    status: HTTPStatus,
    slug: str,
    detail: str | None,
    headers: dict[str, str] | None = None,
    extensions: dict[str, Any] | None = None,
) -> JSONResponse:
    request_id: str | None = getattr(request.state, "request_id", None)
    body: dict[str, Any] = {
        "type": _PROBLEM_TYPE_PREFIX + slug,
        "title": status.phrase,
        "status": status.value,
        "detail": detail,
        "instance": request.url.path,
        "request_id": request_id,
        **(extensions or {}),
    }
    response_headers = dict(headers or {})
    if request_id is not None:
        # Unhandled errors are rendered outside the request-id middleware, so set it here too.
        response_headers["X-Request-ID"] = request_id
    return JSONResponse(body, status_code=status, headers=response_headers, media_type=PROBLEM_JSON)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProblemError)
    async def handle_problem(request: Request, exc: ProblemError) -> JSONResponse:
        return problem_response(
            request,
            status=exc.status,
            slug=exc.slug,
            detail=exc.detail,
            headers=exc.headers,
            extensions=exc.extensions,
        )

    @app.exception_handler(InvalidTransitionError)
    async def handle_invalid_transition(
        request: Request, exc: InvalidTransitionError
    ) -> JSONResponse:
        return problem_response(
            request,
            status=HTTPStatus.CONFLICT,
            slug="invalid-transition",
            detail=str(exc),
            extensions={"current_status": exc.current.value, "target_status": exc.target.value},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
            for err in exc.errors()
        ]
        return problem_response(
            request,
            status=HTTPStatus.UNPROCESSABLE_ENTITY,
            slug="validation-error",
            detail="The request did not pass validation.",
            extensions={"errors": errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        status = HTTPStatus(exc.status_code)
        return problem_response(
            request,
            status=status,
            slug=status.phrase.lower().replace(" ", "-"),
            detail=exc.detail if exc.detail != status.phrase else None,
            headers=dict(exc.headers or {}),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", exc_info=exc)
        return problem_response(
            request,
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            slug="internal-error",
            detail=None,
        )
