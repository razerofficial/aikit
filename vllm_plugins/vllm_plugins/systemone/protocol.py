"""Request parsing for `POST /v1/systemone` (TypeSafe SystemOne wire format).

Validation of a single question (type, option counts, score levels, noul rules) is done
by the vendored `render_question`; this module adds the request envelope and names the
offending question in every error.
"""
from dataclasses import dataclass
from typing import Any

from .systemone import render_question


class RequestError(ValueError):
    """The request body is not a valid SystemOne request (HTTP 422)."""


@dataclass
class ParsedRequest:
    state: Any
    model: str | None
    questions: dict  # id -> raw question spec, in request order
    rendered: dict   # id -> render_question() output, same order


def parse_request(body) -> ParsedRequest:
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")
    if "state" not in body:
        raise RequestError("state is required")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must be a non-empty object of question specs")

    rendered = {}
    for qid, spec in questions.items():
        where = f"questions.{qid}"
        if not isinstance(spec, dict):
            raise RequestError(f"{where}: expected an object")
        try:
            rendered[qid] = render_question(spec)
        except (ValueError, TypeError, KeyError) as e:
            raise RequestError(f"{where}: {e}") from None
    return ParsedRequest(
        state=body["state"],
        model=body.get("model"),
        questions=questions,
        rendered=rendered,
    )
