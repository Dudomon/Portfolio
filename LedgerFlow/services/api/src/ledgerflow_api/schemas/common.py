import uuid
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, PlainSerializer

from ledgerflow_api.domain.money import format_decimal

# Decimals cross the wire as strings ("1530.40"), never as JSON numbers.
DecimalStr = Annotated[Decimal, PlainSerializer(format_decimal, return_type=str, when_used="json")]


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ActorRef(ApiModel):
    id: uuid.UUID
    name: str


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None
