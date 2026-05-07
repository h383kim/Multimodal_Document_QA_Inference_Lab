"""Inference routing — picks OCR / small VLM / large VLM per request."""

from app.routing.policies import RuleBasedPolicy
from app.routing.router import Router
from app.routing.types import RouteDecision, RoutePath

__all__ = ["RouteDecision", "RoutePath", "Router", "RuleBasedPolicy"]
