"""Minimal BDDL (LIBERO problem file) parser: s-expressions to a structured problem."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

SExpr = Union[str, List["SExpr"]]

_TOKEN = re.compile(r"\(|\)|[^\s()]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text)


def parse_sexpr(tokens: List[str]) -> SExpr:
    pos = 0

    def read() -> SExpr:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            out: List[SExpr] = []
            while tokens[pos] != ")":
                out.append(read())
            pos += 1
            return out
        if tok == ")":
            raise ValueError("unexpected ')'")
        return tok

    expr = read()
    if pos != len(tokens):
        raise ValueError("trailing tokens after s-expression")
    return expr


def _typed_names(items: List[str]) -> Dict[str, str]:
    """Parse ``name1 name2 - type1 name3 - type2`` into ``{name: type}``."""
    out: Dict[str, str] = {}
    pending: List[str] = []
    i = 0
    while i < len(items):
        tok = items[i]
        if tok == "-":
            typ = items[i + 1]
            for n in pending:
                out[n] = typ
            pending = []
            i += 2
            continue
        pending.append(tok)
        i += 1
    for n in pending:  # untyped trailing names
        out[n] = n
    return out


@dataclass
class Region:
    name: str
    target: str
    ranges: List[List[float]] = field(default_factory=list)
    yaw_rotation: List[List[float]] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.target}_{self.name}"


@dataclass
class BddlProblem:
    problem_name: str
    domain: str
    language: str
    regions: Dict[str, Region]
    fixtures: Dict[str, str]
    objects: Dict[str, str]
    obj_of_interest: List[str]
    init: List[List[str]]
    goal: List[List[str]]
    source_path: str = ""

    @property
    def entities(self) -> List[str]:
        names = set(self.fixtures) | set(self.objects)
        names |= {r.full_name for r in self.regions.values()}
        return sorted(names)

    def layout_dict(self) -> Dict[str, Any]:
        """Layout = fixtures + objects + regions (what 'same scene' means)."""
        return {
            "fixtures": dict(sorted(self.fixtures.items())),
            "objects": dict(sorted(self.objects.items())),
            "regions": {
                k: {"target": r.target, "ranges": r.ranges, "yaw_rotation": r.yaw_rotation}
                for k, r in sorted(self.regions.items())
            },
        }


def _parse_region(expr: List[SExpr]) -> Region:
    name = str(expr[0])
    target = ""
    ranges: List[List[float]] = []
    yaw: List[List[float]] = []
    for part in expr[1:]:
        if not isinstance(part, list) or not part:
            continue
        key = part[0]
        if key == ":target":
            target = str(part[1])
        elif key == ":ranges":
            ranges = [[float(v) for v in r] for r in part[1]]  # type: ignore[union-attr]
        elif key == ":yaw_rotation":
            yaw = [[float(v) for v in r] for r in part[1]]  # type: ignore[union-attr]
    return Region(name=name, target=target, ranges=ranges, yaw_rotation=yaw)


def _predicates(expr: List[SExpr]) -> List[List[str]]:
    preds: List[List[str]] = []
    for p in expr:
        if isinstance(p, list) and p and isinstance(p[0], str) and p[0].lower() == "and":
            preds.extend(_predicates(p[1:]))
        elif isinstance(p, list):
            preds.append([str(x) for x in p])
    return preds


def parse_bddl(text: str, source_path: str = "") -> BddlProblem:
    expr = parse_sexpr(tokenize(text))
    assert isinstance(expr, list) and expr[0] == "define", "not a BDDL define form"
    problem_name = ""
    domain = ""
    language = ""
    regions: Dict[str, Region] = {}
    fixtures: Dict[str, str] = {}
    objects: Dict[str, str] = {}
    obj_of_interest: List[str] = []
    init: List[List[str]] = []
    goal: List[List[str]] = []
    for section in expr[1:]:
        if not isinstance(section, list) or not section:
            continue
        head = section[0]
        body = section[1:]
        if head == "problem":
            problem_name = str(body[0])
        elif head == ":domain":
            domain = str(body[0])
        elif head == ":language":
            language = " ".join(str(t) for t in body)
        elif head == ":regions":
            for r in body:
                if isinstance(r, list):
                    reg = _parse_region(r)
                    regions[reg.name] = reg
        elif head == ":fixtures":
            fixtures = _typed_names([str(t) for t in body])
        elif head == ":objects":
            objects = _typed_names([str(t) for t in body])
        elif head == ":obj_of_interest":
            obj_of_interest = [str(t) for t in body]
        elif head == ":init":
            init = _predicates(body)
        elif head == ":goal":
            goal = _predicates(body)
    return BddlProblem(
        problem_name=problem_name,
        domain=domain,
        language=language,
        regions=regions,
        fixtures=fixtures,
        objects=objects,
        obj_of_interest=obj_of_interest,
        init=init,
        goal=goal,
        source_path=source_path,
    )


def load_bddl(path: Path) -> BddlProblem:
    return parse_bddl(Path(path).read_text(encoding="utf-8"), source_path=str(path))
