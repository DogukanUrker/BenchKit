"""Symbolic equivalence of two normalized MATH answers, run in a worker process.

``math500`` compares answers as normalized strings first and only asks this
module when the strings differ. It runs as a separate process (see
``math500.SympyChecker``) because simplification can hang or exhaust memory on
adversarial input, and a thread cannot be killed; the parent kills the process
on a deadline instead.

The answers are model-written, so nothing here calls ``eval``, ``sympify`` on
a string, or ``parse_expr``. A small recursive-descent parser turns the LaTeX
subset MATH answers use into SymPy objects directly: numbers, single-letter
variables, ``+ - * / ^``, implicit multiplication, ``\\frac``, ``\\sqrt``,
``\\pi``, ``\\infty``, ``i``, trig and log functions, factorials, base
subscripts (``4210_5``), and the containers around them - ordered tuples and
intervals (bracket types must match), unordered lists and sets, ``\\cup``
unions, ``\\pm``, equations and matrices. Anything else does not parse, and an
answer that does not parse is not equivalent to anything.

Two expressions are equal when ``simplify(a - b) == 0`` or, failing that, when
they agree numerically to a relative 1e-9 at fixed sample points. Decimals
parse as exact rationals, so a rounded ``0.333`` is not ``\\frac{1}{3}``.

This file imports only the standard library and SymPy, so the worker starts
without importing BenchKit. Protocol: one JSON request per line on stdin,
``{"a": ..., "b": ...}``, answered by one line ``{"equal": true|false|null}``,
where ``null`` means one side did not parse.
"""

from __future__ import annotations

import json
import re
import sys

import sympy

I = sympy.I  # noqa: E741 - the imaginary unit

FUNCTIONS = {
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "cot": sympy.cot,
    "sec": sympy.sec,
    "csc": sympy.csc,
    "arcsin": sympy.asin,
    "arccos": sympy.acos,
    "arctan": sympy.atan,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "ln": sympy.log,
    "log": sympy.log,
    "exp": sympy.exp,
}
CONSTANTS = {"pi": sympy.pi, "infty": sympy.oo, "infinity": sympy.oo}
MULTIPLY = {"cdot", "times", "ast"}
COMMANDS = sorted(
    {*FUNCTIONS, *CONSTANTS, *MULTIPLY, "frac", "sqrt", "binom", "div", "pm"},
    key=len,
    reverse=True,
)
SAMPLES = (
    sympy.Rational(7, 5),
    sympy.Rational(-3, 7),
    sympy.Rational(11, 3),
    sympy.Rational(2, 9),
    sympy.Rational(-13, 4),
)


class ParseError(ValueError):
    pass


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    while pos < len(text):
        ch = text[pos]
        if ch.isspace():
            pos += 1
        elif ch.isdigit() or (ch == "." and text[pos + 1 : pos + 2].isdigit()):
            match = re.match(r"\d*\.?\d+|\d+\.?", text[pos:])
            assert match is not None
            tokens.append(match.group())
            pos += match.end()
        elif ch == "\\":
            match = re.match(r"\\([a-zA-Z]+)", text[pos:])
            if match is None:
                # \{ \} \| and friends
                tokens.append(text[pos : pos + 2])
                pos += 2
                continue
            word = match.group(1)
            # Normalization drops spaces, so "\cot x" arrives as "\cotx":
            # take the longest known command and leave the rest as letters.
            command = next((c for c in COMMANDS if word.startswith(c)), None)
            if command is None:
                raise ParseError(f"unknown command \\{word}")
            tokens.append("\\" + command)
            pos += 1 + len(command)
        else:
            tokens.append(ch)
            pos += 1
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise ParseError(f"expected {expected!r}, got {token!r}")
        self.pos += 1
        return token

    def done(self) -> bool:
        return self.pos == len(self.tokens)

    def expr(self) -> sympy.Expr:
        value = self.term()
        while self.peek() in ("+", "-"):
            sign = 1 if self.take() == "+" else -1
            value = value + sign * self.term()
        return value

    def term(self) -> sympy.Expr:
        value = self.unary()
        while True:
            token = self.peek()
            if token in ("*", "\\cdot", "\\times", "\\ast"):
                self.take()
                value = value * self.unary()
            elif token in ("/", "\\div"):
                self.take()
                value = value / self.unary()
            elif self._starts_atom(token):
                # A whole number followed by \frac is a mixed number.
                if (
                    token == "\\frac"
                    and isinstance(value, sympy.Integer)
                    and self.pos > 0
                    and self.tokens[self.pos - 1].isdigit()
                ):
                    fraction = self.power()
                    value = value + fraction if value >= 0 else value - fraction
                else:
                    value = value * self.power()
            else:
                return value

    def unary(self) -> sympy.Expr:
        if self.peek() == "-":
            self.take()
            return -self.unary()
        if self.peek() == "+":
            self.take()
            return self.unary()
        return self.power()

    def power(self) -> sympy.Expr:
        base = self.postfix()
        if self.peek() == "^":
            self.take()
            return base ** self.exponent()
        return base

    def exponent(self) -> sympy.Expr:
        token = self.peek()
        if token == "{":
            return self.group()
        if token == "-":
            self.take()
            return -self.exponent()
        if token is not None and token[0].isdigit():
            # x^23 is x^2 * 3 in LaTeX.
            self.take()
            if len(token) > 1:
                self.tokens.insert(self.pos, token[1:])
            return sympy.Integer(token[0])
        return self.postfix()

    def postfix(self) -> sympy.Expr:
        value = self.atom()
        while self.peek() == "!":
            self.take()
            value = sympy.factorial(value)
        return value

    def group(self) -> sympy.Expr:
        self.take("{")
        value = self.expr()
        self.take("}")
        return value

    def _starts_atom(self, token: str | None) -> bool:
        if token is None:
            return False
        return (
            token[0].isdigit()
            or token[0] == "."
            or token.isalpha()
            or token in ("(", "{", "|")
            or (token.startswith("\\") and token[1:] not in MULTIPLY | {"div", "pm"})
        )

    def atom(self) -> sympy.Expr:
        token = self.take()
        if token[0].isdigit() or token[0] == ".":
            value = sympy.Rational(token)
            if self.peek() == "_":
                return self._base(token)
            return value
        if token.isalpha():
            if self.peek() == "_":
                self.take()
                sub = self.take()
                if sub == "{":
                    parts = []
                    while self.peek() != "}":
                        parts.append(self.take())
                    self.take("}")
                    sub = "".join(parts)
                return sympy.Symbol(f"{token}_{sub}")
            return I if token == "i" else sympy.Symbol(token)
        if token == "(":
            value = self.expr()
            self.take(")")
            return value
        if token == "{":
            value = self.expr()
            self.take("}")
            return value
        if token == "|":
            value = self.expr()
            self.take("|")
            return sympy.Abs(value)
        command = token[1:]
        if command in CONSTANTS:
            return CONSTANTS[command]
        if command == "frac":
            numerator = self._argument()
            return numerator / self._argument()
        if command == "binom":
            n = self._argument()
            return sympy.binomial(n, self._argument())
        if command == "sqrt":
            if self.peek() == "[":
                self.take("[")
                index = self.expr()
                self.take("]")
                return sympy.root(self._argument(), index)
            return sympy.sqrt(self._argument())
        if command in FUNCTIONS:
            exponent = None
            if self.peek() == "^":
                self.take()
                exponent = self.exponent()
            value = FUNCTIONS[command](self.power())
            return value if exponent is None else value**exponent
        raise ParseError(f"unexpected {token!r}")

    def _argument(self) -> sympy.Expr:
        """One \\frac or \\sqrt argument: a braced group or a single token."""
        if self.peek() == "{":
            return self.group()
        token = self.take()
        if token[0].isdigit():
            if len(token) > 1:
                self.tokens.insert(self.pos, token[1:])
            return sympy.Integer(token[0])
        if token.isalpha():
            return I if token == "i" else sympy.Symbol(token)
        self.pos -= 1
        return self.atom()

    def _base(self, digits: str) -> sympy.Expr:
        self.take("_")
        base = self.take()
        if base == "{":
            base = self.take()
            self.take("}")
        try:
            return sympy.Integer(int(digits, int(base)))
        except ValueError as error:
            raise ParseError(str(error)) from error


def parse_expression(text: str) -> sympy.Expr:
    parser = _Parser(_tokenize(text))
    value = parser.expr()
    if not parser.done():
        raise ParseError(f"trailing {parser.peek()!r}")
    return value


def _split_top(text: str, separator: str) -> list[str]:
    """Split on a separator that is not inside braces, brackets or parens."""
    parts, depth, start, pos = [], 0, 0, 0
    while pos < len(text):
        if text.startswith(("\\{", "\\}"), pos):
            depth += 1 if text[pos + 1] == "{" else -1
            pos += 2
            continue
        ch = text[pos]
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif depth == 0 and text.startswith(separator, pos):
            parts.append(text[start:pos])
            pos += len(separator)
            start = pos
            continue
        pos += 1
    parts.append(text[start:])
    return parts


def _expand_pm(item: str) -> list[str]:
    if "\\pm" not in item:
        return [item]
    head, _, tail = item.partition("\\pm")
    return [
        *_expand_pm(f"{head}+{tail}"),
        *_expand_pm(f"{head}-{tail}"),
    ]


def _matched(text: str, open_: str, close: str) -> bool:
    """Whether the opening bracket at 0 closes at the very end."""
    depth = 0
    for pos, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return pos == len(text) - 1
    return False


def parse_answer(text: str) -> tuple:
    """A normalized answer as a comparable structure.

    ``("expr", e)``, ``("eq", e)`` for ``lhs = rhs`` (as ``lhs - rhs``),
    ``("seq", brackets, items)`` for tuples and intervals, ``("set", items)``
    for unordered lists, ``("union", parts)`` and ``("matrix", rows)``.
    """
    text = text.strip()
    matrix = re.fullmatch(r"\\begin\{[pbv]?matrix\}(.*)\\end\{[pbv]?matrix\}", text)
    if matrix:
        rows = [row for row in matrix.group(1).split("\\\\") if row.strip()]
        return (
            "matrix",
            [[parse_expression(cell) for cell in row.split("&")] for row in rows],
        )
    union = _split_top(text, "\\cup")
    if len(union) > 1:
        return ("union", [parse_answer(part) for part in union])
    if text.startswith("\\{") and text.endswith("\\}"):
        items = _split_top(text[2:-2], ",")
        return ("set", [parse_expression(i) for x in items for i in _expand_pm(x)])
    if (
        len(text) >= 2
        and text[0] in "(["
        and text[-1] in ")]"
        and _matched(text, text[0], text[-1])
    ):
        items = _split_top(text[1:-1], ",")
        if len(items) > 1:
            return ("seq", text[0] + text[-1], [parse_expression(i) for i in items])
    items = _split_top(text, ",")
    expanded = [i for x in items for i in _expand_pm(x)]
    if len(expanded) > 1:
        return ("set", [parse_answer(item) for item in expanded])
    if "=" in text:
        sides = text.split("=")
        if len(sides) != 2:
            raise ParseError("chained equation")
        return ("eq", parse_expression(sides[0]) - parse_expression(sides[1]))
    return ("expr", parse_expression(text))


def _close(a: complex, b: complex) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


def expressions_equal(a: sympy.Expr, b: sympy.Expr) -> bool:
    if a == b:
        return True
    if a.has(sympy.oo, -sympy.oo) or b.has(sympy.oo, -sympy.oo):
        return False
    symbols = sorted(a.free_symbols | b.free_symbols, key=str)
    if not symbols:
        try:
            return _close(complex(sympy.N(a, 30)), complex(sympy.N(b, 30)))
        except (TypeError, ValueError, ZeroDivisionError):
            return False
    if sympy.simplify(a - b) == 0:
        return True
    for offset in range(3):
        point = {s: SAMPLES[(i + offset) % len(SAMPLES)] for i, s in enumerate(symbols)}
        try:
            if not _close(
                complex(sympy.N(a.subs(point), 30)), complex(sympy.N(b.subs(point), 30))
            ):
                return False
        except (TypeError, ValueError, ZeroDivisionError):
            return False
    return True


def _equations_equal(a: sympy.Expr, b: sympy.Expr) -> bool:
    """Equations match when one side is a nonzero constant multiple of the other."""
    if expressions_equal(a, b):
        return True
    if b == 0:
        return False
    ratio = sympy.simplify(a / b)
    return ratio.is_number and ratio != 0 and ratio.is_finite is True


def answers_equal(a: tuple, b: tuple) -> bool:
    kind = a[0]
    if kind != b[0]:
        # A one-element list is just its item.
        return False
    if kind == "expr":
        return expressions_equal(a[1], b[1])
    if kind == "eq":
        return _equations_equal(a[1], b[1])
    if kind == "seq":
        return (
            a[1] == b[1]
            and len(a[2]) == len(b[2])
            and all(expressions_equal(x, y) for x, y in zip(a[2], b[2], strict=True))
        )
    if kind == "matrix":
        return [len(row) for row in a[1]] == [len(row) for row in b[1]] and all(
            expressions_equal(x, y)
            for row_a, row_b in zip(a[1], b[1], strict=True)
            for x, y in zip(row_a, row_b, strict=True)
        )
    if kind == "union":
        return len(a[1]) == len(b[1]) and all(
            answers_equal(x, y) for x, y in zip(a[1], b[1], strict=True)
        )
    # Unordered: every item pairs with a distinct equal item.
    if len(a[1]) != len(b[1]):
        return False
    remaining = list(b[1])
    for item in a[1]:
        match = next(
            (
                index
                for index, other in enumerate(remaining)
                if _items_equal(item, other)
            ),
            None,
        )
        if match is None:
            return False
        remaining.pop(match)
    return True


def _items_equal(a: object, b: object) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return answers_equal(a, b)
    if isinstance(a, tuple) or isinstance(b, tuple):
        a = a if isinstance(a, tuple) else ("expr", a)
        b = b if isinstance(b, tuple) else ("expr", b)
        return answers_equal(a, b)
    return expressions_equal(a, b)


def equivalent(a: str, b: str) -> bool | None:
    """Whether two normalized answers are equal; None when one does not parse."""
    try:
        left, right = parse_answer(a), parse_answer(b)
    except (ParseError, ValueError, TypeError, ZeroDivisionError, IndexError):
        return None
    try:
        return bool(answers_equal(left, right))
    except (ValueError, TypeError, ZeroDivisionError, AttributeError):
        return False


def _limit_memory() -> None:
    try:
        import resource

        limit = 2 * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, ValueError, OSError):
        pass


def main() -> None:
    _limit_memory()
    sys.setrecursionlimit(10_000)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            result = equivalent(str(request["a"]), str(request["b"]))
        except (MemoryError, RecursionError):
            result = False
        except Exception:  # anything SymPy raises is a mismatch, not a crash
            result = False
        sys.stdout.write(json.dumps({"equal": result}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
