"""Lexical Scope Management and Symbol Table for RealASTGenome."""
from __future__ import annotations

from .types import Symbol


class SymbolTable:
    """Maintains a lexically scoped symbol registry supporting shadowing and visibility checks."""

    def __init__(self):
        self.symbols: dict[str, list[Symbol]] = {}
        self.scopes: dict[str, set[str]] = {"global": set()}
        self.parent_scopes: dict[str, str] = {}

    def set_parent_scope(self, child_scope: str, parent_scope: str) -> None:
        self.parent_scopes[child_scope] = parent_scope
        if child_scope not in self.scopes:
            self.scopes[child_scope] = set()

    def add_symbol(self, symbol: Symbol) -> None:
        if symbol.name not in self.symbols:
            self.symbols[symbol.name] = []
        self.symbols[symbol.name].append(symbol)

        if symbol.scope not in self.scopes:
            self.scopes[symbol.scope] = set()
        self.scopes[symbol.scope].add(symbol.name)

    def lookup(self, name: str, scope: str = "global") -> Symbol | None:
        """Resolves a symbol using lexical scoping (child -> parent -> global)."""
        if name not in self.symbols:
            return None

        current = scope
        while current:
            candidates = [s for s in self.symbols[name] if s.scope == current]
            if candidates:
                return candidates[-1]
            if current == "global":
                break
            current = self.parent_scopes.get(current, "global")

        # Fallback to global scope check
        candidates = [s for s in self.symbols[name] if s.scope == "global"]
        if candidates:
            return candidates[-1]

        return None

    def get_all_symbols_in_scope(self, scope: str = "global") -> list[Symbol]:
        """Returns all visible symbols accessible from the given scope."""
        visible: dict[str, Symbol] = {}

        # 1. Gather global symbols first
        for name in self.scopes.get("global", set()):
            sym = self.lookup(name, "global")
            if sym:
                visible[name] = sym

        # 2. Walk from parent scopes down to current scope, allowing shadowing
        scope_chain: list[str] = []
        curr = scope
        while curr and curr != "global":
            scope_chain.append(curr)
            curr = self.parent_scopes.get(curr, "global")
        scope_chain.reverse()

        for sc in scope_chain:
            for name in self.scopes.get(sc, set()):
                candidates = [s for s in self.symbols.get(name, []) if s.scope == sc]
                if candidates:
                    visible[name] = candidates[-1]

        return list(visible.values())

    def get_typed_symbols_in_scope(self, scope: str, target_type: str) -> list[Symbol]:
        """Filters visible symbols matching a requested runtime type."""
        all_syms = self.get_all_symbols_in_scope(scope)
        return [
            s for s in all_syms
            if s.symbol_type in ("variable", "parameter")
            and (s.inferred_type == target_type or (target_type in ("int", "float") and s.inferred_type in ("int", "float")))
        ]
