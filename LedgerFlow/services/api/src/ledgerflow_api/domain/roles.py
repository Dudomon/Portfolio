from enum import StrEnum


class Role(StrEnum):
    VIEWER = "viewer"
    REVIEWER = "reviewer"
    ADMIN = "admin"

    @property
    def rank(self) -> int:
        return _RANKS[self]

    def satisfies(self, required: "Role") -> bool:
        """Roles are strictly nested: admin can do everything a reviewer can, and so on."""
        return self.rank >= required.rank


_RANKS = {Role.VIEWER: 0, Role.REVIEWER: 1, Role.ADMIN: 2}
