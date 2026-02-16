# game_engine/utils/helpers.py

import uuid
from datetime import datetime
from typing import List, Any

class EngineHelpers:
    """
    General utility functions for ID generation, data sanitization,
    and formatting for the Laravel response.
    """

    @staticmethod
    def generate_unique_id(prefix: str = "SLIP") -> str:
        """
        Generates a readable, unique identifier for slips.
        Example: SLIP-A1B2C3
        """
        short_uuid = uuid.uuid4().hex[:6].upper()
        return f"{prefix}-{short_uuid}"

    @staticmethod
    def format_money(value: float) -> float:
        """
        Ensures all financial values (stake, return) are rounded
        to exactly 2 decimal places as expected by Laravel's DB.
        """
        return round(float(value), 2)

    @staticmethod
    def sanitize_string(text: str) -> str:
        """
        Cleans up team names or market strings from Laravel
        to ensure consistency in logs and analysis.
        """
        if not text:
            return "unknown"
        return text.strip().title()

    @staticmethod
    def chunk_list(data: List[Any], size: int) -> List[List[Any]]:
        """
        Helper to split a list into smaller groups.
        Useful if the engine needs to process match groups in parallel.
        """
        return [data[i : i + size] for i in range(0, len(data), size)]

    @staticmethod
    def get_timestamp() -> str:
        """Returns ISO format timestamp for engine logging."""
        return datetime.utcnow().isoformat()

    @staticmethod
    def validate_odds_present(matches: List[Any]) -> bool:
        """
        Safety check to ensure Laravel provided at least one 
        market for every match before simulation begins.
        """
        for match in matches:
            if not match.markets or len(match.markets) == 0:
                return False
        return True