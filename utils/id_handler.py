# game_engine/utils/id_handler.py

from typing import Any, Union, Optional
import logging

logger = logging.getLogger(__name__)

class IDFlex:
    """Utility class for handling flexible ID types"""
    
    @staticmethod
    def to_string(value: Any, prefix: str = "") -> str:
        """
        Convert any value to string ID.
        
        Args:
            value: Any value to convert to string
            prefix: Optional prefix to add
            
        Returns:
            String representation of the ID
        """
        try:
            if value is None:
                return f"{prefix}none"
            
            # Handle common numeric types
            if isinstance(value, (int, float, Decimal)):
                return f"{prefix}{int(value)}" if isinstance(value, (int, Decimal)) else f"{prefix}{value}"
            
            # Handle string types
            if isinstance(value, str):
                if value.strip() == "":
                    return f"{prefix}empty"
                return f"{prefix}{value.strip()}"
            
            # Handle objects with ID attribute
            if hasattr(value, 'id'):
                return IDFlex.to_string(value.id, prefix)
            
            # Fallback to string representation
            return f"{prefix}{str(value)}"
            
        except Exception as e:
            logger.warning(f"Failed to convert ID to string: {value}, error: {e}")
            return f"{prefix}error_{hash(str(value))}"
    
    @staticmethod
    def normalize_match_object(match_obj: Any) -> Any:
        """
        Normalize all IDs in a match object to strings.
        
        Args:
            match_obj: Match data object
            
        Returns:
            Normalized object with string IDs
        """
        try:
            # Create a copy or wrapper
            if hasattr(match_obj, '__dict__'):
                # It's an object with attributes
                class NormalizedMatch:
                    def __init__(self, original):
                        self._original = original
                        
                    def __getattr__(self, name):
                        value = getattr(self._original, name, None)
                        
                        # Convert ID fields to strings
                        if name in ['match_id', 'master_slip_id', 'original_master_slip_id', 
                                   'team_id', 'home_team_id', 'away_team_id', 'slip_id']:
                            return IDFlex.to_string(value)
                        
                        return value
                
                return NormalizedMatch(match_obj)
            
            elif isinstance(match_obj, dict):
                # It's a dictionary
                normalized = match_obj.copy()
                
                # Convert ID fields
                id_fields = ['match_id', 'master_slip_id', 'original_master_slip_id', 
                            'team_id', 'home_team_id', 'away_team_id', 'slip_id']
                
                for field in id_fields:
                    if field in normalized:
                        normalized[field] = IDFlex.to_string(normalized[field])
                
                return normalized
            
            return match_obj
            
        except Exception as e:
            logger.error(f"Failed to normalize match object: {e}")
            return match_obj
    
    @staticmethod
    def ensure_string_ids(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively ensure all IDs in a dictionary are strings.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Data with all IDs as strings
        """
        if not isinstance(data, dict):
            return data
        
        result = {}
        
        for key, value in data.items():
            # Check if this is an ID field
            is_id_field = any(id_key in key.lower() for id_key in ['id', 'slip', 'match'])
            
            if is_id_field and value is not None:
                # Convert to string
                result[key] = IDFlex.to_string(value)
            elif isinstance(value, dict):
                # Recursively process dictionaries
                result[key] = IDFlex.ensure_string_ids(value)
            elif isinstance(value, list):
                # Process lists
                result[key] = [
                    IDFlex.ensure_string_ids(item) if isinstance(item, dict) else 
                    (IDFlex.to_string(item) if is_id_field and i == 0 else item)
                    for i, item in enumerate(value)
                ]
            else:
                result[key] = value
        
        return result