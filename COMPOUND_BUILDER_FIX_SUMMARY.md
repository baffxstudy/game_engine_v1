# Compound Slip Builder Refactoring - Fix Summary

## Issues Fixed

### 1. **Missing ExpectedValueCalculator Import**
**Problem:** `ExpectedValueCalculator` was used but not imported, causing `NameError`.

**Fix:** 
- Added import from `slip_builder_maxwin`
- Created wrapper class that supports both float EV values and Slip objects
- Added fallback implementation if import fails

### 2. **EV Score Normalization Signature Mismatch**
**Problem:** `calculate_ev_score_normalized` in MaxWin takes a `Slip` object, but compound builder calls it with a float `ev` value.

**Fix:**
- Enhanced `calculate_ev_score_normalized` to accept both:
  - `float` EV value (for leg-level scoring)
  - `Slip` object (for slip-level scoring)
- Uses type checking to route to appropriate logic

### 3. **Fallback Logic Bugs in EV Filtering**
**Problem:** Fallback code tried to access attributes directly on `SelectionAdapter` instead of using adapter methods.

**Fix:**
- Changed `getattr(a, 'implied_probability', None)` → `a.get_probability()`
- Changed `getattr(a, 'odds', 0)` → `a.get_odds()`
- Changed `getattr(a, 'match_id', '?')` → `a.get_match_id()`
- Changed `getattr(a, 'market_code', '?')` → `a.get_market_code()`

### 4. **Improved Error Handling**
**Problem:** Exception handling was too broad and didn't provide useful diagnostics.

**Fix:**
- Added separate handling for EV calculation errors vs score normalization errors
- Improved logging with match_id and market_code for failed selections
- Added fallback count tracking
- Better diagnostic messages

### 5. **Enhanced Logging**
**Problem:** Insufficient diagnostic information when EV filtering fails.

**Fix:**
- Added EV statistics logging (min/max/avg)
- Logs how many selections passed threshold vs total
- Logs fallback usage count
- More detailed error messages with context

## Key Changes

### Import Section
```python
# Import ExpectedValueCalculator from MaxWin builder
try:
    from .slip_builder_maxwin import ExpectedValueCalculator as MaxWinEVCalc
    
    # Create wrapper that supports both Slip objects and float EV values
    class ExpectedValueCalculator:
        @staticmethod
        def calculate_ev_score_normalized(ev_or_slip, stake: float = 10.0) -> float:
            # Handles both float EV and Slip objects
            if isinstance(ev_or_slip, (int, float)):
                # Leg-level EV scoring
                ...
            else:
                # Slip-level EV scoring
                ...
```

### EV Preparation Method
```python
def _prepare_ev_scored_selections(...):
    # Fixed fallback logic
    try:
        ev = calculate_leg_ev_adapter(a)
        if ev >= leg_ev_threshold:
            score = ExpectedValueCalculator.calculate_ev_score_normalized(ev)  # Now works!
            scored.append((a, score))
    except Exception as e:
        # Fixed fallback uses adapter methods
        prob = a.get_probability()  # Fixed!
        odds = a.get_odds()  # Fixed!
        ...
```

### Progressive Fallback
```python
# Tier 1: Default threshold
scored = _prepare_ev_scored_selections(registry, -5.0, ev_scaling)

# Tier 2: Relaxed threshold
if not scored:
    scored = _prepare_ev_scored_selections(registry, -10.0, ev_scaling)

# Tier 3: No threshold
if not scored:
    scored = _prepare_ev_scored_selections(registry, float('-inf'), ev_scaling)
```

## Testing Recommendations

1. **Test with favorable odds** - Should use default threshold
2. **Test with unfavorable odds** - Should fall back to relaxed threshold
3. **Test with missing probabilities** - Should use fallback scoring
4. **Test with invalid data** - Should handle gracefully with diagnostics

## Expected Behavior

### Success Case
```
[EV_PREP] EV stats: min=-8.45, max=5.30, avg=-2.67, threshold=-5.00, passed=45/120, total_scored=45/120
[COMPOUND] Using default EV threshold -5.0 (45 selections)
```

### Fallback Case
```
[EV_PREP] EV stats: min=-15.20, max=-3.10, avg=-8.50, threshold=-5.00, passed=0/120, total_scored=0/120
[COMPOUND] No selections passed EV threshold -5.0, trying relaxed threshold...
[EV_PREP] EV stats: min=-15.20, max=-3.10, avg=-8.50, threshold=-10.00, passed=85/120, total_scored=85/120
[COMPOUND] Using relaxed EV threshold -10.0 (85 selections)
```

### Complete Failure Case
```
[EV_PREP] No EV values calculated. Errors: 0, Fallbacks: 120, Total adapters: 120, Scored: 120
[COMPOUND] Still no selections with relaxed threshold -10.0, using all available selections...
[COMPOUND] No selections available after all fallbacks. Total selections: 0, Total matches: 8
```

## Benefits

1. ✅ **No more NameError** - ExpectedValueCalculator properly imported
2. ✅ **Flexible EV scoring** - Works with both floats and Slip objects
3. ✅ **Better error handling** - Uses adapter methods correctly
4. ✅ **Improved diagnostics** - Detailed logging for troubleshooting
5. ✅ **Graceful degradation** - Progressive fallback ensures slips can be generated
6. ✅ **Backward compatible** - Works with existing code

## Migration Notes

- No breaking changes to API
- Existing code continues to work
- New fallback behavior is automatic
- Can be disabled by setting very strict thresholds
