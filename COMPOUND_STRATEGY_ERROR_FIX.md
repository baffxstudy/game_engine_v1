# Compound Strategy EV Filtering Error - Fix Documentation

## Error Explanation

### What Happened

The error `"No valid selections for Compound strategy after EV filtering"` occurs when:

1. **Compound Strategy Requirements**: The Compound strategy filters selections based on Expected Value (EV)
2. **EV Threshold**: By default, it only accepts selections with EV ≥ -5.0 (configurable via `COMPOUND_LEG_EV_MIN`)
3. **Filtering Result**: If ALL selections have EV < -5.0, none pass the filter
4. **Failure**: The builder cannot generate any slips, causing the error

### Why This Happens

The Compound strategy is designed for **high-risk, high-reward** accumulators (5-7 legs per slip). It requires:

- **Favorable Expected Value**: Selections should have positive or near-positive EV
- **Quality Odds**: Odds that provide value relative to probabilities
- **Sufficient Data**: At least 7 matches with valid market data

Common causes:
- **Unfavorable odds**: Bookmaker odds don't provide value
- **Low probabilities**: Selections have low win probabilities
- **Market conditions**: Current market conditions don't favor compound accumulators
- **Data quality**: Missing or incomplete probability/odds data

## Fixes Implemented

### 1. **Progressive Fallback Mechanism**

The Compound builder now uses a **three-tier fallback** approach:

```python
# Tier 1: Try default threshold (-5.0)
scored = _prepare_ev_scored_selections(registry, -5.0, ev_scaling)

# Tier 2: If no selections, relax threshold to -10.0
if not scored:
    scored = _prepare_ev_scored_selections(registry, -10.0, ev_scaling)

# Tier 3: If still no selections, use all available (no EV filter)
if not scored:
    scored = _prepare_ev_scored_selections(registry, float('-inf'), ev_scaling)
```

**Benefits:**
- Automatically handles cases where EV threshold is too strict
- Still maintains quality by preferring better EV selections
- Only falls back to all selections as last resort

### 2. **Enhanced Diagnostics**

Added comprehensive logging to understand why selections are filtered:

```python
[EV_PREP] EV stats: min=-12.45, max=2.30, avg=-5.67, threshold=-5.00, passed=15/120
```

**Information Provided:**
- Minimum EV value found
- Maximum EV value found
- Average EV value
- Threshold used
- How many selections passed vs total

### 3. **Better Error Messages**

Improved error messages with actionable suggestions:

**Before:**
```
No valid selections for Compound strategy after EV filtering
```

**After:**
```
No valid selections for Compound strategy after EV filtering. 
Tried thresholds: -5.0, -10.0, -inf. 
Total selections in registry: 120. 
This may indicate insufficient match data or unfavorable odds.
```

### 4. **API-Level Error Handling**

Enhanced API error responses with helpful guidance:

```json
{
  "error": "Compound strategy cannot generate slips with current match data",
  "reason": "No selections passed Expected Value (EV) filtering criteria",
  "suggestion": "Try using 'balanced' or 'maxwin' strategy instead, or ensure matches have sufficient favorable odds and probabilities.",
  "request_id": "req_1234567890_abcd"
}
```

## Configuration Options

### Environment Variables

You can adjust the EV threshold via environment variable:

```bash
# Default: -5.0
export COMPOUND_LEG_EV_MIN=-5.0

# More lenient (allows more selections)
export COMPOUND_LEG_EV_MIN=-10.0

# Very strict (only high-value selections)
export COMPOUND_LEG_EV_MIN=0.0
```

### Other Compound Strategy Settings

```bash
# Number of Monte Carlo simulations
export COMPOUND_MC_SIMULATIONS=10000

# Target coverage percentage
export COMPOUND_TARGET_COVERAGE=0.95

# Minimum slip EV
export COMPOUND_MIN_EV=-15.0

# Maximum total odds
export COMPOUND_MAX_ODDS=1000.0
```

## Recommendations

### When to Use Compound Strategy

✅ **Good for:**
- High-risk, high-reward scenarios
- When you have 7+ matches with favorable odds
- When selections have positive or near-positive EV
- When you want lottery-style payouts

❌ **Not recommended for:**
- Conservative betting strategies
- When odds are unfavorable
- When you have fewer than 7 matches
- When EV values are consistently negative

### Alternative Strategies

If Compound strategy fails, consider:

1. **Balanced Strategy** (`"strategy": "balanced"`)
   - More lenient EV requirements
   - Better for consistent wins
   - Works with 3+ matches

2. **MaxWin Strategy** (`"strategy": "maxwin"`)
   - EV-optimized but less strict than Compound
   - Good middle ground
   - Works with 3+ matches

### Best Practices

1. **Check Match Data Quality**
   - Ensure probabilities are available
   - Verify odds are accurate
   - Check that markets are complete

2. **Monitor EV Values**
   - Review logs for EV statistics
   - Understand why selections are filtered
   - Adjust threshold if needed

3. **Use Appropriate Strategy**
   - Don't force Compound if data doesn't support it
   - Start with Balanced, then try MaxWin
   - Use Compound only when conditions are favorable

## Troubleshooting

### Issue: Still Getting "No valid selections" Error

**Solutions:**
1. Check logs for EV statistics
2. Lower `COMPOUND_LEG_EV_MIN` threshold
3. Verify match data quality
4. Try Balanced or MaxWin strategy instead

### Issue: Too Many Selections Filtered Out

**Solutions:**
1. Review EV statistics in logs
2. Check if probabilities are accurate
3. Verify odds are reasonable
4. Consider if Compound strategy is appropriate

### Issue: Want More Lenient Filtering

**Solutions:**
1. Set `COMPOUND_LEG_EV_MIN=-10.0` or lower
2. The fallback mechanism will automatically relax threshold
3. Consider using Balanced strategy instead

## Technical Details

### EV Calculation

Expected Value for a selection is calculated as:

```
EV = (Probability × (Odds - 1) × Stake) - ((1 - Probability) × Stake)
```

Or simplified:
```
EV = Probability × Odds × Stake - Stake
```

### Filtering Logic

```python
# For each selection:
ev = calculate_leg_ev(selection)

# Only include if EV meets threshold
if ev >= leg_ev_threshold:
    include_selection()
```

### Fallback Logic

```python
# Try default threshold
if selections_found:
    use_selections()
    
# Try relaxed threshold
elif relaxed_selections_found:
    use_relaxed_selections()
    log_warning()
    
# Use all selections (last resort)
elif all_selections_found:
    use_all_selections()
    log_warning()
    
# Fail with diagnostic info
else:
    raise_error_with_diagnostics()
```

## Testing

To test the fix:

```bash
# Test with default threshold
curl -X POST http://localhost:5000/api/v1/generate-slips \
  -H "Content-Type: application/json" \
  -d '{
    "master_slip": {
      "master_slip_id": 12345,
      "strategy": "compound",
      "matches": [...]
    }
  }'

# Check logs for EV statistics
tail -f logs/engine.log | grep EV_PREP
```

## Summary

The error occurs when the Compound strategy's EV filtering is too strict for the available data. The fix:

1. ✅ **Automatically relaxes** EV threshold if no selections pass
2. ✅ **Provides diagnostics** to understand why filtering fails
3. ✅ **Suggests alternatives** when Compound strategy isn't suitable
4. ✅ **Maintains quality** by preferring better EV selections

The system now gracefully handles cases where Compound strategy cannot generate slips, providing clear feedback and automatic fallbacks.
