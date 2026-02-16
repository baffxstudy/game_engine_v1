# This script is for educational and testing purposes only.
# It does not place real bets, submit stakes, or engage in real-money transactions.

import asyncio
import json
import logging
import math
import os
import re
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import async_playwright, Page, ElementHandle

# -----------------------------------------------------------------------------
# Configuration / Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("bet_placer_sim")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(handler)

SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class SimulationAbort(Exception):
    """Raised when the simulation must abort for safety or ambiguity reasons."""


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def safe_text(s: Optional[str]) -> str:
    return (s or "").strip()


def normalize_text(s: str) -> str:
    s = re.sub(r"\(.*?\)", "", s)  # remove parentheses content
    s = re.sub(r"[^0-9A-Za-z\s\-\+\.]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def tokens_of(s: str) -> List[str]:
    return [t for t in re.split(r"[\s\-\_/]+", normalize_text(s)) if t]


def to_float_if_possible(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        try:
            return float(str(v).strip())
        except Exception:
            return None


def timestamped(path_stem: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return SCREENSHOT_DIR / f"{path_stem}_{ts}.png"


# -----------------------------------------------------------------------------
# Selector Resolver - centralizes selector choice and logs choices
# -----------------------------------------------------------------------------
class SelectorResolver:
    def __init__(self, selectors_json: Dict[str, Any]):
        self.data = selectors_json.get("selectors", {})
        # Keep chosen map for observability
        self.chosen: Dict[str, str] = {}

    def _best_item_for_category(self, category: str) -> Optional[Dict[str, Any]]:
        cat = self.data.get(category)
        if not cat or not isinstance(cat.get("items", []), list):
            return None
        items = cat["items"]
        # prefer highest score, stable attributes first
        items_sorted = sorted(items, key=lambda x: float(x.get("score", 0)), reverse=True)
        return items_sorted[0] if items_sorted else None

    def resolve(self, category: str) -> Optional[str]:
        """
        Return a CSS/text selector string for a given category.
        Preference: highest-scored item -> selector field.
        Persist choice for logging.
        """
        item = self._best_item_for_category(category)
        if not item:
            return None
        sel = item.get("selector")
        if sel:
            self.chosen[category] = sel
            LOG.info(f"[SELECTOR] Chosen [{category}] -> {sel} (score={item.get('score')})")
            return sel
        # fallback attempts: try to build selector from attributes
        attrs = item.get("attributes", {})
        if attrs:
            # try stable attributes like data-op, id, role, data-*
            for k in ("data-op", "data-id", "id", "role", "aria-label"):
                if k in attrs:
                    candidate = f'[{k}="{attrs[k]}"]'
                    self.chosen[category] = candidate
                    LOG.info(f"[SELECTOR] Built fallback [{category}] -> {candidate} from attributes")
                    return candidate
        return None

    def all_selectors_in_category(self, category: str) -> List[str]:
        cat = self.data.get(category)
        if not cat:
            return []
        sels = []
        for item in cat.get("items", []):
            if item.get("selector"):
                sels.append(item["selector"])
        return sels


# -----------------------------------------------------------------------------
# Schema validation for slip payload (pre-flight)
# -----------------------------------------------------------------------------
REQUIRED_LEG_FIELDS = {"home_team", "away_team", "market", "selection"}


def validate_slip_schema(payload: Dict[str, Any]) -> None:
    master = payload.get("slip") or payload.get("master_slip") or payload
    if not isinstance(master, dict):
        raise SimulationAbort("Payload missing 'slip' / 'master_slip' root object")
    legs = master.get("legs")
    if not isinstance(legs, list) or not legs:
        raise SimulationAbort("Payload must contain 'legs' list with at least one leg")
    for idx, leg in enumerate(legs):
        if not isinstance(leg, dict):
            raise SimulationAbort(f"Leg {idx} is not an object")
        missing = REQUIRED_LEG_FIELDS - set(k for k in leg.keys() if leg.get(k) is not None)
        if missing:
            raise SimulationAbort(f"Leg {idx} missing required fields: {', '.join(sorted(missing))}")
        # odds optional but if provided must be numeric-like
        odds = leg.get("odds")
        if odds is not None:
            if to_float_if_possible(odds) is None:
                raise SimulationAbort(f"Leg {idx} provided invalid odds: {odds}")
    LOG.info("[VALIDATION] Slip schema validation passed")


# -----------------------------------------------------------------------------
# Match confidence scoring
# -----------------------------------------------------------------------------
BAD_KEYWORDS = {"women", "womens", "w", "u21", "u23", "reserves", "reserve", "u19", "u18", "u17"}


def match_confidence_score(container_text: str, leg: Dict[str, Any]) -> int:
    """
    Scoring:
      +2 per exact team name match
      +1 if match_id found
      -2 if bad keyword detected
    """
    score = 0
    txt = normalize_text(container_text)
    # exact team name matches (word boundaries)
    home = normalize_text(str(leg.get("home_team", "")))
    away = normalize_text(str(leg.get("away_team", "")))
    if home and re.search(rf"\b{re.escape(home)}\b", txt):
        score += 2
    if away and re.search(rf"\b{re.escape(away)}\b", txt):
        score += 2
    # match_id presence
    mid = leg.get("match_id") or leg.get("id") or leg.get("event_id")
    if mid is not None and str(mid).strip() and str(mid) in container_text:
        score += 1
    # penalize presence of unwanted keywords
    for bad in BAD_KEYWORDS:
        if bad in txt:
            score -= 2
            break
    return score


# -----------------------------------------------------------------------------
# Market matching logic inside a match container
# -----------------------------------------------------------------------------
async def find_candidate_markets(match_el: ElementHandle, resolver: SelectorResolver) -> List[Tuple[ElementHandle, str]]:
    """
    Returns list of (element, normalized_text) for market candidates inside the match element.
    Uses market_buttons and market_buttons fallback selectors.
    """
    markets: List[Tuple[ElementHandle, str]] = []
    # try primary category
    for sel in resolver.all_selectors_in_category("market_buttons"):
        try:
            found = await match_el.query_selector_all(sel)
        except Exception:
            found = []
        if found:
            for el in found:
                try:
                    txt = safe_text(await el.inner_text())
                except Exception:
                    txt = ""
                markets.append((el, normalize_text(txt)))
            # prefer the first category that returned results
            if markets:
                return markets
    # fallback: try selection_buttons or odds_buttons as potential market text containers
    for cat in ("selection_buttons", "odds_buttons"):
        for sel in resolver.all_selectors_in_category(cat):
            try:
                found = await match_el.query_selector_all(sel)
            except Exception:
                found = []
            if found:
                for el in found:
                    try:
                        txt = safe_text(await el.inner_text())
                    except Exception:
                        txt = ""
                    markets.append((el, normalize_text(txt)))
        if markets:
            return markets
    return markets


async def choose_market_from_candidates(
    candidates: List[Tuple[ElementHandle, str]],
    requested_market: str,
    requested_odds: Optional[float],
) -> ElementHandle:
    """
    Priority:
      1) Exact normalized text match
      2) Exact odds-context match (if odds provided)
      3) Strict multi-token match (>=2 tokens)
    If ambiguity detected at same confidence level -> raise SimulationAbort
    """
    if not candidates:
        raise SimulationAbort("No market candidates found inside match container")

    req_norm = normalize_text(requested_market)
    req_tokens = tokens_of(requested_market)

    scored: List[Tuple[int, ElementHandle, str]] = []

    # Stage 1: exact normalized text match
    for el, norm in candidates:
        if norm == req_norm:
            scored.append((3, el, norm))

    if scored:
        if len(scored) > 1:
            raise SimulationAbort("Multiple markets match exact normalized text -> abort")
        return scored[0][1]

    # Stage 2: odds-context match
    if requested_odds is not None:
        close_matches = []
        for el, norm in candidates:
            # attempt to find any numeric in norm or inside element text representing odds
            # we inspect norm for floating point substrings
            m = re.findall(r"\d+(?:\.\d+)?", norm)
            for num in m:
                try:
                    v = float(num)
                except Exception:
                    continue
                # allow a tiny tolerance
                if math.isclose(v, requested_odds, rel_tol=1e-3, abs_tol=1e-3):
                    close_matches.append((2, el, norm))
        if close_matches:
            if len(close_matches) > 1:
                raise SimulationAbort("Multiple markets match requested odds -> abort")
            return close_matches[0][1]

    # Stage 3: strict multi-token match (minimum 2 tokens)
    if len(req_tokens) >= 2:
        strict_matches = []
        for el, norm in candidates:
            norm_tokens = tokens_of(norm)
            if all(tok in norm_tokens for tok in req_tokens):
                strict_matches.append((1, el, norm))
        if strict_matches:
            if len(strict_matches) > 1:
                raise SimulationAbort("Multiple markets match strict multi-token criteria -> abort")
            return strict_matches[0][1]

    # No reliable match
    raise SimulationAbort(f"Market '{requested_market}' could not be matched reliably")


# -----------------------------------------------------------------------------
# Selection (outcome) matching and click
# -----------------------------------------------------------------------------
async def find_and_click_selection_in_market(
    page: Page,
    market_el: ElementHandle,
    requested_selection: str,
    resolver: SelectorResolver,
    leg_index: int,
    ensure_added_to_betslip: bool = True
) -> Dict[str, Any]:
    """
    Find the specific selection (outcome) inside a market element and click it.
    Returns structured result.
    """
    request_norm = normalize_text(requested_selection)
    selection_sels = resolver.all_selectors_in_category("selection_buttons")
    if not selection_sels:
        raise SimulationAbort("No selection button selectors available in selectors JSON")

    candidate_buttons: List[Tuple[ElementHandle, str]] = []
    for sel in selection_sels:
        try:
            found = await market_el.query_selector_all(sel)
        except Exception:
            found = []
        for el in found:
            try:
                txt = safe_text(await el.inner_text())
            except Exception:
                txt = ""
            candidate_buttons.append((el, normalize_text(txt)))
        if candidate_buttons:
            break  # use the first category that returned results

    if not candidate_buttons:
        raise SimulationAbort("No selection buttons found inside market element")

    # Try exact match
    exact_matches = [(el, txt) for el, txt in candidate_buttons if txt == request_norm]
    if len(exact_matches) == 1:
        target_el = exact_matches[0][0]
    elif len(exact_matches) > 1:
        raise SimulationAbort("Multiple selection buttons match exact text -> abort")
    else:
        # try partial match (containment) but only if unambiguous
        partial_matches = [(el, txt) for el, txt in candidate_buttons if request_norm in txt or txt in request_norm]
        if len(partial_matches) == 1:
            target_el = partial_matches[0][0]
        elif len(partial_matches) > 1:
            raise SimulationAbort("Multiple selection buttons match partially -> abort")
        else:
            # try token overlap heuristics (at least 2 tokens overlapping)
            req_tokens = set(tokens_of(requested_selection))
            token_matches = []
            for el, txt in candidate_buttons:
                common = req_tokens.intersection(set(tokens_of(txt)))
                if len(common) >= 2 or (req_tokens and len(common) >= 1 and len(req_tokens) == 1):
                    token_matches.append((el, txt))
            if len(token_matches) == 1:
                target_el = token_matches[0][0]
            else:
                raise SimulationAbort(f"Selection '{requested_selection}' ambiguous or not found inside market")

    # Scroll into view and click (simulate)
    try:
        await target_el.scroll_into_view_if_needed()
        await target_el.click(timeout=5000)
        LOG.info(f"[LEG] clicked selection for leg #{leg_index}: '{requested_selection}'")
    except Exception as e:
        raise SimulationAbort(f"Failed to click selection '{requested_selection}': {e}")

    return {"match_found": True, "market_found": True, "selection_clicked": True}


# -----------------------------------------------------------------------------
# Safety checks for stake/submit selectors presence on page
# -----------------------------------------------------------------------------
async def safety_gate_checks(page: Page, resolver: SelectorResolver) -> None:
    # If stake or submit selectors are present, abort (we never interact with them)
    stake_selectors = resolver.all_selectors_in_category("stake_input")
    submit_selectors = resolver.all_selectors_in_category("submit_button")
    present = []
    for sel in stake_selectors + submit_selectors:
        try:
            el = await page.query_selector(sel)
        except Exception:
            el = None
        if el:
            present.append(sel)
    if present:
        raise SimulationAbort(f"Safety: Found stake/submit selectors on page -> {present}")


# -----------------------------------------------------------------------------
# Verify betslip contents
# -----------------------------------------------------------------------------
async def verify_betslip(page: Page, resolver: SelectorResolver, legs: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Verify that betslip contains expected selections.
    Returns (ok, missing_list)
    """
    # find betslip container
    bet_sels = resolver.all_selectors_in_category("betslip_container")
    bet_text = ""
    for sel in bet_sels:
        try:
            el = await page.query_selector(sel)
        except Exception:
            el = None
        if el:
            try:
                bet_text = safe_text(await el.inner_text())
            except Exception:
                bet_text = ""
            if bet_text:
                break
    if not bet_text:
        return False, [f"No betslip container found (tried: {bet_sels})"]

    missing = []
    for idx, leg in enumerate(legs):
        sel_text = normalize_text(str(leg.get("selection", "")))
        if sel_text and sel_text not in normalize_text(bet_text):
            missing.append(f"leg[{idx}] selection '{leg.get('selection')}' not found in betslip")
    return (len(missing) == 0), missing


# -----------------------------------------------------------------------------
# Core Simulation Flow
# -----------------------------------------------------------------------------
async def run_simulation(selectors_path: Path, slip_path: Path, headless: bool = True, slow_mo: int = 0) -> Dict[str, Any]:
    # Load files
    try:
        selectors_json = json.loads(selectors_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SimulationAbort(f"Failed to load selectors JSON: {e}")
    try:
        slip_payload = json.loads(slip_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SimulationAbort(f"Failed to load slip JSON: {e}")

    # Validate slip schema
    validate_slip_schema(slip_payload)
    legs = (slip_payload.get("slip") or slip_payload.get("master_slip") or slip_payload).get("legs", [])

    resolver = SelectorResolver(selectors_json)

    # Choose main site url from scan metadata
    site_url = selectors_json.get("scan_metadata", {}).get("url")
    if not site_url:
        raise SimulationAbort("Selectors JSON missing 'scan_metadata.url'")

    # Launch Playwright and simulate actions
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, slow_mo=slow_mo)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            await page.goto(site_url, wait_until="networkidle", timeout=60000)
            LOG.info(f"[NAV] Opened {site_url}")

            # Global safety gate
            await safety_gate_checks(page, resolver)

            # Resolve match container selectors to query for match containers
            match_container_selectors = resolver.all_selectors_in_category("match_containers")
            if not match_container_selectors:
                raise SimulationAbort("No 'match_containers' selectors available")

            # collect all match elements for scoring
            all_match_elements: List[ElementHandle] = []
            for sel in match_container_selectors:
                try:
                    found = await page.query_selector_all(sel)
                except Exception:
                    found = []
                if found:
                    all_match_elements.extend(found)
            if not all_match_elements:
                raise SimulationAbort("No match container elements found using provided selectors")

            # For each leg, find best match container, then market, then selection
            structured_results = []
            for idx, leg in enumerate(legs):
                LOG.info(f"[LEG] Processing leg #{idx} -> {leg.get('home_team')} vs {leg.get('away_team')} | market={leg.get('market')} selection={leg.get('selection')}")
                # compute confidence scores for each match element
                scored_elements: List[Tuple[int, ElementHandle, str]] = []
                for el in all_match_elements:
                    try:
                        txt = safe_text(await el.inner_text())
                    except Exception:
                        txt = ""
                    sc = match_confidence_score(txt, leg)
                    scored_elements.append((sc, el, txt))
                # pick the highest score, ensure unique and positive
                if not scored_elements:
                    raise SimulationAbort("No candidate match elements to score")
                scored_elements.sort(key=lambda x: x[0], reverse=True)
                top_score = scored_elements[0][0]
                # check ambiguity: count how many with same top_score
                top_count = sum(1 for s, _, _ in scored_elements if s == top_score)
                if top_score <= 0:
                    await page.screenshot(path=str(timestamped(f"abort_leg{idx}_no_positive_score")))
                    raise SimulationAbort(f"No sufficiently confident match found for leg #{idx}")
                if top_count != 1:
                    await page.screenshot(path=str(timestamped(f"abort_leg{idx}_ambiguous_match")))
                    raise SimulationAbort(f"Ambiguous matches for leg #{idx} - top_score={top_score}, candidates={top_count}")
                # got unique best element
                _, match_el, match_text = scored_elements[0]
                LOG.info(f"[LEG] match_found (score={top_score}) for leg #{idx}")

                # find markets inside chosen match element
                candidate_markets = await find_candidate_markets(match_el, resolver)
                if not candidate_markets:
                    await page.screenshot(path=str(timestamped(f"abort_leg{idx}_no_markets")))
                    raise SimulationAbort(f"No markets found for leg #{idx} inside match container")

                # decide market element
                requested_market = leg.get("market")
                requested_odds = to_float_if_possible(leg.get("odds"))
                try:
                    chosen_market_el = await choose_market_from_candidates(candidate_markets, requested_market, requested_odds)
                except SimulationAbort as e:
                    await page.screenshot(path=str(timestamped(f"abort_leg{idx}_market_ambiguous")))
                    raise

                LOG.info(f"[LEG] market_found for leg #{idx}: '{requested_market}'")

                # click selection inside market
                try:
                    click_result = await find_and_click_selection_in_market(page, chosen_market_el, leg.get("selection"), resolver, idx)
                except SimulationAbort as e:
                    await page.screenshot(path=str(timestamped(f"abort_leg{idx}_selection_fail")))
                    raise

                structured_results.append({
                    "leg_index": idx,
                    "match_found": True,
                    "market_found": True,
                    "selection_clicked": True,
                })

            # After all legs processed, verify betslip
            ok, missing = await verify_betslip(page, resolver, legs)
            if not ok:
                await page.screenshot(path=str(timestamped("abort_betslip_verify")))
                raise SimulationAbort(f"Betslip verification failed: {missing}")

            # Do not interact with stake or submit - safety stop
            LOG.info("[SUCCESS] All selections added and verified in betslip. STAGE STOP: no stake or submission will be performed.")
            result = {
                "success": True,
                "message": "Simulated bet setup complete - selections added and verified on betslip. No stake or submission performed.",
                "slip_id": (slip_payload.get("slip") or slip_payload.get("master_slip") or slip_payload).get("slip_id"),
                "selections_added": len(legs),
                "structured_results": structured_results,
            }
            return result

        except SimulationAbort as e:
            LOG.error(f"[ABORT] {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            # unexpected error - capture screenshot and return failure
            LOG.exception("Unexpected error during simulation")
            path = timestamped("error_unexpected")
            try:
                await page.screenshot(path=str(path))
                LOG.info(f"[SCREENSHOT] Saved failure screenshot: {path}")
            except Exception:
                LOG.warning("[SCREENSHOT] Failed to capture screenshot")
            return {"success": False, "error": f"Unexpected error: {e}"}
        finally:
            try:
                await context.close()
                await browser.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="Educational bet placer simulation (Playwright, dry-run)")
    parser.add_argument("--selectors", "-s", type=str, required=True, help="Path to selectors JSON (pre-scanned)")
    parser.add_argument("--slip", "-l", type=str, required=True, help="Path to slip JSON payload")
    parser.add_argument("--headed", action="store_true", help="Run with UI (headed)")
    parser.add_argument("--slow", type=int, default=0, help="Playwright slowMo in ms")
    args = parser.parse_args()

    selectors_path = Path(args.selectors)
    slip_path = Path(args.slip)
    if not selectors_path.exists():
        LOG.error("Selectors JSON not found: %s", selectors_path)
        sys.exit(2)
    if not slip_path.exists():
        LOG.error("Slip JSON not found: %s", slip_path)
        sys.exit(2)

    loop = asyncio.get_event_loop()
    res = loop.run_until_complete(run_simulation(selectors_path, slip_path, headless=not args.headed, slow_mo=args.slow))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()