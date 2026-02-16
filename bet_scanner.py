"""
BETTING SITE SELECTOR SCANNER - EDUCATIONAL TOOL
=================================================
Comprehensive selector identification tool for web automation testing.

DISCLAIMER: This tool is for EDUCATIONAL PURPOSES ONLY.
- Designed for testing automation frameworks on test environments
- Always comply with website terms of service and local laws
- Never use for unauthorized automation or scraping
- Use responsibly and ethically

Features:
- Multi-stage scanning with user interaction
- Pattern-based selector detection
- Betting-specific element identification
- Generates comprehensive JSON selector map
- Screenshot capture for verification
- Detailed logging and statistics

Usage: python betting_selector_scanner.py
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from playwright.async_api import async_playwright, Page, ElementHandle

# Configure logging - FIX for Windows Unicode issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'selector_scan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'  # CRITICAL: Use UTF-8 encoding for emoji support
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure stdout to use UTF-8 on Windows
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


@dataclass
class ElementData:
    """Structured element information"""
    selector: str
    tag_name: str
    classes: List[str]
    id: Optional[str] = None
    text_content: Optional[str] = None
    attributes: Optional[Dict[str, str]] = None
    xpath: Optional[str] = None
    score: int = 0  # Relevance score
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class BettingSelectorScanner:
    """
    Comprehensive betting site selector scanner.
    
    Combines pattern-based scanning with interactive exploration
    to identify all selectors needed for bet placement automation.
    """
    
    # Betting-specific patterns to search for
    BETTING_PATTERNS = {
        "search": ["search", "find", "query", "filter"],
        "match": ["match", "event", "fixture", "game", "contest"],
        "team": ["team", "home", "away", "competitor", "player"],
        "odds": ["odds", "price", "selection", "outcome", "market"],
        "betslip": ["slip", "bet", "ticket", "coupon", "cart"],
        "stake": ["stake", "amount", "wager", "bet-amount"],
        "submit": ["place", "submit", "confirm", "bet-now", "accept"],
        "sport": ["football", "soccer", "sport", "league"],
    }
    
    # Common class/id patterns for betting sites
    COMMON_PATTERNS = [
        "odds", "selection", "match", "event", "slip", "bet",
        "stake", "market", "outcome", "price", "button"
    ]
    
    def __init__(self, url: str, headless: bool = False):
        self.url = url
        self.headless = headless
        self.results = {
            "scan_metadata": {
                "timestamp": datetime.now().isoformat(),
                "url": url,
                "scanner_version": "2.0.0"
            },
            "selectors": {},
            "interaction_flow": [],
            "screenshots": [],
            "statistics": {}
        }
        
        # Create output directories
        self.output_dir = Path("selector_scans")
        self.screenshots_dir = self.output_dir / "screenshots"
        self.output_dir.mkdir(exist_ok=True)
        self.screenshots_dir.mkdir(exist_ok=True)
    
    async def scan(self) -> Dict[str, Any]:
        """Execute complete scanning process"""
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=self.headless)
            page = await browser.new_page()
            
            try:
                logger.info(f"[START] Starting scan of {self.url}")
                
                # Stage 1: Homepage Analysis (CRITICAL - must succeed)
                try:
                    await self._scan_homepage(page)
                except Exception as e:
                    logger.error(f"[ERROR] Stage 1 failed: {e}")
                    self.results["error"] = f"Stage 1 failed: {str(e)}"
                    # Don't exit - try to continue
                
                # Stage 2: Navigation to Sports Section (OPTIONAL)
                try:
                    await self._scan_sports_navigation(page)
                except Exception as e:
                    logger.warning(f"[WARN] Stage 2 failed: {e}")
                    # Continue - not critical
                
                # Stage 2.5: Real Madrid Search (NEW)
                try:
                    await self._search_and_scan_real_madrid(page)
                except Exception as e:
                    logger.warning(f"[WARN] Real Madrid search failed: {e}")
                    # Continue - not critical
                
                # Stage 3: Match List Analysis (OPTIONAL)
                try:
                    await self._scan_match_list(page)
                except Exception as e:
                    logger.warning(f"[WARN] Stage 3 failed: {e}")
                    # Continue - not critical
                
                # Stage 4: Match Detail Analysis (OPTIONAL)
                try:
                    await self._scan_match_details(page)
                except Exception as e:
                    logger.warning(f"[WARN] Stage 4 failed: {e}")
                    # Continue - not critical
                
                # Stage 5: Bet Slip Analysis (OPTIONAL)
                try:
                    await self._scan_betslip_elements(page)
                except Exception as e:
                    logger.warning(f"[WARN] Stage 5 failed: {e}")
                    # Continue - not critical
                
                # Stage 6: Form Elements (OPTIONAL)
                try:
                    await self._scan_form_elements(page)
                except Exception as e:
                    logger.warning(f"[WARN] Stage 6 failed: {e}")
                    # Continue - not critical
                
                # Always calculate statistics (even if partial scan)
                try:
                    self._calculate_statistics()
                except Exception as e:
                    logger.warning(f"[WARN] Statistics calculation failed: {e}")
                    # Set empty statistics if calculation fails
                    self.results["statistics"] = {
                        "total_elements_found": 0,
                        "categories_scanned": 0,
                        "elements_per_category": {},
                        "screenshots_captured": len(self.results.get("screenshots", [])),
                        "interaction_stages": len(self.results.get("interaction_flow", []))
                    }
                
                # Always save results (even if partial)
                try:
                    self._save_results()
                except Exception as e:
                    logger.error(f"[ERROR] Failed to save results: {e}")
                
                logger.info("[SUCCESS] Scan completed successfully")
                
            except Exception as e:
                logger.error(f"[ERROR] Scan failed: {e}", exc_info=True)
                self.results["error"] = str(e)
                
                # Try to save partial results anyway
                try:
                    if "statistics" not in self.results:
                        self.results["statistics"] = {
                            "total_elements_found": 0,
                            "categories_scanned": 0,
                            "elements_per_category": {},
                            "screenshots_captured": len(self.results.get("screenshots", [])),
                            "interaction_stages": len(self.results.get("interaction_flow", []))
                        }
                    self._save_results()
                except Exception:
                    pass
            finally:
                await browser.close()
        
        return self.results
    
    async def _scan_homepage(self, page: Page):
        """Stage 1: Analyze homepage structure"""
        logger.info("[STAGE1] Stage 1: Scanning homepage...")
        
        try:
            # More lenient page load - use domcontentloaded instead of networkidle
            await page.goto(
                self.url,
                wait_until="domcontentloaded",  # More permissive than networkidle
                timeout=60000  # Increased timeout to 60 seconds
            )
            await page.wait_for_timeout(5000)  # Extra wait for dynamic content
        except Exception as e:
            logger.warning(f"[WARN] Initial page load failed: {e}")
            # Try alternative strategy
            try:
                await page.goto(self.url, wait_until="load", timeout=60000)
                await page.wait_for_timeout(5000)
            except Exception as e2:
                logger.error(f"[ERROR] All page load strategies failed: {e2}")
                raise
        
        # CRITICAL: Dismiss any popups/modals that might block interaction
        await self._dismiss_popups(page)
        
        # Take screenshot
        screenshot_path = await self._take_screenshot(page, "01_homepage")
        
        # Scan common elements
        await self._scan_category(page, "search_elements", self.BETTING_PATTERNS["search"])
        await self._scan_category(page, "navigation_links", ["nav", "menu", "header"])
        await self._scan_category(page, "buttons", ["button", "btn"])
        
        logger.info("[OK] Homepage scan complete")
    
    async def _scan_sports_navigation(self, page: Page):
        """Stage 2: Navigate to sports/football section"""
        logger.info("[STAGE2] Stage 2: Navigating to sports section...")
        
        # CRITICAL: Close any modals/popups first
        await self._close_modals(page)
        
        # Try to find and click on football/soccer link
        sport_patterns = ["football", "soccer", "sport"]
        sport_link = None
        
        for pattern in sport_patterns:
            try:
                # Try multiple selector strategies
                selectors = [
                    f'a:has-text("{pattern}")',
                    f'a:has-text("{pattern.upper()}")',
                    f'a:has-text("{pattern.capitalize()}")',
                    f'[data-sport*="{pattern}"]',
                    f'[class*="{pattern}"]'
                ]
                
                for selector in selectors:
                    elements = page.locator(selector)
                    if await elements.count() > 0:
                        sport_link = elements.first
                        logger.info(f"[OK] Found sport link: {selector}")
                        break
                
                if sport_link:
                    break
            except Exception as e:
                logger.debug(f"Pattern {pattern} not found: {e}")
        
        if sport_link:
            try:
                # Close modals again right before clicking
                await self._close_modals(page)
                
                # Click with force option to bypass overlay issues
                await sport_link.click(force=True, timeout=10000)
                await page.wait_for_timeout(3000)
                await self._take_screenshot(page, "02_sports_section")
                self.results["interaction_flow"].append({
                    "stage": "sports_navigation",
                    "action": "clicked_sport_link",
                    "success": True
                })
            except Exception as e:
                logger.warning(f"[WARN] Could not click sport link: {e}")
                # Continue anyway - we might already be on sports page
                await self._take_screenshot(page, "02_sports_section_failed")
                self.results["interaction_flow"].append({
                    "stage": "sports_navigation",
                    "action": "clicked_sport_link",
                    "success": False,
                    "error": str(e)
                })
        else:
            logger.warning("[WARN] Could not find sports navigation link")
            self.results["interaction_flow"].append({
                "stage": "sports_navigation",
                "action": "clicked_sport_link",
                "success": False
            })
    
    async def _dismiss_popups(self, page: Page):
        """Aggressively dismiss any popups, modals, or dialogs blocking interaction"""
        logger.info("[POPUP] Checking for popups/modals...")
        
        try:
            # Strategy 1: Close region/country selector dialog
            region_selectors = [
                '[data-op="region_country-item"]',  # Region selection items
                '[data-op="region-wrap"]',  # Region wrapper
                '#esDialog0',  # The specific dialog ID
                '.es-dialog-wrap',  # Dialog wrapper class
            ]
            
            for selector in region_selectors:
                try:
                    elem = page.locator(selector)
                    if await elem.count() > 0:
                        logger.info(f"[POPUP] Found blocking element: {selector}")
                        # If it's the region item, click it to select
                        if 'region_country-item' in selector:
                            await elem.first.click(force=True, timeout=3000)
                            await page.wait_for_timeout(1000)
                            logger.info("[POPUP] Selected region")
                            return
                except Exception:
                    continue
            
            # Strategy 2: Press Escape key multiple times
            for _ in range(3):
                await page.keyboard.press('Escape')
                await page.wait_for_timeout(500)
            logger.info("[POPUP] Pressed Escape key")
            
            # Strategy 3: Click close buttons
            close_selectors = [
                'button:has-text("Close")',
                'button:has-text("×")',
                'button:has-text("✕")',
                '[aria-label="Close"]',
                '[aria-label*="close"]',
                '.close-btn',
                '.modal-close',
                '.dialog-close',
                '[data-dismiss="modal"]',
            ]
            
            for selector in close_selectors:
                try:
                    close_btn = page.locator(selector)
                    if await close_btn.count() > 0:
                        await close_btn.first.click(force=True, timeout=2000)
                        await page.wait_for_timeout(500)
                        logger.info(f"[POPUP] Closed popup: {selector}")
                        return
                except Exception:
                    continue
            
            # Strategy 4: Click outside modal (on overlay/mask)
            mask_selectors = ['.mask', '.overlay', '.modal-backdrop', '.layout.mask']
            for selector in mask_selectors:
                try:
                    mask = page.locator(selector)
                    if await mask.count() > 0:
                        await mask.first.click(force=True, timeout=2000)
                        await page.wait_for_timeout(500)
                        logger.info(f"[POPUP] Clicked overlay: {selector}")
                        return
                except Exception:
                    continue
            
            logger.info("[POPUP] No blocking popups detected")
            
        except Exception as e:
            logger.warning(f"[POPUP] Error during popup dismissal: {e}")
    
    async def _close_modals(self, page: Page):
        """Close any modal dialogs or popups that might be blocking interaction"""
        await self._dismiss_popups(page)
    
    async def _search_and_scan_real_madrid(self, page: Page):
        """
        NEW: Search for Real Madrid and scan match selectors.
        
        Steps:
        1. Find search input
        2. Type "Real Madrid"
        3. Submit search
        4. Wait for results
        5. Scan match selectors
        6. Click first match
        """
        logger.info("[RM] Searching for Real Madrid...")
        
        # Find search input (try multiple selectors)
        search_input = None
        search_selectors = [
            'input[type="search"]',
            'input[placeholder*="earch" i]',
            'input[placeholder*="ind" i]',
            'input[class*="search" i]',
            'input[type="text"]',  # Fallback
        ]
        
        for selector in search_selectors:
            try:
                elem = page.locator(selector)
                if await elem.count() > 0:
                    first = elem.first
                    if await first.is_visible() and await first.is_enabled():
                        search_input = first
                        logger.info(f"[RM] Found input: {selector}")
                        break
            except Exception:
                continue
        
        if not search_input:
            logger.warning("[RM] No search input found")
            return
        
        try:
            # Type Real Madrid
            await search_input.fill("")
            await search_input.type("Real Madrid", delay=100)
            await page.wait_for_timeout(1000)
            logger.info("[RM] Typed 'Real Madrid'")
            
            # Submit (Enter key)
            await page.keyboard.press('Enter')
            logger.info("[RM] Pressed Enter")
            
            # Wait for results
            await page.wait_for_timeout(5000)
            await self._take_screenshot(page, "02_rm_search")
            
            # Scan results
            await self._scan_betting_specific(page, "rm_matches", self.BETTING_PATTERNS["match"])
            await self._scan_betting_specific(page, "rm_teams", self.BETTING_PATTERNS["team"])
            await self._scan_betting_specific(page, "rm_odds", self.BETTING_PATTERNS["odds"])
            
            # Click first Real Madrid match
            rm_selectors = [
                'a:has-text("Real Madrid")',
                ':text("Real Madrid")',
                '[class*="match"]:has-text("Real Madrid")',
            ]
            
            for selector in rm_selectors:
                try:
                    elem = page.locator(selector)
                    if await elem.count() > 0:
                        await elem.first.click(force=True, timeout=10000)
                        await page.wait_for_timeout(3000)
                        logger.info(f"[RM] Clicked match: {selector}")
                        
                        await self._take_screenshot(page, "03_rm_match")
                        
                        # Scan match details
                        await self._scan_betting_specific(page, "rm_match_odds", self.BETTING_PATTERNS["odds"])
                        await self._scan_betting_specific(page, "rm_match_markets", ["market", "tab"])
                        
                        self.results["interaction_flow"].append({
                            "stage": "rm_search",
                            "action": "success",
                            "success": True
                        })
                        return
                except Exception:
                    continue
            
            logger.warning("[RM] Could not click match")
            
        except Exception as e:
            logger.error(f"[RM] Failed: {e}")

    
    async def _scan_match_list(self, page: Page):
        """Stage 3: Analyze match listing page"""
        logger.info("[STAGE3] Stage 3: Scanning match list...")
        
        # Scan for match containers
        await self._scan_betting_specific(page, "match_containers", self.BETTING_PATTERNS["match"])
        
        # Scan for team names
        await self._scan_betting_specific(page, "team_elements", self.BETTING_PATTERNS["team"])
        
        # Scan for odds buttons
        await self._scan_betting_specific(page, "odds_buttons", self.BETTING_PATTERNS["odds"])
        
        await self._take_screenshot(page, "03_match_list")
        
        logger.info("[OK] Match list scan complete")
    
    async def _scan_match_details(self, page: Page):
        """Stage 4: Click on a match to see detailed view"""
        logger.info("[STAGE4] Stage 4: Scanning match details...")
        
        # Try to click on first match
        match_clicked = False
        
        # Try various match selectors
        match_selectors = [
            "div[class*='match']:first-child",
            "div[class*='event']:first-child",
            "div[class*='fixture']:first-child",
            "[data-match-id]:first-child",
            "article:first-child"
        ]
        
        for selector in match_selectors:
            try:
                element = page.locator(selector)
                if await element.count() > 0:
                    await element.first.click()
                    await page.wait_for_timeout(3000)
                    match_clicked = True
                    logger.info(f"[OK] Clicked match using: {selector}")
                    break
            except Exception as e:
                logger.debug(f"Failed to click {selector}: {e}")
        
        if match_clicked:
            # Scan detailed odds/markets
            await self._scan_betting_specific(page, "market_buttons", ["market", "tab", "option"])
            await self._scan_betting_specific(page, "selection_buttons", self.BETTING_PATTERNS["odds"])
            await self._take_screenshot(page, "04_match_details")
        else:
            logger.warning("[WARN] Could not click on a match")
        
        self.results["interaction_flow"].append({
            "stage": "match_details",
            "action": "clicked_match",
            "success": match_clicked
        })
    
    async def _scan_betslip_elements(self, page: Page):
        """Stage 5: Analyze bet slip elements"""
        logger.info("[STAGE5] Stage 5: Scanning bet slip...")
        
        # Scan for bet slip container
        await self._scan_betting_specific(page, "betslip_container", self.BETTING_PATTERNS["betslip"])
        
        # Scan for stake input
        await self._scan_betting_specific(page, "stake_input", self.BETTING_PATTERNS["stake"])
        
        # Scan for submit button
        await self._scan_betting_specific(page, "submit_button", self.BETTING_PATTERNS["submit"])
        
        # Try to find bet slip by common patterns
        betslip_selectors = [
            "[class*='slip']", "[class*='bet-slip']", "[class*='coupon']",
            "[id*='slip']", "[id*='betslip']",
            "[data-testid*='slip']", "[data-testid*='bet']"
        ]
        
        for selector in betslip_selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    logger.info(f"[OK] Found bet slip elements: {selector} ({len(elements)} found)")
            except Exception:
                pass
        
        await self._take_screenshot(page, "05_betslip")
    
    async def _scan_form_elements(self, page: Page):
        """Stage 6: Scan all form-related elements"""
        logger.info("[STAGE6] Stage 6: Scanning form elements...")
        
        self.results["selectors"]["forms"] = {
            "description": "All form elements on the page",
            "items": []
        }
        
        try:
            forms = await page.query_selector_all("form")
            
            for i, form in enumerate(forms):
                form_data = await form.evaluate("""
                    (form) => {
                        const inputs = [];
                        const inputElements = form.querySelectorAll('input, select, textarea, button');
                        
                        for (let input of inputElements) {
                            inputs.push({
                                tag: input.tagName.toLowerCase(),
                                type: input.type || '',
                                name: input.name || '',
                                id: input.id || '',
                                className: input.className || '',
                                placeholder: input.placeholder || ''
                            });
                        }
                        
                        return {
                            id: form.id || '',
                            name: form.name || '',
                            action: form.action || '',
                            method: form.method || '',
                            className: form.className || '',
                            inputs: inputs
                        };
                    }
                """)
                
                self.results["selectors"]["forms"]["items"].append(form_data)
                logger.info(f"[OK] Scanned form #{i+1}: {len(form_data['inputs'])} inputs")
        
        except Exception as e:
            logger.warning(f"[WARN] Error scanning forms: {e}")
    
    async def _scan_category(self, page: Page, category: str, patterns: List[str]):
        """Scan for elements matching category patterns"""
        self.results["selectors"][category] = {
            "description": f"Elements matching {category} patterns",
            "items": []
        }
        
        for pattern in patterns:
            # Try multiple selector types
            selectors = [
                f'[class*="{pattern}"]',
                f'[id*="{pattern}"]',
                f'[data-testid*="{pattern}"]',
                f'[aria-label*="{pattern}"]',
                f'[placeholder*="{pattern}"]',
                f':text("{pattern}")',
                f':has-text("{pattern}")'
            ]
            
            for selector in selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    
                    for elem in elements[:10]:  # Limit to prevent overwhelming data
                        elem_data = await self._extract_element_data(elem, selector)
                        if elem_data and not self._is_duplicate(category, elem_data):
                            self.results["selectors"][category]["items"].append(elem_data.to_dict())
                
                except Exception as e:
                    logger.debug(f"Selector '{selector}' failed: {e}")
    
    async def _scan_betting_specific(self, page: Page, category: str, patterns: List[str]):
        """Scan for betting-specific elements with scoring"""
        self.results["selectors"][category] = {
            "description": f"Betting-specific: {category}",
            "items": []
        }
        
        tags = ["div", "button", "span", "a", "input", "article", "section"]
        
        for tag in tags:
            for pattern in patterns:
                selectors = [
                    f'{tag}[class*="{pattern}"]',
                    f'{tag}[id*="{pattern}"]',
                    f'{tag}[data-*="{pattern}"]'
                ]
                
                for selector in selectors:
                    try:
                        elements = await page.query_selector_all(selector)
                        
                        for elem in elements[:15]:  # Limit results
                            elem_data = await self._extract_element_data(elem, selector)
                            if elem_data:
                                # Score element based on relevance
                                elem_data.score = self._calculate_relevance_score(elem_data, patterns)
                                
                                if not self._is_duplicate(category, elem_data):
                                    self.results["selectors"][category]["items"].append(elem_data.to_dict())
                    
                    except Exception:
                        pass
        
        # Sort by score (highest first)
        if self.results["selectors"][category]["items"]:
            self.results["selectors"][category]["items"].sort(
                key=lambda x: x.get("score", 0),
                reverse=True
            )
    
    async def _extract_element_data(self, element: ElementHandle, selector: str) -> Optional[ElementData]:
        """Extract comprehensive element information"""
        try:
            data = await element.evaluate("""
                (elem, sel) => {
                    const attrs = {};
                    for (let attr of elem.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                    
                    return {
                        selector: sel,
                        tagName: elem.tagName.toLowerCase(),
                        classes: elem.className.split(' ').filter(c => c),
                        id: elem.id || null,
                        textContent: elem.textContent ? elem.textContent.trim().substring(0, 100) : null,
                        attributes: attrs
                    };
                }
            """, selector)
            
            return ElementData(
                selector=data["selector"],
                tag_name=data["tagName"],
                classes=data["classes"],
                id=data["id"],
                text_content=data["textContent"],
                attributes=data["attributes"]
            )
        
        except Exception as e:
            logger.debug(f"Failed to extract element data: {e}")
            return None
    
    def _calculate_relevance_score(self, elem: ElementData, patterns: List[str]) -> int:
        """Calculate relevance score for element"""
        score = 0
        
        # Check classes
        for cls in elem.classes:
            if any(pattern in cls.lower() for pattern in patterns):
                score += 10
        
        # Check ID
        if elem.id and any(pattern in elem.id.lower() for pattern in patterns):
            score += 15
        
        # Check attributes
        if elem.attributes:
            for attr, value in elem.attributes.items():
                if any(pattern in str(value).lower() for pattern in patterns):
                    score += 5
        
        # Check text content
        if elem.text_content:
            if any(pattern in elem.text_content.lower() for pattern in patterns):
                score += 3
        
        # Bonus for interactive elements
        if elem.tag_name in ["button", "a", "input"]:
            score += 5
        
        return score
    
    def _is_duplicate(self, category: str, elem: ElementData) -> bool:
        """Check if element is already in results"""
        if category not in self.results["selectors"]:
            return False
        
        for item in self.results["selectors"][category]["items"]:
            if item.get("selector") == elem.selector and item.get("id") == elem.id:
                return True
        
        return False
    
    async def _take_screenshot(self, page: Page, name: str) -> str:
        """Capture and save screenshot"""
        try:
            screenshot_path = self.screenshots_dir / f"{name}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            
            self.results["screenshots"].append({
                "name": name,
                "path": str(screenshot_path),
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"[SCREENSHOT] Screenshot saved: {name}.png")
            return str(screenshot_path)
        
        except Exception as e:
            logger.warning(f"[WARN] Failed to capture screenshot: {e}")
            return ""
    
    def _calculate_statistics(self):
        """Calculate scan statistics"""
        total_elements = 0
        category_counts = {}
        
        for category, data in self.results["selectors"].items():
            if "items" in data:
                count = len(data["items"])
                category_counts[category] = count
                total_elements += count
        
        self.results["statistics"] = {
            "total_elements_found": total_elements,
            "categories_scanned": len(self.results["selectors"]),
            "elements_per_category": category_counts,
            "screenshots_captured": len(self.results["screenshots"]),
            "interaction_stages": len(self.results["interaction_flow"])
        }
    
    def _save_results(self):
        """Save scan results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"betting_selectors_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[SAVE] Results saved to: {output_file}")
        
        # Also save a simplified version with just the best selectors
        simplified = self._create_simplified_output()
        simplified_file = self.output_dir / f"betting_selectors_simplified_{timestamp}.json"
        
        with open(simplified_file, 'w', encoding='utf-8') as f:
            json.dump(simplified, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[SAVE] Simplified results saved to: {simplified_file}")
    
    def _create_simplified_output(self) -> Dict[str, Any]:
        """Create simplified output with top-scored selectors only"""
        simplified = {
            "url": self.url,
            "timestamp": datetime.now().isoformat(),
            "recommended_selectors": {}
        }
        
        for category, data in self.results["selectors"].items():
            if "items" in data and data["items"]:
                # Take top 5 by score
                top_items = sorted(
                    data["items"],
                    key=lambda x: x.get("score", 0),
                    reverse=True
                )[:5]
                
                simplified["recommended_selectors"][category] = [
                    {
                        "selector": item.get("selector"),
                        "tag": item.get("tag_name"),
                        "classes": item.get("classes", []),
                        "score": item.get("score", 0)
                    }
                    for item in top_items
                ]
        
        return simplified
    
    def print_summary(self):
        """Print scan summary to console"""
        print("\n" + "="*70)
        print("BETTING SELECTOR SCAN SUMMARY")
        print("="*70)
        print(f"URL: {self.url}")
        
        # Safe access to statistics with defaults
        stats = self.results.get('statistics', {})
        print(f"Total Elements Found: {stats.get('total_elements_found', 0)}")
        print(f"Categories Scanned: {stats.get('categories_scanned', 0)}")
        print(f"Screenshots Captured: {stats.get('screenshots_captured', 0)}")
        
        if 'elements_per_category' in stats:
            print("\nElements by Category:")
            for category, count in stats['elements_per_category'].items():
                print(f"  {category}: {count}")
        
        if 'interaction_flow' in self.results:
            print("\nInteraction Flow:")
            for stage in self.results['interaction_flow']:
                status = "[OK]" if stage.get('success') else "[ERROR]"
                print(f"  {status} {stage['stage']}: {stage['action']}")
        
        print("="*70)


async def main():
    """Main execution function"""
    
    # Configuration
    TARGET_URL = "https://www.sportybet.com/gh/m/"  # SportyBet Ghana mobile
    HEADLESS = True  # Set to True for headless mode
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║   BETTING SELECTOR SCANNER - EDUCATIONAL TOOL v2.0            ║
║                                                               ║
║   [WARN]  FOR EDUCATIONAL AND TESTING PURPOSES ONLY  [WARN]           ║
║                                                               ║
║   This tool identifies CSS selectors for web automation      ║
║   testing frameworks. Use responsibly and ethically.         ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Create and run scanner
    scanner = BettingSelectorScanner(url=TARGET_URL, headless=HEADLESS)
    results = await scanner.scan()
    
    # Print summary
    scanner.print_summary()
    
    print("\n[OK] Scan complete! Check the selector_scans/ directory for results.")
    print("📁 Files generated:")
    print("   - Full results JSON (all elements)")
    print("   - Simplified JSON (top-scored selectors)")
    print("   - Screenshots (visual verification)")
    print("   - Log file (detailed execution log)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[WARN] Scan interrupted by user")
    except Exception as e:
        logger.error(f"[ERROR] Fatal error: {e}", exc_info=True)