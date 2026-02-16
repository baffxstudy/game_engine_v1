# scanSelectors.py
"""
Website Selector Scanner
Educational tool for identifying CSS selectors on a webpage.
For testing and educational purposes only.

Usage: python scanSelectors.py
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright, Page

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SelectorScanner:
    def __init__(self, url: str):
        self.url = url
        self.results = {
            "scan_timestamp": datetime.now().isoformat(),
            "url": url,
            "selectors": {},
            "statistics": {}
        }
    
    async def scan_page(self) -> Dict[str, Any]:
        """Scan the page for various selectors and elements."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            
            try:
                logger.info(f"Navigating to {self.url}")
                await page.goto(self.url, wait_until="networkidle", timeout=30000)
                
                # Wait a bit for dynamic content to load
                await page.wait_for_timeout(3000)
                
                # Scan for different types of elements
                await self._scan_common_elements(page)
                await self._scan_betting_specific_elements(page)
                await self._scan_form_elements(page)
                await self._scan_navigation_elements(page)
                await self._scan_interactive_elements(page)
                
                # Add statistics
                await self._calculate_statistics()
                
                logger.info("Scan completed successfully")
                
            except Exception as e:
                logger.error(f"Error during scanning: {e}")
                self.results["error"] = str(e)
            finally:
                await browser.close()
        
        return self.results
    
    async def _scan_common_elements(self, page: Page):
        """Scan for common webpage elements."""
        logger.info("Scanning common elements...")
        
        common_selectors = {
            "buttons": {
                "description": "All button elements",
                "selectors": ["button", "input[type='button']", "input[type='submit']"]
            },
            "inputs": {
                "description": "Input fields",
                "selectors": [
                    "input[type='text']", "input[type='email']", "input[type='password']",
                    "input[type='number']", "input:not([type])"
                ]
            },
            "links": {
                "description": "Hyperlinks",
                "selectors": ["a[href]"]
            },
            "images": {
                "description": "Images",
                "selectors": ["img[src]", "img[alt]"]
            },
            "headings": {
                "description": "Heading elements",
                "selectors": ["h1", "h2", "h3", "h4", "h5", "h6"]
            }
        }
        
        for category, info in common_selectors.items():
            self.results["selectors"][category] = {
                "description": info["description"],
                "items": []
            }
            
            for selector in info["selectors"]:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        # Get element attributes
                        attrs_js = """
                        (element) => {
                            const attrs = {};
                            for (let attr of element.attributes) {
                                attrs[attr.name] = attr.value;
                            }
                            return {
                                tagName: element.tagName.toLowerCase(),
                                attributes: attrs,
                                textContent: element.textContent.trim().substring(0, 100)
                            };
                        }
                        """
                        element_info = await element.evaluate(attrs_js, element)
                        
                        # Add to results if not already present
                        if element_info not in self.results["selectors"][category]["items"]:
                            self.results["selectors"][category]["items"].append(element_info)
                            
                except Exception as e:
                    logger.warning(f"Error scanning {selector}: {e}")
    
    async def _scan_betting_specific_elements(self, page: Page):
        """Scan for betting-specific elements."""
        logger.info("Scanning betting-specific elements...")
        
        betting_selectors = {
            "match_cards": {
                "description": "Match/event containers",
                "patterns": [
                    "match", "event", "fixture", "game", "contest"
                ],
                "tags": ["div", "article", "section"]
            },
            "team_names": {
                "description": "Team/competitor names",
                "patterns": [
                    "team", "home", "away", "player", "competitor"
                ],
                "tags": ["div", "span", "h1", "h2", "h3", "p"]
            },
            "odds_elements": {
                "description": "Odds/buttons for betting",
                "patterns": [
                    "odds", "price", "selection", "outcome", "market"
                ],
                "tags": ["button", "div", "span"]
            },
            "bet_slip": {
                "description": "Betting slip elements",
                "patterns": [
                    "slip", "bet", "ticket", "coupon"
                ],
                "tags": ["div", "aside", "section"]
            }
        }
        
        for category, info in betting_selectors.items():
            self.results["selectors"][category] = {
                "description": info["description"],
                "items": []
            }
            
            for tag in info["tags"]:
                for pattern in info["patterns"]:
                    # Attribute-based selectors
                    selectors = [
                        f"{tag}[class*='{pattern}']",
                        f"{tag}[id*='{pattern}']",
                        f"{tag}[data-*{pattern}*]",
                        f"{tag}[aria-label*='{pattern}']"
                    ]
                    
                    for selector in selectors:
                        try:
                            elements = await page.query_selector_all(selector)
                            for element in elements:
                                attrs_js = """
                                (element) => {
                                    const attrs = {};
                                    for (let attr of element.attributes) {
                                        attrs[attr.name] = attr.value;
                                    }
                                    return {
                                        selector: arguments[1],
                                        tagName: element.tagName.toLowerCase(),
                                        attributes: attrs,
                                        textContent: element.textContent.trim().substring(0, 100)
                                    };
                                }
                                """
                                element_info = await element.evaluate(attrs_js, element, selector)
                                
                                if element_info not in self.results["selectors"][category]["items"]:
                                    self.results["selectors"][category]["items"].append(element_info)
                                    
                        except Exception as e:
                            # Skip errors for individual selectors
                            pass
    
    async def _scan_form_elements(self, page: Page):
        """Scan for form-related elements."""
        logger.info("Scanning form elements...")
        
        forms = await page.query_selector_all("form")
        self.results["selectors"]["forms"] = {
            "description": "Form elements and their inputs",
            "items": []
        }
        
        for i, form in enumerate(forms):
            try:
                form_info = await form.evaluate("""
                    (form) => {
                        const formData = {
                            id: form.id,
                            name: form.name,
                            action: form.action,
                            method: form.method,
                            className: form.className
                        };
                        
                        // Get form inputs
                        const inputs = [];
                        const inputElements = form.querySelectorAll('input, select, textarea, button');
                        for (let input of inputElements) {
                            inputs.push({
                                tagName: input.tagName.toLowerCase(),
                                type: input.type,
                                name: input.name,
                                id: input.id,
                                placeholder: input.placeholder,
                                className: input.className
                            });
                        }
                        
                        return {
                            ...formData,
                            inputs: inputs
                        };
                    }
                """, form)
                
                self.results["selectors"]["forms"]["items"].append(form_info)
                
            except Exception as e:
                logger.warning(f"Error processing form {i}: {e}")
    
    async def _scan_navigation_elements(self, page: Page):
        """Scan for navigation elements."""
        logger.info("Scanning navigation elements...")
        
        nav_selectors = {
            "navigation": {
                "description": "Navigation menus",
                "selectors": ["nav", "ul[class*='nav']", "div[class*='menu']"]
            },
            "breadcrumbs": {
                "description": "Breadcrumb navigation",
                "selectors": ["nav[aria-label*='breadcrumb']", ".breadcrumb", "[class*='crumb']"]
            }
        }
        
        for category, info in nav_selectors.items():
            self.results["selectors"][category] = {
                "description": info["description"],
                "items": []
            }
            
            for selector in info["selectors"]:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        nav_info = await element.evaluate("""
                            (element) => {
                                const links = [];
                                const linkElements = element.querySelectorAll('a');
                                for (let link of linkElements) {
                                    links.push({
                                        href: link.href,
                                        text: link.textContent.trim(),
                                        className: link.className
                                    });
                                }
                                
                                return {
                                    tagName: element.tagName.toLowerCase(),
                                    className: element.className,
                                    id: element.id,
                                    links: links
                                };
                            }
                        """, element)
                        
                        self.results["selectors"][category]["items"].append(nav_info)
                        
                except Exception as e:
                    logger.warning(f"Error scanning navigation {selector}: {e}")
    
    async def _scan_interactive_elements(self, page: Page):
        """Scan for interactive elements like modals, dropdowns, etc."""
        logger.info("Scanning interactive elements...")
        
        interactive_selectors = {
            "modals": {
                "description": "Modal dialogs and popups",
                "selectors": [
                    "[role='dialog']", "[class*='modal']", "[class*='popup']",
                    "[aria-modal='true']", "[data-modal]"
                ]
            },
            "dropdowns": {
                "description": "Dropdown/select elements",
                "selectors": [
                    "select", "[class*='dropdown']", "[class*='select']",
                    "[role='combobox']", "[aria-haspopup='listbox']"
                ]
            }
        }
        
        for category, info in interactive_selectors.items():
            self.results["selectors"][category] = {
                "description": info["description"],
                "items": []
            }
            
            for selector in info["selectors"]:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        element_info = await element.evaluate("""
                            (element) => {
                                const attrs = {};
                                for (let attr of element.attributes) {
                                    attrs[attr.name] = attr.value;
                                }
                                return {
                                    selector: arguments[1],
                                    tagName: element.tagName.toLowerCase(),
                                    attributes: attrs,
                                    textContent: element.textContent ? element.textContent.trim().substring(0, 100) : ''
                                };
                            }
                        """, element, selector)
                        
                        self.results["selectors"][category]["items"].append(element_info)
                        
                except Exception as e:
                    # Skip individual selector errors
                    pass
    
    async def _calculate_statistics(self):
        """Calculate statistics about the scan results."""
        total_elements = 0
        category_counts = {}
        
        for category, data in self.results["selectors"].items():
            count = len(data["items"])
            category_counts[category] = count
            total_elements += count
        
        self.results["statistics"] = {
            "total_elements_found": total_elements,
            "categories_scanned": len(self.results["selectors"]),
            "elements_per_category": category_counts
        }

async def main():
    # Configuration
    TARGET_URL = "https://www.sportybet.com.gh/me/"  # Change this to your target URL
    
    # Create scanner instance
    scanner = SelectorScanner(TARGET_URL)
    
    # Run scan
    logger.info("Starting selector scan...")
    results = await scanner.scan_page()
    
    # Save results to JSON file
    output_filename = f"selectors_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Scan results saved to {output_filename}")
    
    # Print summary
    print("\n" + "="*50)
    print("SELECTOR SCAN SUMMARY")
    print("="*50)
    print(f"URL Scanned: {TARGET_URL}")
    print(f"Output File: {output_filename}")
    print(f"Total Elements Found: {results.get('statistics', {}).get('total_elements_found', 0)}")
    print(f"Categories Scanned: {results.get('statistics', {}).get('categories_scanned', 0)}")
    print("="*50)
    
    # Print category breakdown
    if "statistics" in results and "elements_per_category" in results["statistics"]:
        print("\nElements by Category:")
        for category, count in results["statistics"]["elements_per_category"].items():
            print(f"  {category}: {count}")

if __name__ == "__main__":
    asyncio.run(main())
