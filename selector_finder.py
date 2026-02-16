# selector_finder.py
import asyncio
import json
from playwright.async_api import async_playwright

async def find_sportybet_selectors():
    """Find actual selectors on SportyBet Ghana"""
    
    selectors_data = {
        "search_selectors": [],
        "match_selectors": [],
        "selection_selectors": [],
        "stake_selectors": [],
        "bet_slip_selectors": [],
        "buttons": []
    }
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        # Step 1: Homepage analysis
        print("🔍 Analyzing homepage...")
        await page.goto('https://www.sportybet.com/gh/m/')
        await page.wait_for_timeout(3000)
        
        # Look for search functionality
        search_inputs = await find_elements_by_keyword(page, ['search', 'Search', 'SEARCH'])
        selectors_data["search_selectors"] = search_inputs
        
        # Step 2: Navigate to football
        print("⚽ Navigating to Football...")
        football_links = await page.locator('a:has-text("Football"), a:has-text("FOOTBALL")')
        if await football_links.count() > 0:
            await football_links.first.click()
        await page.wait_for_timeout(5000)
        
        # Step 3: Find matches
        print("📋 Finding match elements...")
        match_elements = await page.locator('div[class*="match"], div[class*="event"]').all()
        for i, elem in enumerate(match_elements[:3]):
            elem_class = await elem.get_attribute('class') or ''
            elem_text = await elem.text_content() or ''
            selectors_data["match_selectors"].append({
                "index": i,
                "classes": elem_class,
                "text_preview": elem_text[:100]
            })
        
        # Step 4: Click on a match to see selections
        print("🎯 Exploring selections...")
        if match_elements:
            await match_elements[0].click()
            await page.wait_for_timeout(3000)
            
            # Look for selection buttons (odds buttons)
            odds_buttons = await page.locator('button, div[role="button"]').all()
            for i, btn in enumerate(odds_buttons[:10]):
                btn_text = await btn.text_content() or ''
                btn_class = await btn.get_attribute('class') or ''
                if 'odds' in btn_text.lower() or any(x in btn_class.lower() for x in ['odds', 'selection']):
                    selectors_data["selection_selectors"].append({
                        "selector": f'button:nth-child({i+1})',
                        "classes": btn_class,
                        "text": btn_text
                    })
        
        # Step 5: Find bet slip
        print("📝 Looking for bet slip...")
        bet_slip_elements = await page.locator('[class*="slip"], [class*="bet-slip"], [class*="coupon"]').all()
        for elem in bet_slip_elements:
            elem_class = await elem.get_attribute('class') or ''
            selectors_data["bet_slip_selectors"].append({
                "classes": elem_class,
                "html": await elem.inner_html()[:200]
            })
        
        # Step 6: Save findings
        with open('sportybet_selectors.json', 'w') as f:
            json.dump(selectors_data, f, indent=2)
        
        print(f"✅ Selectors saved to sportybet_selectors.json")
        print(f"Found {len(selectors_data['selection_selectors'])} potential selection selectors")
        
        await browser.close()

async def find_elements_by_keyword(page, keywords):
    """Find elements containing specific keywords"""
    found = []
    for keyword in keywords:
        # By text content
        elements = page.locator(f':text("{keyword}"), :has-text("{keyword}")')
        count = await elements.count()
        if count > 0:
            for i in range(min(3, count)):
                elem = elements.nth(i)
                tag = await elem.evaluate('el => el.tagName')
                classes = await elem.get_attribute('class') or ''
                found.append({
                    "keyword": keyword,
                    "tag": tag,
                    "classes": classes
                })
    
    return found

if __name__ == "__main__":
    asyncio.run(find_sportybet_selectors())