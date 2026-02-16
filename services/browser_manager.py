"""
EDUCATIONAL BETTING AUTOMATION FRAMEWORK
For testing and educational purposes only.
Always comply with local laws and website terms of service.

Browser manager for creating anti-detection browser contexts.
"""

import logging
from typing import Optional
from playwright.async_api import async_playwright, BrowserContext, Playwright

from ..config import (
    BROWSER_HEADLESS,
    BROWSER_TIMEOUT,
    VIEWPORT_WIDTH,
    VIEWPORT_HEIGHT
)

logger = logging.getLogger("engine_api.betting")

# Try to import fake_useragent, fallback to default if not available
try:
    from fake_useragent import UserAgent
    UA_AVAILABLE = True
except ImportError:
    UA_AVAILABLE = False
    logger.warning("[BROWSER_MANAGER] fake_useragent not available, using default user agent")


class BrowserManager:
    """
    Manages browser context creation with anti-detection measures.
    
    Features:
    - Anti-bot detection countermeasures
    - Realistic user agent rotation
    - Proper viewport and locale settings
    - Navigator property overrides
    """
    
    def __init__(self, test_mode: bool = True):
        """
        Initialize browser manager.
        
        Args:
            test_mode: Whether running in test mode
        """
        self.test_mode = test_mode
        self.ua = UserAgent() if UA_AVAILABLE else None
        logger.info(f"[BROWSER_MANAGER] Initialized (test_mode={test_mode})")
    
    def _get_user_agent(self) -> str:
        """Get random user agent or default."""
        if self.ua:
            try:
                return self.ua.random
            except Exception:
                pass
        
        # Default Chrome user agent
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    
    async def create_context(self, playwright: Playwright) -> BrowserContext:
        """
        Create anti-detection browser context.
        
        Implements multiple anti-detection measures:
        - Disables automation flags
        - Sets realistic viewport and user agent
        - Overrides navigator properties
        - Sets proper HTTP headers
        
        Args:
            playwright: Playwright instance
            
        Returns:
            BrowserContext with anti-detection settings
        """
        try:
            logger.info("[BROWSER_MANAGER] Creating browser context...")
            
            # Launch browser with anti-detection args
            browser = await playwright.chromium.launch(
                headless=BROWSER_HEADLESS,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-site-isolation-trials',
                    '--disable-setuid-sandbox',
                    '--disable-gpu',
                ]
            )
            
            # Create context with realistic settings
            context = await browser.new_context(
                viewport={
                    'width': VIEWPORT_WIDTH,
                    'height': VIEWPORT_HEIGHT
                },
                user_agent=self._get_user_agent(),
                locale='en-US',
                timezone_id='America/New_York',
                extra_http_headers={
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
            )
            
            # Override detection properties
            await context.add_init_script("""
                // Override webdriver property
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                // Override plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                
                // Override languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
                
                // Override chrome property
                window.chrome = {
                    runtime: {}
                };
            """)
            
            # Set timeout
            context.set_default_timeout(BROWSER_TIMEOUT)
            
            logger.info("[BROWSER_MANAGER] Browser context created successfully")
            return context
            
        except Exception as e:
            logger.error(f"[BROWSER_MANAGER] Failed to create context: {str(e)}")
            raise
