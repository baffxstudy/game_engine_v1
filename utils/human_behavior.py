"""
EDUCATIONAL BETTING AUTOMATION FRAMEWORK
For testing and educational purposes only.
Always comply with local laws and website terms of service.

Human behavior simulator for realistic browser automation.
Implements human-like mouse movements, typing patterns, and delays.
"""

import random
import asyncio
import logging
from typing import Tuple, List
from playwright.async_api import Page

logger = logging.getLogger("engine_api.betting")


class HumanBehaviorSimulator:
    """
    Simulates human-like behavior for browser automation.
    
    Features:
    - Realistic mouse movements using Bezier curves
    - Variable typing speeds with typo simulation
    - Normal distribution delays
    - Pre/post action hesitations
    """
    
    @staticmethod
    async def random_delay(min_ms: int = 500, max_ms: int = 3000) -> None:
        """
        Random delay with normal distribution for realistic timing.
        
        Uses normal distribution centered at midpoint with standard deviation
        based on range, then clamped to min/max bounds.
        
        Args:
            min_ms: Minimum delay in milliseconds
            max_ms: Maximum delay in milliseconds
        """
        # Normal distribution centered at midpoint
        mean = (min_ms + max_ms) / 2
        std_dev = (max_ms - min_ms) / 6  # 99.7% within 3 std devs
        
        delay = random.normalvariate(mean, std_dev)
        delay = max(min_ms, min(max_ms, delay))  # Clamp to bounds
        
        await asyncio.sleep(delay / 1000)
    
    @staticmethod
    def bezier_curve(
        p0: Tuple[float, float],
        p3: Tuple[float, float],
        num_points: int = 50
    ) -> List[Tuple[float, float]]:
        """
        Generate smooth Bezier curve for realistic mouse movement.
        
        Creates a cubic Bezier curve with random control points
        between start and end positions.
        
        Args:
            p0: Start point (x, y)
            p3: End point (x, y)
            num_points: Number of points in curve
            
        Returns:
            List of (x, y) tuples representing the curve path
        """
        # Random control points for natural movement
        p1 = (
            p0[0] + random.uniform(50, 150),
            p0[1] + random.uniform(-50, 50)
        )
        p2 = (
            p3[0] + random.uniform(-150, -50),
            p3[1] + random.uniform(-50, 50)
        )
        
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            
            # Cubic Bezier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
            x = (
                (1 - t) ** 3 * p0[0] +
                3 * (1 - t) ** 2 * t * p1[0] +
                3 * (1 - t) * t ** 2 * p2[0] +
                t ** 3 * p3[0]
            )
            y = (
                (1 - t) ** 3 * p0[1] +
                3 * (1 - t) ** 2 * t * p1[1] +
                3 * (1 - t) * t ** 2 * p2[1] +
                t ** 3 * p3[1]
            )
            points.append((x, y))
        
        return points
    
    @staticmethod
    async def human_click(
        page: Page,
        selector: str,
        click_type: str = 'left'
    ) -> None:
        """
        Simulate human-like click with smooth mouse movement.
        
        Process:
        1. Wait for element to be visible
        2. Get element bounding box
        3. Calculate target with random offset (30-70% of element size)
        4. Move mouse along Bezier curve from current position
        5. Pre-click hesitation (50-200ms)
        6. Click with duration (30-100ms)
        7. Post-click delay (100-500ms)
        
        Args:
            page: Playwright page object
            selector: CSS selector for element to click
            click_type: Type of click ('left', 'right', 'middle')
            
        Raises:
            Exception: If element not found or has no bounding box
        """
        try:
            # Wait for element
            element = await page.wait_for_selector(selector, timeout=10000)
            if not element:
                raise Exception(f"Element not found: {selector}")
            
            # Get bounding box
            box = await element.bounding_box()
            if not box:
                raise Exception(f"Element {selector} has no bounding box")
            
            # Target with random offset (30-70% of element size)
            target_x = box['x'] + box['width'] * random.uniform(0.3, 0.7)
            target_y = box['y'] + box['height'] * random.uniform(0.3, 0.7)
            
            # Get current mouse position (approximate center of viewport)
            current_pos = await page.evaluate("""
                () => [window.innerWidth / 2, window.innerHeight / 2]
            """)
            current_x, current_y = current_pos
            
            # Move along Bezier curve
            path = HumanBehaviorSimulator.bezier_curve(
                (current_x, current_y),
                (target_x, target_y),
                random.randint(20, 40)
            )
            
            for point in path:
                await page.mouse.move(point[0], point[1])
                await asyncio.sleep(random.uniform(0.001, 0.01))
            
            # Pre-click hesitation
            await asyncio.sleep(random.uniform(0.05, 0.2))
            
            # Click with duration
            await page.mouse.down(button=click_type)
            await asyncio.sleep(random.uniform(0.03, 0.1))
            await page.mouse.up(button=click_type)
            
            # Post-click delay
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
        except Exception as e:
            logger.error(f"[HUMAN_BEHAVIOR] Click failed for {selector}: {str(e)}")
            raise
    
    @staticmethod
    async def human_type(page: Page, selector: str, text: str) -> None:
        """
        Simulate human typing with variable speed and typo correction.
        
        Features:
        - Variable typing speed (50-150ms per character)
        - 3% chance of typos with backspace correction
        - Pauses between words (100-500ms)
        - Clicks element first to focus
        
        Args:
            page: Playwright page object
            selector: CSS selector for input element
            text: Text to type
        """
        try:
            # Click element first to focus
            await HumanBehaviorSimulator.human_click(page, selector)
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Clear existing content
            await page.fill(selector, "")
            
            # Type character by character
            for i, char in enumerate(text):
                # Variable typing speed
                delay = random.uniform(50, 150)
                await page.keyboard.type(char, delay=delay)
                
                # 3% chance of typo with correction
                if random.random() < 0.03 and char != ' ':
                    await page.keyboard.press('Backspace', delay=100)
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    await page.keyboard.type(char, delay=delay)
                
                # Pause between words
                if char == ' ':
                    await asyncio.sleep(random.uniform(0.1, 0.5))
                    
        except Exception as e:
            logger.error(f"[HUMAN_BEHAVIOR] Typing failed for {selector}: {str(e)}")
            raise
