import os
import re

from dotenv import load_dotenv
import asyncio
import threading
import multiprocessing
from groq import Groq

# Playwright
import playwright
from playwright._impl._errors import TargetClosedError
from playwright.async_api import async_playwright


load_dotenv()
GROQ_CLIENT = Groq(api_key=os.getenv('GROQ_API_KEY'))


async def bypass_cookie(page):
    """Attempts to click common cookie buttons"""
    selectors = [
        'text=Accept all', 'text=Accept Cookie', 'text=Accept', 'text=I accept cookies',
        'text=Yes, I Accept', 'text=I Accept', 'text=Confirm All','text=Confirm all,' 'text=Consent',
        'text=Allow', 'text=Allow all', 'text=Allow All', 'text=Consent', 'text=I Consent'
    ]

    async def click_cookie(selector):
        try:
            await page.click(selector=selector)
            return True
        except (playwright.async_api.TimeoutError, TargetClosedError):
            return False

    tasks = []
    for selector in selectors:
        tasks.append(asyncio.create_task(click_cookie(selector)))
    for task in asyncio.as_completed(tasks):
        result = await task
        if result:
            tasks.clear()
            return True
        pass


def process(content: str) -> list:
    """
    :param: content: str
    Formats the content
    - removes punctuation
    - removes stop words
    """
    positive_bank = set([
        'major rally', 'helped generate', 'rising phase', 'significant',
        'market opportunity', 'soar', 'soared','struggling economy', 'uptick',
        'stronger appetite', 'strong appetite', 'catching up', 'driving success', 'success',
        'rallies', 'new highs', 'strength continues', 'boosted', 'boosted demand', 'boosted price',
        'gaining', 'gained', 'increased', 'rose', 'markets rose', 'upward', 'upward trend',
        'explosive rise', 'explosive growth', 'price surge', 'recent surge', 'sparks hope',
        'impressed', 'fast', 'pumped', 'pump', 'significant growth', 'strong', 'strong rebound',
        'rallying', 'further growth', 'surges'
    ])
    negative_bank = set([
        'trust issues', 'badly damaged', 'struggling economy', 'bad year',
        'fell', 'sharp', 'decline', 'drops', 'much lower', ' lower', 'expecting much lower',
        'damages', 'shuts down', 'downward', 'downward trend', 'decreased', 'declined',
        'decline', 'dropped', 'price plummit', 'price drop', 'plummit', 'sparks fear', 'dissapointed',
        'dump', 'dumped', 'crash', 'crashed', 'weak', 'lost strength', 'further decline', 'selling pressure'
    ])
    pattern = r"[^\w\s]"
    content = content.strip().split(".")
    cleaned_content = [
        re.sub(pattern, '', item.strip()) for item in content
    ]
    print('cleaned content')
    return list(set(cleaned_content))


async def scrape_source(url):
    """Goes to the page, accepts cookie, grabs the HTML"""
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
            print('booting up page')
            page = await browser.new_page()
            await page.goto(url)

            result = await bypass_cookie(page)
            if not result:
                print('closing browser')
                await browser.close()
                return

            content = await page.locator('body').inner_text()
            print('processing content')
            new_content = process(content)
            print('received cleaned content')
            print(new_content)
        except TimeoutError:
            pass
        except Exception as e:
            print("[SCRAPE SOURCE][ERROR] >> ", str(e))
            pass
        finally:
            await browser.close()


def scrape_source_ignite(url):
    asyncio.run(scrape_source(url))


async def read_sources():
    """
    Reads sources in batches from SOURCE_FILE
    and begins the scrape process concurrently for batch
    """
    try:
        with open(SOURCES_FILE, 'r') as file:
            lines = file.readlines()
        with multiprocessing.Pool(processes=2) as pool:
            pool.map(scrape_source_ignite, lines)
    except Exception as e:
        print(f"[READ SOURCES][ERROR] >> ", str(e))


async def get_sources():
    """
    Retrieves the website links of news headlines and
    saves to file
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        url = ('https://www.google.com/search?q=crypto+news&sca_esv=a5d80f222e3035ff&tbm=nws&sxsrf'
               '=ADLYWILiIJQDisDHfpFU1zAoKF7G785d5Q:1729081168540&ei=UK8PZ_7UIK_JhbIPnrvHkAc&start=10&sa=N&ved'
               '=2ahUKEwj-vNqm8ZKJAxWvZEEAHZ7dEXIQ8NMDegQIBBAW&biw=1601&bih=747&dpr=1.2')

        await page.goto(url)
        accept = page.locator("button:has-text('Accept all')")
        await accept.nth(0).click()

        while True:
            # Waiting for container
            await page.wait_for_selector(".MjjYud")

            # Save to file
            for item in await page.locator(".WlydOe").all():
                text = await item.get_attribute('href')
                save_to_file(text)

            # Next page
            try:
                await page.locator("span:has-text('Next')").click()
                await asyncio.sleep(3)
            except playwright.async_api.TimeoutError:
                print("[SOURCES][TIMEOUT] >> No more pages")
                break
    start_read()


def start_read():
    def func():
        asyncio.run(read_sources())
    thread = threading.Thread(target=func)
    thread.start()



def save_to_file(text):
    """Saves text to SOURCES_FILE"""
    with open(SOURCES_FILE, 'a') as file:
        file.write(text + "\n")



def run():
   # asyncio.run(get_sources())
    asyncio.run(read_sources())




if __name__ == "__main__":
    SOURCES_FILE = 'sources.txt'
    run()
