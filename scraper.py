import asyncio
from playwright.async_api import async_playwright


async def scraper():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        financial_times = 'https://www.ft.com/'
        try:
            await page.goto(financial_times)
            await asyncio.sleep(3000)
        except Exception as e:
            print(f"[SCRAPER][ERROR] >> {str(e)}")


def run():
    asyncio.run(scraper())


if __name__ == "__main__":
    run()
