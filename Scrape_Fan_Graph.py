import os
import time
import shutil
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from dotenv import load_dotenv, find_dotenv

print("📂 Looking for .env file...")

# Load credentials
load_dotenv(find_dotenv())
FG_EMAIL = os.getenv("FG_EMAIL")
FG_PASSWORD = os.getenv("FG_PASSWORD")

if not FG_EMAIL or not FG_PASSWORD:
    raise ValueError("Missing FG_EMAIL or FG_PASSWORD in .env file")

# Setup download path
download_dir = os.path.abspath("downloads")
os.makedirs(download_dir, exist_ok=True)

chrome_options = Options()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "safebrowsing.enabled": True
})
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--disable-software-rasterizer")
chrome_options.add_argument("--disable-webgl")
chrome_options.add_argument("--disable-3d-apis")
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")

urls = {
    "team_standard.csv": "https://www.fangraphs.com/leaders/splits-leaderboards?splitArr=&splitArrPitch=&autoPt=true&splitTeams=false&statType=team&statgroup=1&startDate=2025-03-01&endDate=2025-11-01&groupBy=season",
    "team_advanced.csv": "https://www.fangraphs.com/leaders/splits-leaderboards?splitArr=&splitArrPitch=&autoPt=true&splitTeams=false&statType=team&statgroup=2&startDate=2025-03-01&endDate=2025-11-01&groupBy=season"
}

driver = None
try:
    print("🚀 Launching browser...")
    driver = webdriver.Chrome(options=chrome_options)

    print("🔐 Logging into FanGraphs...")
    driver.get("https://blogs.fangraphs.com/wp-login.php")
    time.sleep(3)

    driver.find_element(By.ID, "user_login").send_keys(FG_EMAIL)
    driver.find_element(By.ID, "user_pass").send_keys(FG_PASSWORD)
    driver.find_element(By.ID, "wp-submit").click()
    time.sleep(4)

    for filename, url in urls.items():
        print(f"\n📊 Visiting {filename} page...")
        driver.get(url)
        time.sleep(4)

        try:
            driver.execute_script("""
                const footer = document.querySelector('[id^="sticky_footer"], .sticky-footer');
                if (footer) footer.remove();
            """)
        except Exception as e:
            print(f"⚠️ Footer removal error: {e}")

        try:
            export = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Export Data"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", export)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", export)
            print("💾 Export clicked.")
        except TimeoutException:
            print(f"❌ Could not export {filename}")
            continue

        downloaded = None
        for _ in range(5):  # check every 3s, for up to 15s
            time.sleep(3)
            candidates = [
                f for f in os.listdir(download_dir)
                if f.endswith(".csv") and "Leader" in f
            ]
            if candidates:
                candidates.sort(key=lambda x: os.path.getmtime(os.path.join(download_dir, x)), reverse=True)
                downloaded = os.path.join(download_dir, candidates[0])
                if time.time() - os.path.getmtime(downloaded) < 60:
                    break

        if not downloaded:
            print(f"⚠️ No file detected for {filename}")
            continue

        print(f"✅ File downloaded: {os.path.basename(downloaded)}")

        final_path = os.path.join(download_dir, filename)
        try:
            if os.path.exists(final_path):
                os.remove(final_path)
            shutil.move(downloaded, final_path)
            print(f"📁 Moved to {filename}")
        except Exception as e:
            print(f"❌ Error moving file: {e}")
            continue

        today = datetime.now().strftime("%Y-%m-%d")
        archive_dir = os.path.join(download_dir, "archive", today)
        os.makedirs(archive_dir, exist_ok=True)
        archive_name = filename.replace(".csv", f"_{today}.csv")
        archive_path = os.path.join(archive_dir, archive_name)

        try:
            shutil.copy2(final_path, archive_path)
            print(f"📦 Archived copy: {archive_name}")
        except Exception as e:
            print(f"❌ Archive failed: {e}")

except Exception as main_err:
    print(f"🚨 Script error: {main_err}")
finally:
    if driver:
        driver.quit()

print("\n🏁 Done.")
