# -*- coding: utf-8 -*-
"""Record a silent screen capture of the Streamlit app, driven automatically.

On your PC, from demos/revenue_recovery, with the app already running on http://localhost:8501 :

    pip install playwright
    python -m playwright install chromium
    python record_app.py

Output: demo_app.webm (about 60 s, 1920x1080). Convert to mp4 if needed:
    ffmpeg -i demo_app.webm -c:v libx264 -pix_fmt yuv420p demo_app.mp4
"""
import shutil
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT_DIR = Path("rec_tmp")


def pause(s):
    time.sleep(s)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080},
                              record_video_dir=str(OUT_DIR), record_video_size={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.goto(URL, wait_until="networkidle")
    page.wait_for_selector("text=Revenue Recovery Agent", timeout=60000)
    page.wait_for_selector('div[data-testid="stDataFrame"]', timeout=60000)
    page.wait_for_selector("text=Open a case", timeout=60000)
    pause(5)  # 1. Page chargee, metriques + tableau Queue visibles. Pause 5 s.

    # 2. Ouvrir le selectbox Open a case, choisir C019. Pause 5 s sur le bandeau orange.
    sel = page.get_by_label("Open a case")
    sel.click(); pause(1)
    sel.fill("C019")
    page.keyboard.press("Enter")
    pause(5)

    # 3. Choisir C011. Pause 6 s. Scroll de 400 px vers le bas pour montrer le brouillon d'e-mail. Pause 4 s.
    sel.click(); pause(1)
    sel.fill("C011")
    page.keyboard.press("Enter")
    pause(6)
    page.mouse.wheel(0, 400); pause(4)

    # 4. Onglet Pending human. Pause 3 s. Vider le champ nom, taper archit@causalengines lettre par lettre. Cliquer Approve. Attendre le rerun. Pause 4 s.
    page.get_by_role("tab", name="Pending human").click(); pause(3)
    name = page.get_by_label("Your name (recorded in the ledger)")
    name.fill(""); name.type("archit@causalengines", delay=65); pause(1)
    page.get_by_role("button", name="Approve").first.click()
    page.wait_for_timeout(2500); pause(4)

    # 5. Onglet Ledger. Pause 5 s sur le bandeau vert. Scroll vers le bas puis retour. Cliquer Tamper with block 1 (demo). Attendre le rerun. Pause 5 s sur le bandeau rouge.
    page.get_by_role("tab", name="Ledger").click(); pause(5)
    page.mouse.wheel(0, 600); pause(2)
    page.mouse.wheel(0, -600); pause(1)
    page.get_by_role("button", name="Tamper with block 1 (demo)").click()
    page.wait_for_timeout(2500); pause(5)

    # 6. Onglet Rules. Pause 6 s. Fin.
    page.get_by_role("tab", name="Rules").click(); pause(6)

    video = page.video
    ctx.close()
    path = Path(video.path())
    shutil.move(str(path), "demo_app.webm")
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    browser.close()
    print("Saved demo_app.webm")
