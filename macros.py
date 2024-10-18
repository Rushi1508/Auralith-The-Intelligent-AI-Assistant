from modules import Functions
import pyautogui
import time

f = Functions.FileSystemManager()
p = Functions.ProcessManager()

def openChrome():
    # Assuming that the 'chrome' environment variable holds the path to the Chrome installation
    chromeDir = 'C:\\Program Files\\Google\\Chrome\\Application'
    chromeDir = chromeDir + '\\chrome.exe'
    p.runCommand(f'"{chromeDir}"')

def searchWeb(data):
    openChrome()  # Now opens Chrome instead of Firefox
    time.sleep(0.5)
    pyautogui.write(data)  # Write the search query
    pyautogui.press("enter")  # Simulate pressing Enter to search
    return "Done!"
