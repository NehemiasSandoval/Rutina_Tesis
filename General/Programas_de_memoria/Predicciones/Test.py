from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import getpass
import os
username = getpass.getuser()
Drivers_path = r"/mnt/c/Users/"+username+r"/OneDrive - Universidad Técnica Federico Santa María/General - Refinamiento Adaptativo/drivers"
chrome_path = os.path.join(Drivers_path,"chrome-win64" , "chrome.exe")
chromedriver_path = os.path.join(Drivers_path,"chromedriver-win64" , "chromedriver.exe")
options = webdriver.ChromeOptions()
options.add_argument("--incognito")
options.binary_location = chrome_path
service = webdriver.chrome.service.Service(chromedriver_path)
service.start()
driver = webdriver.Chrome(service=service)
driver.get("https://facebook.com/")
print(driver.title)
driver.quit()
service.stop()
