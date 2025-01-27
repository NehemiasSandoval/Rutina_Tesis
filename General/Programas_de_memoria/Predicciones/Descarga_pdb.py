import sys
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
options = webdriver.EdgeOptions()
options.use_chromium = True
#options.add_argument("--headless")
driver = webdriver.Edge(options=options)
url = "https://www.rcsb.org/search?request=%7B%22query%22%3A%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_accession_info.initial_release_date%22%2C%22operator%22%3A%22exists%22%7D%2C%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%7D%5D%2C%22logical_operator%22%3A%22and%22%7D%5D%2C%22logical_operator%22%3A%22and%22%2C%22label%22%3A%22text%22%7D%5D%2C%22logical_operator%22%3A%22and%22%7D%2C%22return_type%22%3A%22entry%22%2C%22request_options%22%3A%7B%22paginate%22%3A%7B%22start%22%3A0%2C%22rows%22%3A25%7D%2C%22results_content_type%22%3A%5B%22experimental%22%5D%2C%22sort%22%3A%5B%7B%22sort_by%22%3A%22score%22%2C%22direction%22%3A%22desc%22%7D%5D%2C%22scoring_strategy%22%3A%22combined%22%7D%2C%22request_info%22%3A%7B%22query_id%22%3A%22988fcb9ea0f0d127c26b8f2bf265bc29%22%7D%7D"
driver.get(url)
wait = WebDriverWait(driver, 10)
element = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[3]/div/div/div[3]/div[2]/div[3]/div/div[1]/div[1]/div[2]/div/select")))
element.click()
element.send_keys(Keys.ARROW_DOWN)
element.send_keys(Keys.ENTER)
other_element = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div[3]/div/div/div[3]/div[2]/div[3]/div/div[1]/div[1]/div[3]/div[2]")))
other_element.click()
time.sleep(5)
alert = driver.switch_to.alert
alert.accept()
next_element = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[3]/div/div/div[3]/div[2]/div[3]/div/div[1]/div[3]/div[3]/div[2]/div[1]")))
next_element.click()
driver.switch_to.window(driver.window_handles[-1])
texto = driver.find_element(By.XPATH, "/html/body/pre").text
print("Texto en la nueva ventana:", texto)
time.sleep(5)
elementos = texto.split(',')
Texto_de_moleculas = r"C:\Users\gemel\OneDrive\Desktop\Programas de memoria\Refinamiento_adaptativo\Texto_moleculas.txt"
with open(Texto_de_moleculas, "r") as archivo:
    elementos_existentes = archivo.read().splitlines()
with open(Texto_de_moleculas, "a") as archivo:
    for elemento in elementos:
        if elemento not in elementos_existentes:
            archivo.write(elemento + "\n")
