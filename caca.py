from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Configura la ruta al ChromeDriver
chromedriver_path = "D:\descargas\chromedriver-win64\chromedriver.exe"  # Asegúrate de que esta ruta es correcta

# Configura el servicio de ChromeDriver
service = Service(chromedriver_path)
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

url = "https://www.puntoticket.com/disney-on-ice"

for i in range(100):
    driver.execute_script("window.open('');")  # Abre una nueva pestaña
    driver.switch_to.window(driver.window_handles[-1])  # Cambia a la nueva pestaña
    driver.get(url)  # Navega a la URL
    time.sleep(1)  # Espera 1 segundo entre cada solicitud para no sobrecargar el servidor

print("Proceso completado.")
