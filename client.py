import atexit
from selenium import webdriver

url = 'http://127.0.0.1:8000'

# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--auto-open-devtools-for-tabs")  # Open developer tools
options.add_argument("--start-maximized")  # Open browser maximized
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options)
driver.get(url)  # Open the URL

# Switch to the "console" tab of the developer tools
# driver.execute_script("window.open('');")
# driver.switch_to.window(driver.window_handles[1])
# driver.get('chrome://devtools/console')
