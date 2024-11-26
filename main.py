import requests
import platform
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time
from tqdm import tqdm

token = input("Token: ")
API_URL_1 = f"http://en.axelinh.xyz/apiV1/api.php?token={token}"
PROXY = "http://127.0.0.1:1081"
OUTPUT_FILE = "tempfile.bin"
WAIT_TIME = 120

DOWNLOAD_LINKS = [
    "https://link.testfile.org/500MB",
    "https://link.testfile.org/300MB",
    "https://link.testfile.org/250MB",
    "http://link.testfile.org/150MB",
]

def download_file(url, output_path):
    try:
        response = requests.get(url, stream=True, proxies={"http": PROXY, "https": PROXY})
        total_size = int(response.headers.get('Content-Length', 0))
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))

        return total_size
    except Exception as e:
        return 0

def get_panel_usage(api_url, user_id):
    response = requests.post(api_url, data={'method': 'getusertraffic', 'username': user_id}, timeout=5)
    return int(response.json()['data'][0]['total'])

def reset_panel_usage(api_url, user_id):
    response = requests.post(api_url, data={'method': 'resettraffic', 'username': user_id}, timeout=5)
    if response.status_code == 200:
        print(f"Traffic reset for user: {user_id}")
    else:
        print(f"Failed to reset traffic for user: {user_id}")

def calculate_polynomial_model(real_usage, panel_usage):
    if len(real_usage) < 2 or len(panel_usage) < 2:
        return None

    x = np.array(panel_usage).reshape(-1, 1)
    y = np.array(real_usage)

    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)

    return model

def get_cpu_architecture():
    return platform.machine()

def main(user_id):
    real_usage = []
    panel_usage = []

    for link in DOWNLOAD_LINKS:
        file_size = download_file(link, OUTPUT_FILE)
        
        if file_size == 0:
            continue

        time.sleep(WAIT_TIME)

        panel_data = get_panel_usage(API_URL_1, user_id)# * 1024 * 1024
        real_usage.append(file_size /1024 /1024)
        panel_usage.append(panel_data)

        reset_panel_usage(API_URL_1, user_id)

    model = calculate_polynomial_model(real_usage, panel_usage)
    if model:
        print(f"Polynomial conversion function: y = {model.intercept_} + {model.coef_[1]}*x + {model.coef_[2]}*x^2")
    else:
        print("Could not compute a polynomial model.")

    cpu_arch = get_cpu_architecture()
    print(f"CPU architecture: {cpu_arch}")

    results = {
        "user_id": user_id,
        "real_usage": real_usage,
        "panel_usage": panel_usage,
        "polynomial_model": {
            "intercept": model.intercept_,
            "coef": model.coef_.tolist()
        } if model else None,
        "cpu_architecture": cpu_arch,
    }
    with open(f"results_{user_id}.json", "w") as result_file:
        json.dump(results, result_file, indent=4)

if __name__ == "__main__":
    user_id = input('Username: ')
    reset_panel_usage(API_URL_1, user_id)
    main(user_id)
