import requests

# host = "mlzoomcamp-midterm-env.eba-znmnabvm.ap-southeast-1.elasticbeanstalk.com"
host = "127.0.0.1:9696"
url = f"http://{host}/predict"

flat = { 
    "lease_commence_year": 2000,
    "lease_commence_month": 6,
    "town": "Ang Mo Kio",
    "flat_type": "3 Room",
    "flat_model": "Improved",
    "storey_range": "10 TO 12",
    "floor_area_sqm": 67}

# Make a POST request
try:
    response = requests.post(url, json=flat)
    response.raise_for_status()  # Raise an error for bad HTTP responses
    result = response.json()
    print('Estimated flat price: ', result.get('predicted_price'))
except requests.exceptions.RequestException as e:
    print(f"HTTP error occurred: {e}")
except ValueError as e:
    print(f"Invalid JSON response: {e}")
