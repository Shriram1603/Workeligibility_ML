import requests

data_to_add = ["Excel", 0.2, 1 , 3,'yes']

response = requests.post("http://127.0.0.1:5000/add_data", json=data_to_add)

if response.status_code == 200:
    print("Data added successfully")
else:
    print("Failed to add data to CSV")
