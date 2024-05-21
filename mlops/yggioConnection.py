import requests
import json
import os
import pickle
from starlette.requests import Request


def make_get_request(url):

    try:
        # token generated from /auth/local in yggio swagger
        token = ""
        headers = {
            "Authorization": f"Bearer {token}",
            "Host": "staging.yggio.net"
        }

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print("Request was successful")
            print("Response body:")
            print(response.text)
        else:
            print(f"Request failed with status code: {response.status_code}")
            print("Response body:")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print("An error occurred while making the request:")
        print(e)


def read_file():

    filePath = os.path.abspath('mlops/artifacts/params.pkl')

    try:

        with open(filePath, 'rb') as file:
                result = pickle.load(file)
                return result
    except FileNotFoundError:
        print(f"File with the file path: '{filePath}' not found")
        return
    except pickle.UnpicklingError as e:
        print(f"Error while unpickling: {e}")
        return


def make_post_request(url):

    fileContent = read_file()

    if fileContent:
        print(f"File content: {fileContent}")
    else:
        print("File could not be found")
        return

    try:
        # data to link the app with Yggio
        data = {
            "_id": "663388c7c5812e977c24a63b",
            "AI_data": json.dumps(fileContent)
        }
        # token generated from /auth/local in yggio swagger
        token = ""
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Length": str(len(json.dumps(data))),
            "Host": "staging.yggio.net"
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            print("Request was successful")
        else:
            print(f"POST request failed with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print("Error occurred while making the POST request:")
        print(e)


if __name__ == "__main__":
    url = "https://staging.yggio.net/http-push/generic?identifier=_id"  # Replace with your actual URL
    make_post_request(url)
