import requests

url = "http://jupyter-jkoufopoulos@ieroklis.imsi.athenarc.gr:5050/run_llm"


headers = {
    "Content-Type": "application/json"
}

def invoke_local_llm(text, article_id):
    payload = {
        "text": text,
        "article_id": article_id
    }


    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            print("Success")
            return result
        else:
            print(f"Request failed with status code: {response.status_code}")
            print("Error message:", response.text)
            return None
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)
        return None
