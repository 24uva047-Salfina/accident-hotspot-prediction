import requests

URL = "https://tngis.tn.gov.in/apps/tngis-redesign/tngis_api/api/api_results.php"

test_cases = [
    "get_district",
    "get_districts",
    "district",
    "get_visitor_count"
]

for case in test_cases:

    print("\n================================")
    print("Testing:", case)
    print("================================")

    payload = {
        "case": case,
        "isfirst": 0
    }

    try:
        response = requests.post(
            URL,
            data=payload,
            timeout=30
        )

        print("Status:", response.status_code)
        print("Length:", len(response.text))
        print("Response:", response.text[:1000])

    except Exception as e:
        print("ERROR:", e)