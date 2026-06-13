import requests

def fetch_esp32_data(ip):

    try:

        response=requests.get(
            f"http://{ip.strip()}/readings",
            timeout=2
        )

        if response.status_code==200:

            data=response.json()

            return {

                "temp":data.get("temp",0),

                "hum":data.get("hum",0),

                "air":data.get("air",0),

                "dust":data.get("dust",0),

                "samples":data.get("samples",0),

                "status":data.get("status","UNKNOWN")

            }

        else:

            return None

    except Exception as e:

        print("ESP32 Error:",e)

        return None