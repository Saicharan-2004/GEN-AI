import requests

# uids = ["f20212322","f20210075","f20212090","f20210527"]
uids = ["f20210946","f20210321","f20211815","f20211174","f20212308","f20212078","f20212614","f20213025","f20210746","f20212263","f20211935","f20212384","f20212922","f20210312","f20211320","f20212380","f20212692","f20210567","f20212807","f20212587","f20210247","f20210855","f20212691","f20210640","f20210529","f20210528","f20212084","f20211020","f20210794","f20212168","f20210759","f20212601","f20210565","f20212044","f20211330","f20210542","f20212306","f20212569","f20210560","f20211457"]

cookies = {
    "XSRF-TOKEN": "eyJpdiI6IjhMTnB5b0dXY0tCcmpsUVNRdmxHWlE9PSIsInZhbHVlIjoieFJCM1FJM084NmNUTlNnYWg3Znh1NEgwaGlLXC9jWWRkK2NnK0pGbWdvSVwvYjN1Vm9uMEhMclk1Yk9DT25sbUp4IiwibWFjIjoiMGVlOTY1MTc1NWVlNDQ0N2IwYzU1YjEzYWYxMGIxNjBhYzkwZWVkNGIyYzhjYWIwNzc2MGVmOTQwZDdiMzlkOSJ9",
    "yearbook_nostalgia_session": "eyJpdiI6Ilwva29YbFk1UEpvNGZ2TXFLN1dSRjlRPT0iLCJ2YWx1ZSI6InJyY3kzaDBrZjhaTzVXNWNsYXpmZVJHRG9kOXBxT2tyVVJnVk4rZm56SXV5WUhJeU9SdXBUaFFvNmZMeTRENHgiLCJtYWMiOiJlYWMyZWE4MmYyZTQ2ZmYxMzczMzU1NjM5YjlhZjU0MGY1NTVhMWJlYzM0NWNiYWFlMGNlYzYxZGJhZTc3OTNiIn0%3D"
}

headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,/;q=0.8",
    "accept-language": "en-US,en;q=0.8",
    "cache-control": "max-age=0",
    "content-type": "application/x-www-form-urlencoded",
    "origin": "https://yearbooknostalgia.com",
    "referer": "https://yearbooknostalgia.com/portal/user-testimonial",
    "sec-ch-ua": '"Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "macOS",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "sec-gpc": "1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
}

url = "https://yearbooknostalgia.com/portal/request-testimonial"
print(len(uids))
# for uid in uids:
#     email = f"{uid}@hyderabad.bits-pilani.ac.in"
#     data = {
#         "student_id": email,
#         "req": "Send Request"
#     }
#     print(f"Sending request for {email}...")
#     response = requests.post(url, headers=headers, cookies=cookies, data=data)
#     print(f"Status code: {response.status_code}\n")