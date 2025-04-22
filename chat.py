from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

completions = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role":"user","content":"hello!"}
    ]
)
print(completions.choices[0].message.content)