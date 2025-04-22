import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
import os
from groq import Groq

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key = groq_key
)

def query_db(sql):
    pass

def run_command(command):
    result = os.system(command=command)
    return result



def get_weather(city: str):
    print("🔨 Tool Called: get_weather", city)
    
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"

def add(x, y):
    print("🔨 Tool Called: add", x, y)
    return x + y

import requests

# def get_weather(api_key, city):
#     url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
#     response = requests.get(url)
#     return response.json()

def get_joke():
    url = 'https://icanhazdadjoke.com/'
    response = requests.get(url, headers={'Accept': 'application/json'})
    return response.json()

def get_random_fact():
    url = 'https://uselessfacts.jsph.pl/random.json?language=en'
    response = requests.get(url)
    return response.json()

avaiable_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name as an input and returns the current weather for the city"
    },
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as input to execute on system and returns output"
    },
    "get_joke": {
        "fn": get_joke,
        "description": "returns a random joke if a user asks for it!"
    },
    "get_random_fact": {
        "fn": get_random_fact,
        "description": "returns a random fact if a user asks for it!"
    }
}

system_prompt = f"""
    You are an helpfull AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.
    You are smart in such a way that if you are asked for questions from the user you will always after thinking for a few times will give out an output irrespective of it being whatever you think.
    You also make sure that you do not exceed more than 10 API calls per output.
    You will always land up on the output for sure so make sure that you go in the plan action action output mode always.
    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

    Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city
    - run_command: Takes a command as input to execute on system and returns ouput
    - get_joke: returns a random joke if a user asks for it!
    - get_random_fact: returns a random fact if a user asks for it!
    
    
    Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interseted in weather data of new york" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}
"""

messages = [
    { "role": "system", "content": system_prompt }
]

while True:
    user_query = input('> ')
    messages.append({ "role": "user", "content": user_query })

    while True:
        count = 0
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=messages
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({ "role": "assistant", "content": json.dumps(parsed_output) })

        if parsed_output.get("step") == "plan":
            count = count + 1
            print(f"🧠: {parsed_output.get("content")}")
            continue
        if count == 5:
            break
        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")

            if avaiable_tools.get(tool_name, False) != False:
                output = avaiable_tools[tool_name].get("fn")(tool_input)
                messages.append({ "role": "assistant", "content": json.dumps({ "step": "observe", "output":  output}) })
                continue
        
        if parsed_output.get("step") == "output":
            print(f"🤖: {parsed_output.get("content")}")
            break


    


