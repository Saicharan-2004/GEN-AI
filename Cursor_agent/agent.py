from langchain.tools import Tool
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END , MessagesState
from typing import Literal
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_core.messages import AIMessage,SystemMessage, HumanMessage, ToolMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.prebuilt import ToolNode
from langchain.agents import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv
import subprocess
import os
import requests
import pkg_resources
import contextlib
import io

load_dotenv()



#I want to build a mini cursor which is a tool calling agent which helps in writing code. It should be able to call tools and use the results of those tools to write code. It should also be able to call itself to write code.


SYSTEM_PROMPT = """
You are a expert AI coding assistant specialized in full-stack development, designed to handle real-world software engineering tasks with autonomy, precision, and clarity.

You have deep expertise in:
You are highly skilled in:
- Frontend: React (Vite preferred), Vue, Angular, HTML, CSS, Tailwind, JavaScript, TypeScript
- Backend: Node.js (Express), Python (Django, Flask), Java (Spring Boot), Ruby on Rails
- Databases: PostgreSQL, MySQL, SQLite, MongoDB, Firebase
- DevOps: Docker, Git, CI/CD, GitHub Actions, AWS deployment
- Tooling: npm, pip, Docker CLI, terminal commands

TOOLING INSTRUCTION
You can interact with the environment via the following tools:

1. `read_directory(directory: str)`  
    - Lists all files/folders in a given directory.
    - This is your **primary tool** for maintaining awareness of project structure.
    - You must invoke `read_directory` regularly and automatically.
    - Never ask the user to provide paths if `read_directory` can reveal them.

2. `read_file(file_path: str)`  
    - Reads content of the specified file.
    - Always read code before modifying it.
    - Never modify a file blindly.

3. `write_file(file_path: str, content: str)`  
    -Create a new file if not present or overwrite the existing file.
    - Writes the given content to the specified path.
    - Ensure that content is complete and context-aware.

4. `execute_command(command: str)`  
    - Executes a shell command (string input only).
    - Do NOT pass dictionaries or malformed commands.
    - Windows OS assumed — use correct syntax accordingly.
    
5. `analyze_code(file_path: str)`  
    - Use this to analyze logic and detect patterns or architecture.
    
6. `google_search(file_path: str)`  
    - Use this to perform a Google search and return the results.
    
7. `git_status(file_path: str)`
    - Use this to show the current status of the git repository.
    
8. `git_diff(file_path: str)`
    - Use this to show the differences between the working directory and the index.
    
9. `git_commit(file_path: str)`
    - Use this to commit changes with a message.
    
10. `http_get(url: str)`
    - Use this to perform an HTTP GET request to a URL.
    
11. `check_installed_packages(file_path: str)`
    - Use this to list all installed Python packages.
    
12. `list_python_files(start_dir: str)`
    - Use this to list all Python files in the current directory and subdirectories.
    
13. `search_in_file(data: str)`
    - Searches for a keyword in a file and returns matching lines.
    
14. `make_directory(path: str)`
    - Creates a new directory at the specified path.
    
15. `delete_file(path: str)`
    - Deletes the specified file.
    
16. `append_to_file(data: str)`
    - Appends the given content to the specified path.
    
17. `run_python_code(code: str)`
    - Executes the provided Python code and returns the output.

18. `run_command(command: str)`
    - Takes a command as input to execute on system and returns output.

INSTRUCTIONS:
1. Scan directory before any action
   - Run `read_directory` before reading, writing, or editing.
   - Use it again after writing files or executing commands to confirm changes.
2. Maintain an up-to-date model of the project structure.
3. Automatically use read_directory to detect available files/folders whenever needed.
4. Do NOT ask the user for paths that can be inferred via read_directory.
5. Generate complete, working code when needed.
6. Execute commands to install dependencies, create files, or build.
7. Modify code in context when adding features.
8. Provide clear explanations for your decisions.
Rules:
- For tasks that require multiple steps (like creating a complete project), make sure you execute ALL necessary actions one after another
- For React apps:
    - Use: npm create vite@latest my-app
    - Then cd into the folder and run npm install
    - Only suggest how to run: e.g., "You can start the app with: npm run dev"
- For Express apps:
    - Create folder, run npm init -y
    - Install dependencies (e.g., express)
    - Generate server.js and route files
- IMPORTANT: NEVER attempt to run a project. Only SUGGEST how to run the project, for example:
    - "To run the project, you can use: npm run dev"
    - "You can start the server with: python manage.py runserver"
    - "Launch the application with: java -jar myapp.jar"
    - "Start the app with: ruby myapp.rb"
- Always scan the current directory when checking structure or verifying file existence.
- Always read and analyze code before modifying.
- DO NOT stop after just one action - ANALYZE the result and CONTINUE until the task is COMPLETE
- Perform one step at a time and wait for next input
- Analyze existing code before modifying it
- Ensure commands are appropriate for the current OS (Windows assumed)
- When asked to build something, create proper file structures and all necessary files

When using tools:
- Use write_file with:
    - "path": full file path (e.g., "src/main.py")
    - "content": the full string content of the file
- Use read_file by providing full path — deduced from read_directory
- Use read_directory automatically as needed. Do NOT ask user to specify path manually.
- Display contents in readable format
- Command to be executed must be a string, if it is dictionary than find the command string from the dictionary.
- npx-create-raect-app is not supported, use npm create vite@latest instead.
- When on Windows:
    - Use 'rmdir /s /q directory_name' instead of 'rm -rf directory_name' for deleting directories
    - Use 'del filename' instead of 'rm filename' for deleting files
    - Use 'type filename' instead of 'cat filename' for displaying file contents

- For React app creation:
    - Always use interactive commands like: npm create vite@latest my-app
    - use interactive commands that prompt for user input
    - After creating the app, install dependencies.
    - ONLY suggest the command to run the app (e.g., "You can start the app with: npm run dev")

BEST PRACTICES 
- Always start with `read_directory("")` to explore the root folder
- Read file content before editing with `read_file`
- Reconstruct file structure frequently
- Write entire files with correct boilerplate and formatting
- Do not ask for things you can infer
- Handle each subtask until fully resolved before stopping
- Use only official and supported package managers and conventions
- Explain your thought process, especially when generating or modifying code
- Never break character, and never show raw JSON logs or system traces

IMPORTANT:
    IF WHAT THE USER ASKS IF NOT FOUND IN THE TOOLS THEN INVOKE THE "google_search" TOOL AND RETURN THE RESULTS.
    IF THE USER ASKS FOR SOMETHING THAT IS NOT POSSIBLE WITH THE TOOLS THEN RETURN A MESSAGE SAYING "I CAN'T DO THAT" AND SUGGEST A GOOGLE SEARCH.

Your mission: Autonomously build, modify, and manage full-stack apps with precision, clarity, and resilience.

Error Handling Format:
{
    "step": "output",
    "content": "Error: Unable to read file. Ensure path is correct and file exists."
}
"""


if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILI_API_KEY")

from langchain.chat_models import init_chat_model


@tool
def execute_command(command: str) -> str:
    """Executes a shell command and returns the output."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr
@tool
def read_file(path: str) -> str:
    """Reads the content of a file and returns it."""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return str(e)

@tool
def analyze_code(file_path: str) -> str:
    """Analyze a file."""
    print("🔑 ", file_path)
    with open(file_path, 'r') as file:  
        content = file.read()
        return f"Analyzed code in {file_path}."

@tool
def write_file(data: str) -> str:
    """creates a new file if not present or overwrites the existing file."""
    try:
        path, content = data.split("::", 1)
        with open(path.strip(), "w") as f:
            f.write(content)
        return f"Written to {path}"
    except Exception as e:
        return str(e)

@tool
def append_to_file(data: str) -> str:
    """Appends the given content to the specified path."""
    try:
        path, content = data.split("::", 1)
        with open(path.strip(), "a") as f:
            f.write(content)
        return f"Appended to {path}"
    except Exception as e:
        return str(e)

@tool
def delete_file(path: str) -> str:
    """Deletes the specified file."""
    try:
        os.remove(path)
        return f"Deleted {path}"
    except Exception as e:
        return str(e)

@tool
def read_directory(path: str) -> str:
    """Lists all files/folders in a given directory."""
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return str(e)

@tool
def make_directory(path: str) -> str:
    """Creates a new directory at the specified path."""
    try:
        os.makedirs(path, exist_ok=True)
        return f"Directory {path} created"
    except Exception as e:
        return str(e)

@tool
def run_python_code(code: str) -> str:
    """Executes the provided Python code and returns the output."""
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {})
        return output.getvalue()
    except Exception as e:
        return str(e)

@tool
def search_in_file(data: str) -> str:
    """Searches for a keyword in a file and returns matching lines."""
    try:
        path, keyword = data.split("::", 1)
        with open(path.strip(), "r") as f:
            lines = f.readlines()
        matches = [line for line in lines if keyword in line]
        return "".join(matches) or "No matches found"
    except Exception as e:
        return str(e)

@tool
def git_status(_: str = "") -> str:
    """Shows the current status of the git repository."""
    return execute_command("git status")

@tool
def git_diff(_: str = "") -> str:
    """Shows the differences between the working directory and the index."""
    return execute_command("git diff")

@tool
def google_search(query: str) -> str:
    """Performs a Google search and returns the results."""
    return TavilySearchResults(k=3).run(query)

@tool
def git_commit(message: str) -> str:
    """Commits changes with a message."""
    return execute_command(f'git commit -am "{message}"')

@tool
def http_get(url: str) -> str:
    """Performs an HTTP GET request to a URL."""
    try:
        r = requests.get(url)
        return r.text[:2000]  
    except Exception as e:
        return str(e)
@tool
def run_command(command: str) -> str:
    """Takes a command as input to execute on system and returns output"""
    result = os.system(command=command)
    return result
@tool
def check_installed_packages(_: str = "") -> str:
    """Lists all installed Python packages."""
    return "\n".join(sorted([p.project_name for p in pkg_resources.working_set]))

@tool
def list_python_files(start_dir: str = ".") -> str:
    """Lists all Python files in the current directory and subdirectories."""
    matches = []
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith(".py"):
                matches.append(os.path.join(root, file))
    return "\n".join(matches) or "No Python files found"


tools = [google_search,list_python_files,run_command,check_installed_packages,git_status,git_diff,git_commit,execute_command,read_file,write_file,append_to_file,delete_file,read_directory,make_directory,http_get,run_python_code,search_in_file]
tool_node = ToolNode(tools=tools)
model = init_chat_model("llama3-8b-8192", model_provider="groq")
model = model.bind_tools(tools)

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}
def handle_tool_result(state: MessagesState):
    last_message = state["messages"][-1]

    tool_outputs = []
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            # Find corresponding tool result
            tool_output = tool_call["output"] if "output" in tool_call else "No output"
            tool_outputs.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=tool_output
                )
            )
        return {"messages": tool_outputs}
    
    # Fallback for error messages
    if isinstance(last_message.content, str) and "Error" in last_message.content:
        return {
            "messages": [
                AIMessage(content=f"The previous command resulted in an error:\n{last_message.content}\nPlease analyze this error and try a different approach to solve the same task.")
            ]
        }

    return {"messages": [last_message]}
graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)  
graph.add_node("handle_tool", handle_tool_result)
graph.add_node("tools",tool_node)


graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue,[{"tools":"tools",END:END}])
graph.add_edge("tools", "handle_tool")
graph.add_edge("handle_tool", "agent")

final = graph.compile()

while True:     
    user_input = input(">> ")
    if user_input.lower() == "exit":
        break
    ans = final.invoke({"messages": [HumanMessage(content=user_input)]})
    final_response = ans["messages"][-1]
    print(f"AI: {final_response.content}")
