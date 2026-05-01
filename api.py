import io
import os
import sys
from contextlib import redirect_stdout
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

from admin import ensure_daemon
from helpers import *

# Ensure GEMINI_API_KEY is available
if not os.environ.get("GEMINI_API_KEY"):
    print("WARNING: GEMINI_API_KEY environment variable is missing. API translation will fail.", file=sys.stderr)

app = FastAPI(
    title="Browser Harness API", 
    description="LLM-powered Agent API that translates natural language commands into browser actions with self-healing."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommandRequest(BaseModel):
    command: str

SYSTEM_PROMPT = """
You are an autonomous agent that writes Python scripts to control a web browser using the browser-harness framework. 
Translate the user's natural language command into a valid Python script using the available functions.
Do NOT use markdown code blocks (e.g. ```python) in your response. Return ONLY raw Python code.

Available Functions:
- goto_url(url: str): Navigate to a URL. Use this if the user wants to go to a new website.
- click_at_xy(x: float, y: float, button="left", clicks=1): Click at specific coordinates.
- type_text(text: str): Type text into the currently focused element.
- press_key(key: str, modifiers=0): Press a keyboard key (e.g., 'Enter', 'Tab', 'Escape').
- scroll(x: float, y: float, dy=-300, dx=0): Scroll the page.
- capture_screenshot(path="/tmp/shot.png", full=False): Take a screenshot.
- list_tabs(include_chrome=True): List all open tabs.
- switch_tab(target): Switch to a specific tab target.
- new_tab(url="about:blank"): Open a new tab and navigate to URL.
- wait_for_load(timeout=15.0): Wait for the page to finish loading. Always call this after goto_url, clicking a link, or pressing Enter on a search form.
- page_info(): Returns dict with {url, title, w, h, sx, sy, pw, ph}. Always call `print(page_info())` at the end of your script so the user sees the final state.
- js(expression: str, target_id=None): Run JavaScript on the page and return the result.
- dispatch_key(selector, key="Enter"): Dispatch a DOM KeyboardEvent.

CRITICAL INSTRUCTIONS:
1. You maintain conversation history. You can reference things done in previous steps.
2. If your previous code failed, you will receive the error message. Analyze it and fix your code.
3. Keep scripts minimal and focused on the immediate user request.
4. Always print useful output so the user knows what happened.
"""

# Global state for the Agent
_chat_session = None
_client = None

def get_chat():
    global _chat_session, _client
    if _client is None:
        _client = genai.Client()
    if _chat_session is None:
        _chat_session = _client.chats.create(
            model="gemini-3-flash-preview",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0
            )
        )
    return _chat_session

@app.post("/api/reset")
def api_reset():
    """Clear the agent's conversation history."""
    global _chat_session
    _chat_session = None
    return {"status": "ok", "message": "Chat history cleared."}

@app.post("/api/run")
def api_run(req: CommandRequest):
    """
    Translates a natural language command into Python using Gemini, executes it, and auto-heals if it fails.
    """
    print(f"\n=== RECEIVED COMMAND: '{req.command}' ===")
    
    # 1. Ensure daemon is healthy
    ensure_daemon()
    
    # 2. Get the global chat session
    chat = get_chat()
    
    # 3. Pull current browser context to give the LLM "access to the page"
    try:
        current_state = page_info()
    except Exception as e:
        current_state = f"Could not fetch page_info: {e}"

    # 4. Construct initial prompt
    prompt = f"Current Browser State:\n{current_state}\n\nUser Command:\n{req.command}"
    
    max_retries = 3
    generated_code = ""
    output = ""
    error_msg = ""
    
    # 5. The Agent Loop
    for attempt in range(max_retries):
        print(f"--- Agent Loop Attempt {attempt + 1}/{max_retries} ---")
        try:
            # A. Ask LLM for code
            response = chat.send_message(prompt)
            generated_code = response.text.strip()
            
            # Clean up markdown
            if generated_code.startswith("```python"): generated_code = generated_code[9:]
            if generated_code.startswith("```"): generated_code = generated_code[3:]
            if generated_code.endswith("```"): generated_code = generated_code[:-3]
            generated_code = generated_code.strip()
            
            print("Generated Code:")
            print(generated_code)
            
            # B. Execute the code
            f = io.StringIO()
            with redirect_stdout(f):
                exec(generated_code, globals())
            
            # C. Success!
            output = f.getvalue().strip()
            print("Execution Success. Output:", output)
            
            # Send the success output back to the chat so it remembers what happened
            chat.send_message(f"Execution succeeded. Terminal output:\n{output}")
            
            return {
                "output": output,
                "code": generated_code,
                "retries": attempt,
                "status": "success"
            }
            
        except Exception as e:
            # D. Failure! Catch the error and loop.
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"Execution Failed: {error_msg}")
            
            # The next prompt will just be the error message asking for a fix
            prompt = f"Your previous code failed with the following error:\n{error_msg}\n\nPlease analyze the error, fix the code, and try again. Return ONLY raw Python code."
            
    # If we exhaust retries
    print("Agent Loop Exhausted.")
    raise HTTPException(status_code=500, detail={
        "error": f"Failed after {max_retries} attempts. Last error: {error_msg}",
        "code": generated_code,
        "output": f.getvalue() if 'f' in locals() else ""
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
