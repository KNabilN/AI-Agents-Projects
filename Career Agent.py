# %%
# imports

from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import http.client, urllib


# %%
# The usual start

load_dotenv(override=True)
openai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


# %%
# For pushover

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

# %%
def push(message):
    print(f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)

# %%
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

# %%
def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}

# %%
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

# %%
record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

# %%
tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]

# %%
# This is a more elegant way that avoids the IF statement.

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
    return results

# %%
linkedin_reader = PdfReader("me/karim.pdf")
linkedin = ""
for page in linkedin_reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

cv_reader = PdfReader("me/karim_cv.pdf")
cv = ""
for page in cv_reader.pages:
    text = page.extract_text()
    if text:
        cv += text

with open("me/karim_summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

name = "Karim Nabil"

# %%
system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website from recruiters, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background, CV and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n## CV:\n{cv}"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."


# %%
# Create a Pydantic model for the Evaluation

from pydantic import BaseModel

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str



# %%
evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
The Agent is playing the role of {name} and is representing {name} on their website. \
The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
The Agent has been provided with context on {name} in the form of their summary, CV and LinkedIn details. Here's the information:"

evaluator_system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n## CV:\n{cv}"
evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."

def evaluator_user_prompt(reply, message, history):
    user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
    user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
    user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
    user_prompt += f"Please evaluate the response, replying with whether it is acceptable and your feedback."
    return user_prompt

# %%
def evaluator_user_prompt(reply, message, history):
    user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
    user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
    user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
    user_prompt += f"Please evaluate the response, replying with whether it is acceptable and your feedback."
    return user_prompt

# %%
openai_evaluator = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# %%
def evaluate(reply, message, history) -> Evaluation:

    messages = [{"role": "system", "content": evaluator_system_prompt}] + [{"role": "user", "content": evaluator_user_prompt(reply, message, history)}]
    response = openai_evaluator.beta.chat.completions.parse(model="deepseek/deepseek-chat-v3-0324:free", messages=messages, response_format=Evaluation)
    return response.choices[0].message.parsed

# %%
def rerun(reply, message, history, feedback):
    updated_system_prompt = system_prompt + f"\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
    updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
    updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
    messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content

# %%
def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    done = False
    
    while not done:

        # This is the call to the LLM - see that we pass in the tools json

        response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)

        finish_reason = response.choices[0].finish_reason
        
        # If the LLM wants to call a tool, we do that!
         
        if finish_reason=="tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = handle_tool_calls(tool_calls)
            messages.append(message)
            messages.extend(results)
        else:
            done = True
    
    reply = response.choices[0].message.content
    evaluation = evaluate(reply, message, history)
    
    if evaluation.is_acceptable:
        print("Passed evaluation - returning reply")
    else:
        print("Failed evaluation - retrying")
        print(evaluation.feedback)
        reply = rerun(reply, message, history, evaluation.feedback) 

    return reply

# %%
gr.ChatInterface(chat, type="messages").launch()

# %% [markdown]
# ## And now for deployment
# 
# This code is in `app.py`
# 
# We will deploy to HuggingFace Spaces. Thank you student Robert M for improving these instructions.
# 
# Before you start: remember to update the files in the "me" directory - your LinkedIn profile and summary.txt - so that it talks about you!  
# Also check that there's no README file within the 1_foundations directory. If there is one, please delete it. The deploy process creates a new README file in this directory for you.
# 
# 1. Visit https://huggingface.co and set up an account  
# 2. From the Avatar menu on the top right, choose Access Tokens. Choose "Create New Token". Give it WRITE permissions.
# 3. Take this token and add it to your .env file: `HF_TOKEN=hf_xxx` and see note below if this token doesn't seem to get picked up during deployment  
# 4. From the 1_foundations folder, enter: `uv run gradio deploy` and if for some reason this still wants you to enter your HF token, then interrupt it with ctrl+c and run this instead: `uv run dotenv -f ../.env run -- uv run gradio deploy` which forces your keys to all be set as environment variables   
# 5. Follow its instructions: name it "career_conversation", specify app.py, choose cpu-basic as the hardware, say Yes to needing to supply secrets, provide your openai api key, your pushover user and token, and say "no" to github actions.  
# 
# #### Extra note about the HuggingFace token
# 
# A couple of students have mentioned the HuggingFace doesn't detect their token, even though it's in the .env file. Here are things to try:   
# 1. Restart Cursor   
# 2. Rerun load_dotenv(override=True) and use a new terminal (the + button on the top right of the Terminal)   
# 3. In the Terminal, run this before the gradio deploy: `$env:HF_TOKEN = "hf_XXXX"`  
# Thank you James and Martins for these tips.  
# 
# #### More about these secrets:
# 
# If you're confused by what's going on with these secrets: it just wants you to enter the key name and value for each of your secrets -- so you would enter:  
# `OPENAI_API_KEY`  
# Followed by:  
# `sk-proj-...`  
# 
# And if you don't want to set secrets this way, or something goes wrong with it, it's no problem - you can change your secrets later:  
# 1. Log in to HuggingFace website  
# 2. Go to your profile screen via the Avatar menu on the top right  
# 3. Select the Space you deployed  
# 4. Click on the Settings wheel on the top right  
# 5. You can scroll down to change your secrets, delete the space, etc.
# 
# #### And now you should be deployed!
# 
# Here is mine: https://huggingface.co/spaces/ed-donner/Career_Conversation
# 
# I just got a push notification that a student asked me how they can become President of their country 😂😂
# 
# For more information on deployment:
# 
# https://www.gradio.app/guides/sharing-your-app#hosting-on-hf-spaces
# 
# To delete your Space in the future:  
# 1. Log in to HuggingFace
# 2. From the Avatar menu, select your profile
# 3. Click on the Space itself
# 4. Click the settings wheel on the top right
# 5. Scroll to the Delete section at the bottom
# 

# %% [markdown]
# <table style="margin: 0; text-align: left; width:100%">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/exercise.png" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#ff7800;">Exercise</h2>
#             <span style="color:#ff7800;">• First and foremost, deploy this for yourself! It's a real, valuable tool - the future resume..<br/>
#             • Next, improve the resources - add better context about yourself. If you know RAG, then add a knowledge base about you.<br/>
#             • Add in more tools! You could have a SQL database with common Q&A that the LLM could read and write from?<br/>
#             • Bring in the Evaluator from the last lab, and add other Agentic patterns.
#             </span>
#         </td>
#     </tr>
# </table>

# %% [markdown]
# <table style="margin: 0; text-align: left; width:100%">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/business.png" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#00bfff;">Commercial implications</h2>
#             <span style="color:#00bfff;">Aside from the obvious (your career alter-ego) this has business applications in any situation where you need an AI assistant with domain expertise and an ability to interact with the real world.
#             </span>
#         </td>
#     </tr>
# </table>


