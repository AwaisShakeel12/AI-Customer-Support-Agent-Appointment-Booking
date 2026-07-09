import os
import datetime
import logging
from typing import Literal
from dotenv import load_dotenv

# ==========================================
# 🚨 VERCEL FIX: Set cache directory to /tmp
# This MUST be before importing composio!
# ==========================================
os.environ["COMPOSIO_CACHE_DIR"] = "/tmp/.composio"

# Composio imports
from composio import Composio
from composio_langchain import LangchainProvider

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import MessagesState, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
compose_key = os.getenv("COMPOSIO_API_KEY")

# Initialize Composio with the Langchain Provider
composio_client = Composio(api_key=compose_key, provider=LangchainProvider())


# =====================================================================
# 🚨 PASTE YOUR FULL COMPOSIO USER ID HERE 🚨
# In your screenshot it starts with "pg-test-6e4f17e6" but it is cut off.
# Go to the dashboard, click the COPY icon next to the ID, and paste it below!
# =====================================================================
MY_USER_ID = "pg-test-6e4f17e6-fe46-4aa0-a36a-a08f95ca1cd8" 


schedule_tools_set = composio_client.tools.get(
    user_id=MY_USER_ID,
    tools=[
        "GOOGLECALENDAR_FIND_FREE_SLOTS",
        "GOOGLECALENDAR_CREATE_EVENT",
        "GOOGLEMEET_CREATE_MEET",
        "GMAIL_CREATE_EMAIL_DRAFT",
    ]
)

# Separate out write tools
schedule_tools_write = composio_client.tools.get(
    user_id=MY_USER_ID,
    tools=[
        "GOOGLECALENDAR_CREATE_EVENT",
        "GOOGLEMEET_CREATE_MEET",
        "GMAIL_CREATE_EMAIL_DRAFT",
    ]
)

schedule_tools_write_node = ToolNode(schedule_tools_write)

initial_message = """
You are AS-AI, an AI assistant at a Legal & Financial Services for Law Firms & Lawyers and Tax Consultants & Accountants. Follow these guidelines:
1. Friendly Introduction & Tone
2. Assess User Context (Appointment, Law & Tax inquiry, Online/Physical).
3. Scheduling Requests (Date/Time, Email).
4. Availability Check (Use GOOGLECALENDAR_FIND_FREE_SLOTS, always check 3 days).
5. Responding to Availability (Book if free, suggest alternatives if not).
6. User Confirmation Before Booking.
7. Communication Style (Simple, clear, concise).
8. Privacy of Internal Logic (Never disclose tools or code).
9. Short And simple human style message, not long.
10. Avoid use of ** ## or any symbols, use simple plan text only.

- Reference today's date/time: {today_datetime}.
- Our TimeZone is Pakistan Standard Time GMT+5.
"""

model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", api_key=api_key)
model_with_tools = model.bind_tools(schedule_tools_set)

# Make call_model async and use ainvoke
async def call_model(state: MessagesState):
    today_datetime = datetime.datetime.now().isoformat()
    # Notice the 'await' and 'ainvoke' here!
    response = await model_with_tools.ainvoke([SystemMessage(content=initial_message.format(today_datetime=today_datetime))] + state["messages"])
    return {"messages": [response]}

async def tools_condition(state: MessagesState) -> Literal["find_slots", "create_onlin_meeting", "tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for call in last_message.tool_calls:
            tool_name = call.get("name")
            if tool_name == "GOOGLECALENDAR_FIND_FREE_SLOTS":
                return "find_slots"
            elif tool_name == "GOOGLEMEET_CREATE_MEET":
                return "create_onlin_meeting"
        return "tools"
    return "__end__"

# Fix find_slots async blocking
async def find_slots(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for call in last_message.tool_calls:
            tool_name = call.get("name")
            tool_id = call.get("id")
            args = call.get("args")
            find_free_slots_tool = next((tool for tool in schedule_tools_set if tool.name == tool_name), None)
            if tool_name == "GOOGLECALENDAR_FIND_FREE_SLOTS" and find_free_slots_tool:
                # Use ainvoke!
                res = await find_free_slots_tool.ainvoke(args)
                tool_messages.append(ToolMessage(name=tool_name, content=str(res), tool_call_id=tool_id))
    return {"messages": tool_messages}


async def create_onlin_meeting(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for call in last_message.tool_calls:
            tool_name = call.get("name")
            tool_id = call.get("id")
            args = call.get("args")
            create_meet_tool = next((tool for tool in schedule_tools_set if tool.name == tool_name), None)
            if tool_name == "GOOGLEMEET_CREATE_MEET" and create_meet_tool:
                # Use ainvoke!
                res = await create_meet_tool.ainvoke(args)
                tool_messages.append(ToolMessage(name=tool_name, content=str(res), tool_call_id=tool_id))
    return {"messages": tool_messages}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("find_slots", find_slots) 
workflow.add_node("create_onlin_meeting", create_onlin_meeting) 
workflow.add_node("tools", schedule_tools_write_node) 

workflow.add_edge("__start__", "agent")
workflow.add_conditional_edges("agent", tools_condition, ["tools", "find_slots", "create_onlin_meeting", END])

# Loop the tools back to the agent
workflow.add_edge("tools", "agent")
workflow.add_edge("find_slots", "agent")
workflow.add_edge("create_onlin_meeting", "agent")

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
