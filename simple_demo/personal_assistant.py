
import os

from langchain.tools import tool
from langchain.agents import create_agent

from utils.llm import get_llm, get_single_llm
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.types import Command 
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = os.getenv("DB_URI", "postgresql://gq210:123456@localhost:5433/postgres?sslmode=disable")

# Initialize LLM
model = get_single_llm()
# Enable LangSmith tracing by setting environment variables
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") # type: ignore


# define tools for personal assistant
@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    end_time: str,         # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = ""
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    # Stub: In practice, this would call Google Calendar API, Outlook API, etc.
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@tool
def send_email(
    to: list[str],  # email addresses
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    # Stub: In practice, this would call SendGrid, Gmail API, etc.
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    # Stub: In practice, this would query calendar APIs
    return ["09:00", "14:00", "16:00"]

# create calender agent
CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "If there is no suitable time slot, stop and confirm unavailability in your response. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="Calendar event pending approval",
        ),
    ],
)

# create email agent
EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="Outbound email pending approval",
        ),
    ],
)

# define subagents as tools
@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

# create personal assistant superviser agent

SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
)


if __name__ == "__main__":
    # Example 1,Simple single-domain request
    # query = "Schedule a team standup for tomorrow at 9am"

    # for step in supervisor_agent.stream(
    #     {"messages": [{"role": "user", "content": query}]}
    # ):
    #     for update in step.values():
    #         for message in update.get("messages", []):
    #             message.pretty_print()
    # Example 2,Multi-domain request
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup() 
        supervisor_agent = create_agent(
            model,
            tools=[schedule_event, manage_email],
            system_prompt=SUPERVISOR_PROMPT,
            checkpointer=checkpointer
        )
        query = (
            "Schedule a meeting with jack and mary next Tuesday at 2pm for 1 hour and send them a reminder email about the meeting."
            "address of jack and mary is jack.doe@example.com and mary.smith@example.com."
    )

        config = {"configurable": {"thread_id": "6"}}

        interrupts = []
        for step in supervisor_agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            config,
        ):
            for update in step.values():
                if isinstance(update, dict):
                    for message in update.get("messages", []):
                        message.pretty_print()
                else:
                    interrupt_ = update[0]
                    interrupts.append(interrupt_)
                    print(f"\nINTERRUPTED: {interrupt_.id}")
        
    # Print all interrupts and their associated action requests
        print("\nAll Interrupts and their Action Requests:")
        resume = {}
        for interrupt in interrupts:
            if interrupt_.id == "b269647f904245106c16551d05536a79":
            # Edit email
                edited_action = interrupt_.value["action_requests"][0].copy()
                edited_action["args"]["subject"] = "Mockups reminder"
                resume[interrupt_.id] = {
                    "decisions": [{"type": "edit", "edited_action": edited_action}]
                }
            else:
                resume[interrupt_.id] = {"decisions": [{"type": "approve"}]}
        
        interrupts = []
        for step in supervisor_agent.stream(
            Command(resume=resume),
            config,
        ):
            for update in step.values():
                if isinstance(update, dict):
                    for message in update.get("messages", []):
                        message.pretty_print()
                else:
                    interrupt_ = update[0]
                    interrupts.append(interrupt_)
                    print(f"\nINTERRUPTED: {interrupt_.id}")