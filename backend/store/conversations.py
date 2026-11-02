"""
Conversation storage: create conversations, add and read messages, list a user's chats.
Currently an in-memory dict (CONVERSATIONS); to be replaced with the database.
"""

import uuid
from services.chat_agent import generate_title
# Shape: {conversation_id: {"owner": email, "messages": [{"role": "user", "content": "..."}, ...]}}
CONVERSATIONS = {} # TODO delete when db ready and replace related action with queries

def create_conversation(owner):
    conversation_id = str(uuid.uuid4()) # Random unique id
    CONVERSATIONS[conversation_id] = {"owner": owner, "messages": []}
    return conversation_id

def get_owner(conversation_id):
    conversation = CONVERSATIONS.get(conversation_id)
    if conversation is None:
        return None
    return conversation["owner"]

def get_messages(conversation_id):
    conversation = CONVERSATIONS.get(conversation_id) 
    if conversation is None:
        return None
    return conversation["messages"]

def add_message(conversation_id, role, content):
    conversation = CONVERSATIONS.get(conversation_id)
    if conversation is None:
        return None
    conversation["messages"].append({"role": role, "content": content})

    # first user message -> ask the LLM for a title once
    if role == "user" and not conversation.get("title"):
        title = generate_title(conversation["messages"])
        if title:
            conversation["title"] = title

def set_title(conversation_id, title):
    """Save a generated title on the conversation."""
    conversation = CONVERSATIONS.get(conversation_id)
    if conversation is not None:
        conversation["title"] = title

def list_conversations(owner):
    conversation_list = []
    for conversation_id, conversation in CONVERSATIONS.items():
        if conversation["owner"] == owner:
            title = conversation.get("title")          # saved by set_title
            if not title:                              # no title yet -> fallback
                if conversation["messages"]:
                    title = conversation["messages"][0]["content"][:50]
                else:
                    title = "New Conversation"
            conversation_list.append({"conversation_id": conversation_id, "title": title})
    return conversation_list
