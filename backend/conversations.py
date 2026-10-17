import uuid

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


def list_conversations(owner):
    conversation_list = []
    for conversation_id, conversation in CONVERSATIONS.items():
        if conversation["owner"] == owner:
            if not conversation["messages"]:
                title = "New Conversation"
            else:
                title = conversation["messages"][0]["content"][:50]
            conversation_list.append({"conversation_id": conversation_id, "title": title})
    return conversation_list


