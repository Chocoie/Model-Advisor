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
