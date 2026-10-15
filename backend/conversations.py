import uuid

# Shape: {conversation_id: [{"role": "user", "content": "..."}, ...]}
CONVERSATIONS = {} # TODO delete when db ready and replace related action with queries

def create_conversation():
    conversation_id = str(uuid.uuid4()) # Random unique id
    CONVERSATIONS[conversation_id] = [] 
    return conversation_id

def get_messages(conversation_id):
    return CONVERSATIONS.get(conversation_id) # None if doesn't exist

def add_message(conversation_id, role, content):
    CONVERSATIONS[conversation_id].append({"role": role, "content": content})
