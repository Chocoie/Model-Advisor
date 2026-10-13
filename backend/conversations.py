import uuid

# In memory for now: everything is lost when the server restarts (DB later)
# Shape: {conversation_id: [{"role": "user", "content": "..."}, ...]}
CONVERSATIONS = {}


def create_conversation():
    conversation_id = str(uuid.uuid4()) #random unique id
    CONVERSATIONS[conversation_id] = []
    return conversation_id


def get_messages(conversation_id):
    return CONVERSATIONS.get(conversation_id)  # None if the id doesn't exist


def add_message(conversation_id, role, content):
    CONVERSATIONS[conversation_id].append({"role": role, "content": content})