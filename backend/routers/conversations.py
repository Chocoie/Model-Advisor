"""
Conversation routes (/conversations): create and list conversations,
load a conversation's history, and send a message to the chat agent.
"""

from fastapi import APIRouter, HTTPException, Depends
from schemas import NewConversation, ConversationsList, ConversationHistory, MessageRequest, MessageResponse
from dependencies import get_current_user
from store.conversations import create_conversation, list_conversations, get_owner, get_messages, add_message
from services.chat_agent import next_action, explain_results
from services.ranking import rank_skills

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=NewConversation)
def new_conversation(current_user = Depends(get_current_user)):
    conversation_id = create_conversation(current_user)
    return NewConversation(conversation_id=conversation_id)

@router.get("", response_model=ConversationsList)
def get_conversations_list(current_user: str = Depends(get_current_user)):
    return ConversationsList(conversations=list_conversations(current_user))
    
@router.get("/{conversation_id}", response_model=ConversationHistory)
def get_conversation(conversation_id: str, 
                     current_user = Depends(get_current_user)):
    if get_owner(conversation_id) != current_user: # checks if convo belongs to user
        raise HTTPException(status_code=404, detail="Conversation not found")
    conversation_history = get_messages(conversation_id)
    return ConversationHistory(conversation_id=conversation_id, messages=conversation_history)

@router.post("/{conversation_id}/message", response_model=MessageResponse)
def send_message(conversation_id: str, request: MessageRequest, 
                 current_user = Depends(get_current_user)):
    if get_owner(conversation_id) != current_user:   # checks if convo belongs to user
        raise HTTPException(status_code=404, detail="Conversation not found")
    add_message(conversation_id, "user", request.message) #add user msg to conv
    conversation_history = get_messages(conversation_id)
    action = next_action(conversation_history) # get chat agent next action

    if action["type"] == "question":
        reply = action["question"]
        add_message(conversation_id, "assistant", reply)
        return MessageResponse(reply=reply)
    
    skills = action["skills"]
    results = rank_skills(skills)
    reply = explain_results(skills, results)
    add_message(conversation_id, "assistant", reply)

    return MessageResponse(reply=reply, skills=skills, results=results)