from fastapi import FastAPI, HTTPException
from llama_index.core.bridge.pydantic import BaseModel
from chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from typing import Optional
from template import Template
from chat_title import get_chat_name
from kgpchatroom import KGPChatroomModel
from tags import get_tags
from question_recommendations import question_recommendations

import tracemalloc  
tracemalloc.start()

template = Template.get_template()
LLM = KGPChatroomModel().get_model()

chat_sessions = {}
app = FastAPI()

class ChatRequest(BaseModel):
    conversation_id: str
    user_message: str
    chat_profile: str

class ChatResponse(BaseModel):
    conversation_id: str
    assistant_response: str
    chat_title: Optional[str] = None 
    tags_list: Optional[list] = None
    questions_list: Optional[list] = None

async def get_chat_engine(conversation_id: str, chat_profile: str) -> ContextChatEngine:
    # Initialize the session and title status if it doesn't exist
    if conversation_id not in chat_sessions:
        memory = ChatMemoryBuffer.from_defaults(token_limit=400000)
        pc_index = KGPChatroomModel().load_vector_index(chat_profile=chat_profile)
        chat_engine = ContextChatEngine.from_defaults(retriever=pc_index.as_retriever(), memory=memory, system_prompt=template)
        chat_sessions[conversation_id] = {
            'engine': chat_engine,
            'title_generated': False  # Initialize title status to False
        }
    return chat_sessions[conversation_id]['engine']

@app.post("/chat/{conversation_id}", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        user_message = request.user_message
        conversation_id = request.conversation_id
        chat_profile = request.chat_profile
        chat_engine = await get_chat_engine(conversation_id, chat_profile)

        # Get the assistant response
        response = chat_engine.chat(user_message)

        # Generate title only if it hasn't been generated yet
        title = None
        if not chat_sessions[conversation_id]['title_generated']:
            title = get_chat_name(user_message, response)
            # title = chat_title(user_message, response)
            chat_sessions[conversation_id]['title_generated'] = True  # Set to True after generating title

        history = chat_engine.chat_history
        tags_list, _ = get_tags(history,LLM)
        questions_list, _ = question_recommendations(history,LLM)        
        # Create response object
        response_data = ChatResponse(
            conversation_id=conversation_id,
            assistant_response=str(response),  # Remove newline characters
            chat_title=title,  # Return title only if it was generated
            tags_list = tags_list,
            questions_list = questions_list
        )

        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_reset/{conversation_id}")
async def reset_chat(conversation_id: str):
    try:
        if conversation_id in chat_sessions:
            chat_sessions[conversation_id]['engine'].reset()
            del chat_sessions[conversation_id]  # Reset the entire session
            return {"message": f"Chat history for conversation {conversation_id} has been reset successfully"}
        else:
            raise KeyError
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Conversation ID {conversation_id} does not exist or has already been reset")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/chats/delete_all")
def master_reset():
    try:
        if chat_sessions:
            chat_sessions.clear()  # Clear all chat sessions
            return {"message": "All chat conversations have been deleted successfully! 😈"}
        else:
            raise HTTPException(status_code=404, detail="No active chat conversations to delete 😕")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the app:
# Use: uvicorn main:app --reload
