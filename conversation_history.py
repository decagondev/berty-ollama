conversation_history = ""
def update_conversation_history(user_input, model_response):
    global conversation_history
    conversation_history += f"User: {user_input}\nAI: {model_response}\n"


update_conversation_history("What is the time?", "It is NOW!")

print(conversation_history)
print("------------------------------------")
update_conversation_history("What is the time?", "It is NOW!")

print(conversation_history)
