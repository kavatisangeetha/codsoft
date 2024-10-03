def chatbot_response(user_input):
    user_input = user_input.lower()
    
    if "hello" in user_input or "hi" in user_input:
        return "Hello How can I help you?"
    
    elif "name" in user_input:
        return "I'm a simple chatbot. I am chitti."
    
    elif "who are you" in user_input:
        return "I am a chatbot designed to help with basic queries. What would you like to know?"
    
    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a great day."
    elif "how are you" in user_input:
        return "I'm just a program, so I don't have feelings, but I'm fine."
    
    else:
        return "I'm not sure how to respond to that. Can you ask something else?"

def main():
    print("Chatbot: Hi there. Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["bye", "goodbye"]:
            print("Chatbot: Goodbye. Have a great day.")
            break
        
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
