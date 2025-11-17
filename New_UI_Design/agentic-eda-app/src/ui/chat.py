from streamlit_chat import message as st_message

class ChatInterface:
    def __init__(self):
        self.messages = []

    def display_chat(self):
        for msg in self.messages:
            st_message(msg['content'], is_user=msg['role'] == 'user')

    def add_message(self, role, content):
        self.messages.append({'role': role, 'content': content})
        self.display_chat()

    def clear_chat(self):
        self.messages = []
        st_message("Chat cleared.", is_user=False)