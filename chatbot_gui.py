import tkinter as tk
from tkinter import Scrollbar, Text, Entry, Button

from src import data_loader, chatbot_core


def get_response():
    user_input = entry_field.get()
    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, "You: " + user_input + "\n")

    # Perform table question answering
    table_answer = chain.run(user_input)

    # Generate chatbot response based on table question answer
    chatbot_response = "Chatbot: " + table_answer + "\n"
    chat_display.insert(tk.END, chatbot_response)

    chat_display.config(state=tk.DISABLED)
    entry_field.delete(0, tk.END)  # Clear the input field


# Create the main window
root = tk.Tk()
root.title("Chatbot GUI")
root.configure(bg="#FFB6C1")  # Set main color to pink

# Initialize data and chatbot
docs = data_loader.load_csv_data("src/resources/ml_project1_data.csv")
kwargs = chatbot_core.prompt_engineering()
chain = chatbot_core.initialize_retrieval_qa(docs, kwargs)

# Chatbot label
chatbot_label = tk.Label(root, text="Chatbot", bg="#FFE4E1", font=("Arial", 18, "bold"))
chatbot_label.grid(
    row=0, column=0, columnspan=2, pady=10
)  # Adjust the padding as needed

# Chat display
chat_display = Text(
    root,
    wrap=tk.WORD,
    width=50,
    height=20,
    state=tk.DISABLED,
    bg="#FFE4E1",
    fg="#333333",
    font=("Arial", 12),
)
chat_display.grid(
    row=1, column=0, columnspan=2, sticky="nsew"
)

# Scrollbar for chat display
scrollbar = Scrollbar(root, command=chat_display.yview)
scrollbar.grid(row=1, column=2, sticky="ns")
chat_display.config(yscrollcommand=scrollbar.set)

# User input field
entry_field = tk.Entry(root, width=50, bg="#FFF", fg="#333333", font=("Arial", 12))
entry_field.grid(
    row=2, column=0, padx=10, pady=10, sticky="ew"
)  # Make entry field fill horizontally

# Send button
send_button = tk.Button(root, text="Send", command=get_response, font=("Arial", 12))
send_button.grid(row=2, column=1, padx=10, pady=10)  # Make button fill horizontally

# Bind Enter key to get_response function
root.bind("<Return>", lambda event=None: get_response())

# Configure grid row and column weights to make them expand with window resizing
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

# Start the tkinter main loop
root.mainloop()
