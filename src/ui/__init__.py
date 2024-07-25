import gradio as gr
from ..config import app

custom_css = """
body {
    font-family: Arial, sans-serif;
    background-color: #1a1a1a;
    color: #ffffff;
}
.gradio-container {
    max-width: 800px !important;
    margin: auto;
    padding-top: 2rem;
}
#component-0 {
    height: 400px !important;
    overflow-y: auto;
    border: 1px solid #333;
    border-radius: 8px;
    background-color: #222;
}
#component-1 {
    border: 1px solid #333;
    border-radius: 8px;
    background-color: #222;
}
#component-0, #component-1 {
    margin-bottom: 1rem;
}
.label {
    font-size: 14px;
    color: #aaa;
}
.message {
    font-size: 14px;
    padding: 10px;
    border-radius: 8px;
    background-color: #333;
    color: #ffffff;
    margin-bottom: 8px;
}
.dark input[type="text"] {
    background-color: #333 !important;
    color: #ffffff !important;
}
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; }
"""

def launch_gradio_ui(respond):
    with gr.Blocks() as iface:
        gr.Markdown(app.title)
        gr.Markdown("Ask me anything")

        chat_history = gr.Chatbot(height=250)
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Enter your message here...", 
                label="User Input", 
                scale=4
            )
            send_button = gr.Button("Send", scale = 1)
        clear = gr.ClearButton([msg, chat_history])

        # msg.submit(respond, [msg, chat_history], [msg, chat_history])
        send_button.click(respond, [msg, chat_history], [msg, chat_history])
        print(msg)
        print(chat_history)

    iface.launch(share=True)