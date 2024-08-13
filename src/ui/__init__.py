import gradio as gr
from ..config import app

def launch_gradio_ui(respond):
    with gr.Blocks() as iface:
        gr.Markdown(app.title)
        gr.Markdown("Ask me anything")

        chat_history = gr.Chatbot(height=550)
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Enter your message here...", 
                label="User Input", 
                scale=4
            )
            send_button = gr.Button("Send", scale = 1)
        clear = gr.ClearButton([msg, chat_history])

        msg.submit(respond, [msg, chat_history], [msg, chat_history])
        send_button.click(respond, [msg, chat_history], [msg, chat_history])
        print(msg)
        print(chat_history)

    iface.launch(share=True)