import gradio as gr
import os
import subprocess
from dotenv import set_key, load_dotenv

ENV_FILE = ".env"

def save_env(user_type, c_url, c_key, c_col, c_openai, l_url, l_key, l_col, l_openai):
    # Ensure .env file exists
    if not os.path.exists(ENV_FILE):
        with open(ENV_FILE, "w") as f:
            f.write("")
            
    # Set user_type parameter
    set_key(ENV_FILE, "user_type", user_type)
    
    if user_type == "cloud":
        set_key(ENV_FILE, "QDRANT_URL", c_url)
        set_key(ENV_FILE, "QDRANT_API_KEY", c_key)
        set_key(ENV_FILE, "COLLECTION", c_col)
        set_key(ENV_FILE, "OPENAI_API_KEY", c_openai)
    else:
        set_key(ENV_FILE, "LOCAL_QDRANT_URL", l_url)
        set_key(ENV_FILE, "LOCAL_QDRANT_API_KEY", l_key)
        set_key(ENV_FILE, "LOCAL_COLLECTION", l_col)
        set_key(ENV_FILE, "OPENAI_API_KEY", l_openai)
        
    return f"Successfully saved configuration for '{user_type}' environment to .env!"

def run_ingestion():
    load_dotenv(ENV_FILE, override=True)
    user_type = os.getenv("user_type", "local")
    
    if user_type == "cloud":
        script = "data_preprocess/ingest.py"
    else:
        script = "data_preprocess/ingest-local.py"
        
    try:
        # Open in new terminal window on Windows
        subprocess.Popen(f'start cmd.exe /k python {script}', shell=True)
        return f"Started Data Ingestion ({script}) in a new terminal window."
    except Exception as e:
        return f"Error starting ingestion: {e}"

def run_chat():
    load_dotenv(ENV_FILE, override=True)
    user_type = os.getenv("user_type", "local")
    
    if user_type == "cloud":
        script = "chatbot.py"
    else:
        script = "chatbot-local.py"
        
    try:
        # Open in new terminal window on Windows
        subprocess.Popen(f'start cmd.exe /k python {script}', shell=True)
        return f"Started Chat ({script}) in a new terminal window."
    except Exception as e:
        return f"Error starting chat: {e}"

def update_visibility(choice):
    if choice == "cloud":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

with gr.Blocks(title="System Configuration & Runner") as app:
    gr.Markdown("# System Configuration and Runner")
    
    with gr.Tabs():
        # -- PAGE 1: Configuration --
        with gr.Tab("Configuration"):
            user_type = gr.Radio(
                choices=["cloud", "local"], 
                value="local", 
                label="Select Environment Type",
                info="Choose between Cloud Qdrant + OpenAI Embedding OR Local Docker Qdrant + HuggingFace mixedbread-ai Embedding"
            )
            
            with gr.Group(visible=False) as cloud_group:
                gr.Markdown("### Cloud Environment Setup")
                c_url = gr.Textbox(label="QDRANT_URL", placeholder="https://...")
                c_key = gr.Textbox(label="QDRANT_API_KEY", type="password")
                c_col = gr.Textbox(label="COLLECTION", value="pdfs-store")
                c_openai = gr.Textbox(label="OPENAI_API_KEY", type="password")
                
            with gr.Group(visible=True) as local_group:
                gr.Markdown("### Local Environment Setup")
                l_url = gr.Textbox(label="LOCAL_QDRANT_URL", value="http://localhost:6333")
                l_key = gr.Textbox(label="LOCAL_QDRANT_API_KEY", placeholder="Leave empty for local Docker", type="password")
                l_col = gr.Textbox(label="LOCAL_COLLECTION", value="pdfs-store")
                l_openai = gr.Textbox(label="OPENAI_API_KEY", type="password")
                
            save_btn = gr.Button("Save to .env", variant="primary")
            save_out = gr.Textbox(label="Status", interactive=False)
            
            user_type.change(
                fn=update_visibility,
                inputs=user_type,
                outputs=[cloud_group, local_group]
            )
            
            save_btn.click(
                fn=save_env,
                inputs=[user_type, c_url, c_key, c_col, c_openai, l_url, l_key, l_col, l_openai],
                outputs=save_out
            )
            
        # -- PAGE 2: Actions --
        with gr.Tab("Run Application"):
            gr.Markdown("### Actions")
            gr.Markdown("The actions below will run the respective scripts based on the `user_type` currently saved in `.env`")
            
            with gr.Row():
                ingest_btn = gr.Button("Data Ingestion", variant="secondary")
                chat_btn = gr.Button("Chat (Q&A)", variant="primary")
                
            action_out = gr.Textbox(label="Action Status", interactive=False)
            
            ingest_btn.click(fn=run_ingestion, outputs=action_out)
            chat_btn.click(fn=run_chat, outputs=action_out)

if __name__ == "__main__":
    app.launch(
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    )