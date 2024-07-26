import time

from openai import OpenAI
# from openai import OpenAI
import gradio as gr

from openai import OpenAI
client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://localhost:8000/v1"
)
model_name = client.models.list().data[0].id
start_time = time.time()


def predict(message, history, system_prompt):
    history_openai_format = []
    if len(history_openai_format) == 0:
        history_openai_format.append({"role": "system", "content": system_prompt })

    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = client.chat.completions.create(
      model=model_name,
      messages=history_openai_format,
        temperature=0.8,
        top_p=0.8,
        max_tokens=1000,
        stream=True  # Enable streaming
    )
    # Iterate over the response and print each chunk to the console
    tks = 0
    text = ""
    for chunk in response:
        # print(chunk.choices[0].delta.content, end='', flush=True)
        if chunk.choices[0].delta.content is not None:
            text += chunk.choices[0].delta.content
        yield text
        tks += 1
    print(model_name)

CSS = """
.contain {
    display: flex;
    flex-direction: column;
    height: 50vh;
}
#component-0 {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}
#chatbot {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
}
"""

with gr.Blocks(css=CSS) as block:
    gr.ChatInterface(
        predict,
        show_progress=True,
        additional_inputs=[
            gr.Textbox("You talk like a pirate.", label="System Prompt")
        ]
    )
    block.launch(share=True)
