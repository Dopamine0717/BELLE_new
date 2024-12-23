from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM,GenerationConfig, TextIteratorStreamer
import gradio as gr
import mdtex2html
from gradios.utils import load_model_on_gpus
import time
import torch
from threading import Thread


# 日志模块
log_path = open("/data/luohaichen/BELLE/gradios/log.txt", "a+") 
def writelog(logpath, history):
    logpath.write(f'history: {history}\n')
    logpath.flush()


# 主程序
model_path='/data/luohaichen/power_5_qwen1.5_lr1e-5_epoch2_bs512_1129k'
# model_path='/data/huggingface/qwen2_0.5B/'


tokenizer = AutoTokenizer.from_pretrained(model_path)
#model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     device_map="auto"
# )
model = load_model_on_gpus(model_path, num_gpus=3,base_id=0,multi_number=1, torch_dtype=torch.float16)
model = model.eval()

streamer = TextIteratorStreamer(tokenizer)

generation_config = dict(
        temperature=0.001,
        top_p=0.85,
        top_k=30,
        num_beams=1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.encode('<|im_end|>')[0],
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=512,  # max_length=max_new_tokens+input_sequence
        min_new_tokens=1,  # min_length=min_new_tokens+input_sequence
        repetition_penalty=1,
        do_sample=True,
    )

systems = {
    "电力客服": "你是一名陕西电网的智能电力客服，能够为用户解答绝大多数与电力业务有关的问题。你在回答时需要保持尽可能的礼貌，但是也要保持自己工作的效率、重申自己工作任务的主旨。",
    "电力系统管理辅助专家": "你是一名电力系统管理辅助专家，能够为用户解答绝大多数与电网政策规定有关的问题。你在回答时需要保持尽可能的礼貌，但是也要保持自己工作的效率、重申自己工作任务的主旨。",
    "电力运维专家": "你是一名电力运维专家，能够为用户解答绝大多数与电力运维有关的问题。你在回答时需要保持尽可能的礼貌，但是也要保持自己工作的效率、重申自己工作任务的主旨。",
    "电气工程培训专家": "你是一名电气工程培训专家，能够为用户解答绝大多数与电气工程专业知识有关的问题。你在回答时需要保持尽可能的礼貌，但是也要保持自己工作的效率、重申自己工作任务的主旨。",
    "文档编写助手": "你是一名电力文档编写专家，能够为用户编写技术文档或操作手册。你在回答时需要保持尽可能的礼貌，但是也要保持自己工作的效率、重申自己工作任务的主旨。",
}

"""Override Chatbot.postprocess"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


#past_cxx=[]
# '你好！很高兴见到你。我是一个人工智能助手，有什么我可以帮忙的吗？。然而，我是您的电力服务智能客服。如果您想了解更多关于电力法规、电力市场改革、可再生能源发电计划或太阳能应用示范项目等内容，请随时能与我联系，我将竭诚为您提供帮助。'
def predict(input, chatbot, mode):
    if len(chatbot) > 10:
        chatbot = chatbot[-10:]
    prompt = "<|im_start|>System:\n" + systems[mode] + "<|im_end|>\n"
    for val in chatbot:
        prompt += "<|im_start|>Human: \n" + val[0] + "<|im_end|>\n<|im_start|>Assistant: \n"+val[1]+"<|im_end|>\n"
    prompt += "<|im_start|>Human: \n" + input + "<|im_end|>\n<|im_start|>Assistant: \n"
    prompt = prompt.replace('<p>','').replace('</p>','')
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", truncation=True).to(model.device)
    
    chatbot.append([input, ''])
    model_config = dict(
                input_ids = inputs["input_ids"], 
                attention_mask = inputs['attention_mask'],
                **generation_config,
                streamer = streamer
            )
    t = Thread(target=model.generate, kwargs=model_config)
    t.start()
    response  = ""
    for new_token in streamer:
        # if new_token != '<':
        response += new_token
        output = response[len(prompt):].replace('<|im_end|>','')
        chatbot[-1] = [input,output]
        # chatbot[-1] = [input,response]
        yield "", chatbot
    writelog(log_path, chatbot)
    
    # generated_ids = model.generate(
    # input_ids=inputs['input_ids'],
    # attention_mask = inputs['attention_mask'],
    # **generation_config
    # )
    # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], generated_ids)]
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # chatbot.append((input, response))
    # return "", chatbot


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return []


with gr.Blocks() as demo:

    log_path.write('#####\n')
    log_path.write('#####\n')
    log_path.write('#####\n')
    log_path.write('\n')
    modified_time_str = time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime())
    log_path.write(modified_time_str)
    log_path.write('\n')
    log_path.write('\n')


    
    # 标题介绍
    gr.Markdown(
    """
    <div align=center><img src="https://s1.ax1x.com/2023/04/29/p91NLee.png" width="400px"></div>
    
    <div align="center">
    <table rules="none">
    <tr>
    <td>
    <p><font size=4>研发单位</font></p>
    </td>
    <td>
    <img src="https://s1.ax1x.com/2023/04/28/p9lr5lR.png" width="260px"/>
    </td>
    </tr>
    </table>    
    </div>
      
    """)
    # 自我介绍
    gr.Markdown(
            """
            我是PowerChat，一个大规模语言模型，经过大量文本数据的训练，具有丰富的信息和业务能力，能以5个不同的身份为您解答不同场景的问题。
            电力客服：
            - 什么叫分时电价？/ 你的解释不够好。 / 我想要知道具体的时间段。 
            电力系统管理辅助专家：
            - 什么是安全注意事项？/ 检查现场有哪些准备措施需要做好？ / 在雷雨天气时能进行接地极检查吗？...
            电力运维专家：
            - 我在维护电压互感器时发现有部分有断裂现象，您能帮我分析一下可能的原因吗？/ 裂痕在...
            电气工程培训专家：
            - 我想了解一下电机是如何将机械能转换为电能的？/ 请问电动机和发电机在能量转换中有什么区别呢？ / 电机在能量转换过程中会产生哪些损耗？...
            文档编写助手：
            - 我需要撰写一份关于城市电网安全规则的文件，请你帮忙编写。/ 本文件应包含电网设计安全、施工安全、维护检修安全、应急处理和个人防护装备章节。 / 城市电网安全规则完整内容

            我会尽我最大的努力听从您的指示，用我的知识，以全面和翔实的方式回答您的问题。我仍在不断学习，会根据您的提问不断提高与完善自己。如果您有任何与电力有关的问题，请随时向我提问。
            """)
    
    
    with gr.Row():
        with gr.Column(scale=4):
            mode = gr.Radio(["电力客服", "电力系统管理辅助专家", "电力运维专家", "电气工程培训专家", "文档编写助手"], value="电力客服", label="模型身份")
            chatbot = gr.Chatbot(height=600)
            
        with gr.Column(scale=1):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
            emptyBtn = gr.Button("Clear History")

            # max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            # top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            # temperature = gr.Slider(0, 1, value=0.01, step=0.01, label="Temperature", interactive=True)
    # 示例
    gr.Examples(
        label="示例",
        examples=[            
            "什么叫分时电价？",
            "我家停电了，怎么办？",
            "微信交电费的方式？",
            "充电桩能装快充吗？",
            "缴费后为何没来电？",
            "你回答的不够好，我不满意。",
            "能具体解释一下吗？",
            "服务太差了，我要投诉你们！",
            "谢谢，你的回答很详细很到位。"
                ],
        inputs=user_input,
    )
    # history = gr.State([])
    #history=[] # 为什么这一段不能改成[]这样一个参数，为什么要用gr.state
    # past_key_values = gr.State(None)
    # user_input.submit(predict, [user_input, chatbot, history, past_key_values],
    #                 [chatbot, history, past_key_values], show_progress=True)
    # submitBtn.click(predict, [user_input, chatbot, history, past_key_values],
    #                 [chatbot, history, past_key_values], show_progress=True)
    # submitBtn.click(reset_user_input, [], [user_input])

    # emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

    user_input.submit(predict, [user_input, chatbot, mode],
                    [user_input, chatbot], show_progress=True)
    submitBtn.click(predict, [user_input, chatbot, mode],
                    [user_input, chatbot], show_progress=True)
    # submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True, queue=False)

if __name__ == '__main__':
    index = 0
    gr.close_all()
    demo.title = "人工智能助手"
    # demo.queue(concurrency_count=2)
    demo.queue().launch(share=False, inbrowser=True,server_name='0.0.0.0',server_port=6206)
