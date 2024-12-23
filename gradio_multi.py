from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM,GenerationConfig, TextIteratorStreamer
import gradio as gr
import mdtex2html
from gradios.utils import load_model_on_gpus
import time
import torch
from threading import Thread
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_path",
                    type=str,
                    default="/mnt/afs/luohaichen/models/power_qwen1.5_lr1e-5_epoch1_bs256_1100k")
parser.add_argument("--port",
                    type=int,
                    default=6206)
args = parser.parse_args()
# 日志模块
log_path = open("gradios/log.txt", "a+") 
def writelog(logpath, history):
    logpath.write(f'history: {history}\n')
    logpath.flush()


# 主程序
# model_path='/mnt/afs/luohaichen/models/power_qwen1.5_lr1e-5_epoch1_bs256_1100k'
model_path=args.model_path


tokenizer = AutoTokenizer.from_pretrained(model_path)
#model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# model = load_model_on_gpus(model_path, num_gpus=3,base_id=0,multi_number=1, torch_dtype=torch.float16)
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
def predict(input, chatbot):
    if len(chatbot) > 10:
        chatbot = chatbot[-10:]
    prompt = ''
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
    # return chatbot


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
            我是陕西电网的智能电力客服，一个大规模语言模型，经过大量文本数据的训练，具有丰富的信息和业务能力，能够为您解答绝大多数与电力业务有关的问题，例如电费查询、停电计划或用电安全。
            比如：
            - 什么叫分时电价？/ 你的解释不够好。 / 我想要知道具体的时间段。 / 怎么涨了这么多...
            - 我家停电了，怎么办？/ 我没有欠费。 / 但是你们电话打不通。 / 你们到底什么时候来修？ / 我要投诉你们...
            - 微信交电费的方式？/ 具体怎么操作？ / 我没有注册啊。 / 我前几天交了但是没有到账。 / 我是在生活缴费交的...
            - 充电桩能装快充吗？/ 为什么不用三相电？ / 你的规章制度有问题啊。 / 我记得是可以用380V充电的...
            - 缴费后为何没来电？/ 我已经交过费了。 / 支付宝交的费。 / 可是我检查过了网络没有问题啊...
            - 我的户号是1377777？ / 我的户号是多少你还记得吗？ / 你回答的不对，你再想想。 / 我已经说过了，我想让你复述一遍...

            我会尽我最大的努力听从您的指示，用我的知识，以全面和翔实的方式回答您的问题。我仍在不断学习，会根据您的提问不断提高与完善自己。如果您有任何电力业务需要，请随时向我提问。
            """)
    
    
    with gr.Row():
        with gr.Column(scale=4):
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

    user_input.submit(predict, [user_input, chatbot],
                    [user_input, chatbot], show_progress=True)
    submitBtn.click(predict, [user_input, chatbot],
                    [user_input, chatbot], show_progress=True)
    # submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True, queue=False)

if __name__ == '__main__':
    index = 0
    gr.close_all()
    demo.title = "人工智能助手"
    # demo.queue(concurrency_count=2)
    demo.queue().launch(share=True, inbrowser=True,server_name='0.0.0.0',server_port=args.port)
