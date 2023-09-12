import gradio as gr
import scipy
from datasets import load_dataset
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from transformers import AutoProcessor, MusicgenForConditionalGeneration, pipeline


def load_input():
    dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
    sample = next(iter(dataset))["audio"]
    # take the first half of the audio sample
    sample["array"] = sample["array"][: len(sample["array"]) // 8]
    return sample["array"]


def imageToText(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-large"
    )
    text = image_to_text(url)
    return text[0]["generated_text"]


def storyGeneratorGPT(user_input):
    template = """
    You are a music story teller;
    You can suggest music that suits the scenario;
    The suggested music should include the genre of the music as well as the style where it is inpired from;
    The suggestion should be no more than 20 words.
   
    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    prompt.format(scenario=user_input)
    story_chain = LLMChain(
        llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1),
        prompt=prompt,
        verbose=True,
    )
    story = story_chain.run(user_input)
    # print(story)
    return story


def generate(text):
    print("generate..")
    print(text)
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    inputs = processor(
        # audio=load_input(),
        text=[text],
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs, max_new_tokens=256)  # 256
    sampling_rate = model.config.audio_encoder.sampling_rate
    resultFile = "musicgen_out.wav"
    scipy.io.wavfile.write(
        resultFile,
        rate=sampling_rate,
        data=audio_values[0, 0].numpy(),
    )
    return resultFile


series_1 = gr.Interface(
    fn=imageToText,
    inputs="pil",
    outputs="text",
    examples=["beatles.png"],
)
series_2 = gr.Interface(fn=storyGeneratorGPT, inputs="text", outputs="text")
series_3 = gr.Interface(fn=generate, inputs="text", outputs="video")
demo = gr.Series(series_1, series_2, series_3)


if __name__ == "__main__":
    demo.launch()
