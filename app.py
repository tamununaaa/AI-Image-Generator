import streamlit as st                          # To build web UI
from diffusers import StableDiffusionPipeline   # Hugging Face Diffusers to load the image generation model
import torch                                    

# Set up the Streamlit UI page
st.set_page_config(page_title="POSTER GENERATOR", layout="centered")
st.title("🖼️ Image Generator")
st.markdown("Describe your image so I can bring it to life!")

# Input field to get user prompt
prompt = st.text_input("Enter your prompt", placeholder="e.g: A dog eating pizza on Saturn, sitting on a bench")

# Method to load AI model and keep it in cache so it doesn't reload every time
@st.cache_resource
def load_model():
    with st.spinner("Hold on... Let me grab my brushes and paints!🖌️🎨"):
        #model_id = "CompVis/stable-diffusion-v1-4"
        model_id = "stabilityai/sdxl-turbo"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

# Generate button
if st.button("Generate Poster"):
    if prompt:
        with st.spinner("Painting your masterpiece...🎨"):
            try:
                #image = pipe(prompt).images[0]
                image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
                st.image(image, caption="Ready!✨", use_column_width=True)
            except Exception as e:
                st.error(f"Uh-oh! Something went wrong: {e}")
    else:
        st.warning("Let me know what to create!")
