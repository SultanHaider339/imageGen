# app.py - version with NO token required

import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import time

st.set_page_config(page_title="Text → Image • No Login", layout="wide")
st.title("Free Text-to-Image (no HuggingFace login required)")
st.caption("Using publicly accessible model • Quality is limited in 2026 without login")

@st.cache_resource(show_spinner="Loading model... (happens once)")
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "dreamlike-art/dreamlike-diffusion-1.0",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    pipe.scheduler = pipe.scheduler_class.from_config(pipe.scheduler.config)  # default is ok
    
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()   # crucial for Streamlit Cloud
    
    return pipe.to("cpu")


pipe = load_model()

# ── UI ────────────────────────────────────────────────────────────
col1, col2 = st.columns([5,2])

with col1:
    prompt = st.text_area("Prompt", 
                         "beautiful anime girl, detailed eyes, fantasy landscape background",
                         height=100)

with col2:
    steps = st.slider("Steps", 15, 50, 28)
    guidance = st.slider("Guidance", 4.0, 12.0, 7.5, 0.5)
    seed = st.number_input("Seed (-1 = random)", -1, 2147483647, -1)

if st.button("Generate", type="primary"):
    if not prompt.strip():
        st.error("Please write a prompt")
        st.stop()

    with st.spinner("Generating... (2–8 minutes on free CPU)"):
        start = time.time()
        
        generator = None if seed == -1 else torch.Generator().manual_seed(seed)
        
        try:
            image = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                height=512,
                width=512
            ).images[0]
            
            st.image(image, use_column_width=True)
            
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(), "image.png", "image/png")
            
            st.caption(f"Done in {time.time()-start:.1f} seconds")
            
        except Exception as e:
            st.error("Generation failed")
            st.exception(e)

st.markdown("---")
st.caption("Limitations: older model • lower quality than current private/token models • no negative prompt support in this lightweight version")
