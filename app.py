import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import io
import time

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

MODEL_ID = "stabilityai/stable-diffusion-2-1"           # ← good quality / still manageable on CPU
# Alternative lighter/faster options:
# MODEL_ID = "runwayml/stable-diffusion-v1-5"
# MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"   # ← much heavier, usually too slow on CPU

USE_FP16 = False               # fp16 is usually worse on CPU → we use float32
DEVICE = "cpu"

# ──────────────────────────────────────────────────────────────
# Load model with caching
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model... (first time can take 2–5 minutes)")
def load_pipeline():
    if not st.secrets.get("HF_TOKEN"):
        st.error("Hugging Face token (HF_TOKEN) not found in Streamlit secrets!")
        st.stop()

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        variant=None,                # remove "_fp16" variant on CPU
        use_auth_token=st.secrets["HF_TOKEN"],
        safety_checker=None          # remove NSFW filter (optional - be responsible)
    )

    # Better scheduler for fewer steps
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(DEVICE)

    # Memory optimizations (very important on CPU!)
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()      # ← most important for CPU RAM

    return pipe


# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Text → Image • Free", layout="wide")

st.title("Free Text-to-Image Generator")
st.caption(f"Model: **{MODEL_ID.split('/')[-1]}**  •  Running on CPU (slow but free)")

# ── Inputs ───────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    prompt = st.text_area("**Prompt**", 
                         "cyberpunk city at night, neon lights, rain, cinematic, highly detailed",
                         height=110, key="prompt")

    negative = st.text_area("**Negative prompt** (what to avoid)", 
                           "blurry, low quality, deformed, ugly, bad anatomy, watermark, text, signature",
                           height=90, key="negative")

with col2:
    st.write(" ")
    st.write(" ")
    steps = st.slider("Inference steps", 15, 60, 28, 1)
    guidance = st.slider("Guidance scale (CFG)", 3.0, 15.0, 7.5, 0.5)
    width = st.select_slider("Width", options=[512, 640, 768], value=512)
    height = st.select_slider("Height", options=[512, 640, 768, 896], value=640)
    seed = st.number_input("Seed (-1 = random)", -1, 2147483647, -1)

# ── Generate button ──────────────────────────────────────────────
if st.button("✨ Generate Image", type="primary", use_container_width=True):
    if not prompt.strip():
        st.error("Please write something in the prompt!")
        st.stop()

    pipe = load_pipeline()

    progress = st.progress(0)
    status = st.empty()

    try:
        start = time.time()

        generator = None if seed == -1 else torch.Generator(device=DEVICE).manual_seed(seed)

        status.info("Generating... (can take 2–10+ minutes on CPU)")

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative if negative.strip() else None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=height,
                width=width,
                generator=generator,
                callback=lambda i, t, latents: progress.progress((i + 1) / steps)
            ).images[0]

        took = time.time() - start

        st.success(f"Done! Generated in **{took:.1f} seconds**")

        # Show image big
        st.image(image, use_column_width=True)

        # Download
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            "⬇️ Download PNG",
            data=byte_im,
            file_name=f"generated_{int(time.time())}.png",
            mime="image/png",
            use_container_width=True
        )

    except Exception as e:
        st.error("Generation failed")
        st.exception(e)

# Tips & info footer
st.markdown("---")
st.caption(
    "Tip: 20–35 steps usually enough with DPM++ • "
    "Lower guidance = more creative • "
    "Higher guidance = follow prompt more strictly\n\n"
    "Running on free CPU → be patient :)"
)
