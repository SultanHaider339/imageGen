import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import time

st.set_page_config(page_title="Very Basic Txt→Img", layout="wide")

st.title("Minimal Text-to-Image (no token, no GPU)")
st.caption("Very limited quality & size • Probably still crashes due to RAM • For demo only")

@st.cache_resource(show_spinner="Loading tiny-ish model... (may fail due to RAM)")
def load_model():
    try:
        # One of the smallest remaining truly public models
        pipe = StableDiffusionPipeline.from_pretrained(
            "nitrosocke/Arcane-Diffusion",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True if torch.__version__ >= "2.0" else False
        )
        
        # NO offloading → we hope it fits
        # pipe.enable_attention_slicing()          # ← sometimes helps a bit, try uncomment
        pipe = pipe.to("cpu")
        
        return pipe
    
    except Exception as e:
        st.error("Model loading failed (most likely RAM limit on free tier)")
        st.exception(e)
        st.stop()


pipe = load_model()

# ── Interface ────────────────────────────────────────────────────
prompt = st.text_area(
    "Prompt",
    "league of legends arcane style, jinx shooting rockets, dramatic lighting",
    height=100
)

steps = st.slider("Steps", 15, 45, 25)
guidance = st.slider("Guidance", 5.0, 12.0, 7.5, 0.5)

if st.button("Generate (very slow & low-res)", type="primary"):
    if not prompt.strip():
        st.warning("Write something first :)")
    else:
        with st.spinner("Working... (3–15+ minutes if it doesn't crash)"):
            try:
                start = time.time()
                
                image = pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=384,   # smaller = better chance to fit in RAM
                    width=384
                ).images[0]
                
                st.success(f"Done in {time.time() - start:.1f} s (if lucky)")
                st.image(image, use_column_width=True)
                
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button("Download", buf.getvalue(), "output.png", "image/png")
                
            except Exception as e:
                st.error("Generation failed – very likely out of memory")
                st.exception(e)


st.markdown("---")
st.caption(
    "Reality check 2026:\n"
    "• Free Streamlit Cloud → ~1 GB RAM\n"
    "• Almost no Stable Diffusion model fits reliably\n"
    "• Best solution = **paid tier** / local run / Google Colab / RunPod / ..."
)
