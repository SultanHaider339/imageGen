import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import time
from PIL import Image
import io

st.set_page_config(
    page_title="Text → Image • No Login",
    page_icon="🎨",
    layout="wide"
)

st.title("Free Text-to-Image Generator")
st.caption("No Hugging Face token needed • Using public model • Quality is modest")

# ==============================================
#   MODEL LOADING (cached)
# ==============================================
@st.cache_resource(show_spinner="Loading model... (first time: ~2–4 min)")
def load_model():
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "nitrosocke/Ghibli-Diffusion",          # one of few truly public models left
            # "dreamlike-art/dreamlike-diffusion-1.0",   ← alternative (also public)
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            variant=None
        )

        # Some memory optimizations for free CPU environment
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()

        return pipe.to("cpu")

    except Exception as e:
        st.error("Failed to load model")
        st.exception(e)
        st.stop()


# Load once
pipe = load_model()

# ==============================================
#               INTERFACE
# ==============================================
col1, col2 = st.columns([5, 2])

with col1:
    prompt = st.text_area(
        "Your prompt",
        value="a magical forest with glowing mushrooms, Studio Ghibli style, soft light, detailed background",
        height=110,
        key="prompt"
    )

with col2:
    st.write("")  # spacing
    st.write("")
    steps = st.slider("Steps", 15, 60, 30, 5)
    guidance = st.slider("Guidance scale", 4.0, 14.0, 7.5, 0.5)
    seed_input = st.number_input("Seed (-1 = random)", value=-1, min_value=-1)

# Generate button
if st.button("✨ Generate", type="primary", use_container_width=True):

    if not prompt.strip():
        st.warning("Please write something in the prompt field")
        st.stop()

    with st.spinner("Generating... (usually 3–12 minutes on free CPU)"):
        try:
            start_time = time.time()

            # Seed handling
            generator = None
            if seed_input != -1:
                generator = torch.Generator(device="cpu").manual_seed(seed_input)

            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                height=512,
                width=512
            ).images[0]

            took = time.time() - start_time

            st.success(f"Done in {took:.1f} seconds!")

            # Show result
            st.image(image, use_column_width=True)

            # Download
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            byte_data = buf.getvalue()

            st.download_button(
                label="💾 Download PNG",
                data=byte_data,
                file_name=f"ghibli-style-{int(time.time())}.png",
                mime="image/png",
                use_container_width=True
            )

        except Exception as e:
            st.error("Something went wrong during generation")
            st.exception(e)


# Footer / info
st.markdown("---")
st.caption(
    "Model: nitrosocke/Ghibli-Diffusion (public, no token needed)\n"
    "Style: anime / Studio Ghibli inspired\n"
    "Expectations: modest quality • slow generation (CPU only)\n"
    "Tip: 25–40 steps usually gives best balance"
)
