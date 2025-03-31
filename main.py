import streamlit as st
import os
import base64
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI

# Load env + OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="Cocktail Image Generator", layout="centered")
st.title("🍸 Cocktail Image Generator")
st.caption("Upload 3 elements and describe your cocktail. GPT-4o will help construct a photorealistic image prompt.")

# Upload images
col1, col2, col3 = st.columns(3)
with col1:
    glass_img = st.file_uploader("Glass", type=["jpg", "jpeg", "png"], key="glass")
    if glass_img: st.image(glass_img, caption="Glass", width=120)
with col2:
    garniture_img = st.file_uploader("Garniture (on rim)", type=["jpg", "jpeg", "png"], key="garniture")
    if garniture_img: st.image(garniture_img, caption="Garniture", width=120)
with col3:
    bite_img = st.file_uploader("Bite (floating inside)", type=["jpg", "jpeg", "png"], key="bite")
    if bite_img: st.image(bite_img, caption="Bite", width=120)

color = st.text_input("Cocktail liquid color", "amber")

# GPT-4o Vision
def describe_image(file, label):
    image_bytes = file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are a visual assistant. Briefly describe this {label} for photorealistic image generation."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error analyzing {label}: {str(e)}"

# Button 1: Analyze images
if st.button("🧠 Analyze & Generate Prompt"):
    if not all([glass_img, garniture_img, bite_img]):
        st.warning("Please upload all 3 images.")
    else:
        with st.spinner("Describing images with GPT-4o..."):
            glass_desc = describe_image(glass_img, "glass")
            garniture_desc = describe_image(garniture_img, "garniture")
            bite_desc = describe_image(bite_img, "bite")

            prompt = (
                f"Create a photorealistic image of a cocktail in a glass that looks like this: {glass_desc}. "
                f"Place a garniture on the rim that resembles this: {garniture_desc}, and a floating bite similar to this: {bite_desc}. "
                f"The cocktail liquid should be {color}. No ice cube floating. White background."
            )

            st.session_state["final_prompt"] = prompt

# Show editable prompt if available
if "final_prompt" in st.session_state:
    st.markdown("### ✏️ Final Prompt")
    st.session_state["final_prompt"] = st.text_area("You can edit this prompt before generating:", value=st.session_state["final_prompt"], height=200)
    if st.button("🎨 Generate Image"):
        with st.spinner("Generating image using DALL·E 3..."):
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=st.session_state["final_prompt"],
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                image_url = response.data[0].url
                st.image(image_url, caption="Generated Cocktail", use_container_width=True)
                st.markdown(f"[🔽 Download Image]({image_url})", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
