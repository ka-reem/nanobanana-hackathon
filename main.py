import os
import pygame
from PIL import Image
import requests
import base64
import math

# Configuration
IMAGE_FILE = "images/elon-camera.png"  # Single image file
FRAME_DELAY = 500  # Delay between frames in milliseconds
GRID_SIZE = (4, 4)  # 4x4 grid

# Text overlay configuration
TEXTS = ["First text", "Second text", "Third text", "Fourth text"]  # one text per group
TEXT_CHANGE_EVERY = 4  # change text every N frames
FONT_SIZE = 48

# Model configuration
MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct-FP8"
# Provide either LLAMA_API_URL + LLAMA_API_KEY in environment, or OPENAI_API_KEY for OpenAI (fallback)
LLAMA_API_URL = os.environ.get("LLAMA_API_URL")
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# Load .env (if python-dotenv is available) or parse .env file manually
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v


def _local_fallback_generate(image_path, groups):
    """Create deterministic, clearly fictional captions and a short story when no LLM is available."""
    base = os.path.splitext(os.path.basename(image_path))[0].replace("-", " ").title()
    phrases = [
        "BREAKING: MUSK SELLS TESLA",
        "CONFIRMED: $500B DEAL",
        "EXCLUSIVE: NEW CEO NAMED", 
        "URGENT: STOCK SURGES",
        "DEVELOPING: MERGER TALKS",
        "ALERT: MARKET REACTION",
    ]
    captions = [phrases[i % len(phrases)] for i in range(groups)]
    story = (
        f"BREAKING NEWS - FICTIONAL REPORT: In this realistic but completely invented business story, "
        "sources report that Elon Musk has announced plans to sell a majority stake in Tesla to focus on SpaceX's Mars colonization project. "
        "The fictional transaction, valued at approximately $500 billion, would reportedly involve a consortium of major automotive and tech companies. "
        "In this imaginary scenario, Musk stated his intention to concentrate on sustainable space technology and interplanetary transport. "
        "Market analysts in this fictional report predict significant implications for the EV industry and Tesla's future direction. "
        "This is entirely fictional business speculation - no real transactions, announcements, or business decisions are represented here. "
        "This is a simulated realistic news report used as a local fallback for the demo."
    )
    return captions, story


def crop_image(image_path, grid_size):
    """Crop a single image into a grid of smaller images."""
    image = Image.open(image_path)
    img_width, img_height = image.size
    cell_width = img_width // grid_size[1]
    cell_height = img_height // grid_size[0]

    cropped_images = []
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            left = col * cell_width
            upper = row * cell_height
            right = left + cell_width
            lower = upper + cell_height
            cropped_image = image.crop((left, upper, right, lower))

            # Resize the cropped image to be 3x larger
            resized_image = cropped_image.resize((cell_width * 3, cell_height * 3))

            # Convert to a format pygame can use
            cropped_images.append(pygame.image.fromstring(
                resized_image.tobytes(), resized_image.size, resized_image.mode
            ))

    return cropped_images


def play_stop_motion(images, frame_delay, texts=None, change_every=4):
    """Play the stop-motion animation and overlay text that changes every `change_every` frames."""
    pygame.init()

    # Prepare font
    pygame.font.init()
    font = pygame.font.SysFont(None, FONT_SIZE)

    # Calculate the display size based on the first image
    img_width, img_height = images[0].get_size()
    screen = pygame.display.set_mode((img_width, img_height))
    pygame.display.set_caption("Stop Motion Video")

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Play the animation frame by frame
        for i, image in enumerate(images):
            screen.fill((0, 0, 0))  # Clear the screen
            screen.blit(image, (0, 0))  # Display the image full screen

            # Draw text overlay for every group of frames
            if texts:
                group_index = i // change_every
                text = texts[group_index % len(texts)]

                # Render text with a simple shadow for readability
                text_surface = font.render(text, True, (255, 255, 255))
                shadow_surface = font.render(text, True, (0, 0, 0))
                tx = (img_width - text_surface.get_width()) // 2
                ty = img_height - text_surface.get_height() - 20
                screen.blit(shadow_surface, (tx + 2, ty + 2))
                screen.blit(text_surface, (tx, ty))

            pygame.display.flip()
            pygame.time.delay(frame_delay)

        clock.tick(60)

    pygame.quit()


def generate_story_and_captions(image_path, model_name, groups, api_url=None, api_key=None):
    """Send the image to the LLM endpoint and request a fictional 'crazy fake news' story plus short captions.

    Returns (captions_list, story_text). Captions_list will have length == groups.
    The prompt forces the model to produce clearly fictional content and avoid real people.
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt_text = (
        f"Create a realistic, plausible FICTIONAL breaking news story about Elon Musk announcing he's selling Tesla or making major business changes. "
        "Make it sound like a credible news report with realistic business reasons like: "
        "selling Tesla to focus on SpaceX and Mars colonization, stepping down as CEO to pursue new ventures, "
        "announcing a strategic partnership or merger, restructuring the company for sustainability goals, "
        "divesting to concentrate on AI and robotics development, or transitioning leadership to focus on Neuralink. "
        "Include realistic quotes about innovation, the future of transportation, or sustainable technology. "
        "Add plausible market reactions, analyst predictions, and business implications. "
        "Use realistic buyer scenarios like major automotive companies, tech giants, or investment consortiums. "
        "Keep everything clearly FICTIONAL but believable — this is realistic speculative news, NOT real events!\n"
        f"Also produce exactly {groups} professional news overlay captions (one per group). Each caption must be 3-6 words, newsworthy and professional, suitable for business news headlines (use words like 'BREAKING', 'EXCLUSIVE', 'URGENT', 'CONFIRMED').\n"
        "Output format MUST be: captions separated by `||` on the first line, then a line with `===STORY===`, then the full story text. Example:\n"
        "BREAKING: MUSK SELLS TESLA || CONFIRMED: $500B DEAL || EXCLUSIVE: NEW CEO NAMED || ...\n===STORY===\nBREAKING NEWS - In this FICTIONAL but realistic business report...\n\n"
        "Write it like a professional business news story with credible tone and realistic details. Include the word FICTIONAL prominently for clarity."
    )

    # Build compatible chat payload (works for OpenAI-style and Llama's chat endpoint)
    chat_payload = {
        "model": model_name,
        "messages": [{
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                }
            ]
        }],
        "max_tokens": 800,
    }

    headers = {}

    # If a LLAMA_API_KEY is provided but no LLAMA_API_URL, default to Llama's public API endpoint
    if api_key and not api_url and os.environ.get("LLAMA_API_KEY"):
        api_url = os.environ.get("LLAMA_API_URL") or "https://api.llama.com/v1/chat/completions"
        print(f"Info: LLAMA_API_KEY present, using default LLama endpoint: {api_url}")

    # Try provider-specific endpoints
    if api_url and api_key:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(api_url, json=chat_payload, headers=headers, timeout=30)
    elif OPENAI_API_KEY:
        # Fallback to OpenAI chat completions -- send as a single user message
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        resp = requests.post("https://api.openai.com/v1/chat/completions", json=chat_payload, headers=headers, timeout=30)
    else:
        # No API configuration at all — fallback to local deterministic generator
        print("Warning: No API configuration found. Falling back to deterministic local captions/story.")
        return _local_fallback_generate(image_path, groups)

    resp.raise_for_status()
    data = resp.json()
    
    # Debug: print the response to see what we're getting
    print("DEBUG: API Response:", data)

    # Extract text from provider response
    text = None
    if "choices" in data and len(data["choices"]) > 0:
        # OpenAI-style
        text = data["choices"][0].get("message", {}).get("content") or data["choices"][0].get("text")
    elif "completion_message" in data:
        # Llama API style
        content = data["completion_message"].get("content", {})
        if isinstance(content, dict) and "text" in content:
            text = content["text"]
        elif isinstance(content, str):
            text = content
    else:
        # Generic field
        text = data.get("output") or data.get("text") or data.get("response")

    if not text:
        print("DEBUG: Could not extract text from response. Available keys:", list(data.keys()))
        if "completion_message" in data:
            print("DEBUG: completion_message keys:", list(data["completion_message"].keys()))
        raise RuntimeError("Failed to extract text from model response")

    # Parse expected format
    parts = text.split("===STORY===")
    captions_part = parts[0].strip() if len(parts) > 0 else ""
    story_part = parts[1].strip() if len(parts) > 1 else text.strip()

    captions = [c.strip() for c in captions_part.split("||") if c.strip()]

    # If model did not return enough captions, pad by repeating the last caption or generate dummy captions
    if len(captions) < groups:
        last = captions[-1] if captions else "Breaking News"
        while len(captions) < groups:
            captions.append(last)

    return captions[:groups], story_part


def main():
    # reload env after attempting to load .env
    LLAMA_API_URL = os.environ.get("LLAMA_API_URL")
    LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    print("DEBUG: LLAMA_API_URL set?", bool(LLAMA_API_URL))
    print("DEBUG: LLAMA_API_KEY set?", bool(LLAMA_API_KEY))
    print("DEBUG: OPENAI_API_KEY set?", bool(OPENAI_API_KEY))

    # Crop the single image into a grid of smaller images
    images = crop_image(IMAGE_FILE, GRID_SIZE)
    total_frames = len(images)
    expected = GRID_SIZE[0] * GRID_SIZE[1]
    if total_frames != expected:
        print(f"Error: Expected {expected} images, but found {total_frames}.")
        return

    # Determine number of caption groups
    groups = math.ceil(total_frames / TEXT_CHANGE_EVERY)

    # Generate story and captions from the model (or fallback)
    try:
        api_url = LLAMA_API_URL
        api_key = LLAMA_API_KEY
        captions, story = generate_story_and_captions(IMAGE_FILE, MODEL_NAME, groups, api_url=api_url, api_key=api_key)
        print("Generated fictional story:\n", story)
    except Exception as e:
        print("Warning: failed to generate story from model:", e)
        # Use deterministic local fallback (clearly fictional) instead of the static TEXTS
        captions, story = _local_fallback_generate(IMAGE_FILE, groups)
        print("Using local fallback story/captions.")

    # Convert captions into repeated list matching frames
    texts_per_frame = []
    for i in range(total_frames):
        group_index = i // TEXT_CHANGE_EVERY
        texts_per_frame.append(captions[group_index % len(captions)])

    # Play the stop-motion video with text overlays
    play_stop_motion(images, FRAME_DELAY, texts=texts_per_frame, change_every=1)


if __name__ == "__main__":
    main()