import os
import base64
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def prompt_llm(user_text):
    """
    Basic function to send user text to GPT-4 mini and return the response.

    Args:
        user_text (str): The input text from the user

    Returns:
        str: The response text from GPT-4 mini
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4 mini model
            messages=[{"role": "user", "content": user_text}],
            max_tokens=1000,
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


def prompt_llm_image(image_path, prompt_text):
    """
    Function to send an image and text prompt to GPT-4 Vision and return the response.

    Args:
        image_path (str): Path to the image file
        prompt_text (str): The text prompt to ask about the image

    Returns:
        str: The response text from GPT-4 Vision
    """
    try:
        # Check if image file exists
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"

        # Read and encode the image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Determine the image format
        image_extension = os.path.splitext(image_path)[1].lower()
        if image_extension in [".jpg", ".jpeg"]:
            image_format = "jpeg"
        elif image_extension == ".png":
            image_format = "png"
        elif image_extension == ".gif":
            image_format = "gif"
        elif image_extension == ".webp":
            image_format = "webp"
        else:
            return f"Error: Unsupported image format {image_extension}. Supported formats: jpg, jpeg, png, gif, webp"

        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 with vision capabilities
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


def parse_bullet_points(llm_output):
    """
    Parse LLM output and extract bullet points into a list of individual items.

    Args:
        llm_output (str): The output text from the LLM containing bullet points

    Returns:
        list: A list of individual bullet point items (up to 5 items)
    """
    if (
        not llm_output
        or isinstance(llm_output, str)
        and llm_output.startswith("Error:")
    ):
        return []

    # Split by lines and process each line
    lines = llm_output.strip().split("\n")
    bullet_points = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Look for bullet points with various formats:
        # - Traditional bullets: -, *, •
        # - Numbered lists: 1., 2., etc.
        # - Unicode bullets: ◦, ▪, ▫, etc.
        bullet_patterns = [
            r"^[-*•◦▪▫]\s*(.+)",  # - * • ◦ ▪ ▫ bullets
            r"^\d+\.\s*(.+)",  # 1. 2. 3. numbered
            r"^[a-zA-Z]\.\s*(.+)",  # a. b. c. lettered
            r"^\d+\)\s*(.+)",  # 1) 2) 3) numbered with parenthesis
        ]

        # Try each pattern
        for pattern in bullet_patterns:
            match = re.match(pattern, line)
            if match:
                bullet_text = match.group(1).strip()
                # Remove any trailing punctuation and clean up
                bullet_text = bullet_text.rstrip(".,;:!?")
                if bullet_text:  # Only add non-empty items
                    bullet_points.append(bullet_text)
                break

    # Return up to 5 items as requested
    return bullet_points[:5]


# Example usage
if __name__ == "__main__":
    # # Test the text function
    # print("=== Testing Text Function ===")
    # user_input = "Hello! Can you explain what machine learning is in simple terms?"
    # response = prompt_llm(user_input)
    # print(f"User: {user_input}")
    # print(f"AI: {response}")

    print("\n=== Testing Image Function ===")
    # Test the image function (assuming there's an image.jpg in the project)
    image_path = "image.jpg"
    if os.path.exists(image_path):
        # ===============================
        # First LLM ImageSummLLM
        # ===============================
        image_prompt = """
        You are helpful museum guide who can explain the history of different items
        
        What do you see in this image? 
        
        * instructions:
        - give me a bullet point list of the 5 most interesting items 
        - each bullet point should be 8 words max
        - there should be 5 bullet points max
        """
        image_response = prompt_llm_image(image_path, image_prompt)
        print(f"Image: {image_path}")
        print(f"Prompt: {image_prompt}")
        print(f"AI Response:\n{image_response}")

        # Parse the bullet points into a list
        bullet_list = parse_bullet_points(image_response)
        print(f"\nParsed Bullet Points ({len(bullet_list)} items):")
        for i, item in enumerate(bullet_list, 1):
            print(f"{i}. {item}")

        # ===============================
        # Second LLM HistoryLLM
        # ===============================
        item_number = input("Choose item number to explain: ")
        item = bullet_list[int(item_number) - 1]
        history_prompt = f"""\n
        explain the item: {item}
        * instructions:
        - a short quick history time period of the item
        - make your response structured as follows:
            # History of the item
             - Circa 1890
            # Where is it popular
             - United States
            # What is it made of
             - Gold
            # What is it used for
             - Jewelry
            # Who made it
             - Tiffany & Co.
            # Who owned it
             - John D. Rockefeller

        """
        item_response = prompt_llm(history_prompt)
        print(f"Item Response:\n{item_response}")
    else:
        print(
            f"No test image found at {image_path}. To test image functionality, add an image file and update the path."
        )
