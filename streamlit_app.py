import streamlit as st
from PIL import Image
import requests
import google.generativeai as genai
import json
import urllib.parse

buy_me_coffee_link = "https://www.buymeacoffee.com/ataulhaque"

def extract_text_from_image(image_file):
    """Extracts text from an image using Google Cloud Vision API.

    Args:
        image_file: The image file to process.

    Returns:
        The extracted text from the image.
    """

    # Replace 'YOUR_API_KEY' with your actual Google Cloud Vision API key
    api_key = "YOUR_API_KEY"
    endpoint = "https://vision.googleapis.com/v1/images:annotate"

    # Prepare the image data
    image_content = image_file.read()
    image_bytes = base64.b64encode(image_content).decode('utf-8')

    # Construct the request body
    request_body = {
        "requests": [
            {
                "image": {
                    "content": image_bytes
                },
                "features": [
                    {"type": "DOCUMENT_TEXT_DETECTION"}
                ]
            }
        ]
    }

    # Send the request
    response = requests.post(endpoint, headers={'Authorization': 'Bearer ' + api_key}, json=request_body)

    # Parse the response
    if response.status_code == 200:
        response_json = json.loads(response.text)
        text = response_json['responses'][0]['fullTextAnnotation']['text']
        return text
    else:
        st.error("Error extracting text from image. Please try again.")
        return None

def solve_math_problem(question, hf_api_key):
    url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    parameters = {
        "max_new tokens": 5000,
        "temperature": 0.01,
        "top_k": 50,
        "top p": 0.95,
        "return_full_text": False,
    }
    prompt = """
    </begin of_text |></start_header_id|>system< | end_header_id>You are a helpful and smart Math professor. You accurately solve the math problem thinking step by step.<leot_id|></start_header_id|>user< end header id > Here is the
Math question: ```{question}``` if the question is unrelated to Math, respond politely to user to ask Math related questions only.<leot_id/></start_header_id|>assistant<|end_header_id|>
"""
    headers = {
        'Authorization': f'Bearer {hf_api_key}',
        'Content-Type': 'application/json'
        }

    #query += """ role: You are android app tester and should respond for ios related question that I don't know about ios"""
    prompt = prompt.replace("{question)", question)
    #print(prompt)
    payload = {
        "inputs": prompt,
        "parameters": parameters
    }
    response = requests.post(url, headers=headers, json=payload)
    response_text = response.json()[0]['generated_text'].strip()
    return response_text


def verify_solution(question, solution):
    # Mock GPT verification logic
    prompt = f"Is the following solution to the math question correct?\nQuestion: {question}\nSolution: {solution}. If you could verify the solution reply with 'AI_Verified=YES', else 'AI_Verified=NO'"
    # Call GPT-based model for verification
    genai.configure(api_key="YOUR_API_KEY")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    print(response.text)
    verified = "AI_Verified=YES" in response.text  # Replace with actual GPT API call result
    return verified

def notify_admin_via_whatsapp(question):
    """Encodes the question and creates a WhatsApp chat link for user feedback.
    Args:
      question: The user's question to be encoded.
    Returns:
      None
    """
    # URL encode the question
    url_encoded_question = urllib.parse.quote(question)
    # Informative message and WhatsApp chat link
    st.info("Connect with us on WhatsApp Chat for any feedback or consultation.")
    st.markdown("""
      <a aria-label="Chat on WhatsApp" href="https://wa.me/917205467646?text={url_encoded_question}">
        <img alt="Chat on WhatsApp" src="https://image.pngaaa.com/326/2798326-middle.png" width="150" height="auto"/>
      </a><br/><br/>
  """, unsafe_allow_html=True)

# Show title and description.
st.title("📄 Math Solver AI with Human Evaluation Backup")
st.write(
    "Upload a Math problem image below and ask a question about it – AI will answer! "
    "If AI couldn't solve, question will be passed to human for evaluation."
)

question_type = st.radio("Choose Math Question Format:", ["Text", "Image"])
question = ""

if question_type == "Text":
    question = st.text_area("Enter your math question here:")
elif question_type == "Image":
    uploaded_file = st.file_uploader("Upload an image with the math question:", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        question = extract_text_from_image(uploaded_file)

if st.button("Solve"):
    if question.strip():
        solution = solve_math_problem(question)
        verified = verify_solution(question, solution)
        if verified is False:
            st.error("The AI could not verify the solution. The question has been sent for human evaluation.")
            notify_admin_via_whatsapp(question)
            st.markdown(f"[Buy Me a Coffee for Human Evaluation]({buy_me_coffee_link})")
        else:
            st.success(f"The solution is: {solution}")
    else:
        st.warning("Please provide a valid math question.")
