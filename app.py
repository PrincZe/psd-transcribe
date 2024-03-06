from flask import Flask, request, jsonify, render_template
import replicate
import os
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Access environment variables for sensitive information
aws_access_key = os.environ.get("AWS_ACCESS_KEY")
aws_secret_key = os.environ.get("AWS_SECRET_KEY")
replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
bucket_name = os.environ.get("S3_BUCKET_NAME")

# Initialize AWS S3 client
s3 = boto3.client(
    "s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
)

# Initialize Replicate client
model = replicate.Client(api_token=replicate_api_token)

# render HTML
@app.route("/")
def index():
    return render_template("index.html")

# Function to transcribe audio using Replicate
@app.route("/process-audio", methods=["POST"])
def process_audio_data():
    try:
        audio_file = request.files["audio"]

        print("Processing audio...")

        def read_file_chunks(file, chunk_size=8192):
            """Generator function to read a file in chunks"""
            while True:
                data = file.read(chunk_size)
                if not data:
                    break
                yield data

        # Upload audio directly to S3 bucket in chunks
        s3.upload_fileobj(read_file_chunks(audio_file), bucket_name, audio_file.filename)

        audio_data_uri = f"https://{bucket_name}.s3.amazonaws.com/{audio_file.filename}"

        # Run Replicate model for transcription
        output = model.run(
            "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
            input={
                "task": "transcribe",
                "audio": audio_data_uri,
                "language": "english",
                "timestamp": "chunk",
                "batch_size": 64,
                "diarise_audio": False,
            },
        )

        results = output["text"]

        return jsonify({"transcript": results})

    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"error": "An error occurred while processing the audio"}), 500

# Function to generate suggestion using Replicate
@app.route("/get-suggestion", methods=["POST"])
def get_suggestion():
    try:
        data = request.get_json()  # Parse JSON data from the request
        transcript = data.get("transcript", "")  # Extract transcript
        prompt_text = data.get("prompt", "")  # Extract prompt text

        prompt = f"""
        {transcript}
        ------
        {prompt_text}
        """

        suggestion = ""
        for event in model.stream(
            "mistralai/mistral-7b-instruct-v0.2",
            input={
                "debug": False,
                "top_k": 50,
                "top_p": 0.9,
                "prompt": prompt,
                "temperature": 0.6,
                "max_new_tokens": 512,
                "min_new_tokens": -1,
                "prompt_template": "<s>[INST] {prompt} [/INST] ",
                "repetition_penalty": 1.15,
            },
        ):
            suggestion += str(event)  # Accumulate the output

        return jsonify({"suggestion": suggestion})  # Send as JSON response

    except Exception as e:
        print(f"Error generating suggestion: {e}")
        return jsonify({"error": "An error occurred while generating suggestion"}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)