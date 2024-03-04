from flask import Flask, request, jsonify, render_template
import replicate
import os
import boto3
from io import BytesIO

aws_access_key = os.environ.get("AWS_ACCESS_KEY")
aws_secret_key = os.environ.get("AWS_SECRET_KEY")
bucket_name = "realtimeconversationupload"

s3 = boto3.client(
    "s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
)

app = Flask(__name__)
model = replicate


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process-audio", methods=["POST"])
def process_audio_data():
    audio_data = request.files["audio"].read()

    print("Processing audio...")
    try:
        # Stream audio data to S3 bucket
        s3_object_key = "audio.wav"
        s3.put_object(Bucket=bucket_name, Key=s3_object_key, Body=BytesIO(audio_data))

        # Get S3 object URL
        s3_object_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_object_key}"

        # Run Replicate model on S3 object URL
        output = model.run(
            "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
            input={
                "task": "transcribe",
                "audio": s3_object_url,
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
        return jsonify({"error": str(e)}), 500


@app.route("/get-suggestion", methods=["POST"])
def get_suggestion():
    data = request.get_json()
    transcript = data.get("transcript", "")
    prompt_text = data.get("prompt", "")

    prompt = f"{transcript}\n------\n{prompt_text}"

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
        suggestion += str(event)

    return jsonify({"suggestion": suggestion})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
