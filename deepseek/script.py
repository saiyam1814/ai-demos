import os
import sys
import requests
from youtube_transcript_api import YouTubeTranscriptApi

def get_youtube_transcript(video_url):
    """
    Fetch the transcript of a YouTube video given its URL.
    """
    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def split_transcript(transcript, max_tokens, overlap=50):
    """
    Split the transcript into chunks that fit within the LLM token limit, with overlapping context.
    """
    words = transcript.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1  # Account for spaces
        if current_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]  # Keep overlap words for context
            current_length = sum(len(w) + 1 for w in current_chunk)
        current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_with_ollama(chunk):
    """
    Send a chunk of text to the Ollama API for summarization.
    """
    base_url = "http://127.0.0.1:11434/api/generate"  # Correct endpoint
    model = "deepseek-r1:7b"

    payload = {
        "model": model,
        "prompt": f"Summarize the following text in a concise and clear manner:\n{chunk}",
        "stream": False  # Get the response in a single reply
    }

    try:
        response = requests.post(base_url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No summary generated.")  # Extract the 'response' field
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama API: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <YouTube URL>")
        sys.exit(1)

    video_url = sys.argv[1]
    print("Fetching transcript...")
    transcript = get_youtube_transcript(video_url)

    if transcript.startswith("Error"):
        print(transcript)
        sys.exit(1)

    print("Transcript fetched successfully:")
    print(transcript[:500] + "...\n")  # Print the first 500 characters for verification

    # Split the transcript into manageable chunks
    max_tokens = 16000  # Adjust based on your LLM's limit
    chunks = split_transcript(transcript, max_tokens, overlap=50)

    summaries = []
    print("Generating summary...")
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        print(f"Chunk {i + 1}: {chunk[:200]}...\n")  # Print the first 200 characters of the chunk

        summary = summarize_with_ollama(chunk)
        print(f"Summary for chunk {i + 1}: {summary}\n")  # Print the summary for debugging
        summaries.append(summary)

    final_summary = " ".join(summaries)
    print("Final Summary:")
    print(final_summary)

if __name__ == "__main__":
    main()

