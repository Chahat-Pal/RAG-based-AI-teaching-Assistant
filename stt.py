import  whisper
import json

model = whisper.load_model("base")

result = model.transcribe("audio/1.mp3")
chunks=[]
for segment in result['segments']:
    chunks.append({"start": segment['start'], 'end': segment['end'], 'text': segment['text']})
print(chunks)
with open("output.json", "w") as f:
    json.dump(chunks,f)