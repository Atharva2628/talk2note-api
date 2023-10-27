from flask import Flask, request
import requests
# import json
from flask_cors import CORS
from io import BytesIO
# import json
# import uuid
import openai
from pydub import AudioSegment
from pydub.utils import make_chunks
import requests

app = Flask(__name__)
CORS(app, resources={r"/talk2notecomputex": {"origins": "*", "supports_credentials": True}})

@app.route('/talk2notecomputex')
def hello_world():
    audio_uri = request.args.get('audio_uri')
    api_key = request.args.get('api_key')

    if audio_uri and api_key == 'tbx827b.8x7[2h]dgbyu3g(hh88CRYN#QIiuyrnh2879iOIUYNjxknw9*2fbtydIUBTUfb27ybcify)noi3huih:893YNCDYBU=TCNkuhncfiicuwi':
        
        openai.api_key = "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"
        
        response = requests.get(audio_uri)

        myaudio = AudioSegment.from_file(BytesIO(response.content), format="mp4")
        chunk_length_ms = 1000*60*4 # pydub calculates in millisec (4 mins each)
        chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

        transcriptionMatrix = []
        for i, chunk in enumerate(chunks):
            exported_io = BytesIO()
            exported_io.name = f'chunk{i}.mp4'
            chunk.export(exported_io, format="mp4")
            transcriptx = openai.Audio.transcribe("whisper-1", exported_io)
            transcriptionMatrix.append(transcriptx)
        
        # print(transcriptionMatrix)

        allText = ""
        for i in range(len(transcriptionMatrix)):
            allText += " "+transcriptionMatrix[i].text

        
        x=10000*4 # max length of 40k characters (~10k tokens)
        res=[allText[y-x:y] for y in range(x, len(allText)+x,x)]
        # res[1]

        
        responseArray = []
        
        for segment in res:
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                "role": "system",
                "content": "You create comprehensive lecture notes from lecture transcripts in this format:\n\nTITLE: Reasonable title.\n\nNOTES: Explain each topic in high detail with clear headings, not more than one sentence each. \n\nPOTENTIAL EXAM QUESTIONS: Create three potential exam questions."
                },
                {
                "role": "user",
                "content": segment
                },
            ],
            temperature=0.75,
            max_tokens=4096,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
            responseArray.append(response)

        from typing_extensions import final
        finalTitle = "TITLE:"
        finalNotes = "NOTES:"
        finalExamQuestions = "POTENTIAL EXAM QUESTIONS:"
        
        for response in responseArray:
            text = response.choices[0].message.content
            title = text.split("TITLE:")[1].split("NOTES:")[0].replace("\n\n","")
            notes = text.split("NOTES:")[1].split("POTENTIAL EXAM QUESTIONS:")[0].replace("\n\n","")
            examQuestions = text.split("POTENTIAL EXAM QUESTIONS:")[1].replace("\n\n","")
        
            finalTitle =  finalTitle + "\n" + title
            finalNotes = finalNotes + "\n" + notes
            finalExamQuestions = finalExamQuestions + "\n" + examQuestions
        
        finalTitle = finalTitle.replace("\n\n","\n").replace("\n \n","\n")
        finalNotes = finalNotes.replace("\n\n","\n").replace("\n \n","\n")
        finalExamQuestions = finalExamQuestions.replace("\n\n","\n").replace("\n \n","\n")
        
        # print(finalTitle)
        # print(finalNotes)
        # print(finalExamQuestions)

        return [finalTitle, finalNotes, finalExamQuestions]
    
    return 'No Response'
