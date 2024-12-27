import streamlit as st
from transformers import pipeline
import torch

# ------------------------------
# Load Whisper Model
# ------------------------------
@st.cache_resource
def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    print("Loading the Whisper model...")
    whisper = pipeline("automatic-speech-recognition", "openai/whisper-tiny", chunk_length_s = 30)
    print("Whisper model loaded!")
    return whisper

# ------------------------------
# Load NER Model
# ------------------------------
@st.cache_resource
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    print("Loading BERT...")
    ner = pipeline("ner", model = "dslim/bert-base-NER", tokenizer = "dslim/bert-base-NER")
    print("BERT loaded!")
    return ner

# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file, whisper_pipeline):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
    Returns:
        str: Transcribed text from the audio file.
    """
    print("Transcribing audio...")
    transcription = whisper_pipeline(uploaded_file)
    print("Transcription done!")
    return transcription['text']

# ------------------------------
# Entity Extraction
# ------------------------------
def extract_entities(text, ner_pipeline):
    """
    Extract entities from transcribed text using the NER model.
    Args:
        text (str): Transcribed text.
        ner_pipeline: NER pipeline loaded from Hugging Face.
    Returns:
        tuple: Lists of entity words per their groups(PERs, ORGs and LOCs).
    """
    print("Extracting entities...")
    results = ner_pipeline(text)
    per_set = set()
    org_set = set()
    loc_set = set()

    words_list = []

    for i in range(0, len(results)):

        #print(results[i])
        #print("Words_list: ", words_list)
        word = results[i]['word']
        tag = results[i]['entity'][-3:]

        if (results[i]['entity'][0] == "I"):
            word = words_list[-1][0] + " " + word
            word = word.replace(" ##", "") # "##" is used to signal continuation, we've already added a space. If that combination exists, unify the chars!
            words_list.pop(-1) #The old word was not the full word, we need to get rid of it and add in the correct one.
        
        words_list.append([word, tag])

    for couple in words_list:
        word = couple[0]
        tag = couple[1]
        if (tag == "PER"):
            per_set.add(word)
        elif (tag == "ORG"):
            org_set.add(word)
        elif (tag == "LOC"):
            loc_set.add(word)
        else:
            pass #We got "B-MISC, which we do not care about."

    #Now, for the PER processing.
    real_per_list = [per for per in per_set if " " in per] #We know that people whose names are more than one word are proper persons. They are straightforward enough.
    vetting_per_list = [per for per in per_set if not " " in per] #The others.

    #It is unlikely, but possible, that someone might just be referred to by their name or surname, with their full name never being used
    #in the recording. We will try to spot such cases here.

    names = []
    for real_per in real_per_list:
        names += real_per.split(" ") #We first get all the names and surnames separately.
    
    for per in vetting_per_list:
        if(per in names): #If the per waiting to be vetted is among the names we already know, we can safely assume that the token is *not* referring to someone new.
            continue 
        #But if we get a clean per, that means we are talking about someone new. Therefore, they need to be added as a real per, and another name to be checked for.
        real_per_list.append(per)
        names.append(per)


    print("Entities extracted!")
    return (real_per_list, list(org_set), list(loc_set))
            


# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    transcription = ""
    st.title("Meeting Transcription and Entity Extraction")

    # You must replace below
    STUDENT_NAME = "Metin Furkan Amarat"
    STUDENT_ID = "150230301"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")
    st.write("You can upload an audio recording of a discussion to;")
    st.write("")
    st.write("1. Transcribe the audio into text.")
    st.write("2. Extract key entities from the transcription such as Persons, Organisations, Dates, and Locations.")

    st.divider()
    uploaded_file = st.file_uploader("Upload an audio file (.wav format)", type = ".wav")
    if uploaded_file is not None:
        st.info("Please wait, the transcription may take time.")
        whisper = load_whisper_model()
        transcription = transcribe_audio(uploaded_file.getvalue(), whisper)
        st.success("Transcription completed!")

        st.divider()
    
        st.header("Transcription")
        st.write(transcription)
    
        st.divider()
    
        ner = load_ner_model()
        entities = extract_entities(transcription, ner)
        #print(entities)
    
        st.header("Extracted Entities")
        st.subheader("Persons(PERs)")
        for word in entities[0]:
            st.write("* " + word)
        st.subheader("Organisations(ORGs)")
        for word in entities[1]:
            st.write("* " + word)
        st.subheader("Locations(LOCs)")
        for word in entities[2]:
            st.write("* " + word)

if __name__ == "__main__":
    main()
