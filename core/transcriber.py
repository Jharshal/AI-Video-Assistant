
import whisper
import os


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None


def load_model():

    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")

        _model = whisper.load_model(WHISPER_MODEL)

        print("Whisper model loaded successfully!")

    return _model


def transcribe_chunk(
    chunk_path: str,
    language: str = "english",
    translate: bool = False
) -> str:

    model = load_model()

    task = "translate" if translate else "transcribe"

    result = model.transcribe(
        chunk_path,
        task=task,
        language=language
    )

    return result["text"].strip()


def transcribe_all(
    chunks: list,
    language: str = "english",
    translate: bool = False
) -> str:

    full_transcript = []

    for i, chunk in enumerate(chunks):

        print(f"Transcribing chunk {i+1}/{len(chunks)}...")

        text = transcribe_chunk(
            chunk,
            language=language,
            translate=translate
        )

        full_transcript.append(text)

    print("\nTranscription completed!")

    return " ".join(full_transcript)
