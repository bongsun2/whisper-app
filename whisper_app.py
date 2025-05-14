import streamlit as st
import yt_dlp
import whisper
import os
from transformers import pipeline
from datetime import datetime

st.title("🎧 Whisper 기반 유튜브 대본 추출기")
st.write("유튜브 영상 링크를 입력하면 대본과 요약을 자동으로 생성해줍니다.")

url = st.text_input("유튜브 영상 URL을 입력하세요:")

def download_audio(youtube_url):
    ydl_opts = {
        'ffmpeg_location': '/usr/bin/ffmpeg',
        'format': 'bestaudio/best',

        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

if url:
    if st.button("🎬 실행하기"):
        with st.spinner("유튜브 오디오 다운로드 중..."):
            download_audio(url)

        with st.spinner("Whisper로 대본 추출 중..."):
            model = whisper.load_model("base")
            result = model.transcribe("audio.mp3")
            text = result.get("text", "").strip()
            if not text:
                text = "[⚠️ 대본 추출 실패 또는 음성 없음]"

        with st.spinner("요약 생성 중..."):
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

            def chunk_text(text, max_len=1000):
                return [text[i:i+max_len] for i in range(0, len(text), max_len)]

            chunks = chunk_text(text)
            summaries = []
            for chunk in chunks:
                try:
                    out = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                    if out and isinstance(out, list) and 'summary_text' in out[0]:
                        summaries.append(out[0]['summary_text'])
                    else:
                        summaries.append("[요약 실패]")
                except Exception as e:
                    summaries.append(f"[요약 에러: {str(e)}]")

            summary = "\n".join(summaries)

        with st.spinner("결과 저장 중..."):
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"whisper_result_{now}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("📝 [전문 대본]\n")
                f.write(text)
                f.write("\n\n📘 [요약본]\n")
                f.write(summary)

        st.success("✅ 작업 완료! 결과 파일을 다운로드하세요.")
        with open(filename, "rb") as f:
            st.download_button(label="📥 결과 다운로드", data=f, file_name=filename, mime="text/plain")

        os.remove("audio.mp3")
        os.remove(filename)
