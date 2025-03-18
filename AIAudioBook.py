import torch
from typing import Optional, Tuple, List
from models import build_model, generate_speech, list_available_voices
from tqdm.auto import tqdm
import soundfile as sf
from pathlib import Path
import numpy as np
import time
import os
import ebooklib
from bs4 import BeautifulSoup, Comment
from ebooklib import epub
from pydub import AudioSegment
import PyPDF2

# Constants
SAMPLE_RATE = 24000
DEFAULT_MODEL_PATH = 'kokoro-v1_0.pth'
DEFAULT_CHUNK_SIZE = 500  # Words per chunk
DEFAULT_OUTPUT_DIR = "output_chunks"
FINAL_OUTPUT_FILE = "final_output.wav"

# Configure tqdm for better Windows console support
tqdm.monitor_interval = 0

def extract_epub_text(epub_path: str) -> str:
    """Extract and clean text content from an EPUB file."""
    book = epub.read_epub(epub_path)
    text = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            try:
                content = item.get_content().decode('utf-8')
                soup = BeautifulSoup(content, 'html.parser')
                for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                    comment.extract()
                cleaned_text = soup.get_text(separator=' ', strip=True)
                text.append(cleaned_text)
            except UnicodeDecodeError:
                tqdm.write(f"Warning: Could not decode content of item {item.get_name()}. Skipping...")
    return ' '.join(text)

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF using PyPDF2."""
    text = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return ' '.join(text)

def split_text_into_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Split text into chunks of approximately `chunk_size` words."""
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def save_audio_with_retry(audio_data: np.ndarray, sample_rate: int, output_path: Path, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
    """Attempt to save audio with retries."""
    for attempt in range(max_retries):
        try:
            sf.write(output_path, audio_data, sample_rate)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                tqdm.write(f"Failed to save audio (attempt {attempt + 1}/{max_retries})")
                tqdm.write(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                tqdm.write(f"Error: Could not save audio after {max_retries} attempts.")
                return False
    return False

def combine_wav_files(chunk_files: List[Path], output_file: str):
    """Combine WAV files into one."""
    combined = AudioSegment.empty()
    for chunk_file in tqdm(chunk_files, desc="Combining WAV files", position=0):
        combined += AudioSegment.from_wav(chunk_file)
    combined.export(output_file, format="wav")

def select_voice(voices: List[str]) -> str:
    """Interactive voice selection."""
    tqdm.write("Available voices:")
    for i, voice in enumerate(voices, 1):
        tqdm.write(f"{i}. {voice}")
    while True:
        try:
            choice = input("Select a voice number (or press Enter for default 'af_bella'): ").strip()
            if not choice:
                return "af_bella"
            choice = int(choice)
            if 1 <= choice <= len(voices):
                return voices[choice - 1]
            tqdm.write("Invalid choice. Please try again.")
        except ValueError:
            tqdm.write("Please enter a valid number.")

def get_speed() -> float:
    """Get speech speed from user."""
    while True:
        try:
            speed = input("Enter speech speed (0.5-2.0, default 1.0): ").strip()
            if not speed:
                return 1.0
            speed = float(speed)
            if 0.5 <= speed <= 2.0:
                return speed
            tqdm.write("Speed must be between 0.5 and 2.0")
        except ValueError:
            tqdm.write("Please enter a valid number.")

def ask_compression() -> bool:
    """Ask user if they want compression."""
    while True:
        choice = input("Compress final audio? (Y/N, default N): ").strip().lower()
        if not choice:
            return False
        if choice in ['y', 'n']:
            return choice == 'y'
        tqdm.write("Please enter 'y' or 'n'")

def process_input_to_audio(input_path: str, model, voice: str, speed: float, compress: bool):
    """Process input file into audio."""
    # Extract text
    tqdm.write(f"Extracting text from {Path(input_path).suffix.upper()}...")
    text = extract_epub_text(input_path) if input_path.endswith('.epub') else extract_pdf_text(input_path)
    
    # Split into chunks
    tqdm.write("Splitting text into chunks...")
    chunks = split_text_into_chunks(text)
    
    # Create output directory
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Generate audio for each chunk
    tqdm.write("Generating audio for each chunk...")
    chunk_files = []
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks", position=0), start=1):
        output_path = output_dir / f"chunk_{i}.wav"
        all_audio = []
        generator = model(chunk, voice=f"voices/{voice}.pt", speed=speed, split_pattern=r'[.!?]+\s+')
        
        # Inner progress bar for speech generation
        with tqdm(desc=f"Generating speech for chunk {i}", unit="segment", position=1, leave=False) as pbar:
            for gs, ps, audio in generator:
                if audio is not None:
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    all_audio.append(audio)
                    pbar.update(1)
        
        if all_audio:
            final_audio = torch.cat(all_audio, dim=0)
            if save_audio_with_retry(final_audio.numpy(), SAMPLE_RATE, output_path):
                chunk_files.append(output_path)
    
    # Combine WAV files
    tqdm.write("Combining all WAV files...")
    combine_wav_files(chunk_files, FINAL_OUTPUT_FILE)
    
    # Compress if requested
    output_path = Path(FINAL_OUTPUT_FILE)
    if compress:
        try:
            flac_path = output_path.with_suffix('.flac')
            tqdm.write("Compressing audio...")
            audio = AudioSegment.from_wav(output_path)
            audio.export(flac_path, format="flac", parameters=["-compression_level", "8"])
            output_path.unlink()
            output_path = flac_path
            tqdm.write(f"Compressed audio saved to {flac_path.absolute()}")
        except Exception as e:
            tqdm.write(f"Compression failed: {str(e)}")
    
    tqdm.write(f"Final audio saved to {output_path.absolute()}")
    # Clean up chunks
    for chunk_file in chunk_files:
        chunk_file.unlink()

def main() -> None:
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tqdm.write(f"Using device: {device}")
        
        # Build model
        with tqdm(total=1, desc="Building model", position=0) as pbar:
            model = build_model(DEFAULT_MODEL_PATH, device)
            pbar.update(1)
        
        # Get user inputs
        input_path = input("Enter the path to the input file (EPUB or PDF): ").strip()
        voices = list_available_voices()
        if not voices:
            tqdm.write("No voices found! Please check the voices directory.")
            return
        
        voice = select_voice(voices)
        speed = get_speed()
        compress = ask_compression()
        
        process_input_to_audio(input_path, model, voice, speed, compress)
    
    except Exception as e:
        tqdm.write(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()