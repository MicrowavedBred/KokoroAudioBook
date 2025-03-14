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
                # Extract raw content
                content = item.get_content().decode('utf-8')
                # Clean HTML tags and comments
                soup = BeautifulSoup(content, 'html.parser')
                # Remove comments
                for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                    comment.extract()
                # Get clean text
                cleaned_text = soup.get_text(separator=' ', strip=True)
                text.append(cleaned_text)
            except UnicodeDecodeError:
                print(f"Warning: Could not decode content of item {item.get_name()}. Skipping...")
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
    """
    Attempt to save audio data to file with retry logic.
    Returns True if successful, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            sf.write(output_path, audio_data, sample_rate)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\nFailed to save audio (attempt {attempt + 1}/{max_retries})")
                print("The output file might be in use by another program (e.g., media player).")
                print(f"Please close any programs that might be using '{output_path}'")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"\nError: Could not save audio after {max_retries} attempts.")
                print(f"Please ensure '{output_path}' is not open in any other program and try again.")
                return False
    return False


def combine_wav_files(chunk_files: List[Path], output_file: str):
    """Combine multiple WAV files into one."""
    combined = AudioSegment.empty()
    for chunk_file in tqdm(chunk_files, desc="Combining WAV files"):
        combined += AudioSegment.from_wav(chunk_file)
    combined.export(output_file, format="wav")


def select_voice(voices: List[str]) -> str:
    """Interactive voice selection."""
    print("\nAvailable voices:")
    for i, voice in enumerate(voices, 1):
        print(f"{i}. {voice}")
    while True:
        try:
            choice = input("\nSelect a voice number (or press Enter for default 'af_bella'): ").strip()
            if not choice:
                return "af_bella"
            choice = int(choice)
            if 1 <= choice <= len(voices):
                return voices[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def get_speed() -> float:
    """Get speech speed from user."""
    while True:
        try:
            speed = input("\nEnter speech speed (0.5-2.0, default 1.0): ").strip()
            if not speed:
                return 1.0
            speed = float(speed)
            if 0.5 <= speed <= 2.0:
                return speed
            print("Speed must be between 0.5 and 2.0")
        except ValueError:
            print("Please enter a valid number.")


def ask_compression() -> bool:
    """Ask user if they want lossless FLAC compression."""
    while True:
        choice = input("\nCompress final audio? (Y/N, default N): ").strip().lower()
        if not choice:
            return False
        if choice in ['y', 'n']:
            return choice == 'y'
        print("Please enter 'y' or 'n'")


def process_input_to_audio(input_path: str, model, voice: str, speed: float, compress: bool):
    """Process an input file (EPUB or PDF) into audio chunks and combine them."""
    # Extract text based on file type
    if input_path.lower().endswith('.epub'):
        print("\nExtracting text from EPUB...")
        text = extract_epub_text(input_path)
    elif input_path.lower().endswith('.pdf'):
        print("\nExtracting text from PDF...")
        text = extract_pdf_text(input_path)
    else:
        raise ValueError("Unsupported file type. Only .epub and .pdf are supported.")

    # Split text into chunks
    print("\nSplitting text into chunks...")
    chunks = split_text_into_chunks(text)

    # Create output directory for chunks
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Generate audio for each chunk
    print("\nGenerating audio for each chunk...")
    chunk_files = []
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        output_path = output_dir / f"chunk_{i + 1}.wav"
        all_audio = []
        generator = model(chunk, voice=f"voices/{voice}.pt", speed=speed, split_pattern=r'\n+')
        with tqdm(desc=f"Generating speech for chunk {i + 1}") as pbar:
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

    # Combine all WAV files
    print("\nCombining all WAV files...")
    combine_wav_files(chunk_files, FINAL_OUTPUT_FILE)

    # Compress if requested
    output_path = Path(FINAL_OUTPUT_FILE)
    if compress:
        try:
            flac_path = output_path.with_suffix('.flac')
            print("\nCompressing audio...")
            audio = AudioSegment.from_wav(output_path)
            audio.export(flac_path, format="flac", parameters=["-compression_level", "8"])
            output_path.unlink()  # Remove original WAV
            output_path = flac_path
            print(f"Compressed audio saved to {flac_path.absolute()}")
        except Exception as e:
            print(f"Compression failed: {str(e)}")
            print("Saving as WAV instead")

    print(f"\nFinal audio saved to {output_path.absolute()}")

    # Clean up chunk files
    for chunk_file in chunk_files:
        chunk_file.unlink()


def main() -> None:
    try:
        # Set up device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Build model
        print("\nInitializing model...")
        with tqdm(total=1, desc="Building model") as pbar:
            model = build_model(DEFAULT_MODEL_PATH, device)
            pbar.update(1)

        # Get user inputs
        input_path = input("\nEnter the path to the input file (EPUB or PDF): ").strip()

        # List available voices
        voices = list_available_voices()
        if not voices:
            print("No voices found! Please check the voices directory.")
            return

        # Select voice
        voice = select_voice(voices)

        # Get speech speed
        speed = get_speed()

        # Ask for compression
        compress = ask_compression()

        # Process input file to audio
        process_input_to_audio(input_path, model, voice, speed, compress)

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
