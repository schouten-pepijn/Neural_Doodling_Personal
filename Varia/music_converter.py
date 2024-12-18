from pydub import AudioSegment
import glob

def convert_flac_file_to_mp3(flac_file_path, mp3_file_path):
    """
    Given paths to a .flac file and a target .mp3 file, convert the .flac file to .mp3
    and save it at the target path.
    """
    audio_segment = AudioSegment.from_file(flac_file_path, format="flac")
    audio_segment.export(mp3_file_path, format="mp3")

def get_flac_files(directory_path):
    """
    Returns a list of all .flac files in the specified directory and its subdirectories.
    """
    return glob.glob(f"{directory_path}*/*.flac", recursive=True)
