import os
import shutil
from frechet_audio_distance import FrechetAudioDistance


# --- Initialize FAD ---
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=False
)

# --- 1) Single-file evaluation ---
with open(output_scores_file, "w", encoding="utf-8") as f_out:
    # Walk through all subdirectories/files in eval_data
    for root, dirs, files in os.walk(eval_data):
        for filename in files:
            if filename.lower().endswith(".wav"):
                # Create a directory named after the WAV file (without extension)
                sample_dirname = os.path.splitext(filename)[0]
                sample_dir = os.path.join(root, sample_dirname)
                os.makedirs(sample_dir, exist_ok=True)

                # Full path to the original WAV
                src_wav_path = os.path.join(root, filename)
                # Destination path (inside the newly created directory)
                dst_wav_path = os.path.join(sample_dir, filename)
                
                # Copy the WAV file into that directory
                shutil.copyfile(src_wav_path, dst_wav_path)

                try:
                    # Compute FAD score comparing bg_data to the newly created directory
                    fad_score = frechet.score(
                        bg_data,
                        sample_dir,
                        background_embds_path=bg_emb,
                        dtype="float32"
                    )
                    # Write filename and FAD score to the output text file
                    f_out.write(f"{filename}\t{fad_score}\n")

                finally:
                    # Clean up: delete the directory and its contents
                    shutil.rmtree(sample_dir)


# --- 2) Whole-folder evaluation ---
# Compute one FAD score for the entire eval_data folder
whole_folder_score = frechet.score(
    bg_data,
    eval_data,
    background_embds_path=bg_emb,
    dtype="float32"
)

# Write the whole-folder FAD score into a separate text file
with open(whole_folder_scores_file, "w", encoding="utf-8") as f_out_folder:
    f_out_folder.write(f"FAD Score for entire folder '{eval_data}': {whole_folder_score}\n")

print(f"Individual file FAD scores written to: {output_scores_file}")
print(f"Whole-folder FAD score written to: {whole_folder_scores_file}")
