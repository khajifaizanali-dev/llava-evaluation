from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="FreedomIntelligence/MileBench",
    local_dir="/scratch/user/khajifaizanali/nlpproject/MileBench",
    local_dir_use_symlinks=False  # Optional: store full files instead of symlinks
)

from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="FreedomIntelligence/MileBench", local_dir="/scratch/user/khajifaizanali/nlpproject/MileBench", repo_type="dataset")