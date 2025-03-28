# Run Sbatch to evaluate a particular dataset
sbatch sbatch.job

Datasets could be found under data/MileBench

# Run generate to generate responses from llava
python generate_small.py     --data_dir data/MileBench     --dataset_name IEdit    --model_name llava-v1.5-7b     --output_dir outputs 
# Run Evaluate script to evaluate the generated responses
python evaluate.py --data-dir data/MileBench --dataset IEdit --result-dir outputs/llava-v1.5-7b
