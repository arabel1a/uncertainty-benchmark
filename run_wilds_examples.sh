source venv/bin/activate
export PATH=$(pwd)/wilds_git/examples:$PATH

# python3 wilds/examples/run_expt.py --algorithm ERM --dataset iwildcam --root_dir data

# python3 wilds_git/examples/run_expt.py --algorithm ERM --dataset civilcomments --root_dir data

python3 evaluate_uncertainty.py --algorithm ERM --dataset civilcomments --root_dir data
