
import os 

# TRAIN EXPANDED NAMES
# os.system("python3 train/train_expanded_names_cds.py \
#     --num_female_names=2000 --num_male_names=2000 --dest=models_final/expanded_names_cds_unbalanced_2000_2000.kv \
#     --name_matching_probabilities=data/2000_2000_names.npy")

# os.system("python3 train/train_expanded_names_cds.py \
#     --num_female_names=1500 --num_male_names=2500 --dest=models_final/expanded_names_cds_unbalanced_1500_2500.kv \
#     --name_matching_probabilities=data/1500_2500_names_new.npy")

# os.system("python3 train/train_expanded_names_cds.py \
#     --num_female_names=2000 --num_male_names=2000 --dest=models_final/expanded_names_cds_unbalanced_2000_2000_less.kv \
#     --name_matching_probabilities=data/2000_2000_names.npy \
#     --less_target=True")


# os.system("python3 train/train_expanded_names_cds.py \
#     --num_female_names=1500 --num_male_names=2500 --dest=models_final/expanded_names_cds_unbalanced_1500_2500_less.kv \
#     --name_matching_probabilities=data/1500_2500_names_new.npy \
#     --less_target=True")




# # TRAIN ORIG CDS
# os.system("python3 train/train_orig_cds.py \
#     --dest=models_final/orig_cds.kv")

# os.system("python3 train/train_orig_cds.py \
#     --dest=models_final/orig_cds_less.kv \
#     --less_target=True")




# # TRAIN ORIG NAMES CDS
# os.system("python3 train/train_orig_names_cds.py \
#     --bipartite_matching=data/bipartite_name_matches_2000.txt --dest=models_final/orig_names_cds_2000.kv")

# os.system("python3 train/train_orig_names_cds.py \
#     --bipartite_matching=data/bipartite_name_matches.txt --dest=models_final/orig_names_cds_2500.kv")

# os.system("python3 train/train_orig_names_cds.py \
#     --bipartite_matching=data/bipartite_name_matches_2000.txt --dest=models_final/orig_names_cds_2000_less.kv \
#     --less_target=True")

# os.system("python3 train/train_orig_names_cds.py \
#     --bipartite_matching=data/bipartite_name_matches.txt --dest=models/orig_names_cds_2500_less.kv \
#     --less_target=True")


# TRAIN BASELINE
# os.system("python3 train/train_baseline.py")