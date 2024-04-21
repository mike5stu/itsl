# itsl
Instruction Tuning model with Self-critical Learning
![image](https://github.com/mike5stu/itsl/assets/166886594/22dc73f3-e751-4b92-99cd-8598666af304)
Given input texts, video, and audio modalities, we first encode samples by a frozen multimodal encoder to extract semantic information. Those embeddings are then transformed into a pretrained language model LLaMA space via an adapter module. 
Subsequently, we feed the aligned embeddings with a suite of learnable prompt vectors into LLaMA to predict the intent class and rationale simultaneously via the self-critical learning module.

# Abstract
Multimodal Intent Recognition (MIR) is a critical task involving the discernment of underlying intentions or purposes within textual, visual, or auditory inputs. While prevailing MIR methods can yield desirable intent classification results, they often lack a  rationale for interpreting the decision-making process of intent classification. To fill this gap, we introduce a novel task named Explainable Multimodal Intent Recognition (EMIR),  not only furnishing intent recognition results but also generating comprehensible free-text rationales. In this work, we mainly solve two primary challenges in EMIR: (1) the lack of faithful human-annotated rationales and (2) the integration of intent classification with the generation of explanations into a unified end-to-end model. Specifically, to address the former, we exploit instruction tuning for the training of explanation generation by pre-trained language model, e.g., GPT-$4$. Regarding the latter, we design a self-critical learning strategy that facilitates the concurrent training of intent classification and explanation generation through a multi-task paradigm. 

# Comparison
![image](https://github.com/mike5stu/itsl/assets/166886594/6eb273d5-973a-498c-aefd-a9bb1287a951)

# Use of the Code
## Environment
To begin, create a virtual environment for the project. This code floder "Imagebind_LLM" is placed in LLaMA-Adapter project to ultilize their functions and pretrained parameters. The Mintrec data path is hardcoded on top of code files.

### Setup
conda create -n llama_adapter -y python=3.8
conda activate llama_adapter
conda install pytorch cudatoolkit -c pytorch -y
pip install -r requirements.txt
pip install -e .

### Training
torchrun --nproc_per_node 8 engine_finetune.py \
         --model Llama7B_adapter \
         --llama_model_path $TARGET_FOLDER/ \
         --data_path $DATA_PATH/alpaca_data.json \
         --adapter_layer 30 \
         --adapter_len 10 \
         --max_seq_len 512 \
         --batch_size 4 \
         --epochs 5 \
         --warmup_epochs 2 \
         --blr 9e-3 \
         --weight_decay 0.02 \
         --output_dir ./checkpoint/

### Inference
1. Run avt_get_emb.py to get embeddings (adjust input in code to ablate modalities).
2. Run classifier.py to get results.
