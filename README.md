# itsl
Instruction Tuning model with Self-critical Learning
![image](https://github.com/mike5stu/itsl/assets/166886594/22dc73f3-e751-4b92-99cd-8598666af304)
Given input texts, video, and audio modalities, we first encode samples by a frozen multimodal encoder to extract semantic information. Those embeddings are then transformed into a pretrained language model LLaMA space via an adapter module. 
Subsequently, we feed the aligned embeddings with a suite of learnable prompt vectors into LLaMA to predict the intent class and rationale simultaneously via the self-critical learning module.
