# Transformer-Based Python Code Generator

This project explores the potential of transformer-based models for automated Python code generation. By systematically comparing multiple model architectures, this research aims to advance our understanding of machine learning's role in software development.

## **Objective**
To develop and evaluate transformer models capable of generating syntactically correct and semantically meaningful Python code, with a focus on:
1. Developing models using various transformer architectures.
2. Assessing their performance in generating accurate Python code.
3. Identifying challenges in machine-driven code generation.
4. Comparing performance across different configurations.

## **Datasets**
1. **CodeParrot Collection**: ~50GB of Python code from GitHub repositories, showcasing diverse programming styles.
2. **Alpaca 18k Dataset**: 18,000 instruction-code pairs for fine-tuning model precision.

## **Methodology**
- Developed **four models**: Baseline GPT-2, GPT-2 Small, Salesforce CodeGen Mono, and Llama-2 7B.
- Key features include:
  - Baseline: Simplified GPT-2 with six heads, 384 embeddings, and six layers.
  - Llama-2: Fine-tuned for three epochs on two NVIDIA A100 GPUs.
  - Advanced data streaming and distributed parallelism for efficient training.
- Rigorous evaluation using metrics like loss, perplexity, and syntactic accuracy.

## **Results**
- **Baseline**: Limited code generation capabilities with issues in semantic coherence and token repetition.
- **GPT-2 Small**: Improved performance but struggled with consistent quality.
- **CodeGen Mono**: Superior syntactic accuracy, though with unique whitespace behaviors.
- **Llama-2**: Outperformed all models, leveraging its pre-trained depth for coherent and contextually appropriate Python code.

## **Challenges & Limitations**
- Token repetition and semantic understanding issues.
- Computational resource constraints.
- Dataset biases and limited contextual code generation.

## **Conclusion**
This research provides insights into the strengths and limitations of transformer-based models for Python code generation. While challenges remain, the study highlights promising directions for improving automated programming tools through advanced architectures and pre-training strategies.


### About resources:
0. Indiana University has provided GPU A100 for training the model.
1. Llama open weights and model https://www.llama.com/llama2/
2. Code-parrot-clean dataset https://huggingface.co/datasets/codeparrot/codeparrot-clean
3. Alpaca 18k dataset https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca
4. Reference for training code https://huggingface.co/blog/codeparrot
5. Training code for baseline model and llama model https://github.com/Jags96/CodeGen
6. CodeGen mono and GPT-2 model and pre-trained weights are obtained from hugging face library
‘transformers’
