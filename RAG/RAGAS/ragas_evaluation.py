"""
RAGAS Framework Implementation for RAG Evaluation
This script demonstrates how to evaluate a RAG system using RAGAS metrics.
"""

# Install required packages first:
# pip install ragas langchain openai pandas datasets

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
    answer_correctness,
    answer_similarity
)
import pandas as pd

# Set your OpenAI API key (RAGAS uses LLMs for evaluation)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Alternative: Use other LLM providers
# from langchain_community.chat_models import ChatOpenAI
# from ragas.llms import LangchainLLMWrapper


def create_sample_data():
    """
    Create sample RAG evaluation data.
    In production, this would come from your RAG system outputs.
    """
    data = {
        "question": [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is photosynthesis?"
        ],
        "answer": [
            "The capital of France is Paris. It is located in northern France.",
            "William Shakespeare wrote Romeo and Juliet in the 1590s.",
            "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll."
        ],
        "contexts": [
            [
                "Paris is the capital and most populous city of France.",
                "Paris is located in northern central France on the River Seine."
            ],
            [
                "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
                "The play was written between 1591 and 1595."
            ],
            [
                "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
                "Chlorophyll is the green pigment involved in photosynthesis."
            ]
        ],
        "ground_truth": [
            "Paris is the capital of France.",
            "William Shakespeare wrote Romeo and Juliet.",
            "Photosynthesis is the process plants use to convert light energy into chemical energy."
        ]
    }
    
    return Dataset.from_dict(data)


def evaluate_rag_system(dataset, metrics_to_use=None):
    """
    Evaluate RAG system using RAGAS metrics.
    
    Args:
        dataset: Dataset containing question, answer, contexts, and ground_truth
        metrics_to_use: List of metrics to evaluate. If None, uses all metrics.
    
    Returns:
        Evaluation results
    """
    
    # Default metrics if none specified
    if metrics_to_use is None:
        metrics_to_use = [
            faithfulness,           # Answer faithful to context
            answer_relevancy,       # Answer relevant to question
            context_precision,      # Relevant contexts ranked higher
            context_recall,         # Context contains ground truth info
            context_relevancy,      # Proportion of relevant context
            answer_correctness,     # Combined accuracy metric
            answer_similarity       # Semantic similarity to ground truth
        ]
    
    print("Starting RAGAS evaluation...")
    print(f"Evaluating {len(dataset)} samples with {len(metrics_to_use)} metrics\n")
    
    # Run evaluation
    results = evaluate(
        dataset,
        metrics=metrics_to_use,
    )
    
    return results


def display_results(results):
    """
    Display evaluation results in a readable format.
    """
    print("\n" + "="*60)
    print("RAGAS EVALUATION RESULTS")
    print("="*60)
    
    # Overall scores
    print("\n📊 Overall Metric Scores:")
    print("-" * 60)
    for metric, score in results.items():
        if metric != 'per_question_scores':
            print(f"{metric:.<30} {score:.4f}")
    
    # Per-question breakdown if available
    if hasattr(results, 'to_pandas'):
        df = results.to_pandas()
        print("\n📝 Per-Question Scores:")
        print("-" * 60)
        print(df.to_string(index=False))
        
        # Save to CSV
        df.to_csv("ragas_evaluation_results.csv", index=False)
        print("\n💾 Detailed results saved to 'ragas_evaluation_results.csv'")


def evaluate_without_ground_truth(dataset):
    """
    Evaluate RAG system when ground truth is not available.
    Uses reference-free metrics only.
    """
    print("\n" + "="*60)
    print("EVALUATION WITHOUT GROUND TRUTH")
    print("="*60)
    
    # Metrics that don't require ground truth
    reference_free_metrics = [
        faithfulness,        # Does answer match context?
        answer_relevancy,    # Is answer relevant to question?
        context_relevancy,   # Is context relevant?
    ]
    
    # Remove ground_truth from dataset
    dataset_no_gt = dataset.remove_columns(['ground_truth'])
    
    results = evaluate(
        dataset_no_gt,
        metrics=reference_free_metrics,
    )
    
    return results


def main():
    """
    Main execution function demonstrating RAGAS evaluation.
    """
    
    print("🚀 RAGAS RAG Evaluation Demo\n")
    
    # Create sample dataset
    print("1️⃣ Creating sample dataset...")
    dataset = create_sample_data()
    print(f"   Created dataset with {len(dataset)} samples\n")
    
    # Full evaluation with ground truth
    print("2️⃣ Running full evaluation (with ground truth)...")
    results_full = evaluate_rag_system(dataset)
    display_results(results_full)
    
    # Evaluation without ground truth
    print("\n3️⃣ Running evaluation without ground truth...")
    results_no_gt = evaluate_without_ground_truth(dataset)
    display_results(results_no_gt)
    
    print("\n✅ Evaluation complete!")


# Example: Custom RAG system integration
def integrate_with_your_rag_system():
    """
    Example of how to integrate RAGAS with your actual RAG system.
    """
    
    # Your RAG system outputs
    rag_outputs = []
    
    # For each question in your test set:
    for question in ["your", "test", "questions"]:
        # 1. Get answer from your RAG system
        contexts = your_retrieval_function(question)  # Your retrieval logic
        answer = your_generation_function(question, contexts)  # Your generation logic
        
        # 2. Collect the data
        rag_outputs.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": get_ground_truth(question)  # If available
        })
    
    # 3. Create dataset and evaluate
    dataset = Dataset.from_list(rag_outputs)
    results = evaluate_rag_system(dataset)
    
    return results


if __name__ == "__main__":
    # Note: Make sure to set your OPENAI_API_KEY before running
    if os.environ.get("OPENAI_API_KEY") == "your-api-key-here":
        print("⚠️  WARNING: Please set your OPENAI_API_KEY environment variable")
        print("   You can set it in the script or as an environment variable:")
        print("   export OPENAI_API_KEY='your-actual-key'\n")
    
    main()
