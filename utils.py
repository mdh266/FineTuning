import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import arxiv
from typing import Dict
import numpy as np
from torch.utils.data import DataLoader
import evaluate


def get_arxiv_data()
    client = arxiv.Client()

    # artificial intelligence abstraccs
    ai_results = [{
                    "id": res.entry_id,
                    "code": res.primary_category,
                    "text": res.summary
                } for res in client.results(
                    arxiv.Search(
                        query = "cat:cs.AI",
                        max_results = 1000
                        )
                    )
    ]

    # information retervial abstracts
    ir_results = [{
                    "id": res.entry_id,
                    "code": res.primary_category,
                    "text": res.summary
                } for res in client.results(
                    arxiv.Search(
                        query = "cat:cs.IR",
                        max_results = 1000
                        )
                    )
    ]

    # robotics abstracts
    ro_results = [{
                    "id": res.entry_id,
                    "code": res.primary_category,
                    "text": res.summary
                } for res in client.results(
                    arxiv.Search(
                        query = "cat:cs.RO",
                        max_results = 100
                        )
                    )
    ]

   return pd.DataFrame(ai_results + ir_results + ro_results)


def plot_target_distribution_combined(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col:str = 'label'
) -> None:
    """
    Create a bar plot showing percentage distribution of target values across datasets
    """
    # Calculate percentages for each dataset
    datasets = {'Train': train_df, 'Validation': val_df, 'Test': test_df}
    percentage_data = []
    
    for dataset_name, df in datasets.items():
        # Calculate percentage for each target value
        value_counts = df[target_col].value_counts()
        percentages = (value_counts / len(df)) * 100
        
        # Create rows for the dataframe
        for target_value, percentage in percentages.items():
            percentage_data.append({
                'Dataset': dataset_name,
                'Label': str(target_value),
                'Percentage': percentage
            })
    
    # Convert to dataframe
    percentage_df = pd.DataFrame(percentage_data)
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    sns.barplot(data=percentage_df, x='Dataset', y='Percentage', hue='Label')
    plt.title('Class Label Value Percentage Distribution Across Datasets')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Dataset')
    plt.legend(title='Class Label')
    
    # Add percentage labels on bars
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='edge')
    
    plt.tight_layout()
    plt.show()


def calculate_roc_auc(model, loader: DataLoader) -> Dict[str, np.float64]:

  roc_auc_score = evaluate.load("roc_auc", "multiclass")
  model.eval()
  for batch in loader:
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
          outputs = model(**batch)
          scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
          roc_auc_score.add_batch(references=batch["labels"],
                                prediction_scores=scores)

  return roc_auc_score.compute(multi_class="ovr")
