Title: I Joined a Kaggle Playground Series and Failed Miserably  
Date: 2025-12-29 11:00  
Category: Machine Learning  
Tags: machine learning, kaggle  
Slug: diabetes-challenge  
Authors: Tung Thanh Tran  
Summary: My first Kaggle Playground Series experience — and how I failed spectacularly.


This was the **first time I joined a Kaggle Playground Series challenge**, even though I’ve been working as an AI Engineer for a while.

The challenge was a **Diabetes Prediction** task with tabular data.  
And let me tell you upfront: **I failed miserably.**

## The Reality Check

It had been a *long* time since I last worked seriously with tabular datasets.  
So I had to relearn - or at least re-get-used-to - things like:

- Exploratory Data Analysis (EDA)
- Feature engineering
- Data preprocessing pipelines
- Evaluation metrics and threshold tuning

…all over again.

Model selection was another headache.  
For tabular data, the usual suspects are:

- Logistic Regression
- Decision Trees
- Gradient Boosting (XGBoost / LightGBM - *usually the best choice*)

But instead of being sensible, I went with what I was most familiar with: **A neural network**

Yes.  
This also explains why my rank was so low 😅

## The Model (a.k.a. My Mistake)

I used a residual MLP architecture in PyTorch.

### Residual Block

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.res_path = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.res_path(x)
        return x + out
```

### Residual MLP

```python
class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_blocks=2):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(n_blocks)]
        )

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)
```

Was it over-engineered for tabular data?
Absolutely.

Was it fun to build?
Also yes.

## Data Processing (The Part That Actually Helped)

I did some basic but necessary preprocessing:
- Grouped categorical columns
- Used `OrdinalEncoder` for ordered features
- One-hot encoded nominal features
- Standardized numerical columns

```python
ordinal_cols = ["education_level", "income_level"]
ordinal_categories = [
    ["No formal", "Highschool", "Graduate", "Postgraduate"], 
    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]
]

nominal_cols = ["gender", "ethnicity", "smoking_status", "employment_status"]

binary_cols = [
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history"
]

numerical_cols = [
    'age', 'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day',
    'screen_time_hours_per_day', 'bmi',
    'waist_to_hip_ratio', 'systolic_bp',
    'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol',
    'ldl_cholesterol', 'triglycerides'
]

preprocessor = ColumnTransformer(
    transformers=[
        ("ordinal", OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), nominal_cols),
        ("binary", "passthrough", binary_cols),
        ("numeric", StandardScaler(), numerical_cols)
    ]
)
```

## Threshold Tuning (Small Win)

I also tuned the classification threshold instead of blindly using 0.5:

```
thresholds = np.arange(0.1, 0.9, 0.1)
f1_scores = []

for t in thresholds:
    preds = (all_probs >= t).astype(int)
    f1_scores.append(f1_score(all_targets, preds))

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)
```

This gave me a tiny boost — but not enough to save the model choice.


## The Result (Brace Yourself)

Best F1 score: ~0.6825

Leaderboard rank: 2990 / ~3800

Yes.

That’s barely top 80%.

(Lol.)


## What I Learned

1. **Tabular data ≠ Neural Networks (most of the time)**

    Gradient boosting is king for a reason.

2. **Strong baselines matter**

    A simple logistic regression would probably have beaten my model.

3. **EDA and feature engineering matter more than architecture**

    Especially in Kaggle Playground challenges.

4. **Failing publicly is actually useful**

    I learned more from this than from an easy win.

## Conclusion

Despite the terrible ranking, this was a fun and valuable learning experience.

Next time:

- I’ll start with a strong baseline
- I’ll try LightGBM / XGBoost first
- And I definitely won’t jump straight into neural networks for tabular data

…probably.

If you’re also joining Kaggle for the first time:
fail fast, learn faster, and enjoy the process.