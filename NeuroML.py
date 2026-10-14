# type: ignore
import streamlit as st
import pandas as pd
import numpy as np


from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import matplotlib.pyplot as plt
import importlib
import utils


st.set_page_config(
    page_title="No-Code ML for Neuroscience",
    page_icon="🧪",
    layout="wide"
)

### Streamlit session state initialization ###

if 'model' not in st.session_state:
    st.session_state.model = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'model_dict' not in st.session_state:
    try:
        st.session_state.model_dict = utils.read_models_from_json('model_dict.json')
    except Exception as e:
        st.error(str(e))

### Model-specific Functions ###
def permutation_test(model, X, y, original_score, n_permutations=1000, n_jobs=1):
    '''
    Perform a permutation test to assess the significance of the model's performance.
    
    Parameters:
    - model: The trained model (Pipeline).
    - X: Feature data (DataFrame).
    - y: Target variable (Series or array).
    - original_score: The original model score.
    - n_permutations: Number of permutations to perform.
    - n_jobs: Number of parallel jobs for cross-validation.
    
    Returns:
    - p_value: The p-value from the permutation test.
    '''
    perm_progress = st.progress(0)
    perm_scores = []
    perm_test_model = clone(model)  # Clone the model to avoid altering the original
    permutation_ids = utils.generate_permutation_ids(len(y), n_permutations) #initialize permutation ids
    
    for i, perm_indices in enumerate(permutation_ids, start=1):
        y_permuted = y[perm_indices]
        perm_score = cross_validate(
            perm_test_model,
             X, y_permuted,
             cv=5,
             return_train_score=False,
             n_jobs=n_jobs
             )["test_score"].mean()
        perm_scores.append(perm_score)
        perm_progress.progress(i / n_permutations, text=f"Permutation {i}/{n_permutations} completed.")

    perm_progress.empty()  # Clear the progress bar after completion
    p_value = (np.count_nonzero(np.array(perm_scores) >= original_score) + 1) / (n_permutations + 1)  # Add 1 to numerator and denominator for continuity correction
    
    return p_value


# Title and description for the Streamlit app
st.title("🧪 No-Code ML for Neuroscience")
st.markdown("""
A user-friendly program for conducting multivariate analysis (i.e. MVPA) on your tabulated data.
Simply upload your data in the sidebar, select your model and analysis parameters, and let the program do the rest.
The Results tab will provide relevant visualizations and scores for your analysis.
""")

# Side panel for loading data
with st.sidebar:
    st.header("Load training data")
    
    uploaded_file = st.file_uploader(
        "Must select file", 
        type=['csv'],
        help="Upload your data in csv format, with the first row containing feature names"
    )

    # Reading the data from the uploaded CSV file
    if uploaded_file is not None:
        try:
            # Read the CSV file
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            input_df = None
        else:         
            # Present data and basic info
            st.subheader("Data view")
            st.dataframe(input_df.head(10))
            st.metric("Number of subjects", len(input_df))

        # Identifying target variable and other relevant columns
        if input_df is not None:
            form = st.form("feature_selection_form")
            form.subheader("Identify features and target variable")
            
            target_variable = form.selectbox(
                "Select target variable (label column)",
                options=input_df.columns.tolist(),
                index=0,
                help="Choose which column contains the target values to predict"
            )

            subject_id_variable = form.selectbox(
                "Select Subject ID variable (optional)",
                options=input_df.columns.tolist(),
                index=None,
                help="Choose which column contains unique subject identifiers (optional)"
            )

            # covariate feature selection placeholder
            
            #confirm the data identification
            data_identified = form.form_submit_button("Confirm")

            if data_identified:
                # Separate features and target variable
                feature_cols = [col for col in input_df.columns if col not in [target_variable, subject_id_variable]]
                X = input_df[feature_cols]
                y = input_df[[target_variable]]

                # Identify and one-hot encode categorical columns in X
                X_cat = X.select_dtypes(include=['object']).columns.tolist()
                if X_cat:
                    X_encode = X.copy()
                    encoded_features = pd.get_dummies(X_encode[X_cat], prefix='Is', drop_first=False)
                    # Replace original string columns
                    X_encode = X_encode.drop(columns=X_cat)
                    X = pd.concat([X_encode, encoded_features], axis=1)

                # Find binary columns in X and convert them to 0/1
                X_binary = []
                for col in X.columns:
                    col_unique = X[col].unique()
                    if len(col_unique) == 2:
                        X_binary.append(col)
                        if set(col_unique) != {0, 1}:
                            # Map the two unique values to 0 and 1
                            mapping = {col_unique[0]: 0, col_unique[1]: 1}
                            X[col] = X[col].map(mapping)


                # Identify if the target variable contains non-numeric values
                y_cat = y.select_dtypes(include=['object']).columns.tolist()
                if y_cat:
                    st.write("Note: the selected target variable contains non-numeric values and has been one-hot encoded")
                    y_encode = pd.get_dummies(y, prefix='Is', drop_first=True)
                    y = y_encode
                    if len(y.columns) > 1:
                        st.write("Multiple categories detected in the target variable, multiclass classification will be performed.")
                
                # save train data and features in session state for later use in prediction tab
                st.session_state.train_data = X.copy()
                st.session_state.train_features = X.columns.tolist()
                st.session_state.cat_features = X_binary
                st.session_state.target_variable = y.copy()
                st.session_state.subject_id_feature = subject_id_variable
                # st.session_state.covariate_features = covariate_variables             

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Load and train", "Statistical Analysis", "📈 Results"])

# Tab 1: Preprocessing and model training
with tab1:

    st.header("1. Model Settings")
    
    st.subheader("Model Selection")
    model_name = st.selectbox(
        "Model type",
        list(st.session_state.model_dict.keys()),
        help="Choose regression for continuous values or classification for categorical data"
    )
    model_callable = st.session_state.model_dict[model_name]['model']
    module_name, _, class_name = model_callable.rpartition('.')
    try:
        model_module = importlib.import_module(module_name)
        model_class = getattr(model_module, class_name)
    except ImportError as e:
        st.error(f"Error importing model: {e}")

    model_params = {}
    if model_name is not None:
        for param_name, param_options in st.session_state.model_dict[model_name]['params'].items():
            if not isinstance(param_options, list):
                param_value = param_options
            elif isinstance(param_options[0], (int, float)):
                param_value = st.number_input(
                    label=param_name,
                    min_value=param_options[0],
                    max_value=param_options[1],
                    value=param_options[2])
            else:
                param_value = st.segmented_control(
                    label=param_name,
                    options=param_options,
                    default = param_options[0]
                )
            model_params[param_name] = param_value

    model_type = st.session_state.model_dict[model_name]['type']

    if st.session_state.train_data is None:
        st.info("Please upload training data to visualize feature space.")
    else:
        st.subheader("2. Model Training")
        n_jobs = 1
        parallelize = st.checkbox("Parallelization Acceleration", value=False, help="Enable parallelization  across multiple coresto speed up model training if your system supports it.")
        if parallelize:
            n_jobs = -1  # Use all available cores for parallel processing
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                X = st.session_state.train_data
                y = st.session_state.target_variable.values.ravel()
                scale_features = [
                    column for column in X.columns
                    if column not in st.session_state.cat_features
                ]
                scaler = ColumnTransformer(
                    [("standard_scaler", StandardScaler(), scale_features)],
                    remainder="passthrough"
                )
                model = Pipeline([
                    ("scaler", scaler),
                    ("model", model_class(**model_params, random_state=42))
                ])
                cv_scores = cross_validate(
                    model,
                    X,
                    y,
                    cv=5,
                    return_train_score=True,
                    n_jobs=n_jobs
                )
                model.fit(X, y)
            
            # Save the model to session state
            st.session_state.model = model
            st.session_state.train_scores = cv_scores["train_score"]
            st.session_state.test_scores = cv_scores["test_score"]

            st.success("✅ Model trained successfully!")
    
# Tab 2: Statistical Analysis
with tab2:
    st.subheader("Permutation Test for Model Significance")
    perm_test_done = False
    if st.session_state.model is not None:
        n_permutations = st.number_input(
            "Number of Permutations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Specify the number of permutations to perform for the permutation test"
        )
        perm_jobs = 1
        parallelize_perm =st.checkbox("Parallelize Permutation Test", value=False, help="Enable parallelization across multiple cores to speed up the permutation test if your system supports it.")
        if parallelize_perm:
            perm_jobs = -1  # Use all available cores for parallel processing

        # Permutation Test
        if st.button("Run Permutation Test"):
            with st.spinner("Running permutation test..."):
                X = st.session_state.train_data
                y = st.session_state.target_variable.values.ravel()
                model = st.session_state.model
                n_samples = len(y)
                p_value = permutation_test(
                    model,
                    X, y,
                    st.session_state.test_scores.mean(),
                    n_permutations=n_permutations,
                    n_jobs=perm_jobs)
                

            st.success("✅ Permutation test completed!")
            perm_test_done = True
            
        if perm_test_done:
            model_type = st.session_state.model_dict[model_name]['type']
            if model_type == 'regressor':
                st.write(f"Original Model R2 Score: {st.session_state.test_scores.mean():.4f}")
                st.write(f"P-value from Permutation Test: {p_value:.4f}")        
            elif model_type == 'classifier':
                st.write(f"Original Model Accuracy: {st.session_state.test_scores.mean():.4f}")
                st.write(f"P-value from Permutation Test: {p_value:.4f}")
    else:
        st.info("The model has not been trained yet. Please complete model training to view statistical analysis.")


# Tab 3: Results and Information
with tab3:      
    st.subheader("Data Statistics")
    if st.session_state.model is not None:
        model_type = st.session_state.model_dict[model_name]['type']

        # Display descriptive statistics of the target variable
        data_cols = st.columns([1,2])
        desc_stats = st.session_state.target_variable.describe()
        
        data_cols[0].dataframe(desc_stats)

        # Histogram of target variable (regression)
        if model_type == 'regressor':
            fig_target, ax = plt.subplots(figsize=(10, 4))
            ax.hist(st.session_state.target_variable.values, bins=30, alpha=0.7,
                color='purple', edgecolor='black')
            ax.set_xlabel('Label Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of the Target Variable', pad=20)
            ax.grid(True, alpha=0.3)
            data_cols[1].pyplot(fig_target)

        if model_type == 'classifier':
            fig_target, ax = plt.subplots(figsize=(10, 4))
            target_np = st.session_state.target_variable.values
            target_counts = np.unique_counts(target_np)
            ax.pie(target_counts.counts, labels=target_counts.values)
            ax.set_title('Distribution of the Target Variable', pad=20)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            data_cols[1].pyplot(fig_target)

        results_cols = st.columns(2)
        
        train_scores = st.session_state.train_scores
        test_scores = st.session_state.test_scores
        train_score = np.mean(train_scores)
        test_score = np.mean(test_scores)
        results_cols[0].subheader("Model Performance")
        if model_type == 'regressor':
            results_cols[0].metric(
                "Mean CV Training R² Score",
                f"{train_score:.4f} ± {np.std(train_scores):.4f}"
            )
            results_cols[0].metric(
                "Mean CV Test R² Score",
                f"{test_score:.4f} ± {np.std(test_scores):.4f}"
            )
        elif model_type == 'classifier':
            results_cols[0].metric(
                "Mean CV Training Accuracy",
                f"{train_score:.4f} ± {np.std(train_scores):.4f}"
            )
            results_cols[0].metric(
                "Mean CV Test Accuracy",
                f"{test_score:.4f} ± {np.std(test_scores):.4f}"
            )
        results_cols[0].caption(
            f"5-fold training scores: {np.array2string(train_scores, precision=4)}"
        )
        results_cols[0].caption(
            f"5-fold test scores: {np.array2string(test_scores, precision=4)}"
        )


    else:
        st.info("The model has not been trained yet. Please complete model training to view results.")
