# type: ignore
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error, root_mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import cross_validate, GridSearchCV
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
            input_form = st.form("feature_selection_form")
            input_form.subheader("Identify features and target variable")
            
            target_variable = input_form.selectbox(
                "Select target variable (label column)",
                options=input_df.columns.tolist(),
                index=0,
                help="Choose which column contains the target values to predict"
            )

            covariate_variables = input_form.multiselect(
                "Select covariate features (optional)",
                options=input_df.columns.tolist(),
                help="Choose which features to include as covariates in the analysis. Certain statistical methods can account for covariates separately"
            )
            
            excluded_variables = input_form.multiselect(
                "Select variables to exclude (optional)",
                options=input_df.columns.tolist(),
                help="Choose which features to exclude from the analysis, i.e. subject ID columns or other non-predictive features"
            )
            
            #confirm the data identification
            data_identified = input_form.form_submit_button("Confirm")

            if data_identified:
                # Separate features and target variable
                feature_cols = [col for col in input_df.columns if col not in [target_variable] + excluded_variables]
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
                st.session_state.excluded_features = excluded_variables
                st.session_state.covariate_features = covariate_variables             

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Load and train", "Statistical Analysis", "📈 Results"])

# Tab 1: Preprocessing and model training
with tab1:

    st.header("1. Model Settings")
    model_params = {}
    gridsearch_params = {}
    model_container = st.container()
    with model_container:
        st.subheader("Model Selection")
        model_name = st.selectbox(
        "Model type",
        list(st.session_state.model_dict.keys()),
        help="Choose regression models for continuous values or classification for categorical data"
        )
        model_callable = st.session_state.model_dict[model_name]['model']
        module_name, _, class_name = model_callable.rpartition('.')
        try:
            model_module = importlib.import_module(module_name)
            model_class = getattr(model_module, class_name)
        except ImportError as e:
            st.error(f"Error importing model: {e}")

    with model_container:
        if model_name is not None:
            gridsearch_column, parameter_column = st.columns([1, 4])
            with gridsearch_column:
                st.subheader("Grid Search")
            with parameter_column:
                st.subheader("Parameters")

            for param_name, param_options in st.session_state.model_dict[model_name]['params'].items():
                model_param_name = "model__" + param_name
                param_cols = st.columns([1, 4])
                with param_cols[0]:
                    if isinstance(param_options, list):
                        use_gridsearch = st.checkbox(
                            param_name,
                            value=False,
                            key=f"gridsearch_{model_name}_{param_name}",
                            label_visibility="hidden"
                        )
                with param_cols[1]:
                    if not isinstance(param_options, list):
                        param_value = param_options
                    elif isinstance(param_options[0], (int, float)):
                        param_value = st.number_input(
                            label=param_name,
                            min_value=param_options[0],
                            max_value=param_options[1],
                            value=param_options[2],
                            disabled=use_gridsearch
                        )
                        if use_gridsearch:
                            param_grid = np.linspace(param_options[0], param_options[1], 5, dtype=type(param_options[0]))
                    else:
                        param_value = st.segmented_control(
                            label=param_name,
                            options=param_options,
                            default=param_options[0],
                            disabled=use_gridsearch
                        )
                        if use_gridsearch:
                            param_grid = param_options

                if use_gridsearch:
                    gridsearch_params[model_param_name] = param_grid
                else:
                    model_params[model_param_name] = param_value

            gridsearch = bool(gridsearch_params)

    model_type = st.session_state.model_dict[model_name]['type']

    if st.session_state.train_data is None:
        st.info("Please upload training data to visualize feature space.")
    else:
        st.subheader("2. Model Training")
        train_container = st.container()
        with train_container:
            n_jobs = 1
            parallelize = st.checkbox("Parallelization Acceleration", value=False, help="Enable parallelization  across multiple coresto speed up model training if your system supports it.")
            if parallelize:
                n_jobs = -1  # Use all available cores for parallel processing
            cv = st.number_input(
                "Number of Cross-Validation Folds",
                min_value=2,
                max_value=20,
                value=5,
                step=1,
                help="Specify the number of cross-validation k-folds for model training"
            )
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    X = st.session_state.train_data
                    y = st.session_state.target_variable.values.ravel()
                    model, cv_scores = utils.train_model(
                        model_class,
                        model_params,
                        X=X,
                        y=y,
                        cat_features=st.session_state.cat_features,
                        gridsearch=gridsearch,
                        gridsearch_params=gridsearch_params,
                        cv=cv,
                        n_jobs=n_jobs
                    )

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
        parallelize_perm =st.checkbox(
            "Parallelize Permutation Test",
            value=False,
            help="Enable parallelization across multiple cores to speed up the permutation test if your system supports it."
            )
        if parallelize_perm:
            perm_jobs = -1  # Use all available cores for parallel processing

        covariate_exclusion = st.checkbox(
            "Covariate Exclusion",
            value=False,
            help="Traditional Permutation Test permutes labels across all input features. Enabling this option will exclude specified covariate features from permutation,\nlinking them to the original labels. This limits the statistical analysis to non-covariate features only."
            )

        # Permutation Test
        if st.button("Run Permutation Test"):
            progress_bar = st.progress(
                0,
                text=f"Permutations tested: 0/{n_permutations}"
            )

            def update_permutation_progress(iteration):
                progress_bar.progress(
                    iteration / n_permutations,
                    text=f"Permutations tested: {iteration}/{n_permutations}"
                )

            with st.spinner("Running permutation test..."):
                X = st.session_state.train_data
                y = st.session_state.target_variable.values.ravel()
                model = st.session_state.model
                n_samples = len(y)
                perm_scores = utils.permutation_test(
                    model,
                    X, y,
                    covariate_exclusion=covariate_exclusion,
                    covariate_variables=st.session_state.covariate_features,
                    n_permutations=n_permutations,
                    n_jobs=perm_jobs,
                    progress_callback=update_permutation_progress
                )
                

            st.success("✅ Permutation test completed!")
            st.session_state.perm_scores = perm_scores
            
    else:
        st.info("The model has not been trained yet. Please complete model training to view statistical analysis.")


# Tab 3: Results and Information
with tab3:
    if st.session_state.model is not None:
        model_type = st.session_state.model_dict[model_name]['type']
        st.subheader("Data Statistics")
        # Display descriptive statistics of the target variable
        if model_type == 'classifier':
            target_values = st.session_state.target_variable.to_numpy()
            target_labels = np.unique(target_values, return_counts=True)
            target_percents = target_labels[1] / len(st.session_state.target_variable)
            desc_stats = pd.DataFrame({'Label': target_labels[0], 'Count': target_labels[1], 'Percent': target_percents})
        else:
            desc_stats = st.session_state.target_variable.describe()

        data_cols = st.columns([1, 2])
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
        # Pie chart of target variable (classification)
        if model_type == 'classifier':
            fig_target, ax = plt.subplots(figsize=(10, 4))
            target_np = st.session_state.target_variable.values
            target_counts = np.unique_counts(target_np)
            ax.pie(target_counts.counts, labels=target_counts.values)
            ax.set_title('Distribution of the Target Variable', pad=20)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            data_cols[1].pyplot(fig_target)

        # Model training results
        train_scores = st.session_state.train_scores
        test_scores = st.session_state.test_scores
        train_score = np.mean(train_scores)
        test_score = np.mean(test_scores)
        results_cols = st.columns(2)
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

        if 'perm_scores' in st.session_state:
            results_cols[1].subheader("Statistical Test Results")
            stat_test = "Permutation"
            results_cols[1].write(f"Statistical Test: {stat_test} Test")
            if model_type == 'regressor':
                score_metric = "R²"
            elif model_type == 'classifier':
                score_metric = "Accuracy"
            results_cols[1].metric(
                f"Permutation {score_metric} Score",
                f"{np.mean(st.session_state.perm_scores):.4f} ± {np.std(st.session_state.perm_scores):.4f}"
            )
            p_value = (np.sum(st.session_state.perm_scores >= test_score) + 1) / (len(st.session_state.perm_scores) + 1) 
            results_cols[1].metric(
                "Permutation Test p-value",
                f"{p_value:.4f}"
            )

    else:
        st.info("The model has not been trained yet. Please complete model training to view results.")
