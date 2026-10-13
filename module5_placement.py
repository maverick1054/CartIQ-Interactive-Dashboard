import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def render_module5(cust_df, order_df):
    st.title("🎯 Module 5: Placement Prediction & PCA Analytics")
    st.markdown("---")

    if cust_df.empty or len(cust_df) < 5 or order_df.empty or len(order_df) < 5:
        st.warning("⚠️ Insufficient data for placement prediction & PCA modeling under current filter selection. Please broaden sidebar filters.")
        return

    sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs([
        "📉 PCA Method & Visual Projections",
        "🎯 All Classification Models (Full vs PCA)",
        "📊 Model Benchmark Leaderboard",
        "🎮 Interactive Placement Simulator"
    ])

    # -------------------------------------------------------------
    # PREPARE DATASET FOR PLACEMENT PREDICTION
    # Target: Customer High-Value Placement (is_high_value)
    # Features: age, total_orders, total_items, avg_delivery_days, avg_order_value + gender & state
    # -------------------------------------------------------------
    feature_cols_num = ['age', 'total_orders', 'total_items', 'avg_delivery_days', 'avg_order_value']
    feature_cols_cat = ['gender', 'state']

    X_raw = cust_df[feature_cols_num + feature_cols_cat]
    y_placement = cust_df['is_high_value']

    X_encoded = pd.get_dummies(X_raw, columns=feature_cols_cat, drop_first=True, dtype=int)

    # Filter out columns with zero variance
    X_encoded = X_encoded.loc[:, X_encoded.std() > 0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # -------------------------------------------------------------
    # SUB TAB 1: PCA METHOD & VISUAL PROJECTIONS
    # -------------------------------------------------------------
    with sub_t1:
        st.subheader("1️⃣ Principal Component Analysis (PCA) Dimensionality Reduction")
        st.markdown("""
        **PCA (Principal Component Analysis)** is an unsupervised feature extraction technique that transforms high-dimensional correlated features 
        into a set of linearly uncorrelated principal components ($PC_1, PC_2, \\dots$), maximizing variance retention.
        """)

        col_pca_cfg1, col_pca_cfg2 = st.columns(2)
        with col_pca_cfg1:
            max_comp_avail = min(X_encoded.shape[1], X_encoded.shape[0])
            n_components = st.slider("Select Number of PCA Components", 2, min(10, max_comp_avail), 3, key="pca_n_comp")
        with col_pca_cfg2:
            st.info(f"Transforming **{X_encoded.shape[1]} original features** into **{n_components} Principal Components**.")

        # Fit PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pca_cols = [f"PC{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols)
        df_pca['Placement_Status'] = y_placement.map({1: 'High-Value Placed 🌟', 0: 'Standard Tier 📦'})

        exp_var = pca.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)

        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Top Component (PC1) Variance", f"{exp_var[0]*100:.1f}%")
        c_m2.metric("PC2 Variance", f"{exp_var[1]*100:.1f}%")
        c_m3.metric("Cumulative Variance Retained", f"{cum_var[-1]*100:.1f}%")

        # Scree Plot
        st.markdown("#### 📈 Scree Plot: Explained Variance Ratio")
        df_var = pd.DataFrame({
            'Component': pca_cols,
            'Individual Variance': exp_var,
            'Cumulative Variance': cum_var
        })

        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=df_var['Component'], y=df_var['Individual Variance'], name='Individual Variance', marker_color='#2563EB'))
        fig_scree.add_trace(go.Scatter(x=df_var['Component'], y=df_var['Cumulative Variance'], name='Cumulative Variance', line=dict(color='#DC2626', width=3)))
        fig_scree.update_layout(title="PCA Explained Variance Ratio & Cumulative Curve", xaxis_title="Principal Components", yaxis_title="Variance Ratio", yaxis_tickformat=".0%")
        st.plotly_chart(fig_scree, width='stretch')

        # Feature Loadings Heatmap
        st.markdown("#### 🧬 PCA Feature Loadings Heatmap")
        st.markdown("Understanding how original features contribute to each Principal Component:")
        loadings_df = pd.DataFrame(pca.components_.T, columns=pca_cols, index=X_encoded.columns)
        fig_loadings = px.imshow(loadings_df, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                                 title="PCA Feature Loadings Matrix (Original Features vs Principal Components)")
        st.plotly_chart(fig_loadings, width='stretch')

        # 2D & 3D Projections
        col_proj1, col_proj2 = st.columns(2)
        with col_proj1:
            st.markdown("#### 🗺️ 2D PCA Component Scatter Projection")
            fig_2d = px.scatter(df_pca, x='PC1', y='PC2', color='Placement_Status',
                                title="2D Projection: PC1 vs PC2",
                                color_discrete_map={'High-Value Placed 🌟': '#059669', 'Standard Tier 📦': '#6B7280'},
                                hover_data=pca_cols)
            st.plotly_chart(fig_2d, width='stretch')

        with col_proj2:
            st.markdown("#### 🧊 3D PCA Component Scatter Projection")
            if n_components >= 3:
                fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Placement_Status',
                                       title="3D Projection: PC1 vs PC2 vs PC3",
                                       color_discrete_map={'High-Value Placed 🌟': '#059669', 'Standard Tier 📦': '#6B7280'})
                st.plotly_chart(fig_3d, width='stretch')
            else:
                st.info("Select at least 3 PCA components to render the 3D projection.")

    # -------------------------------------------------------------
    # SUB TAB 2: ALL CLASSIFICATION MODELS (FULL VS PCA)
    # -------------------------------------------------------------
    with sub_t2:
        st.subheader("2️⃣ Machine Learning Classification Suite: Placement Prediction")
        st.markdown("Train and evaluate predictive models for **Placement Status (High-Value Tier)** using original features or PCA-reduced space.")

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            feature_space = st.radio("Select Feature Space for Model Training", ["Original High-Dimensional Features", "PCA Reduced Components"], key="p_feat_space")
        with col_cfg2:
            model_choice = st.selectbox(
                "Select Machine Learning Model",
                [
                    "Logistic Regression",
                    "Decision Tree Classifier",
                    "Random Forest Classifier",
                    "Support Vector Machine (SVM)",
                    "K-Nearest Neighbors (KNN)",
                    "Gaussian Naive Bayes",
                    "Gradient Boosting Classifier",
                    "XGBoost Classifier",
                    "LightGBM Classifier",
                    "Multi-Layer Perceptron (MLP Neural Net)"
                ],
                key="p_model_choice"
            )
        with col_cfg3:
            test_size_p = st.slider("Train-Test Split Ratio (% Test Set)", 10, 40, 20, step=5, key="p_split") / 100.0

        # Choose dataset (Original vs PCA)
        if feature_space == "Original High-Dimensional Features":
            X_model_data = X_scaled
        else:
            pca_m = PCA(n_components=min(n_components, X_scaled.shape[1]), random_state=42)
            X_model_data = pca_m.fit_transform(X_scaled)

        min_class_cnt = y_placement.value_counts().min() if y_placement.nunique() > 1 else 0
        use_stratify = y_placement if (y_placement.nunique() > 1 and (min_class_cnt * test_size_p) >= 1) else None

        X_tr, X_te, y_tr, y_te = train_test_split(X_model_data, y_placement, test_size=test_size_p, random_state=42, stratify=use_stratify)

        if y_tr.nunique() < 2:
            st.warning("⚠️ Filtered dataset contains only one target class. Broaden sidebar filters to evaluate models.")
        else:
            # Instantiate model
            model = None
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_choice == "Decision Tree Classifier":
                model = DecisionTreeClassifier(max_depth=5, random_state=42)
            elif model_choice == "Random Forest Classifier":
                model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
            elif model_choice == "Support Vector Machine (SVM)":
                model = SVC(probability=True, random_state=42)
            elif model_choice == "K-Nearest Neighbors (KNN)":
                model = KNeighborsClassifier(n_neighbors=5)
            elif model_choice == "Gaussian Naive Bayes":
                model = GaussianNB()
            elif model_choice == "Gradient Boosting Classifier":
                model = GradientBoostingClassifier(n_estimators=50, random_state=42)
            elif model_choice == "XGBoost Classifier":
                if HAS_XGB:
                    model = XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
                else:
                    st.warning("XGBoost package not installed. Falling back to Gradient Boosting Classifier.")
                    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
            elif model_choice == "LightGBM Classifier":
                if HAS_LGB:
                    model = LGBMClassifier(n_estimators=50, random_state=42, verbosity=-1)
                else:
                    st.warning("LightGBM package not installed. Falling back to Random Forest Classifier.")
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
            elif model_choice == "Multi-Layer Perceptron (MLP Neural Net)":
                model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)

            # Fit model
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            if hasattr(model, "predict_proba") and len(model.classes_) > 1 and 1 in model.classes_:
                class_idx = np.where(model.classes_ == 1)[0][0]
                y_prob = model.predict_proba(X_te)[:, class_idx]
            else:
                y_prob = np.zeros(len(y_te))

            acc = accuracy_score(y_te, y_pred)
            prec = precision_score(y_te, y_pred, zero_division=0)
            rec = recall_score(y_te, y_pred, zero_division=0)
            f1 = f1_score(y_te, y_pred, zero_division=0)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc * 100:.1f}%")
            c2.metric("Precision", f"{prec * 100:.1f}%")
            c3.metric("Recall", f"{rec * 100:.1f}%")
            c4.metric("F1 Score", f"{f1:.3f}")

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                cm = confusion_matrix(y_te, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                   x=['Standard Tier (0)', 'High-Value Placed (1)'],
                                   y=['Standard Tier (0)', 'High-Value Placed (1)'],
                                   title=f"Confusion Matrix ({model_choice})",
                                   labels={'x': 'Predicted Label', 'y': 'Actual Label'})
                st.plotly_chart(fig_cm, width='stretch')

            with col_m2:
                if len(np.unique(y_te)) > 1:
                    fpr, tpr, _ = roc_curve(y_te, y_prob)
                    roc_auc = auc(fpr, tpr)
                else:
                    fpr, tpr, roc_auc = [0, 1], [0, 1], 0.5

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_choice} (AUC = {roc_auc:.3f})',
                                             line=dict(color='#2563EB', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline (AUC = 0.5)',
                                             line=dict(color='gray', width=2, dash='dash')))
                fig_roc.update_layout(title="Receiver Operating Characteristic (ROC) Curve",
                                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, width='stretch')

    # -------------------------------------------------------------
    # SUB TAB 3: BENCHMARK LEADERBOARD (ORIGINAL VS PCA)
    # -------------------------------------------------------------
    with sub_t3:
        st.subheader("3️⃣ Model Benchmark Leaderboard: Original Features vs PCA Space")
        st.markdown("Compare all 10 classification algorithms trained on **Original Full Features** versus **PCA Reduced Components**.")

        pca_bench = PCA(n_components=min(n_components, X_scaled.shape[1]), random_state=42)
        X_pca_bench = pca_bench.fit_transform(X_scaled)

        min_class_cnt = y_placement.value_counts().min() if y_placement.nunique() > 1 else 0
        use_stratify = y_placement if (y_placement.nunique() > 1 and (min_class_cnt * 0.2) >= 1) else None

        X_tr_orig, X_te_orig, y_tr, y_te = train_test_split(X_scaled, y_placement, test_size=0.2, random_state=42, stratify=use_stratify)
        X_tr_pca, X_te_pca, _, _ = train_test_split(X_pca_bench, y_placement, test_size=0.2, random_state=42, stratify=use_stratify)

        models_dict = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Gaussian Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
            "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        }

        if HAS_XGB:
            models_dict["XGBoost"] = XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
        if HAS_LGB:
            models_dict["LightGBM"] = LGBMClassifier(n_estimators=50, random_state=42, verbosity=-1)

        leaderboard_rows = []

        for name, m_cls in models_dict.items():
            # Original Space
            m_orig = m_cls.__class__(**m_cls.get_params())
            m_orig.fit(X_tr_orig, y_tr)
            pred_orig = m_orig.predict(X_te_orig)
            leaderboard_rows.append({
                'Model': name,
                'Feature Space': 'Original Features',
                'Accuracy (%)': accuracy_score(y_te, pred_orig) * 100,
                'Precision (%)': precision_score(y_te, pred_orig, zero_division=0) * 100,
                'Recall (%)': recall_score(y_te, pred_orig, zero_division=0) * 100,
                'F1 Score': f1_score(y_te, pred_orig, zero_division=0)
            })

            # PCA Space
            m_pca = m_cls.__class__(**m_cls.get_params())
            m_pca.fit(X_tr_pca, y_tr)
            pred_pca = m_pca.predict(X_te_pca)
            leaderboard_rows.append({
                'Model': name,
                'Feature Space': f'PCA ({n_components} Components)',
                'Accuracy (%)': accuracy_score(y_te, pred_pca) * 100,
                'Precision (%)': precision_score(y_te, pred_pca, zero_division=0) * 100,
                'Recall (%)': recall_score(y_te, pred_pca, zero_division=0) * 100,
                'F1 Score': f1_score(y_te, pred_pca, zero_division=0)
            })

        df_lb = pd.DataFrame(leaderboard_rows).sort_values(by='F1 Score', ascending=False).reset_index(drop=True)

        st.markdown("#### 🏆 Leaderboard Summary Table")
        st.dataframe(df_lb.style.format({
            'Accuracy (%)': '{:.1f}%',
            'Precision (%)': '{:.1f}%',
            'Recall (%)': '{:.1f}%',
            'F1 Score': '{:.3f}'
        }), width='stretch')

        # Comparison Bar Chart
        fig_lb_comp = px.bar(df_lb, x='Model', y='F1 Score', color='Feature Space', barmode='group',
                             title="Model F1 Score Comparison: Original Features vs PCA Components",
                             color_discrete_sequence=['#2563EB', '#7C3AED'])
        st.plotly_chart(fig_lb_comp, width='stretch')

    # -------------------------------------------------------------
    # SUB TAB 4: INTERACTIVE PLACEMENT SIMULATOR
    # -------------------------------------------------------------
    with sub_t4:
        st.subheader("4️⃣ Interactive Customer Placement Simulator")
        st.markdown("Enter hypothetical customer details to predict **Placement Status (High-Value Placement Tier)** in real-time.")

        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            in_age = st.number_input("Customer Age", min_value=18, max_value=90, value=35, key="sim_p_age")
            in_orders = st.number_input("Total Lifetime Orders", min_value=1, max_value=50, value=5, key="sim_p_orders")
        with sim_col2:
            in_items = st.number_input("Total Purchased Items", min_value=1, max_value=200, value=12, key="sim_p_items")
            in_delivery = st.number_input("Avg Delivery Days", min_value=1.0, max_value=30.0, value=10.0, step=0.5, key="sim_p_del")
        with sim_col3:
            in_aov = st.number_input("Average Order Value ($)", min_value=10.0, max_value=10000.0, value=500.0, step=25.0, key="sim_p_aov")
            gen_opts = cust_df['gender'].dropna().unique().tolist()
            in_gender = st.selectbox("Gender", options=gen_opts if gen_opts else ['Female', 'Male'], key="sim_p_gen")

        st_opts = cust_df['state'].dropna().unique().tolist()
        in_state = st.selectbox("State", options=st_opts if st_opts else ['NSW'], key="sim_p_st")

        # Build single input vector matching X_encoded columns
        input_df = pd.DataFrame(0, index=[0], columns=X_encoded.columns)
        input_df['age'] = in_age
        input_df['total_orders'] = in_orders
        input_df['total_items'] = in_items
        input_df['avg_delivery_days'] = in_delivery
        input_df['avg_order_value'] = in_aov

        g_col = f"gender_{in_gender}"
        if g_col in input_df.columns:
            input_df[g_col] = 1

        s_col = f"state_{in_state}"
        if s_col in input_df.columns:
            input_df[s_col] = 1

        # Standardize & transform using PCA
        input_scaled = scaler.transform(input_df)
        pca_sim = PCA(n_components=min(n_components, X_scaled.shape[1]), random_state=42)
        X_pca_all = pca_sim.fit_transform(X_scaled)
        input_pca = pca_sim.transform(input_scaled)

        # Train Random Forest & Logistic Regression for quick inference simulation
        rf_sim = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42).fit(X_pca_all, y_placement)
        lr_sim = LogisticRegression(max_iter=1000, random_state=42).fit(X_pca_all, y_placement)

        rf_prob = rf_sim.predict_proba(input_pca)[0, 1] if len(rf_sim.classes_) > 1 and 1 in rf_sim.classes_ else 0.0
        lr_prob = lr_sim.predict_proba(input_pca)[0, 1] if len(lr_sim.classes_) > 1 and 1 in lr_sim.classes_ else 0.0

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            status_rf = "High-Value Placed 🌟" if rf_prob >= 0.5 else "Standard Tier 📦"
            st.success(f"🌲 **Random Forest Prediction (PCA Space)**:\n### **{status_rf}**")
            st.metric("High-Value Probability Score", f"{rf_prob * 100:.1f}%")

        with res_col2:
            status_lr = "High-Value Placed 🌟" if lr_prob >= 0.5 else "Standard Tier 📦"
            st.info(f"📈 **Logistic Regression Prediction (PCA Space)**:\n### **{status_lr}**")
            st.metric("High-Value Probability Score", f"{lr_prob * 100:.1f}%")
