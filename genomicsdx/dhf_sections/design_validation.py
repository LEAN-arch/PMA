# --- SME OVERHAUL: Definitive, Compliance-Focused Version (PMA-Focused V4-Final) ---
"""
Renders the Design Validation section of the DHF dashboard.

This module provides a comprehensive UI for documenting design validation,
which confirms that the complete diagnostic service (Assay + SaMD) meets
user needs and intended uses. It is structured to manage evidence from pivotal
clinical trials and human factors (usability) studies, as required by
21 CFR 820.30(g) and FDA guidance.
"""

# --- Standard Library Imports ---
import logging
from typing import Any, Dict, List

# --- Third-party Imports ---
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.metrics import confusion_matrix

# --- Local Application Imports ---
from genomicsdx.utils.session_state_manager import SessionStateManager
from genomicsdx.utils.plot_utils import create_roc_curve, create_confusion_matrix_heatmap

# --- Setup Logging ---
logger = logging.getLogger(__name__)

def render_design_validation(ssm: SessionStateManager) -> None:
    """
    Renders the UI for the Design Validation section, focusing on clinical
    and usability validation evidence.

    Args:
        ssm (SessionStateManager): The session state manager to access DHF data.
    """
    st.header("8. Design Validation")
    st.markdown("""
    *As per 21 CFR 820.30(g), ISO 13485:2016 Section 7.3.6, and IEC 62366.*

    Validation ensures the final, complete diagnostic service meets the defined **user needs** and **intended uses**. This is accomplished through objective evidence from clinical trials and usability studies conducted under actual or simulated use conditions. It answers the ultimate question: **"Did we build the right product for our users and patients?"**
    """)
    st.info("Use the tabs to manage evidence from the pivotal clinical study and all human factors validation activities.", icon="✅")
    st.divider()

    try:
        # --- 1. Load Data and Prepare Dependencies ---
        validation_data: Dict[str, Any] = ssm.get_data("clinical_study") or {}
        inputs_data: List[Dict[str, Any]] = ssm.get_data("design_inputs", "requirements") or []
        logger.info("Loaded design validation and clinical study data.")

        user_needs = [req for req in inputs_data if req.get('type') == 'User Need']
        user_need_options: List[str] = [""] + sorted([f"{un.get('id', '')}: {un.get('description', '')[:60]}..." for un in user_needs])
        user_need_map = {opt: opt.split(':')[0] for opt in user_need_options}

        # --- 2. Define Tabs for Each Validation Stream ---
        tab_titles = [
            "1. Pivotal Clinical Trial Dashboard",
            "2. Human Factors (Usability) Validation",
            "3. Adverse Event & Safety Monitoring"
        ]
        tab1, tab2, tab3 = st.tabs(tab_titles)

        # --- Tab 1: Pivotal Clinical Trial ---
        with tab1:
            st.subheader("Pivotal Clinical Trial Dashboard")
            st.caption("This dashboard tracks the progress and interim performance of the pivotal study validating the end-to-end diagnostic service (Assay + SaMD).")
            
            # --- Overview & Enrollment ---
            overview_cols = st.columns([2, 1])
            with overview_cols[0]:
                st.markdown(f"**Protocol:** `{validation_data.get('protocol_id', 'N/A')}`")
                st.markdown(f"**NCT ID:** `{validation_data.get('nct_id', 'N/A')}`")
                st.markdown(f"**Primary Endpoints:** {validation_data.get('primary_endpoints', 'N/A')}")
            with overview_cols[1]:
                enrollment_df = pd.DataFrame(validation_data.get("enrollment", []))
                if not enrollment_df.empty:
                    total_enrolled = enrollment_df['enrolled'].sum()
                    total_target = enrollment_df['target'].sum()
                    overall_progress = (total_enrolled / total_target) * 100 if total_target > 0 else 0
                    st.metric("Overall Enrollment", f"{total_enrolled:,} / {total_target:,}", f"{overall_progress:.1f}% Complete")
                    st.progress(overall_progress / 100)
            st.dataframe(enrollment_df, use_container_width=True, hide_index=True)
            
            # --- Interim Performance Metrics & Visuals ---
            st.subheader("Interim Clinical Performance (Based on Locked Algorithm)")
            perf_cols = st.columns(2)
            with perf_cols[0]:
                st.markdown("**Key Performance Metrics**")
                perf_df = pd.DataFrame(validation_data.get("performance_metrics", []))
                if not perf_df.empty:
                    st.dataframe(perf_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No performance metrics available yet.")
            
            with perf_cols[1]:
                # Mock clinical data for visualization
                rng = np.random.default_rng(42)
                scores = np.concatenate([rng.normal(0.8, 0.2, 50), rng.normal(0.2, 0.15, 950)]).clip(0, 1)
                truth = np.concatenate([np.ones(50), np.zeros(950)])
                clinical_df = pd.DataFrame({'score': scores, 'truth': truth})
                
                st.markdown("**Receiver Operating Characteristic (ROC)**")
                roc_fig = create_roc_curve(clinical_df, 'score', 'truth', title="")
                roc_fig.update_layout(height=300, title_text="", margin=dict(t=10, b=40, l=40, r=10))
                st.plotly_chart(roc_fig, use_container_width=True)

            st.markdown("**Confusion Matrix (at optimal threshold)**")
            optimal_threshold = 0.5 # For demonstration
            predictions = (clinical_df['score'] >= optimal_threshold).astype(int)
            cm = confusion_matrix(clinical_df['truth'], predictions)
            cm_fig = create_confusion_matrix_heatmap(cm, ['No Cancer', 'Cancer'])
            cm_fig.update_layout(height=400, title_text="")
            st.plotly_chart(cm_fig, use_container_width=True)

        # --- Tab 2: Human Factors Validation ---
        with tab2:
            st.subheader("Human Factors & Usability Validation Studies (IEC 62366)")
            st.caption("Document formative and summative usability studies that validate the user interface is safe, effective, and that use-related risks have been controlled.")

            hf_studies_data = validation_data.get("hf_studies", [])
            hf_studies_df = pd.DataFrame(hf_studies_data)
            
            if 'user_need_validated' in hf_studies_df.columns:
                 hf_studies_df['user_need_display'] = hf_studies_df['user_need_validated'].map({v:k for k,v in user_need_map.items() if v})
            else:
                 hf_studies_df['user_need_display'] = ""

            edited_hf_df = st.data_editor(
                hf_studies_df, num_rows="dynamic", use_container_width=True, key="hf_validation_editor",
                column_config={
                    "id": st.column_config.TextColumn("Study ID", required=True),
                    "study_name": st.column_config.TextColumn("Study/Protocol Name", width="large"),
                    "user_interface_validated": st.column_config.SelectboxColumn("UI Validated", options=["Sample Collection Kit & IFU", "Clinical Report", "LIMS/Portal"]),
                    "user_need_display": st.column_config.SelectboxColumn("Links to User Need", options=user_need_options, required=True),
                    "confirms_risk_control": st.column_config.CheckboxColumn("Summative (Confirms Risk Control?)"),
                    "status": st.column_config.SelectboxColumn("Status", options=["Not Started", "In Progress", "Completed"]),
                    "report_link": st.column_config.LinkColumn("Link to Report"),
                    "user_need_validated": None
                }, hide_index=True
            )
            
            if edited_hf_df.to_dict('records') != hf_studies_df.to_dict('records'):
                df_to_save = edited_hf_df.copy()
                df_to_save['user_need_validated'] = df_to_save['user_need_display'].map(user_need_map)
                df_to_save.drop(columns=['user_need_display'], inplace=True)
                validation_data["hf_studies"] = df_to_save.to_dict('records')
                ssm.update_data(validation_data, "clinical_study")
                st.toast("Human Factors validation studies updated!", icon="✅"); st.rerun()

        # --- Tab 3: Adverse Events ---
        with tab3:
            st.subheader("Adverse Event (AE) & Safety Monitoring")
            st.caption("Log all adverse events from clinical and usability studies to monitor device safety.")
            ae_data = validation_data.get("adverse_events", [])
            ae_df = pd.DataFrame(ae_data)
            
            edited_ae_df = st.data_editor(
                ae_df, num_rows="dynamic", use_container_width=True, key="ae_editor",
                column_config={
                    "id": st.column_config.TextColumn("AE ID", required=True),
                    "site": st.column_config.TextColumn("Site"),
                    "description": st.column_config.TextColumn("Event Description", width="large"),
                    "severity": st.column_config.SelectboxColumn("Severity", options=["Mild", "Moderate", "Severe"]),
                    "related_to_device": st.column_config.CheckboxColumn("Device Related?"),
                }, hide_index=True
            )
            
            if edited_ae_df.to_dict('records') != ae_data:
                validation_data["adverse_events"] = edited_ae_df.to_dict('records')
                ssm.update_data(validation_data, "clinical_study")
                st.toast("Adverse Events log updated!", icon="✅"); st.rerun()

    except Exception as e:
        st.error("An error occurred while displaying the Design Validation section. The data may be malformed.")
        logger.error(f"Failed to render design validation: {e}", exc_info=True)
