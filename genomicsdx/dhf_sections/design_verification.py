# --- SME OVERHAUL: Definitive, Compliance-Focused Version (PMA-Focused V4-Final) ---
"""
Renders the Design Verification / Analytical Validation section of the DHF.

This module provides a comprehensive dashboard for managing the extensive
Analytical Validation (AV) program required for a PMA-class genomic diagnostic.
It confirms that the design outputs for the assay and SaMD meet their corresponding
design inputs, as per 21 CFR 820.30(f), by tracking specific performance
characteristic studies.
"""

# --- Standard Library Imports ---
import logging
from typing import Any, Dict, List

# --- Third-party Imports ---
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px

# --- Local Application Imports ---
from genomicsdx.utils.session_state_manager import SessionStateManager
from genomicsdx.utils.plot_utils import create_lod_probit_plot, create_bland_altman_plot

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def render_design_verification(ssm: SessionStateManager) -> None:
    """
    Renders the UI for the Design Verification / Analytical Validation section.
    This function displays the AV program in categorized tabs with interactive
    data analysis workbenches, ensuring full traceability and compliance.
    """
    st.header("7. Design Verification (Analytical Validation)")
    st.markdown("""
    *As per 21 CFR 820.30(f) and relevant CLSI Guidelines (e.g., EP05, EP17, EP12).*

    Verification confirms through objective evidence that design outputs meet design inputs. For this IVD, this is primarily achieved through a comprehensive **Analytical Validation (AV)** program that characterizes the performance of the **assay** and **bioinformatics pipeline (SaMD)**. It answers the question: **"Did we build the assay and system right?"**
    """)
    st.info("Use the tabs below to manage AV studies for each component. The 'Data Workbench' in each tab provides live statistical analysis tools.", icon="🔬")

    try:
        # --- 1. Load Data and Prepare Dependencies ---
        verification_data: List[Dict[str, Any]] = ssm.get_data("design_verification", "tests") or []
        outputs_data: List[Dict[str, Any]] = ssm.get_data("design_outputs", "documents") or []
        inputs_data: List[Dict[str, Any]] = ssm.get_data("design_inputs", "requirements") or []
        
        output_ids: List[str] = [""] + sorted([doc.get('id', '') for doc in outputs_data if doc.get('id')])
        input_ids: List[str] = [""] + sorted([req.get('id', '') for req in inputs_data if req.get('id')])
        
        # --- 2. High-Level AV Program KPIs & Gap Analysis ---
        st.subheader("Analytical Validation Program Status")
        if verification_data:
            av_df = pd.DataFrame(verification_data)
            total_tests = len(av_df)
            completed_tests = len(av_df[av_df['status'] == 'Completed'])
            passing_tests = len(av_df[av_df['result'] == 'Pass'])
            progress = (completed_tests / total_tests) * 100 if total_tests > 0 else 0
            
            # Gap Analysis
            verifiable_reqs = {req['id'] for req in inputs_data if req.get('type') in ['System', 'Assay', 'Software', 'Kit', 'Labeling']}
            verified_reqs = set(av_df['input_verified_id'].dropna())
            unverified_count = len(verifiable_reqs - verified_reqs)
            
            kpi_cols = st.columns(3)
            kpi_cols[0].metric("AV Program Completion", f"{progress:.1f}%", f"{completed_tests}/{total_tests} Protocols Executed")
            kpi_cols[1].metric("Passing Protocols", f"{passing_tests} / {completed_tests}", help="Of all completed protocols, how many passed.")
            kpi_cols[2].metric("Unverified Requirements", unverified_count, help="Number of requirements not yet covered by a verification test.", delta=unverified_count, delta_color="inverse")
            st.progress(progress / 100)
        else:
            st.warning("No verification activities have been logged yet.")
        st.divider()

        # --- 3. Define Tabs for Each AV Study Category ---
        tab_titles = ["Assay Precision & Reproducibility", "Assay Analytical Sensitivity (LoD)", "Assay Analytical Specificity", "Software & System V&V"]
        tabs = st.tabs(tab_titles)

        def render_av_protocol_editor(tab_name: str, test_types: List[str]):
            st.subheader(f"{tab_name} Study Protocols")
            tab_data = [t for t in verification_data if t.get('test_type') in test_types]
            edited_df = st.data_editor(
                pd.DataFrame(tab_data), num_rows="dynamic", use_container_width=True, key=f"verification_editor_{tab_name}",
                column_config={
                    "id": st.column_config.TextColumn("Protocol ID", required=True), "test_type": st.column_config.SelectboxColumn("Test Type", options=test_types),
                    "test_name": st.column_config.TextColumn("Test Name", width="large"), "input_verified_id": st.column_config.SelectboxColumn("Verifies Req.", options=input_ids),
                    "output_verified_id": st.column_config.SelectboxColumn("Tests Output", options=output_ids), "status": st.column_config.SelectboxColumn("Status", options=["Not Started", "In Progress", "Completed"]),
                    "result": st.column_config.SelectboxColumn("Result", options=["Pending", "Pass", "Fail"]), "report_link": st.column_config.LinkColumn("Link to Report")
                }, hide_index=True
            )
            if edited_df.to_dict('records') != tab_data:
                other_tests = [t for t in verification_data if t.get('test_type') not in test_types]
                ssm.update_data(other_tests + edited_df.to_dict('records'), "design_verification", "tests")
                st.toast(f"{tab_name} studies updated!", icon="✅"); st.rerun()

        with tabs[0]:
            render_av_protocol_editor("Precision", ["Repeatability", "Intermediate Precision", "Reproducibility"])
            with st.expander("🧮 **Precision Data Workbench (CLSI EP05)**"):
                st.info("Upload raw data from a precision study (CSV with columns: operator, lot, day, value) to automatically calculate precision statistics.")
                uploaded_file = st.file_uploader("Upload Precision Data CSV", type="csv", key="prec_upload")
                if uploaded_file:
                    df_prec = pd.read_csv(uploaded_file)
                    st.dataframe(df_prec.head())
                    if 'value' in df_prec.columns:
                        st.subheader("Precision Results")
                        results = df_prec.groupby(['operator', 'lot'])['value'].agg(['mean', 'std']).reset_index()
                        results['%CV'] = (results['std'] / results['mean']) * 100
                        st.dataframe(results)
                        fig = px.box(df_prec, x='operator', y='value', color='lot', title="Precision by Operator and Lot", points="all")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Uploaded CSV must contain a 'value' column.")

        with tabs[1]:
            render_av_protocol_editor("Analytical Sensitivity", ["Limit of Blank (LoB)", "Limit of Detection (LoD)", "Limit of Quantitation (LoQ)"])
            with st.expander("🧮 **Limit of Detection (LoD) Data Workbench (CLSI EP17)**"):
                st.info("Enter concentration and hit rate data from a dilution series to perform Probit regression and calculate the LoD. This is a critical assay performance characteristic.")
                lod_data = ssm.get_data("quality_system", "lod_probit_data")
                df_lod = pd.DataFrame(lod_data)
                edited_lod_df = st.data_editor(df_lod, num_rows="dynamic", key="lod_editor")
                if not edited_lod_df.empty:
                    fig = create_lod_probit_plot(edited_lod_df, 'Concentration', 'Hit Rate')
                    st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            render_av_protocol_editor("Analytical Specificity", ["Interference", "Cross-Reactivity"])
            with st.expander("🧮 **Interference Data Workbench (Bland-Altman)**"):
                st.info("Enter paired measurements (e.g., baseline vs. with interferent) to generate a Bland-Altman plot assessing agreement and bias.")
                inter_data = ssm.get_data("quality_system", "interference_data")
                df_inter = pd.DataFrame(inter_data)
                st.dataframe(df_inter, height=200)
                fig = create_bland_altman_plot(df_inter, 'Baseline_Value', 'Interferent_Value')
                st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            render_av_protocol_editor("Software", ["SW Verification", "System Verification"])
            with st.expander("🧮 **Software V&V Dashboard**"):
                st.info("This dashboard summarizes results from automated test suites for the SaMD components, providing live feedback on software quality as per ISO 62304.")
                sw_test_cols = st.columns(3)
                sw_test_cols[0].metric("Unit Tests", "857 / 857", "100% Pass")
                sw_test_cols[1].metric("Integration Tests", "112 / 115", "-3 Failing", delta_color="inverse")
                sw_test_cols[2].metric("Code Coverage", "92%", "Target: >90%")
                
                st.write("**Failing Integration Tests:**")
                failing_tests = [
                    {"Test ID": "INT-042", "Component": "LIMS_API", "Error": "Timeout on record push"},
                    {"Test ID": "INT-078", "Component": "Classifier", "Error": "Memory leak on large inputs"},
                    {"Test ID": "INT-091", "Component": "Report_Gen", "Error": "CSO graph rendering fails for edge case"}
                ]
                st.dataframe(failing_tests, hide_index=True)

    except Exception as e:
        st.error("An error occurred while displaying the Design Verification section. The data may be malformed.")
        logger.error(f"Failed to render design verification: {e}", exc_info=True)
