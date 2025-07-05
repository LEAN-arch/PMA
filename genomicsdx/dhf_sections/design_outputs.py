# --- SME OVERHAUL: Definitive, Compliance-Focused Version (PMA-Focused V4-Final) ---
"""
Renders the Design Outputs section of the DHF dashboard.

This module provides a structured, categorized UI for managing all tangible
outputs of the design process, which collectively form the basis for the
Device Master Record (DMR). It ensures each output for the assay and software
is version-controlled, its status is tracked, and it is traceably linked to a
design input, as required by 21 CFR 820.30(d).
"""

# --- Standard Library Imports ---
import logging
from typing import Any, Dict, List

# --- Third-party Imports ---
import pandas as pd
import streamlit as st

# --- Local Application Imports ---
from ..utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def render_design_outputs(ssm: SessionStateManager) -> None:
    """
    Renders the UI for managing categorized Design Outputs.

    This function displays outputs in separate tabs for clarity (Assay, Software,
    etc.). It provides an editable table within each tab to manage
    the outputs, including their version, status, and the critical traceability
    link back to a specific design input requirement. It also features a
    dashboard for analyzing the health and completeness of the DMR.

    Args:
        ssm (SessionStateManager): The session state manager to access DHF data.
    """
    st.header("5. Design Outputs (Device Master Record Basis)")
    st.markdown("""
    *As per 21 CFR 820.30(d).*

    Design Outputs are the full set of specifications, procedures, and artifacts that define the entire diagnostic service, including both the **IVD assay** and the **SaMD algorithm**. They are the tangible results of the design process and form the basis for the **Device Master Record (DMR)**. **Crucially, each output must be traceable to a design input, be approved, and be placed under formal version control.**
    """)
    st.info("Use the tabs below to manage the outputs for each part of the system. Every output must be linked to an input requirement. Changes are saved automatically.", icon="📝")

    try:
        # --- 1. Load Data and Prepare Dependencies ---
        outputs_data: List[Dict[str, Any]] = ssm.get_data("design_outputs", "documents") or []
        inputs_data: List[Dict[str, Any]] = ssm.get_data("design_inputs", "requirements") or []
        logger.info(f"Loaded {len(outputs_data)} design output records and {len(inputs_data)} input records.")

        if not inputs_data:
            st.warning("⚠️ No Design Inputs found. Please add requirements in '4. Design Inputs' before creating outputs.", icon="❗")
            return
        
        df_outputs = pd.DataFrame(outputs_data)
        df_inputs = pd.DataFrame(inputs_data)
        
        # --- 2. DMR Health Dashboard ---
        st.subheader("Device Master Record (DMR) Health Dashboard")
        kpi_cols = st.columns(3)
        
        if not df_outputs.empty:
            status_counts = df_outputs['status'].value_counts()
            approved_count = status_counts.get("Approved", 0)
            total_outputs = len(df_outputs)
            
            non_user_needs_ids = set(df_inputs[df_inputs['type'] != 'User Need']['id']) if 'type' in df_inputs.columns else set()
            linked_inputs = set(df_outputs.dropna(subset=['linked_input_id'])['linked_input_id'])
            coverage_pct = (len(linked_inputs.intersection(non_user_needs_ids)) / len(non_user_needs_ids)) * 100 if non_user_needs_ids else 0
            
            untraced_outputs = df_outputs[df_outputs['linked_input_id'].isnull() | (df_outputs['linked_input_id'] == '')]

            kpi_cols[0].metric("Approved Outputs", f"{approved_count} / {total_outputs}", f"{(approved_count/total_outputs if total_outputs > 0 else 0):.1%}")
            kpi_cols[1].metric("Input-to-Output Coverage", f"{coverage_pct:.1f}%", help="Percentage of formal requirements (non-User Needs) that are covered by at least one Design Output.")
            kpi_cols[2].metric("Untraced Outputs", len(untraced_outputs), help="Number of Design Outputs not linked to a Design Input. This is a critical compliance gap.", delta=len(untraced_outputs), delta_color="inverse")
            
            with st.expander("View DMR Status Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Outputs by Status")
                    st.dataframe(status_counts)
                with col2:
                    st.write("Untraced Design Outputs")
                    st.dataframe(untraced_outputs[['id', 'title', 'type']], use_container_width=True)
        else:
            st.warning("No Design Outputs logged yet.")

        st.divider()
        
        req_options_map = {
            f"{req.get('id', '')}: {req.get('description', '')[:70]}...": req.get('id', '')
            for req in inputs_data
        }
        reverse_req_map = {v: k for k, v in req_options_map.items()}

        # --- 3. Define Tabs for Each Output Category ---
        tab_titles = [
            "Assay & Reagent Specifications",
            "Software & Algorithm Outputs",
            "Hardware & Kit Specifications",
            "Labeling & Procedures"
        ]
        tabs = st.tabs(tab_titles)

        def render_editor_tab(df: pd.DataFrame, key_suffix: str, type_options: List[str], help_text: str):
            st.subheader(f"{key_suffix.replace('_', ' ').title()}")
            st.caption(help_text)
            
            tab_df = df[df['type'].isin(type_options)].copy()
            
            if 'approval_date' in tab_df.columns:
                tab_df['approval_date'] = pd.to_datetime(tab_df['approval_date'], errors='coerce')
            
            original_records = tab_df.to_dict('records')

            tab_df['linked_input_descriptive'] = tab_df['linked_input_id'].map(reverse_req_map)

            edited_tab_df = st.data_editor(
                tab_df, num_rows="dynamic", use_container_width=True, key=f"output_editor_{key_suffix}",
                column_config={
                    "id": st.column_config.TextColumn("ID", required=True, help="Unique identifier for the design output document/artifact."),
                    "type": st.column_config.SelectboxColumn("Type", options=type_options, required=True),
                    "title": st.column_config.TextColumn("Title", width="large", required=True),
                    "version": st.column_config.TextColumn("Version", required=True),
                    "status": st.column_config.SelectboxColumn("Status", options=["Draft", "In Review", "Approved", "Obsolete"], required=True),
                    "approval_date": st.column_config.DateColumn("Approval Date", format="YYYY-MM-DD"),
                    "linked_input_descriptive": st.column_config.SelectboxColumn("Traces to Requirement", options=list(req_options_map.keys()), required=True),
                    "linked_input_id": None, 
                    "link_to_artifact": st.column_config.LinkColumn("Link to Artifact")
                }, hide_index=True
            )

            if str(edited_tab_df.to_dict('records')) != str(original_records):
                edited_tab_df['linked_input_id'] = edited_tab_df['linked_input_descriptive'].map(req_options_map)
                df_to_save = edited_tab_df.drop(columns=['linked_input_descriptive'])
                
                if 'approval_date' in df_to_save.columns:
                    df_to_save['approval_date'] = df_to_save['approval_date'].dt.strftime('%Y-%m-%d').replace({pd.NaT: None})

                other_outputs = df_outputs[~df_outputs['type'].isin(type_options)].to_dict('records')
                updated_all_outputs = other_outputs + df_to_save.to_dict('records')
                ssm.update_data(updated_all_outputs, "design_outputs", "documents")
                logger.info(f"Design outputs for '{key_suffix}' updated.")
                st.toast(f"{key_suffix.replace('_', ' ').title()} saved!", icon="✅")
                st.rerun()

        with tabs[0]:
            render_editor_tab(df_outputs, "assay_reagent", ["Assay Spec"], "Define the specifications for all wet-lab components, including oligo sequences, buffer formulations, and reagent QC tests.")
        with tabs[1]:
            render_editor_tab(df_outputs, "software_algorithm", ["Software Spec", "Algorithm"], "List the Software Design Specifications (SDS), locked algorithm models, and source code repositories that define the bioinformatics pipeline and classifier.")
        with tabs[2]:
            render_editor_tab(df_outputs, "hardware_kit", ["Hardware Spec"], "Define the specifications for the physical components of the service, such as the blood collection tube, packaging, and shipping materials.")
        with tabs[3]:
            render_editor_tab(df_outputs, "labeling_procedures", ["Labeling", "Procedure"], "List all controlled documents that will be part of the final service, including the Instructions for Use (IFU), lab SOPs, and the Clinical Report template.")

    except Exception as e:
        st.error("An error occurred while displaying the Design Outputs section. The data may be malformed.")
        logger.error(f"Failed to render design outputs: {e}", exc_info=True)
