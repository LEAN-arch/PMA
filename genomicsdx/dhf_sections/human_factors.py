# --- SME OVERHAUL: Definitive, Compliance-Focused Version (PMA-Focused V4-Final) ---
"""
Renders the Human Factors & Usability Engineering File of the DHF dashboard.

This module provides a structured UI for documenting the complete Usability
Engineering process in alignment with IEC 62366 and FDA guidance. It includes
defining the use specification, user profiles, and conducting a Use-Related Risk
Analysis (URRA) for each user interface (kit, lab, and report), and linking risks
to controls and validation activities.
"""

# --- Standard Library Imports ---
import logging
from typing import Any, Dict, List

# --- Third-party Imports ---
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Local Application Imports ---
from ..utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def render_human_factors(ssm: SessionStateManager) -> None:
    """Renders the UI for the Human Factors & Usability Engineering section."""
    st.header("3. Human Factors & Usability Engineering (IEC 62366)")
    st.markdown("This section documents the usability engineering process to ensure the diagnostic service can be used safely and effectively by its intended users in the intended use environments. The process focuses on identifying and mitigating use-related risks that could lead to patient harm (e.g., from a sample collection error or a misinterpreted result).")
    
    try:
        # --- 1. Load Data and Prepare Dependencies ---
        hf_data: Dict[str, Any] = ssm.get_data("human_factors") or {}
        rmf_data: Dict[str, Any] = ssm.get_data("risk_management_file") or {}
        val_data: Dict[str, Any] = ssm.get_data("clinical_study") or {}
        
        hazards: List[Dict[str, Any]] = rmf_data.get("hazards", [])
        hazard_ids: List[str] = [""] + sorted([h.get('id', '') for h in hazards if h.get('id')])

        hf_val_studies: List[Dict[str, Any]] = val_data.get("hf_studies", [])
        hf_val_ids: List[str] = [""] + sorted([s.get('id', '') for s in hf_val_studies if s.get('id')])
        
        # --- 2. HFE Risk Dashboard ---
        st.subheader("Usability Risk & Validation Dashboard")
        all_urra_tasks = hf_data.get("kit_urra", []) + hf_data.get("lab_urra", []) + hf_data.get("report_urra", [])
        if all_urra_tasks:
            df_urra = pd.DataFrame(all_urra_tasks)
            total_tasks = len(df_urra)
            # A task is controlled if it has a non-empty risk control measure
            controlled_tasks = df_urra['risk_control_measure'].str.len().gt(0).sum()
            # A task is validated if its validation link is not empty
            validated_tasks = df_urra[df_urra['validation_link'].notna() & (df_urra['validation_link'] != '')].shape[0]

            kpi_cols = st.columns(3)
            kpi_cols[0].metric("Identified Critical Tasks", total_tasks)
            kpi_cols[1].metric("Tasks with Risk Controls", f"{controlled_tasks} / {total_tasks}", f"{controlled_tasks/total_tasks:.1%}")
            kpi_cols[2].metric("Validated Risk Controls", f"{validated_tasks} / {controlled_tasks}", f"{(validated_tasks/controlled_tasks if controlled_tasks > 0 else 0):.1%}")

            # Risk Matrix Plot (simplified S vs P)
            df_urra['S'] = df_urra['related_hazard_id'].apply(lambda x: next((h['initial_S'] for h in hazards if h['id'] == x), 1) if x else 1)
            # Mock probability for visualization
            rng = np.random.default_rng(hash("urra"))
            df_urra['P'] = rng.integers(1, 4, size=len(df_urra))
            fig = go.Figure(data=go.Scatter(
                x=df_urra['S'], y=df_urra['P'],
                mode='markers+text', text=df_urra['id'], textposition='top center',
                marker=dict(size=15, color=df_urra['P'], colorscale='Reds'),
                hovertemplate='<b>%{text}</b><br>Task: %{customdata[0]}<br>Severity: %{x}<br>Probability: %{y}<extra></extra>',
                customdata=df_urra[['user_task']]
            ))
            fig.update_layout(title="<b>Use-Related Risk Matrix</b>", xaxis_title="Severity of Potential Harm (S)", yaxis_title="Probability of Use Error (P)", yaxis=dict(range=[0.5, 3.5]), xaxis=dict(range=[0.5, 5.5]))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        
        # --- 3. Define Tabs for HFE Process ---
        st.info("Use the tabs to define the use context and conduct a Use-Related Risk Analysis (URRA) for each user interface. Changes are saved automatically.", icon="👥")
        tab_titles = ["1. Use Specification", "2. User Profiles", "3. URRA: Sample Kit", "4. URRA: Lab", "5. URRA: Report"]
        tabs = st.tabs(tab_titles)

        def render_editor_tab(table_key: str, df: pd.DataFrame, column_config: dict):
            original_data = df.to_dict('records')
            edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"hf_editor_{table_key}", column_config=column_config, hide_index=True)
            if edited_df.to_dict('records') != original_data:
                cleaned_records = [record for record in edited_df.to_dict('records') if any(val not in [None, ''] for val in record.values())]
                hf_data[table_key] = cleaned_records
                ssm.update_data(hf_data, "human_factors")
                st.toast(f"{table_key.replace('_', ' ').title()} data updated!", icon="✅"); st.rerun()

        with tabs[0]:
            st.subheader("Use Specification (Application Specification per IEC 62366)")
            st.caption("This section defines the intended medical indication, patient population, part of the body/type of tissue interacted with, and the operating principle.")
            spec_data = hf_data.get("use_specification", {})
            with st.form("use_spec_form"):
                use_profile = st.text_area("Intended Medical Indication & Use Profile", value=spec_data.get('intended_use_profile', ''))
                user_profile = st.text_area("Intended User Profile(s)", value=spec_data.get('intended_user_profile', ''))
                use_env = st.text_area("Intended Use Environment", value=spec_data.get('intended_use_environment', ''))
                op_func = st.text_area("Key Operational Functions (of Assay and SaMD)", value=spec_data.get('key_operational_functions', ''))
                if st.form_submit_button("Save Use Specification"):
                    hf_data['use_specification'] = {
                        'intended_use_profile': use_profile, 'intended_user_profile': user_profile,
                        'intended_use_environment': use_env, 'key_operational_functions': op_func
                    }
                    ssm.update_data(hf_data, "human_factors")
                    st.toast("Use Specification saved!", icon="✅")

        with tabs[1]:
            st.subheader("Intended Users and Use Environments")
            st.caption("Define the characteristics of each user group and the context in which they will interact with the diagnostic service.")
            user_profiles_df = pd.DataFrame(hf_data.get("user_profiles", []))
            render_editor_tab("user_profiles", user_profiles_df, {
                "user_group": "User Group", "description": st.column_config.TextColumn("User Profile Description", width="large"),
                "training": "Assumed Training", "use_environment": st.column_config.TextColumn("Use Environment", width="large")
            })

        urra_column_config = {
            "id": "URRA ID", "user_task": st.column_config.TextColumn("Critical User Task", width="large"),
            "potential_use_error": st.column_config.TextColumn("Potential Use Error", width="medium"),
            "potential_harm": st.column_config.TextColumn("Resulting Harm", width="medium"),
            "related_hazard_id": st.column_config.SelectboxColumn("Links to System Hazard", options=hazard_ids),
            "risk_control_measure": st.column_config.TextColumn("Risk Control Measure", width="large"),
            "validation_link": st.column_config.SelectboxColumn("HFE Validation Link", options=hf_val_ids)
        }

        with tabs[2]:
            st.subheader("URRA: Sample Collection Kit & IFU")
            st.caption("Identify tasks, potential errors, and resulting harms related to the phlebotomist's interaction with the kit and Instructions For Use (IFU).")
            kit_urra_df = pd.DataFrame(hf_data.get("kit_urra", []))
            render_editor_tab("kit_urra", kit_urra_df, urra_column_config)

        with tabs[3]:
            st.subheader("URRA: Laboratory Processing")
            st.caption("Identify tasks, potential errors, and resulting harms related to the lab technologist's interaction with samples, reagents, instruments, and LIMS.")
            lab_urra_df = pd.DataFrame(hf_data.get("lab_urra", []))
            render_editor_tab("lab_urra", lab_urra_df, urra_column_config)

        with tabs[4]:
            st.subheader("URRA: Clinical Test Report")
            st.caption("Identify tasks, potential errors, and resulting harms related to the oncologist's interpretation of the final test report generated by the SaMD.")
            report_urra_df = pd.DataFrame(hf_data.get("report_urra", []))
            render_editor_tab("report_urra", report_urra_df, urra_column_config)

    except Exception as e:
        st.error("An error occurred while displaying the Human Factors section. The data may be malformed.")
        logger.error(f"Failed to render human factors: {e}", exc_info=True)
