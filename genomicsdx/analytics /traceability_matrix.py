# --- SME OVERHAUL: Definitive, Consolidated, and Unabridged Final Version (PMA-Focused V4-Final) ---
"""
Module for rendering the Multi-Level DHF Traceability Command Center.

This module provides the critical, multi-faceted traceability views required
for a PMA-class genomic diagnostic. It demonstrates linkage from clinical needs
down to analytical (assay) and software (SaMD) verification, in alignment with
21 CFR 820.30(g), ISO 13485, ISO 14971, and ISO 62304. This is the definitive
traceability module for the project, featuring bi-directional tracing and gap analysis.
"""

# --- Standard Library Imports ---
import logging
from typing import Dict, List, Tuple, Any

# --- Third-party Imports ---
import pandas as pd
import streamlit as st

# --- Local Application Imports ---
from genomicsdx.utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)

# --- Styling and Configuration ---
def style_trace_cell(val: str, **kwargs) -> str:
    """Applies CSS styling to a cell based on its status icon."""
    val_str = str(val)
    color = "inherit"
    bg_color = "rgba(214, 39, 40, 0.1)"  # Red for untraced/fail by default
    font_weight = "normal"

    if '✅' in val_str:
        bg_color = 'rgba(44, 160, 44, 0.15)'  # Green for Pass
        color = '#1E4620'
        font_weight = "bold"
    elif '❌' in val_str and 'Fail' in val_str:
        bg_color = 'rgba(214, 39, 40, 0.2)'  # Darker Red for explicit Fail
        color = '#5A0001'
        font_weight = "bold"
    elif '⏳' in val_str:
        bg_color = 'rgba(255, 127, 14, 0.2)'  # Orange for In Progress
        color = '#663000'
        font_weight = "bold"
    elif 'N/A' in val_str or pd.isna(val):
        bg_color = "#f8f9fa" # Default light grey for empty cells
        color = '#6c757d'
    elif 'Untraced' in val_str:
        color = '#B30000'
        font_weight = 'normal'

    return f'background-color: {bg_color}; color: {color}; font-weight: {font_weight}; text-align: center;'

def highlight_risk_control(row):
    """Highlights a row if it corresponds to a risk control requirement."""
    if row.get('is_risk_control'):
        return ['background-color: #e7f3ff; font-weight: bold;'] * len(row)
    return [''] * len(row)

@st.cache_data
def generate_all_trace_data(
    requirements: List[Dict],
    verifications: List[Dict],
    validations: List[Dict],
    risk_controls: List[Dict]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pre-processes and merges all data for all traceability matrices.
    This expensive operation is cached for performance.
    Returns:
        - clinical_matrix
        - analytical_matrix
        - sw_matrix
        - risk_matrix
    """
    if not requirements:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    reqs_df = pd.DataFrame(requirements)
    ver_df = pd.DataFrame(verifications) if verifications else pd.DataFrame(columns=['id', 'result', 'input_verified_id'])
    val_df = pd.DataFrame(validations) if validations else pd.DataFrame(columns=['id', 'status', 'user_need_validated'])
    risk_df = pd.DataFrame(risk_controls) if risk_controls else pd.DataFrame(columns=['id', 'risk_control_measure', 'verification_link'])

    status_icon_map_ver = {'Pass': '✅', 'Fail': '❌ Fail', 'In Progress': '⏳', 'Pending': '⏳', 'Not Started': '⏳'}
    status_icon_map_val = {'Completed': '✅', 'Pass': '✅', 'Fail': '❌ Fail', 'In Progress': '⏳', 'Not Started': '⏳'}

    # --- 1. Clinical Traceability Data (User Needs -> Validation) ---
    user_needs_df = reqs_df[reqs_df['type'] == 'User Need'].copy()
    clinical_matrix_df = user_needs_df[['id', 'description']].set_index(['id', 'description'])
    if not val_df.empty and 'user_need_validated' in val_df.columns:
        val_status_map = val_df.set_index('user_need_validated')['status'].to_dict()
        val_id_map = val_df.set_index('user_need_validated')['id'].to_dict()
        
        clinical_matrix_df['Status'] = clinical_matrix_df.index.get_level_values('id').map(val_status_map)
        clinical_matrix_df['Link'] = clinical_matrix_df.index.get_level_values('id').map(val_id_map)
        
        trace_text = clinical_matrix_df['Status'].map(status_icon_map_val).fillna('❌ Untraced') + ' ' + clinical_matrix_df['Link'].fillna('')
        clinical_matrix_df['Validation Trace'] = trace_text
    else:
        clinical_matrix_df['Validation Trace'] = '❌ Untraced'

    # --- 2. Generic Wide Matrix Creation Logic ---
    def create_wide_matrix(req_subset_df, ver_val_df, trace_col, status_col, link_col):
        if ver_val_df.empty or req_subset_df.empty or trace_col not in ver_val_df.columns:
            return pd.DataFrame(index=pd.MultiIndex.from_frame(req_subset_df[['id', 'description', 'is_risk_control']]))
        
        ver_val_df['status_icon'] = ver_val_df[status_col].map(status_icon_map_ver).fillna('⏳')
        ver_val_df['trace_link'] = ver_val_df['status_icon'] + ' ' + ver_val_df['id']

        merged = pd.merge(
            req_subset_df[['id', 'description', 'is_risk_control']],
            ver_val_df[[trace_col, 'id', 'trace_link']],
            left_on='id', right_on=trace_col, how='left'
        )
        
        if merged['id_y'].notna().any():
            matrix = merged.pivot_table(index=['id_x', 'description', 'is_risk_control'], columns='id_y', values='trace_link', aggfunc='first')
        else:
            matrix = pd.DataFrame(index=pd.MultiIndex.from_frame(req_subset_df[['id', 'description', 'is_risk_control']]))

        matrix = matrix.reindex(pd.MultiIndex.from_frame(req_subset_df[['id', 'description', 'is_risk_control']]))
        matrix.index.names = ['id', 'description', 'is_risk_control']
        return matrix

    # --- 3. Analytical & Software Traceability Data ---
    assay_reqs_df = reqs_df[reqs_df['type'].isin(['Assay', 'System'])].copy()
    analytical_matrix_df = create_wide_matrix(assay_reqs_df, ver_df, 'input_verified_id', 'result', 'id')

    sw_reqs_df = reqs_df[reqs_df['type'] == 'Software'].copy()
    sw_matrix_df = create_wide_matrix(sw_reqs_df, ver_df, 'input_verified_id', 'result', 'id')
    
    # --- 4. Risk Control Traceability (Hazards -> Verification) ---
    if not risk_df.empty:
        # Use a copy to avoid SettingWithCopyWarning
        risk_df_copy = risk_df.copy()
        risk_df_copy['is_risk_control'] = True # Add this column for consistent styling
        risk_matrix_df = create_wide_matrix(risk_df_copy, ver_df, 'verification_link', 'result', 'id')
        risk_matrix_df.index.names = ['id', 'description', 'is_risk_control']
    else:
        risk_matrix_df = pd.DataFrame()

    return clinical_matrix_df, analytical_matrix_df, sw_matrix_df, risk_matrix_df

def render_traceability_matrix(ssm: SessionStateManager):
    st.subheader("Multi-Level Traceability Command Center")
    st.markdown("This provides end-to-end, status-driven traceability across the entire DHF for both the **IVD assay** and the **SaMD algorithm**. This is a critical component for PMA submission and audit readiness.")

    legend_cols = st.columns(4)
    legend_cols[0].markdown("<span style='color: #1E4620; font-weight: bold;'>✅ Pass/Complete</span>", unsafe_allow_html=True)
    legend_cols[1].markdown("<span style='color: #5A0001; font-weight: bold;'>❌ Fail / Untraced</span>", unsafe_allow_html=True)
    legend_cols[2].markdown("<span style='color: #663000; font-weight: bold;'>⏳ In Progress</span>", unsafe_allow_html=True)
    legend_cols[3].markdown("<span style='background-color: #e7f3ff; font-weight: bold; padding: 2px 5px; border-radius: 3px;'>Risk Control Item</span>", unsafe_allow_html=True)
    st.divider()

    try:
        # Defensive data loading with .get() and providing default empty lists
        all_requirements = ssm.get_data("design_inputs", "requirements") or []
        all_verifications = ssm.get_data("design_verification", "tests") or []
        all_validations = ssm.get_data("clinical_study", "hf_studies") or []
        all_hazards = ssm.get_data("risk_management_file", "hazards") or []
        
        clinical_matrix, analytical_matrix, sw_matrix, risk_matrix = generate_all_trace_data(
            all_requirements,
            all_verifications,
            all_validations,
            all_hazards
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "**1. Clinical Traceability (Needs → Validation)**",
            "**2. Assay Traceability (Requirements → AV)**",
            "**3. SaMD Traceability (Requirements → SW V&V)**",
            "**4. Risk Control Traceability (Hazards → V&V)**",
            "**5. Bi-Directional Trace Explorer**"
        ])
        
        def render_kpis(matrix_df: pd.DataFrame, is_wide: bool):
            if matrix_df.empty:
                st.warning("No data available to calculate KPIs.")
                return
                
            total_items = len(matrix_df)
            if is_wide:
                traced_items = matrix_df.notna().any(axis=1).sum()
                has_pass = matrix_df.apply(lambda row: row.str.contains('✅', na=False).any(), axis=1).sum()
                passing_items = has_pass
            else:
                trace_col_name = matrix_df.columns[0]
                traced_items = matrix_df[~matrix_df[trace_col_name].str.contains("Untraced", na=False)].shape[0]
                passing_items = matrix_df[matrix_df[trace_col_name].str.contains("✅", na=False)].shape[0]
            
            trace_coverage = (traced_items / total_items * 100) if total_items > 0 else 0
            pass_coverage = (passing_items / traced_items * 100) if traced_items > 0 else 0

            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Traceability Coverage", f"{trace_coverage:.1f}%", help="Percentage of items in this view that are linked to at least one downstream artifact.")
            kpi2.metric("Items with Passing V&V", f"{passing_items} / {traced_items}", help="Of the traced items, how many are linked to at least one 'Pass' result.")
            kpi3.metric("Overall Pass Rate", f"{pass_coverage:.1f}%", help="Percentage of traced items that have a 'Pass' status.")

        with tab1:
            st.subheader("Clinical Needs vs. Clinical & Usability Validation")
            st.caption("Answers: 'Are we building the right test for patients?' This matrix links each high-level Clinical Need to the validation study that proves it is met.")
            if not clinical_matrix.empty:
                render_kpis(clinical_matrix, is_wide=False)
                st.dataframe(
                    clinical_matrix.reset_index()[['id', 'description', 'Validation Trace']].style.apply(lambda s: s.map(lambda x: style_trace_cell(x)), subset=['Validation Trace']),
                    use_container_width=True, hide_index=True,
                    column_config={"id": "Need ID", "description": "Clinical Need Description", "Validation Trace": "Validation Study (Status & ID)"}
                )
            else:
                st.warning("No 'User Need' type requirements found to build this matrix.")

        def render_wide_matrix(title: str, caption: str, matrix_df: pd.DataFrame):
            st.subheader(title)
            st.caption(caption)
            if not matrix_df.empty:
                render_kpis(matrix_df, is_wide=True)
                display_df = matrix_df.reset_index().fillna('N/A')
                display_df.rename(columns={'id_x': 'id', 'description_x': 'description'}, inplace=True, errors='ignore')
                
                trace_cols = [col for col in display_df.columns if col not in ['id', 'description', 'is_risk_control', 'risk_control_measure']]
                styler = display_df.style.apply(highlight_risk_control, axis=1)
                styler = styler.map(style_trace_cell, subset=trace_cols)

                st.dataframe(styler, use_container_width=True, hide_index=True,
                             column_config={"id": "Requirement ID", "description": "Requirement Description", "is_risk_control": None})
            else:
                st.warning("No relevant requirements or verification tests found to build this matrix.")

        with tab2:
            render_wide_matrix("Assay & System Requirements vs. Analytical Validation Protocols", "Answers: 'Did we build the assay correctly?' This matrix links each assay performance requirement to the AV protocol that verifies it.", analytical_matrix)
        with tab3:
            render_wide_matrix("Software Requirements vs. Software V&V Tests", "Answers: 'Did we build the software correctly?' This matrix links each software requirement to the test proving its implementation (Ref: ISO 62304).", sw_matrix)
        with tab4:
            st.subheader("Risk Controls vs. Verification & Validation")
            st.caption("Answers: 'Have we proven our risk mitigations are effective?' This matrix links each hazard's risk control measure to the V&V activity that proves it works.")
            if not risk_matrix.empty:
                render_kpis(risk_matrix, is_wide=True)
                display_df = risk_matrix.reset_index().fillna('N/A')
                trace_cols = [col for col in display_df.columns if col not in ['id', 'description', 'is_risk_control']]
                styler = display_df.style.map(style_trace_cell, subset=trace_cols)
                st.dataframe(styler, use_container_width=True, hide_index=True,
                            column_config={"id": "Hazard ID", "description": "Risk Control Measure Description", "is_risk_control": None})
            else:
                st.warning("No risk control measures with verification links found in the Risk Management File.")
        
        with tab5:
            st.subheader("Bi-Directional Traceability Explorer")
            st.info("Select any DHF artifact to see all its upstream and downstream connections. This is a powerful tool for impact analysis and audit preparation.")
            
            all_artifacts = []
            req_df = pd.DataFrame(all_requirements)
            ver_df = pd.DataFrame(all_verifications)
            val_df = pd.DataFrame(all_validations)
            haz_df = pd.DataFrame(all_hazards)
            out_df = pd.DataFrame(ssm.get_data("design_outputs", "documents") or [])
            
            if not req_df.empty and 'id' in req_df.columns: all_artifacts.extend([f"REQ: {id}" for id in req_df['id']])
            if not ver_df.empty and 'id' in ver_df.columns: all_artifacts.extend([f"VER: {id}" for id in ver_df['id']])
            if not val_df.empty and 'id' in val_df.columns: all_artifacts.extend([f"VAL: {id}" for id in val_df['id']])
            if not haz_df.empty and 'id' in haz_df.columns: all_artifacts.extend([f"HAZ: {id}" for id in haz_df['id']])
            if not out_df.empty and 'id' in out_df.columns: all_artifacts.extend([f"OUT: {id}" for id in out_df['id']])
            
            selected_artifact = st.selectbox("Select an artifact to trace:", sorted(list(set(all_artifacts))))

            if selected_artifact:
                art_type, art_id = selected_artifact.split(": ")
                st.write(f"#### Traceability for **{selected_artifact}**")
                
                # Upstream Traces
                with st.container(border=True):
                    st.write("##### 🔼 **Upstream Traces (What does this fulfill?)**")
                    found_upstream = False
                    if art_type == "VER" and not ver_df.empty and 'input_verified_id' in ver_df.columns and not ver_df[ver_df['id'] == art_id].empty:
                        req_id = ver_df.loc[ver_df['id'] == art_id, 'input_verified_id'].iloc[0]
                        if req_id: st.write(f"- Verifies Requirement: **REQ: {req_id}**"); found_upstream = True
                    if art_type == "VAL" and not val_df.empty and 'user_need_validated' in val_df.columns and not val_df[val_df['id'] == art_id].empty:
                        need_id = val_df.loc[val_df['id'] == art_id, 'user_need_validated'].iloc[0]
                        if need_id: st.write(f"- Validates User Need: **REQ: {need_id}**"); found_upstream = True
                    if art_type == "OUT" and not out_df.empty and 'linked_input_id' in out_df.columns and not out_df[out_df['id'] == art_id].empty:
                        req_id = out_df.loc[out_df['id'] == art_id, 'linked_input_id'].iloc[0]
                        if req_id: st.write(f"- Fulfills Requirement: **REQ: {req_id}**"); found_upstream = True
                    if art_type == "REQ" and not req_df.empty and 'parent_id' in req_df.columns and not req_df[req_df['id'] == art_id].empty:
                        parent_id = req_df.loc[req_df['id'] == art_id, 'parent_id'].iloc[0]
                        if parent_id: st.write(f"- Decomposes from Parent Requirement: **REQ: {parent_id}**"); found_upstream = True
                    if not found_upstream: st.caption("No upstream traces found.")
                
                # Downstream Traces
                with st.container(border=True):
                    st.write("##### 🔽 **Downstream Traces (How is this fulfilled?)**")
                    found_downstream = False
                    if art_type == "REQ":
                        if 'parent_id' in req_df.columns: child_reqs = req_df[req_df['parent_id'] == art_id]
                        else: child_reqs = pd.DataFrame()
                        if not child_reqs.empty: found_downstream = True; [st.write(f"- Decomposed into Requirement: **REQ: {cid}**") for cid in child_reqs['id']]
                        
                        if 'input_verified_id' in ver_df.columns: child_vers = ver_df[ver_df['input_verified_id'] == art_id]
                        else: child_vers = pd.DataFrame()
                        if not child_vers.empty: found_downstream = True; [st.write(f"- Verified by Protocol: **VER: {cid}**") for cid in child_vers['id']]
                        
                        if 'user_need_validated' in val_df.columns: child_vals = val_df[val_df['user_need_validated'] == art_id]
                        else: child_vals = pd.DataFrame()
                        if not child_vals.empty: found_downstream = True; [st.write(f"- Validated by Study: **VAL: {cid}**") for cid in child_vals['id']]
                        
                        if 'linked_input_id' in out_df.columns: child_outs = out_df[out_df['linked_input_id'] == art_id]
                        else: child_outs = pd.DataFrame()
                        if not child_outs.empty: found_downstream = True; [st.write(f"- Realized by Output: **OUT: {cid}**") for cid in child_outs['id']]
                            
                    if art_type == "HAZ" and not haz_df.empty and 'verification_link' in haz_df.columns and not ver_df.empty and not haz_df[haz_df['id'] == art_id].empty:
                        ver_link = haz_df.loc[haz_df['id'] == art_id, 'verification_link'].iloc[0]
                        if ver_link: found_downstream = True; st.write(f"- Risk Control Verified by: **VER: {ver_link}**")

                    if not found_downstream: st.caption("No downstream traces found.")

    except Exception as e:
        st.error("An error occurred while generating the traceability matrix. The data may be incomplete or malformed.")
        logger.error(f"Failed to render traceability matrix: {e}", exc_info=True)
