# --- SME OVERHAUL: Definitive, Compliance-Focused Version (PMA-Focused V4-Final) ---
"""
Renders the Design Inputs section of the DHF dashboard.

This module provides a structured, hierarchical UI for managing all product
requirements, which serve as the foundation for the DHF. It categorizes inputs
into clinical (User Needs), assay performance, and software (SaMD) domains,
as required for a complex genomic diagnostic service under 21 CFR 820.30(c).
It includes interactive tools for visualizing traceability and analyzing completeness.
"""

# --- Standard Library Imports ---
import logging
from typing import Any, Dict, List

# --- Third-party Imports ---
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# --- Local Application Imports ---
from genomicsdx.utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)

def create_requirements_tree(df: pd.DataFrame) -> go.Figure:
    """Creates an interactive sunburst chart to visualize requirement hierarchy."""
    if df.empty or 'parent_id' not in df.columns:
        return go.Figure()

    df_copy = df.copy()
    df_copy['parent_id'] = df_copy['parent_id'].fillna('') # Ensure root nodes have a blank parent
    
    fig = go.Figure(go.Sunburst(
        ids=df_copy.id,
        labels=df_copy.id,
        parents=df_copy.parent_id,
        hovertext=df_copy.description,
        hovertemplate='<b>%{label}</b><br>%{hovertext}<extra></extra>',
        branchvalues="total",
        marker=dict(colors=px.colors.qualitative.Plotly)
    ))
    fig.update_layout(
        margin=dict(t=20, l=20, r=20, b=20),
        title_text="<b>Interactive Requirements Hierarchy</b>",
        title_x=0.5,
        height=500
    )
    return fig

def render_design_inputs(ssm: SessionStateManager) -> None:
    """
    Renders the hierarchical UI for managing all Design Inputs.

    This function displays requirements in categorized tabs for clarity and
    manages traceability between requirement levels (e.g., System -> Software)
    and to risk controls. It also provides analytics on requirement health.
    """
    st.header("4. Design Inputs (Requirements)")
    st.markdown("""
    *As per 21 CFR 820.30(c), ISO 13485:2016 Section 7.3.3, and ISO 14971.*

    This section captures all requirements for the diagnostic service. This includes high-level **User Needs**, which are decomposed into testable **System, Assay, and Software Requirements**. Requirements derived from risk analysis (**Risk Controls**) are also managed here. Each requirement must be unambiguous, verifiable, and traceable.
    """)
    st.info("Use the tabs below to manage requirements for each component of the system. The traceability tree and health dashboard provide a live analysis of completeness and gaps.", icon="ℹ️")

    try:
        # --- 1. Load All Necessary Data ---
        inputs_data: List[Dict[str, Any]] = ssm.get_data("design_inputs", "requirements") or []
        df_all = pd.DataFrame(inputs_data) if inputs_data else pd.DataFrame(columns=['id', 'type', 'description', 'parent_id', 'status', 'is_risk_control', 'related_hazard_id'])
        
        hazards: List[Dict[str, Any]] = ssm.get_data("risk_management_file", "hazards") or []
        hazard_ids: List[str] = [""] + sorted([h.get('id', '') for h in hazards if h.get('id')])
        
        verifications = ssm.get_data("design_verification", "tests") or []
        verified_req_ids = {v['input_verified_id'] for v in verifications if v.get('input_verified_id')}

        # --- 2. Requirements Health & Traceability Dashboard ---
        st.subheader("Requirements Health & Traceability")
        kpi_cols = st.columns(2)
        
        # KPI 1: Untraced Requirements
        untraced_reqs = pd.DataFrame()
        if not df_all.empty and 'type' in df_all.columns:
            non_user_needs = df_all[df_all['type'] != 'User Need']
            untraced_reqs = non_user_needs[non_user_needs['parent_id'].fillna('').eq('')]
        kpi_cols[0].metric("Untraced Requirements", len(untraced_reqs), help="Requirements that do not trace back to a parent User Need or System Requirement. These are critical gaps.")
        
        # KPI 2: Unverified Requirements
        unverified_reqs = pd.DataFrame()
        if not df_all.empty and 'id' in df_all.columns:
            verifiable_reqs = df_all[df_all['type'] != 'User Need']
            unverified_reqs = verifiable_reqs[~verifiable_reqs['id'].isin(verified_req_ids)]
        kpi_cols[1].metric("Unverified Requirements", len(unverified_reqs), help="Requirements not yet covered by a verification test. These must be addressed before design freeze.")

        with st.expander("View Interactive Requirements Hierarchy Tree & Gap Analysis Tables"):
            if not df_all.empty:
                st.plotly_chart(create_requirements_tree(df_all), use_container_width=True)
                
                analysis_cols = st.columns(2)
                with analysis_cols[0]:
                    st.write("🔴 **Untraced Requirements**")
                    st.dataframe(untraced_reqs[['id', 'type', 'description']], use_container_width=True, hide_index=True)
                with analysis_cols[1]:
                    st.write("🔴 **Unverified Requirements**")
                    st.dataframe(unverified_reqs[['id', 'type', 'description']], use_container_width=True, hide_index=True)
            else:
                st.caption("No requirements data to display.")
        
        st.divider()

        # --- 3. Define Tabs for Each Requirement Category ---
        tab_titles = [
            "User Needs & Intended Use",
            "System & Assay Requirements",
            "Software (SaMD) Requirements",
            "Sample Kit & Labeling Requirements"
        ]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        def render_editor_tab(title: str, df: pd.DataFrame, key: str, column_config: Dict, help_text: str):
            st.subheader(title)
            st.caption(help_text)

            original_tab_data = df.to_dict('records')
            edited_tab_df = st.data_editor(
                df, num_rows="dynamic", use_container_width=True, key=key,
                column_config=column_config, hide_index=True
            )

            if edited_tab_df.to_dict('records') != original_tab_data:
                other_reqs = [req for req in inputs_data if req.get('type') not in df['type'].unique()]
                updated_all_reqs = other_reqs + edited_tab_df.to_dict('records')
                ssm.update_data(updated_all_reqs, "design_inputs", "requirements")
                logger.info(f"Design inputs for '{title}' updated.")
                st.toast(f"{title} saved!", icon="✅")
                st.rerun()

        # --- 4. Render Each Tab with Context-Aware Dropdowns ---
        user_need_ids = [""] + sorted(list(df_all[df_all['type'] == 'User Need']['id'].unique()))
        system_req_ids = [""] + sorted(list(df_all[df_all['type'].isin(['System', 'Assay'])]['id'].unique()))

        with tab1: # User Needs
            un_df = df_all[df_all['type'] == 'User Need'].copy()
            render_editor_tab("User Needs & Intended Use", un_df, "un_editor", {
                "id": st.column_config.TextColumn("ID", help="E.g., UN-001", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=["User Need"], default="User Need", required=True),
                "description": st.column_config.TextColumn("Description", width="large", help="High-level need from a user perspective.", required=True),
                "status": st.column_config.SelectboxColumn("Status", options=["Active", "Obsolete"], default="Active", required=True),
                "parent_id": None, "is_risk_control": None, "related_hazard_id": None
            }, "Define the highest-level clinical and user needs. These are the inputs to the Design Validation process.")

        with tab2: # System & Assay Requirements
            sa_df = df_all[df_all['type'].isin(['System', 'Assay'])].copy()
            render_editor_tab("System & Assay Requirements", sa_df, "sa_editor", {
                "id": st.column_config.TextColumn("ID", help="E.g., SYS-001, ASY-001", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=["System", "Assay"], required=True),
                "description": st.column_config.TextColumn("Description", width="large", help="Testable performance requirement for the assay or overall system.", required=True),
                "parent_id": st.column_config.SelectboxColumn("Traces to User Need", options=user_need_ids, help="Which User Need does this requirement satisfy?"),
                "status": st.column_config.SelectboxColumn("Status", options=["Active", "Obsolete"], default="Active", required=True),
                "is_risk_control": st.column_config.CheckboxColumn("Is Risk Control?", default=False, help="Is this requirement a mitigation for a hazard?"),
                "related_hazard_id": st.column_config.SelectboxColumn("Mitigates Hazard ID", options=hazard_ids, help="Which hazard from the RMF does this control mitigate?")
            }, "Define the top-level, quantitative performance requirements for the entire service and the core assay. These are the inputs to the Analytical Validation process.")

        with tab3: # Software Requirements
            sw_df = df_all[df_all['type'] == 'Software'].copy()
            render_editor_tab("Software (SaMD) Requirements", sw_df, "sw_editor", {
                "id": st.column_config.TextColumn("ID", help="E.g., SW-001", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=["Software"], default="Software", required=True),
                "description": st.column_config.TextColumn("Description", width="large", help="Specific, testable software function.", required=True),
                "parent_id": st.column_config.SelectboxColumn("Traces to System Req.", options=system_req_ids, help="Which System/Assay requirement does this software function support?"),
                "status": st.column_config.SelectboxColumn("Status", options=["Active", "Obsolete"], default="Active", required=True),
                "is_risk_control": st.column_config.CheckboxColumn("Is Risk Control?", default=False),
                "related_hazard_id": st.column_config.SelectboxColumn("Mitigates Hazard ID", options=hazard_ids)
            }, "Define the detailed requirements for the bioinformatics pipeline, classifier, LIMS interface, and reporting tool. Governed by ISO 62304.")

        with tab4: # Kit & Labeling Requirements
            kl_df = df_all[df_all['type'].isin(['Kit', 'Labeling'])].copy()
            render_editor_tab("Sample Kit & Labeling Requirements", kl_df, "kl_editor", {
                "id": st.column_config.TextColumn("ID", help="E.g., KIT-001, LBL-001", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=["Kit", "Labeling"], required=True),
                "description": st.column_config.TextColumn("Description", width="large", help="Requirement for physical kit or user-facing text.", required=True),
                "parent_id": st.column_config.SelectboxColumn("Traces to User Need", options=user_need_ids, help="Which User Need does this support?"),
                "status": st.column_config.SelectboxColumn("Status", options=["Active", "Obsolete"], default="Active", required=True),
                "is_risk_control": st.column_config.CheckboxColumn("Is Risk Control?", default=False),
                "related_hazard_id": st.column_config.SelectboxColumn("Mitigates Hazard ID", options=hazard_ids)
            }, "Define requirements for the physical sample collection kit (e.g., tube, packaging) and all labeling (e.g., IFU, Report Text). Governed by 21 CFR 809.10 and Human Factors Engineering.")

    except Exception as e:
        st.error("An error occurred while displaying the Design Inputs section. The data may be malformed.")
        logger.error(f"Failed to render design inputs: {e}", exc_info=True)
