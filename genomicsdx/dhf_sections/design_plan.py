# --- SME OVERHAUL: Definitive, Compliance-Focused Version (PMA-Focused V4-Final) ---
"""
Renders the Design and Development Plan section of the DHF dashboard.

This module provides a comprehensive, visually-driven UI for creating and editing
the foundational planning document for the MCED diagnostic program, as required
by 21 CFR 820.30(b). It explicitly outlines the parallel development tracks for
the physical IVD assay and the software (SaMD) components.
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

def create_org_chart(team_df: pd.DataFrame) -> go.Figure:
    """Creates a hierarchical organization chart."""
    if team_df.empty or 'role' not in team_df.columns or 'name' not in team_df.columns:
        return go.Figure()
    
    # A simple hierarchy based on keywords in roles
    parents = []
    for role in team_df['role']:
        if 'Lead' in role and 'Program Lead' not in role:
            parents.append("Program Lead")
        elif 'Program Lead' in role:
            parents.append("") # Root node
        else:
            # Assign team members to their respective leads
            if "Assay" in role: parents.append("Assay Development Lead")
            elif "Bioinformatics" in role: parents.append("Bioinformatics Lead")
            elif "Clinical" in role: parents.append("Clinical Lead")
            else: parents.append("Program Lead") # Default case

    fig = go.Figure(go.Treemap(
        ids=team_df['role'],
        labels=team_df.apply(lambda x: f"<b>{x['role']}</b><br>{x['name']}", axis=1),
        parents=parents,
        root_color="lightgrey",
        hovertemplate='<b>%{label}</b><br>Responsibility: %{customdata}<extra></extra>',
        customdata=team_df['responsibility'],
        textinfo="label",
        pathbar=dict(visible=True)
    ))
    fig.update_layout(
        margin=dict(t=20, l=10, r=10, b=10),
        title_text="<b>Project Organization Chart</b>",
        title_x=0.5,
        height=400
    )
    return fig

def render_design_plan(ssm: SessionStateManager) -> None:
    """
    Renders the UI for the Design and Development Plan section.
    """
    st.header("1. Design and Development Plan")
    st.markdown("""
    *As per 21 CFR 820.30(b) and ISO 13485:2016, Section 7.3.*

    This is the master plan that describes and references all design and development activities for the complete diagnostic service. It outlines the project scope, team, regulatory pathway, and the strategies for proving the safety and effectiveness of both the **IVD assay** and the **SaMD algorithm**. This living document will be updated as the project evolves.
    """)
    st.info("Use the sections below to manage the plan. Use the 'Save Plan Changes' button at the bottom to commit all edits.", icon="📝")
    st.divider()

    try:
        # --- 1. Load Data ---
        plan_data: Dict[str, Any] = ssm.get_data("design_plan")
        logger.info("Loaded design plan data.")

        # --- 2. Render UI Sections within a single form for atomic updates ---
        with st.form("design_plan_form"):
            
            # --- Project Overview & Strategy ---
            st.subheader("Project Charter & Strategy")
            col1, col2 = st.columns(2)
            with col1:
                project_name_val = st.text_input("**Project Name**", value=plan_data.get("project_name", ""))
                intended_use_val = st.text_area("**Intended Use Statement**", value=plan_data.get("intended_use", ""), height=200, help="This is a formal regulatory statement.")
            with col2:
                scope_val = st.text_area("**Project Scope**", value=plan_data.get("scope", ""), height=300, help="Define the boundaries of the project, including all components of the service.")

            # --- Team & Phasing Visualizations ---
            st.subheader("Project Organization & Phasing")
            vis_col1, vis_col2 = st.columns(2)
            with vis_col1:
                team_df_vis = pd.DataFrame(plan_data.get("team_members", []))
                st.plotly_chart(create_org_chart(team_df_vis), use_container_width=True)
            with vis_col2:
                reviews_df_vis = pd.DataFrame(plan_data.get("design_review_plan", []))
                if not reviews_df_vis.empty:
                    reviews_df_vis['planned_date'] = pd.to_datetime(reviews_df_vis['planned_date'])
                    reviews_df_vis['end_date'] = reviews_df_vis['planned_date'] + pd.Timedelta(days=5)
                    fig = px.timeline(reviews_df_vis, x_start="planned_date", x_end="end_date", y="phase_name", title="<b>Planned Project Phase Gates</b>")
                    fig.update_yaxes(autorange="reversed")
                    fig.update_layout(height=400, margin=dict(t=50, l=10, r=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

            # --- Detailed Planning Editors ---
            st.subheader("Plan Details & Configuration")
            with st.expander("Expand to Edit Detailed Plan Sections"):
                # --- Regulatory & Compliance ---
                st.markdown("**Regulatory Strategy & Compliance Framework**")
                reg_cols = st.columns(3)
                reg_cols[0].text_input("**Device Classification**", value="Class III", disabled=True)
                reg_cols[1].text_input("**Submission Pathway**", value="Premarket Approval (PMA)", disabled=True)
                reg_cols[2].text_input("**Special Designations**", value="Breakthrough Device Designation", disabled=True)
                
                st.markdown("**Applicable Standards & Guidance Documents**")
                standards_df = pd.DataFrame(plan_data.get("standards", []))
                edited_standards_df = st.data_editor(
                    standards_df, num_rows="dynamic", use_container_width=True, key="dp_standards_editor",
                    column_config={
                        "id": st.column_config.TextColumn("Document ID", required=True),
                        "title": st.column_config.TextColumn("Title/Description", required=True, width="large"),
                        "category": st.column_config.SelectboxColumn("Category", options=["Regulation", "ISO Standard", "IEC Standard", "FDA Guidance", "CLSI Guideline"], required=True)
                    }
                )

                # --- V&V and Other Plans ---
                st.markdown("**Development, V&V, and Lifecycle Plans**")
                plan_cols = st.columns(2)
                clinical_dev_plan_val = plan_cols[0].text_area("**Clinical Strategy Summary**", value=plan_data.get("clinical_dev_plan", ""), height=150)
                av_master_plan_ref_val = plan_cols[1].text_input("**Assay AV Master Plan Document ID**", value=plan_data.get("av_master_plan_ref", ""))
                
                loc_options = ["Major", "Moderate", "Minor"]
                current_loc = plan_data.get("software_level_of_concern", "Major")
                loc_index = loc_options.index(current_loc) if current_loc in loc_options else 0
                software_level_of_concern_val = plan_cols[0].selectbox("**Software Level of Concern (per FDA Guidance):**", options=loc_options, index=loc_index)
                sw_dev_plan_val = plan_cols[0].text_area("**Bioinformatics & SaMD Development Plan Summary**", value=plan_data.get("sw_dev_plan", ""), height=150)

                plan_cols[1].text_input("**Risk Management Plan Document ID**", value=plan_data.get("risk_management_plan_ref", "RMP-001"))
                plan_cols[1].text_input("**Human Factors Plan Document ID**", value=plan_data.get("human_factors_plan_ref", "HFE-PLAN-001"))
                plan_cols[1].text_input("**Configuration Management Plan Document ID**", value=plan_data.get("config_management_plan_ref", "CM-PLAN-001"))

                # --- Design Review Plan Editor ---
                st.markdown("**Design Review Plan**")
                reviews_df = pd.DataFrame(plan_data.get("design_review_plan", []))
                if 'planned_date' in reviews_df.columns:
                    reviews_df['planned_date'] = pd.to_datetime(reviews_df['planned_date'], errors='coerce')
                edited_reviews_df = st.data_editor(
                    reviews_df, num_rows="dynamic", use_container_width=True, key="dp_reviews_editor",
                    column_config={
                        "phase_name": st.column_config.TextColumn("Phase/Gate Name", required=True),
                        "description": st.column_config.TextColumn("Description", width="large"),
                        "planned_date": st.column_config.DateColumn("Planned Date", format="YYYY-MM-DD")
                    }
                )

                # --- Team Editor ---
                st.markdown("**Team and Responsibilities**")
                team_df = pd.DataFrame(plan_data.get("team_members", []))
                edited_team_df = st.data_editor(
                    team_df, num_rows="dynamic", use_container_width=True, key="design_plan_team_editor",
                    column_config={
                        "role": st.column_config.TextColumn("Role", required=True), "name": st.column_config.TextColumn("Assigned Member", required=True),
                        "responsibility": st.column_config.TextColumn("Key Responsibilities", width="large"),
                    }, hide_index=True
                )
            
            # --- Save Button ---
            submitted = st.form_submit_button("Save All Plan Changes", use_container_width=True, type="primary")
            if submitted:
                # Aggregate all changes into the plan_data dictionary
                plan_data.update({
                    "project_name": project_name_val, "scope": scope_val, "intended_use": intended_use_val,
                    "standards": edited_standards_df.to_dict('records'), "clinical_dev_plan": clinical_dev_plan_val,
                    "av_master_plan_ref": av_master_plan_ref_val, "software_level_of_concern": software_level_of_concern_val,
                    "sw_dev_plan": sw_dev_plan_val, "team_members": edited_team_df.to_dict('records')
                })
                # Handle date conversion for review plan
                reviews_to_save_df = edited_reviews_df.copy()
                if 'planned_date' in reviews_to_save_df.columns:
                     reviews_to_save_df['planned_date'] = reviews_to_save_df['planned_date'].dt.strftime('%Y-%m-%d').replace({pd.NaT: None})
                plan_data["design_review_plan"] = reviews_to_save_df.to_dict('records')
                
                # Update session state
                ssm.update_data(plan_data, "design_plan")
                logger.info("Design plan data updated in session state.")
                st.toast("Design & Development Plan saved!", icon="✅")
                st.rerun()

    except Exception as e:
        st.error("An error occurred while displaying the Design Plan section. The data may be malformed.")
        logger.error(f"Failed to render design plan: {e}", exc_info=True)
