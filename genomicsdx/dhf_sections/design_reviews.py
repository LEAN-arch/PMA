# --- SME OVERHAUL: Definitive, Compliance-Focused Version (PMA-Focused V4-Final) ---
"""
Module for rendering the Design Reviews section of the DHF Explorer.

This component provides a structured, formal interface to document and manage
records from phase-gate, technical, and other formal design reviews, as
required by 21 CFR 820.30(e). It includes analytics on action items and a
feature to generate formal meeting minutes for key milestones like assay
design freeze and algorithm lock.
"""

# --- Standard Library Imports ---
import logging
from typing import Dict, List, Any
from datetime import date

# --- Third-party Imports ---
import pandas as pd
import streamlit as st
import plotly.express as px

# --- Local Application Imports ---
from ..utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)

def generate_meeting_minutes(review: Dict[str, Any]) -> str:
    """Generates a formatted string for meeting minutes from a review dictionary."""
    minutes = f"# Design Review Minutes: {review.get('id', 'N/A')} - {review.get('type')} Review\n\n"
    minutes += f"## {review.get('phase', 'N/A')}\n\n"
    minutes += f"**Date:** {review.get('date')}\n"
    minutes += f"**Independent Reviewer:** {review.get('independent_reviewer')}\n"
    minutes += f"**Attendees:**\n"
    for attendee in review.get('attendees', []):
        minutes += f"- {attendee}\n"
    
    minutes += f"\n---\n\n"
    minutes += f"### Scope & Purpose\n{review.get('scope', 'Not documented.')}\n\n"
    minutes += f"### Documents Reviewed\n"
    docs_reviewed = review.get('documents_reviewed', [])
    if docs_reviewed:
        for doc in docs_reviewed:
            minutes += f"- {doc}\n"
    else:
        minutes += "None specified.\n"
        
    minutes += f"\n### Summary & Notes\n{review.get('notes', 'Not documented.')}\n\n"
    minutes += f"### Outcome\n**{review.get('outcome')}**\n\n"
    
    minutes += f"---\n\n"
    minutes += f"### Action Items\n"
    action_items = review.get('action_items', [])
    if not action_items:
        minutes += "None.\n"
    else:
        df = pd.DataFrame(action_items)
        try:
            minutes += df.to_markdown(index=False)
        except ImportError:
            logger.warning("`tabulate` package not found. Falling back to plain text table for meeting minutes.")
            minutes += df.to_string(index=False)
        
    return minutes

def render_design_reviews(ssm: SessionStateManager) -> None:
    """
    Renders an editable view of all formal design review records.
    
    This function displays each design review in a separate, structured container,
    allowing for focused viewing and editing. It enforces the documentation of

    key compliance elements and provides analytics on generated action items.
    """
    st.header("6. Design Reviews")
    st.markdown("""
    *As per 21 CFR 820.30(e) and ISO 13485:2016, Section 7.3.5.*

    Document and track formal reviews of the design at appropriate, planned stages of the project lifecycle (e.g., Assay Freeze, Algorithm Lock). Each review must include an individual who does not have direct responsibility for the design stage being reviewed, as well as any specialists needed. The results, including the design review date, attendees, and any action items, must be documented in the DHF.
    """)
    
    try:
        # --- 1. Load Data and Prepare Dependencies ---
        reviews = ssm.get_data("design_reviews", "reviews") or []
        team_members_data = ssm.get_data("design_plan", "team_members") or []
        owner_options = sorted([member.get('name') for member in team_members_data if member.get('name')])
        all_outputs = ssm.get_data("design_outputs", "documents") or []
        output_options = sorted([f"{o.get('id')}: {o.get('title')}" for o in all_outputs])
        
        # --- 2. High-Level Analytics ---
        st.subheader("Action Item Health Dashboard (from all Reviews)")
        all_actions = [item for r in reviews for item in r.get("action_items", [])]
        if all_actions:
            df_actions = pd.DataFrame(all_actions)
            if 'due_date' in df_actions.columns:
                now = pd.to_datetime(date.today())
                df_actions['due_date'] = pd.to_datetime(df_actions['due_date'], errors='coerce')
                df_actions.loc[(df_actions['due_date'] < now) & (df_actions['status'] != 'Completed'), 'status'] = 'Overdue'
            
            status_counts = df_actions['status'].value_counts() if 'status' in df_actions.columns else pd.Series()
            
            kpi_cols = st.columns(4)
            overdue_count = int(status_counts.get("Overdue", 0))
            kpi_cols[0].metric("Total Actions", len(df_actions))
            kpi_cols[1].metric("Completed", int(status_counts.get("Completed", 0)))
            kpi_cols[2].metric("Open", int(status_counts.get("Open", 0) + status_counts.get("In Progress", 0)))
            kpi_cols[3].metric("Overdue", overdue_count, delta=overdue_count, delta_color="inverse")
            
            if not status_counts.empty:
                fig = px.pie(df_actions, names='status', title='Action Items by Status',
                             color='status', color_discrete_map={'Completed':'green', 'Open':'orange', 'In Progress':'blue', 'Overdue':'red'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No action items have been generated from design reviews yet.")

        st.divider()
        st.info("Select a review from the dropdown to view its details, or log a new review record.", icon="ℹ️")

        reviews.sort(key=lambda r: pd.to_datetime(r.get('date', '1900-01-01')), reverse=True)

        if not reviews:
            st.warning("No design review records found.")
        else:
            review_options = {f"{pd.to_datetime(r.get('date')).strftime('%Y-%m-%d')}: {r.get('type', 'Review')} ({r.get('id')})": r.get('id') for r in reviews}
            selected_review_display = st.selectbox("**Select a Design Review to View/Edit**", options=review_options.keys())
            if selected_review_display:
                selected_review_id = review_options[selected_review_display]
                review = next((r for r in reviews if r.get('id') == selected_review_id), None)
                
                if review:
                    with st.container(border=True):
                        render_review_form(review, reviews, ssm, owner_options, output_options)

        if st.button("📝 Log New Design Review", use_container_width=True):
            new_id = f"DR-{len(reviews) + 1:03d}"
            new_review = {
                "id": new_id, "date": str(date.today()), "type": "Technical", "phase": "N/A",
                "attendees": [], "independent_reviewer": "", "scope": "", "outcome": "Pending", "notes": "", "documents_reviewed": [], "action_items": []
            }
            reviews.insert(0, new_review)
            ssm.update_data(reviews, "design_reviews", "reviews")
            st.success(f"Created new draft review {new_id}. Select it from the dropdown to edit.")
            st.rerun()

    except Exception as e:
        st.error("An error occurred while displaying the Design Reviews section. The data may be malformed.")
        logger.error(f"Failed to render design reviews: {e}", exc_info=True)

def render_review_form(review: Dict, all_reviews: List[Dict], ssm: SessionStateManager, owner_options: List, output_options: List):
    """Renders the form for a single design review."""
    with st.form(key=f"review_form_{review.get('id')}"):
        st.subheader(f"Design Review Record: {review.get('id')}")

        cols = st.columns(3)
        date_val = cols[0].date_input("**Date**", value=pd.to_datetime(review.get('date')))
        type_options = ["Phase-Gate", "Technical", "Risk", "Software", "Assay", "Usability"]
        current_type = review.get('type', 'Technical')
        type_val = cols[1].selectbox("**Review Type**", options=type_options, index=type_options.index(current_type) if current_type in type_options else 0)
        phase_val = cols[2].text_input("**Project Phase Reviewed**", value=review.get('phase', ''), help="E.g., 'Assay Freeze'")
        
        scope_val = st.text_area("**Scope & Purpose**", value=review.get('scope', ''), height=100)
        docs_val = st.multiselect("**Documents Reviewed**", options=output_options, default=review.get('documents_reviewed', []))

        st.markdown("**Attendees & Review Outcome**")
        att_cols = st.columns(3)
        attendees_val = att_cols[0].multiselect("**Attendees**", options=owner_options, default=review.get('attendees', []))
        
        current_reviewer = review.get('independent_reviewer', '')
        reviewer_index = owner_options.index(current_reviewer) if current_reviewer in owner_options else 0
        independent_reviewer_val = att_cols[1].selectbox("**Independent Reviewer (Required)**", options=owner_options, index=reviewer_index)
        
        current_outcome = review.get('outcome', 'Pending')
        outcome_options = ["Go", "Go with Conditions", "No-Go", "Pending"]
        outcome_val = att_cols[2].selectbox("**Formal Outcome**", options=outcome_options, index=outcome_options.index(current_outcome) if current_outcome in outcome_options else 3)
        
        notes_val = st.text_area("**Summary & Notes**", value=review.get('notes', ''), height=150)

        st.markdown("**Action Items**")
        action_items_df = pd.DataFrame(review.get("action_items", []))
        if 'due_date' in action_items_df.columns:
            action_items_df['due_date'] = pd.to_datetime(action_items_df['due_date'])
        edited_actions_df = st.data_editor(
            action_items_df, num_rows="dynamic", use_container_width=True, key=f"action_item_editor_{review['id']}",
            column_config={
                "id": st.column_config.TextColumn("ID", required=True), "description": st.column_config.TextColumn("Description", width="large", required=True),
                "owner": st.column_config.SelectboxColumn("Owner", options=owner_options, required=True), "due_date": st.column_config.DateColumn("Due Date", format="YYYY-MM-DD", required=True),
                "status": st.column_config.SelectboxColumn("Status", options=["Open", "In Progress", "Completed"], required=True),
                "risk_priority": st.column_config.SelectboxColumn("Priority", options=["High", "Medium", "Low"], default="Medium", required=True)
            }, hide_index=True,
        )
        
        form_cols = st.columns([3,1])
        with form_cols[0]:
            submitted = st.form_submit_button(f"💾 Save Changes for Review {review.get('id')}", use_container_width=True, type="primary")
        with form_cols[1]:
            st.download_button(
                label="📄 Export Minutes",
                data=generate_meeting_minutes(review),
                file_name=f"Minutes_{review.get('id')}_{review.get('date')}.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"download_minutes_{review.get('id')}"
            )
        
        if submitted:
            edited_actions_list = edited_actions_df.to_dict('records')
            for item in edited_actions_list:
                item['due_date'] = str(pd.to_datetime(item['due_date']).date()) if pd.notna(item['due_date']) else None
            
            review_index = next((i for i, r in enumerate(all_reviews) if r.get('id') == review.get('id')), None)
            if review_index is not None:
                all_reviews[review_index] = {
                    "id": review.get('id'), "date": str(date_val), "type": type_val, "phase": phase_val,
                    "scope": scope_val, "documents_reviewed": docs_val, "attendees": attendees_val,
                    "independent_reviewer": independent_reviewer_val, "outcome": outcome_val,
                    "notes": notes_val, "action_items": edited_actions_list
                }
                ssm.update_data(all_reviews, "design_reviews", "reviews")
                st.toast(f"Design Review {review.get('id')} updated!", icon="✅")
                st.rerun()
