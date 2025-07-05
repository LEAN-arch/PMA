# --- SME OVERHAUL: Definitive, Compliance-Focused Version (PMA-Focused V4-Final) ---
"""
Renders the Design Change Control section of the DHF dashboard.

This module provides a structured UI for documenting and managing formal
design change control workflows, as required by 21 CFR 820.30(i). It ensures
a comprehensive impact assessment on both the assay and software components,
traceable linkage to validation activities, and integrated management of
action items required to implement the change.
"""

# --- Standard Library Imports ---
import logging
from typing import Any, Dict, List
from datetime import date

# --- Third-party Imports ---
import pandas as pd
import streamlit as st

# --- Local Application Imports ---
from genomicsdx.utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def render_design_changes(ssm: SessionStateManager) -> None:
    """
    Renders the UI for the Design Change Control section.

    This function displays a summary table of all Design Change Requests (DCRs)
    and provides a detailed, dossier-style interface for viewing and editing
    individual records. It enforces a compliant workflow, including structured
    impact analysis, V&V planning, and management of implementation actions.

    Args:
        ssm (SessionStateManager): The session state manager to access DHF data.
    """
    st.header("10. Design Change Control")
    st.markdown("""
    *As per 21 CFR 820.30(i) and ISO 13485:2016, Section 7.3.7.*

    This section documents the formal control of all design changes made after initial design approval.
    Each change must be identified, documented, reviewed, and approved before implementation.
    The process must include a re-assessment of risk and a determination of the effects of the change
    on the constituent parts of the diagnostic service (assay, software, kit, etc.).
    """)

    try:
        # --- 1. Load Data & Initialize State ---
        changes_data: List[Dict[str, Any]] = ssm.get_data("design_changes", "changes") or []
        team_members_data = ssm.get_data("design_plan", "team_members") or []
        owner_options = sorted([member.get('name') for member in team_members_data if member.get('name')])

        if "selected_dcr_id" not in st.session_state:
            st.session_state.selected_dcr_id = None
        if "dcr_edit_mode" not in st.session_state:
            st.session_state.dcr_edit_mode = False

        # --- 2. High-Level KPIs & DCR Log ---
        st.subheader("Design Change Control Log")
        kpi_cols = st.columns(3)
        open_dcrs = [d for d in changes_data if d.get('approval_status') not in ['Closed', 'Rejected']]
        kpi_cols[0].metric("Open DCRs", len(open_dcrs))
        kpi_cols[1].metric("Pending Approval", len([d for d in open_dcrs if d.get('approval_status') == 'Pending']))
        kpi_cols[2].metric("Total DCRs Logged", len(changes_data))
        
        if not changes_data:
            st.warning("No design change records have been logged yet.")
        else:
            changes_df = pd.DataFrame(changes_data)
            changes_df['approval_date'] = pd.to_datetime(changes_df['approval_date'], errors='coerce').dt.date
            
            st.data_editor(
                changes_df,
                column_config={
                    "id": "DCR ID",
                    "description": "Description",
                    "initiator": "Initiator",
                    "approval_status": "Status",
                    "approval_date": "Approval Date",
                    # Hide other columns from this summary view
                    "reason": None, "request_date": None, "impact_analytical": None,
                    "impact_risk": None, "vv_plan": None, "approvers": None,
                    "action_items": None, "impact_clinical": None, "impact_software": None,
                    "impact_lab_ops": None, "impact_analysis_details": None
                },
                column_order=['id', 'description', 'initiator', 'approval_status', 'approval_date'],
                use_container_width=True, hide_index=True, on_select="rerun",
                selection_mode="single-row", key="dcr_selection_table",
            )
            
            selection = st.session_state.get("dcr_selection_table", {}).get("selection", {})
            if selection.get("rows"):
                selected_index = selection["rows"][0]
                newly_selected_id = changes_df.iloc[selected_index]['id']
                if st.session_state.selected_dcr_id != newly_selected_id:
                    st.session_state.selected_dcr_id = newly_selected_id
                    st.session_state.dcr_edit_mode = False
            
        # --- 3. DCR Creation/Editing ---
        st.divider()

        if st.button("📝 Log New Design Change Request"):
            new_id = f"DCR-{len(changes_data) + 1:03d}"
            st.session_state.selected_dcr_id = new_id
            st.session_state.dcr_edit_mode = True
            new_dcr = {"id": new_id, "description": "", "reason": "", "initiator": "", "request_date": str(date.today()), "impact_clinical": False, "impact_analytical": False, "impact_software": False, "impact_lab_ops": False, "impact_risk": False, "impact_analysis_details": "", "vv_plan": "", "approval_status": "Pending", "approval_date": None, "action_items": [], "approvers": []}
            changes_data.append(new_dcr)
            ssm.update_data(changes_data, "design_changes", "changes")
            st.rerun()

        dcr_to_display = next((d for d in changes_data if d.get('id') == st.session_state.selected_dcr_id), None)
        
        if not dcr_to_display:
            st.info("Select a change from the table to view its details, or log a new change request.", icon="ℹ️")
            return
            
        # --- 4. DCR Dossier (View and Edit Modes) ---
        if st.session_state.dcr_edit_mode:
            render_dcr_edit_form(dcr_to_display, changes_data, ssm, owner_options)
        else:
            render_dcr_dossier_view(dcr_to_display)

    except Exception as e:
        st.error("An error occurred while displaying the Design Changes section. The data may be malformed.")
        logger.error(f"Failed to render design changes: {e}", exc_info=True)


def render_dcr_dossier_view(dcr: Dict[str, Any]):
    """Displays the DCR in a read-only, professional dossier format."""
    st.subheader(f"DCR Dossier: {dcr.get('id', 'N/A')}")
    
    if st.button("✏️ Edit this DCR", use_container_width=True):
        st.session_state.dcr_edit_mode = True
        st.rerun()

    st.markdown(f"**Description:** {dcr.get('description', '*Not specified*')}")
    
    meta_cols = st.columns(3)
    meta_cols[0].metric("Status", dcr.get('approval_status', 'N/A'))
    meta_cols[1].metric("Initiator", dcr.get('initiator', 'N/A'))
    meta_cols[2].metric("Request Date", str(dcr.get('request_date', 'N/A')))
    
    st.markdown("**Reason for Change:**")
    st.info(dcr.get('reason', '*Not specified*'))

    st.markdown("**Impact Analysis:**")
    impact_cols = st.columns(2)
    impact_cols[0].checkbox("Clinical/Regulatory Impact", value=dcr.get("impact_clinical", False), disabled=True)
    impact_cols[1].checkbox("Assay Performance Impact", value=dcr.get("impact_analytical", False), disabled=True)
    impact_cols[0].checkbox("Software/Algorithm Impact", value=dcr.get("impact_software", False), disabled=True)
    impact_cols[1].checkbox("Lab Operations Impact", value=dcr.get("impact_lab_ops", False), disabled=True)
    impact_cols[0].checkbox("Risk Management Impact", value=dcr.get("impact_risk", False), disabled=True)
    with st.expander("View Impact Analysis Details"):
        st.markdown(dcr.get('impact_analysis_details', '*No details provided.*'))

    st.markdown("**V&V Plan:**")
    st.info(dcr.get('vv_plan', '*Not specified*'))

    st.markdown("**Implementation Action Items:**")
    action_items = dcr.get("action_items", [])
    if action_items:
        st.dataframe(pd.DataFrame(action_items), use_container_width=True, hide_index=True)
    else:
        st.caption("No action items required for this change.")

    st.markdown("**Approval:**")
    app_cols = st.columns(2)
    app_cols[0].markdown(f"**Approval Date:** {dcr.get('approval_date', 'N/A')}")
    app_cols[1].markdown(f"**Approvers:** {', '.join(dcr.get('approvers', [])) or 'N/A'}")


def render_dcr_edit_form(dcr: Dict[str, Any], all_dcrs: List[Dict[str, Any]], ssm: SessionStateManager, owner_options: List[str]):
    """Renders the form for editing a DCR."""
    st.subheader(f"Editing DCR: {dcr.get('id', 'N/A')}")

    with st.form(key=f"dcr_form_{dcr.get('id')}"):
        dcr_id = dcr.get("id")
        description = st.text_area("**Change Description**", value=dcr.get("description", ""), height=100)
        reason = st.text_area("**Reason for Change**", value=dcr.get("reason", ""), height=100)
        initiator = st.selectbox("**Initiator**", options=owner_options, index=owner_options.index(dcr.get("initiator")) if dcr.get("initiator") in owner_options else 0)

        st.markdown("**Impact Analysis (Required)**")
        impact_cols = st.columns(2)
        impact_clinical = impact_cols[0].checkbox("Clinical/Regulatory Impact", value=dcr.get("impact_clinical", False))
        impact_analytical = impact_cols[1].checkbox("Assay Performance Impact", value=dcr.get("impact_analytical", False))
        impact_software = impact_cols[0].checkbox("Software/Algorithm Impact", value=dcr.get("impact_software", False))
        impact_lab_ops = impact_cols[1].checkbox("Lab Operations Impact", value=dcr.get("impact_lab_ops", False))
        impact_risk = impact_cols[0].checkbox("Risk Management Impact", value=dcr.get("impact_risk", False))
        impact_analysis_details = st.text_area("Impact Analysis Details", value=dcr.get("impact_analysis_details", ""), height=150)

        st.markdown("**Required Verification & Validation Activities**")
        vv_plan = st.text_area("V&V Plan", value=dcr.get("vv_plan", ""), height=150)

        st.markdown("**Implementation Action Items**")
        action_items_df = pd.DataFrame(dcr.get("action_items", []))
        edited_actions_df = st.data_editor(
            action_items_df, num_rows="dynamic", use_container_width=True, key=f"dcr_action_editor_{dcr_id}", hide_index=True,
            column_config={
                "id": st.column_config.TextColumn("ID", required=True), "description": st.column_config.TextColumn("Action Description", required=True, width="large"),
                "owner": st.column_config.SelectboxColumn("Owner", options=owner_options, required=True), "due_date": st.column_config.DateColumn("Due Date", format="YYYY-MM-DD", required=True),
                "status": st.column_config.SelectboxColumn("Status", options=["Open", "In Progress", "Completed"], required=True)
            }
        )
        
        st.markdown("**Approval**")
        approval_cols = st.columns(3)
        status_options = ["Pending", "Approved", "Rejected", "Implementation Pending", "Closed"]
        current_status = dcr.get("approval_status", "Pending")
        approval_status = approval_cols[0].selectbox("Approval Status", options=status_options, index=status_options.index(current_status))
        
        approval_date_val = pd.to_datetime(dcr.get("approval_date"), errors='coerce')
        approval_date = approval_cols[1].date_input("Approval Date", value=approval_date_val if pd.notna(approval_date_val) else None)
        
        approvers = approval_cols[2].multiselect("Approvers", options=owner_options, default=dcr.get("approvers", []))
        
        submit_cols = st.columns(2)
        if submit_cols[0].form_submit_button("✅ Save & Exit Edit Mode", use_container_width=True, type="primary"):
            updated_dcr_data = dcr.copy()
            updated_dcr_data.update({
                "description": description, "reason": reason, "initiator": initiator, "impact_clinical": impact_clinical,
                "impact_analytical": impact_analytical, "impact_software": impact_software, "impact_lab_ops": impact_lab_ops,
                "impact_risk": impact_risk, "impact_analysis_details": impact_analysis_details, "vv_plan": vv_plan,
                "approval_status": approval_status, "approval_date": str(approval_date) if approval_date else None,
                "approvers": approvers, "action_items": edited_actions_df.to_dict('records')
            })

            dcr_index = next((i for i, item in enumerate(all_dcrs) if item["id"] == dcr_id), None)
            if dcr_index is not None:
                all_dcrs[dcr_index] = updated_dcr_data
            
            ssm.update_data(all_dcrs, "design_changes", "changes")
            logger.info(f"Design change record '{dcr_id}' saved/updated.")
            st.toast(f"DCR '{dcr_id}' saved successfully!", icon="✅")
            st.session_state.dcr_edit_mode = False
            st.rerun()

        if submit_cols[1].form_submit_button("❌ Cancel", use_container_width=True):
            st.session_state.dcr_edit_mode = False
            if not dcr.get('description'):
                all_dcrs.pop()
                ssm.update_data(all_dcrs, "design_changes", "changes")
                st.session_state.selected_dcr_id = None
            st.rerun()
