# --- SME OVERHAUL: Definitive, Compliance-Focused Version (PMA-Focused V4-Final) ---
"""
Renders the centralized Compliance Action & CAPA Management Tool.

This module provides the logic for aggregating all compliance-related actions
from across the DHF (e.g., from Design Reviews, Audits, Complaints, CAPAs)
into a single, interactive, and risk-prioritized view. This tool is designed
to meet the requirements of 21 CFR 820.100 (CAPA) and ISO 13485, providing
an auditable, "single source of truth" for all corrective and preventive actions.
"""

# --- Standard Library Imports ---
import logging
from datetime import datetime
from typing import Any, Dict, List

# --- Third-party Imports ---
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- Local Application Imports ---
from genomicsdx.utils.session_state_manager import SessionStateManager
from genomicsdx.utils.plot_utils import create_pareto_chart

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def render_action_item_tracker(ssm: SessionStateManager) -> None:
    """
    Aggregates, analyzes, and displays all compliance actions from the QMS.

    This function fetches actions from various sources (Design Reviews, Changes,
    CAPAs, Audits, Complaints, Non-conformances), enriches the data (e.g.,
    identifies overdue items), displays risk-based KPIs and Pareto charts,
    and provides an interactive, filterable table for project-wide oversight
    and audit readiness.

    Args:
        ssm (SessionStateManager): The session state manager to access DHF data.
    """
    st.subheader("Centralized Compliance Action & CAPA Tracker")
    st.markdown("This table consolidates all action items from Design Reviews, Change Controls, CAPAs, Non-Conformances, and Audits for project-wide oversight and compliance management under **21 CFR 820.100**.")
    st.info("This is a single source of truth for all required compliance actions. Prioritize items based on their risk level and source.", icon="🎯")

    try:
        # --- 1. Aggregate actions from all potential QMS sources ---
        all_actions: List[Dict[str, Any]] = []
        sources_checked = []

        # Source 1: Design Reviews
        reviews = ssm.get_data("design_reviews", "reviews") or []
        sources_checked.append("Design Reviews")
        for review in reviews:
            if not isinstance(review, dict): continue
            for action in review.get("action_items", []):
                if not isinstance(action, dict): continue
                action_copy = action.copy()
                action_copy['source'] = f"Review: {review.get('id', 'N/A')}"
                action_copy['type'] = action.get('type', 'Action Item')
                all_actions.append(action_copy)

        # Source 2: Design Changes
        changes = ssm.get_data("design_changes", "changes") or []
        sources_checked.append("Design Changes")
        for change in changes:
            if not isinstance(change, dict): continue
            for action in change.get("action_items", []):
                if not isinstance(action, dict): continue
                action_copy = action.copy()
                action_copy['source'] = f"DCR-{change.get('id', 'N/A')}"
                action_copy['type'] = action.get('type', 'Change Action')
                all_actions.append(action_copy)

        # Source 3: CAPA Records (Ref: 21 CFR 820.100)
        capas = ssm.get_data("quality_system", "capa_records") or []
        sources_checked.append("CAPAs")
        for capa in capas:
            if not isinstance(capa, dict): continue
            for action in capa.get("action_plan", []):
                if not isinstance(action, dict): continue
                action_copy = action.copy()
                action_copy['source'] = f"CAPA-{capa.get('id', 'N/A')}"
                action_copy['type'] = 'CAPA Action'
                all_actions.append(action_copy)
        
        # Source 4: Non-Conformance Reports (NCRs)
        ncrs = ssm.get_data("quality_system", "ncr_records") or []
        sources_checked.append("NCRs")
        for ncr in ncrs:
            if not isinstance(ncr, dict): continue
            for action in ncr.get("correction_actions", []):
                if not isinstance(action, dict): continue
                action_copy = action.copy()
                action_copy['source'] = f"NCR-{ncr.get('id', 'N/A')}"
                action_copy['type'] = 'NCR Correction'
                all_actions.append(action_copy)

        logger.info(f"Aggregated a total of {len(all_actions)} action items from sources: {', '.join(sources_checked)}.")

        if not all_actions:
            st.success("🎉 No compliance actions have been recorded in the QMS yet.")
            return

        actions_df = pd.DataFrame(all_actions)
        actions_df.drop_duplicates(subset=['id'], keep='first', inplace=True)

        # --- 2. Enrich the data for better insights ---
        required_cols = ['id', 'description', 'owner', 'due_date', 'status', 'source', 'risk_priority', 'type', 'voe_status']
        for col in required_cols:
            if col not in actions_df.columns:
                actions_df[col] = None

        actions_df['due_date'] = pd.to_datetime(actions_df['due_date'], errors='coerce')
        now = pd.to_datetime(datetime.now().date())
        is_overdue = (actions_df['due_date'].notna()) & (actions_df['due_date'] < now) & (actions_df['status'] != 'Completed')
        actions_df.loc[is_overdue, 'status'] = 'Overdue'

        # --- 3. Create insightful, risk-based KPIs ---
        total_items = len(actions_df)
        completed_items = len(actions_df[actions_df['status'] == 'Completed'])
        open_items = total_items - completed_items
        overdue_count = len(actions_df[actions_df['status'] == 'Overdue'])
        high_risk_open_count = len(actions_df[(actions_df['risk_priority'] == 'High') & (actions_df['status'] != 'Completed')])

        st.subheader("Compliance Health KPIs")
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Total Open Actions", open_items, help="All actions that are not yet 'Completed'.")
        kpi_cols[1].metric("Overdue Actions", overdue_count, help="Open actions that are past their due date.", delta=overdue_count, delta_color="inverse")
        kpi_cols[2].metric("High-Risk Open Actions", high_risk_open_count, help="Open actions with 'High' risk priority. These require immediate attention.", delta=high_risk_open_count, delta_color="inverse")
        kpi_cols[3].metric("VoE Pending", len(actions_df[actions_df['voe_status'] == 'Pending']), help="Completed CAPA actions awaiting Verification of Effectiveness. This is a critical step for CAPA closure.")

        # --- 4. Add advanced visualization ---
        st.subheader("Analysis of Open Actions")
        
        viz_tab1, viz_tab2 = st.tabs(["Breakdown by Status & Risk", "Pareto Analysis by Owner"])
        
        with viz_tab1:
            if open_items > 0:
                open_df = actions_df[actions_df['status'] != 'Completed'].copy()
                risk_status_summary = open_df.groupby(['status', 'risk_priority']).size().reset_index(name='count')
                status_order = ["Overdue", "In Progress", "Open"]
                risk_order = ["High", "Medium", "Low"]

                fig = pd.pivot_table(risk_status_summary, values='count', index='status', columns='risk_priority').reindex(index=status_order, columns=risk_order).fillna(0)
                
                bar_fig = go.Figure()
                for risk_level in risk_order:
                    if risk_level in fig.columns:
                        bar_fig.add_trace(go.Bar(
                            y=fig.index,
                            x=fig[risk_level],
                            name=risk_level,
                            orientation='h',
                            text=fig[risk_level].apply(lambda x: int(x) if x > 0 else ''),
                            textposition='inside',
                            marker_color={'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#1f77b4'}[risk_level]
                        ))
                
                bar_fig.update_layout(
                    barmode='stack',
                    title="Breakdown of Open Compliance Actions by Status and Risk",
                    xaxis_title="Number of Actions",
                    yaxis_title="Status",
                    legend_title_text='Risk Priority',
                    yaxis={'categoryorder':'array', 'categoryarray': status_order}
                )
                st.plotly_chart(bar_fig, use_container_width=True)
            else:
                st.success("No open actions to visualize.")
        
        with viz_tab2:
            if open_items > 0:
                open_df = actions_df[actions_df['status'] != 'Completed'].copy()
                pareto_fig = create_pareto_chart(open_df, category_col='owner', title="Pareto Analysis of Open Action Item Workload")
                st.plotly_chart(pareto_fig, use_container_width=True)
                st.caption("This chart identifies the 'vital few' owners with the majority of the open action items, helping to focus management and support resources effectively.")
            else:
                st.success("No open actions to analyze.")


        # --- 5. Create interactive filters ---
        st.divider()
        st.subheader("Filter and Export Compliance Actions")
        filter_cols = st.columns(4)
        
        status_options = sorted(actions_df['status'].unique())
        owner_options = sorted(actions_df['owner'].dropna().unique())
        type_options = sorted(actions_df['type'].dropna().unique())
        risk_options = sorted(actions_df['risk_priority'].dropna().unique())
        
        default_status = [s for s in status_options if s != 'Completed']

        status_filter = filter_cols[0].multiselect("Filter by Status:", options=status_options, default=default_status)
        owner_filter = filter_cols[1].multiselect("Filter by Owner:", options=owner_options)
        type_filter = filter_cols[2].multiselect("Filter by Type:", options=type_options)
        risk_filter = filter_cols[3].multiselect("Filter by Risk:", options=risk_options)

        filtered_df = actions_df.copy()
        if status_filter:
            filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
        if owner_filter:
            filtered_df = filtered_df[filtered_df['owner'].isin(owner_filter)]
        if type_filter:
            filtered_df = filtered_df[filtered_df['type'].isin(type_filter)]
        if risk_filter:
            filtered_df = filtered_df[filtered_df['risk_priority'].isin(risk_filter)]

        # --- 6. Display the styled, filtered DataFrame ---
        def style_row_by_status_and_risk(row: pd.Series) -> List[str]:
            """Applies a background color based on status and risk priority."""
            base_style = ''
            if row.status == 'Overdue':
                base_style = 'background-color: #ffcccc' # Red for overdue
            elif row.risk_priority == 'High':
                base_style = 'background-color: #ffebcc' # Orange for high risk
            elif row.risk_priority == 'Medium':
                base_style = 'background-color: #fff9cc' # Yellow for medium risk
            return [base_style for _ in row]
        
        st.dataframe(
            filtered_df[['id', 'description', 'owner', 'due_date', 'status', 'risk_priority', 'type', 'source', 'voe_status']].style.apply(style_row_by_status_and_risk, axis=1),
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("Action ID", help="Unique identifier for the action item."),
                "description": st.column_config.TextColumn("Description", width="large", help="The specific task to be completed."),
                "owner": st.column_config.TextColumn("Owner", help="The individual responsible for completing the action."),
                "due_date": st.column_config.DateColumn("Due Date", format="YYYY-MM-DD"),
                "status": st.column_config.TextColumn("Status", help="Current status of the action item. 'Overdue' is set automatically."),
                "risk_priority": st.column_config.TextColumn("Risk", help="Risk priority of the action item (High, Medium, Low)."),
                "type": st.column_config.TextColumn("Type", help="Category of the action (e.g., CAPA Action, Audit Finding)."),
                "source": st.column_config.TextColumn("Source", help="The QMS record this action originated from (e.g., CAPA-001, DCR-002)."),
                "voe_status": st.column_config.TextColumn("VoE Status", help="Verification of Effectiveness Status for CAPA actions. 'Pending' means the action is done but its effectiveness has not yet been confirmed.")
            },
            hide_index=True
        )

        # --- 7. Add an Export Button ---
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(filtered_df)
        st.download_button(
            label="📥 Export Filtered View to CSV",
            data=csv,
            file_name=f"compliance_actions_export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="export_compliance_actions"
        )

    except Exception as e:
        st.error("An error occurred while generating the action item tracker. The data may be incomplete or malformed.")
        logger.error(f"Failed to render action item tracker: {e}", exc_info=True)
