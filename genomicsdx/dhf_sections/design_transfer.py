# --- SME OVERHAUL: Definitive, Compliance-Focused Version (PMA-Focused V4-Final) ---
"""
Module for rendering the Assay Transfer & Lab Operations Readiness section.

This component provides a structured, multi-faceted command center to manage and
document activities related to transferring the validated assay design and SaMD
from R&D into a CLIA/CAP-certified clinical laboratory for commercial operation,
as required by 21 CFR 820.30(h) and 21 CFR 820.170.
"""

# --- Standard Library Imports ---
import logging
from typing import Any, Dict, List

# --- Third-party Imports ---
import pandas as pd
import streamlit as st
import plotly.express as px

# --- Local Application Imports ---
from genomicsdx.utils.session_state_manager import SessionStateManager

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def render_design_transfer(ssm: SessionStateManager) -> None:
    """Renders a comprehensive dashboard for Assay Transfer & Lab Operations Readiness."""
    st.header("9. Design Transfer & Lab Operations Readiness")
    st.markdown("""
    *As per 21 CFR 820.30(h) Design Transfer & 21 CFR 820.170 Production and Process Controls.*

    This section documents the activities that ensure the validated diagnostic service design—including both the **IVD assay** and the **SaMD algorithm**—is correctly translated into production procedures and specifications for the clinical laboratory. This is the critical bridge between R&D and a scalable, compliant commercial testing service under **CLIA** and **ISO 15189**.
    """)
    st.info("This command center provides a real-time view of operational readiness for commercial launch.", icon="🚀")
    
    try:
        # --- 1. Load Data ---
        transfer_data: Dict[str, Any] = ssm.get_data("lab_operations") or {}
        personnel_data = ssm.get_data("design_plan", "team_members") or []
        sop_data = transfer_data.get("sops", [])
        lab_techs = [p.get('name') for p in personnel_data if 'Tech' in p.get('role', '')]
        logger.info("Loaded lab operations and design transfer data.")

        # --- 2. High-Level Readiness KPIs ---
        st.subheader("Launch Readiness Dashboard")
        infra_data = transfer_data.get("infrastructure", [])
        ppq_data = transfer_data.get("ppq_runs", [])

        sop_total = len(sop_data)
        sop_approved = len([s for s in sop_data if s.get('status') == 'Approved'])
        sop_progress = (sop_approved / sop_total) * 100 if sop_total > 0 else 0

        infra_total = len(infra_data)
        infra_qualified = len([i for i in infra_data if i.get('status') == 'PQ Complete'])
        infra_progress = (infra_qualified / infra_total) * 100 if infra_total > 0 else 0
        
        ppq_required = 3
        ppq_passed = len([p for p in ppq_data if p.get('result') == 'Pass'])
        ppq_progress = (ppq_passed / ppq_required) * 100 if ppq_required > 0 else 0

        kpi_cols = st.columns(3)
        with kpi_cols[0]:
            st.metric("SOP Readiness", f"{sop_progress:.0f}%", f"{sop_approved}/{sop_total} Approved")
            st.progress(sop_progress / 100)
        with kpi_cols[1]:
            st.metric("Infrastructure Qualification", f"{infra_progress:.0f}%", f"{infra_qualified}/{infra_total} PQ Complete")
            st.progress(infra_progress / 100)
        with kpi_cols[2]:
            st.metric("Process Qualification (PPQ)", f"{ppq_progress:.0f}%", f"{ppq_passed}/{ppq_required} Runs Passed")
            st.progress(ppq_progress / 100)

        st.divider()

        # --- 3. Define Tabs for Each Transfer Stream ---
        tab_titles = [
            "1. Process & SOPs",
            "2. Personnel & Training",
            "3. Materials & Suppliers",
            "4. Lab Infrastructure & LIMS",
            "5. SaMD Deployment (Bioinformatics)",
            "6. PPQ & Stability"
        ]
        tabs = st.tabs(tab_titles)

        def render_editor_tab(df_key: str, column_config: Dict):
            data_list = transfer_data.get(df_key, [])
            df = pd.DataFrame(data_list)
            
            date_columns = [col for col, config in column_config.items() if isinstance(config, dict) and config.get("type") == "date"]
            
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            original_records = df.to_dict('records')
            edited_df = st.data_editor(
                df, num_rows="dynamic", use_container_width=True, 
                key=f"transfer_editor_{df_key}", column_config=column_config, hide_index=True
            )
            
            df_to_save = edited_df.copy()
            for col in date_columns:
                if col in df_to_save.columns:
                     df_to_save[col] = df_to_save[col].dt.strftime('%Y-%m-%d').replace({pd.NaT: None})

            if df_to_save.to_dict('records') != original_records:
                ssm.update_data(df_to_save.to_dict('records'), "lab_operations", df_key)
                st.toast(f"{df_key.replace('_', ' ').title()} updated!", icon="✅"); st.rerun()

        with tabs[0]:
            st.subheader("Standard Operating Procedures (SOPs)")
            st.caption("Track the transfer of R&D processes into formal, version-controlled SOPs for the clinical lab.")
            render_editor_tab("sops", {
                "doc_id": "SOP ID", "title": st.column_config.TextColumn("SOP Title", width="large"), "version": "Version",
                "status": st.column_config.SelectboxColumn("Status", options=["Draft", "In Review", "Approved"])
            })

        with tabs[1]:
            st.subheader("Personnel Qualification & Training Matrix")
            st.caption("Track and document that all lab personnel are trained on all effective, approved SOPs before performing patient testing.")
            approved_sops = [s.get('doc_id') for s in sop_data if s.get('status') == 'Approved']
            if approved_sops and lab_techs:
                training_matrix_df = pd.DataFrame(index=approved_sops, columns=lab_techs).fillna(False)
                st.data_editor(training_matrix_df, use_container_width=True)
            else:
                st.info("Define approved SOPs and lab technologists to generate the training matrix.")

        with tabs[2]:
            st.subheader("Critical Materials & Supplier Management")
            st.caption("Manage the list of critical reagents and consumables, the qualification status of their suppliers, and the testing of incoming lots.")
            supplier_audits = ssm.get_data("quality_system", "supplier_audits") or []
            st.markdown("**Supplier Qualification Status**")
            st.dataframe(pd.DataFrame(supplier_audits), use_container_width=True, hide_index=True)
            st.markdown("**Incoming Reagent Lot Qualification**")
            lot_qual = transfer_data.get("readiness", {}).get("reagent_lot_qualification", {})
            total = lot_qual.get('total', 0)
            passed = lot_qual.get('passed', 0)
            st.metric(f"Lot Qualification Pass Rate", f"{(passed / total) * 100 if total > 0 else 0:.1f}%", f"{passed}/{total} Passed")

        with tabs[3]:
            st.subheader("Laboratory Equipment & LIMS Qualification")
            st.caption("Document the IQ/OQ/PQ of all critical lab instruments and the validation of the Laboratory Information Management System (LIMS).")
            render_editor_tab("infrastructure", {
                "asset_id": "Asset ID", "equipment_type": "Equipment/System",
                "status": st.column_config.SelectboxColumn("Qualification Status", options=["Pending", "IQ Complete", "OQ Complete", "PQ Complete"]),
                "qualification_report_link": st.column_config.LinkColumn("IQ/OQ/PQ Report")
            })

        with tabs[4]:
            st.subheader("SaMD Deployment (Bioinformatics Pipeline & Classifier)")
            st.caption("Track the formal, controlled deployment of the locked bioinformatics pipeline and classifier algorithm to the validated production infrastructure, as per ISO 62304.")
            render_editor_tab("software_deployment", {
                "component": "Software Component", "version": "Deployed Version/Hash", "deployment_date": st.column_config.DateColumn("Deployment Date", format="YYYY-MM-DD", type="date"),
                "validation_protocol": "Validation Protocol ID", "validation_report_link": st.column_config.LinkColumn("Validation Report")
            })

        with tabs[5]:
            st.subheader("Process Performance Qualification (PPQ) & Stability")
            st.caption("Document the capstone PPQ runs demonstrating process robustness, and track ongoing stability studies for the sample collection kit and reagents.")
            st.markdown("**PPQ Runs (End-to-End Process)**")
            render_editor_tab("ppq_runs", {
                "run_id": "PPQ Run ID", "description": st.column_config.TextColumn("Run Description", width="large"),
                "run_date": st.column_config.DateColumn("Run Date", format="YYYY-MM-DD", type="date"), "result": st.column_config.SelectboxColumn("Result", options=["Not Started", "In Progress", "Pass", "Fail"]),
                "summary_report_link": st.column_config.LinkColumn("Summary Report")
            })
            st.markdown("**Stability Program**")
            stability_df = pd.DataFrame(transfer_data.get("readiness", {}).get("sample_stability_studies", []))
            st.dataframe(stability_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("An error occurred while displaying the Design Transfer section. The data may be malformed.")
        logger.error(f"Failed to render design transfer: {e}", exc_info=True)
