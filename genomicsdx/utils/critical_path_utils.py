# --- SME REVIEW: Definitive, Hardened, and Compliance-Documented Version (PMA-Focused V4-Final) ---
"""
Utility for calculating the critical path of a project plan.

This module provides the core logic for implementing the Critical Path Method (CPM)
on a set of tasks defined in a pandas DataFrame. Its role within a regulated
(21 CFR 820) environment is as a planning and monitoring tool. The algorithm's
correctness is established during software V&V and is not meant to be modified
during routine use.
"""

# --- Standard Library Imports ---
import logging
from typing import Dict, List, Set, Any

# --- Third-party Imports ---
import pandas as pd

# --- Setup Logging ---
logger = logging.getLogger(__name__)

# --- Type Aliases for clarity ---
TaskId = str
TaskMap = Dict[TaskId, Dict[str, Any]]


def find_critical_path(tasks_df: pd.DataFrame) -> List[TaskId]:
    """
    Identifies the critical path in a project task list using the Critical Path Method (CPM).

    This implementation performs a forward and backward pass to calculate the
    earliest and latest start/finish times for each task. Tasks with zero
    "slack" or "float" (where Early Start equals Late Start) are considered
    to be on the critical path.

    SME Note for QMS (Quality Management System) File:
    - **Purpose:** This utility serves as a project management aid for visualizing
      and monitoring the timeline as part of Design and Development Planning (21 CFR 820.30(b)).
      It helps identify tasks that, if delayed, will directly impact the final
      project completion date (e.g., PMA submission date).
    - **Validation:** The correctness of this utility is verified as part of the
      overall software validation for the GenomicsDx Command Center application,
      in accordance with internal SOPs for software development and validation
      (which align with ISO 62304 principles). Unit tests with known inputs and
      expected outputs form the basis of this verification.
    - **Assumptions (for validation purposes):**
      1. The task dependency graph is a Directed Acyclic Graph (DAG). It does not
         contain circular dependencies (e.g., Task A -> Task B -> Task A).
         The presence of cycles will lead to a recursion error or incorrect results.
         Input validation in the UI should prevent the creation of such cycles.
      2. Task durations are calculated in whole days.
      3. The 'dependencies' column is a well-formed, comma-separated string of task IDs.
      4. Dates are valid and have been pre-processed into datetime objects.

    Args:
        tasks_df (pd.DataFrame):
            A DataFrame of tasks. It must contain the following columns:
            - 'id': A unique identifier for the task (str).
            - 'start_date': The task's start date (datetime object).
            - 'end_date': The task's end date (datetime object).
            - 'dependencies': A comma-separated string of prerequisite task IDs.

    Returns:
        List[TaskId]: A list of task IDs that form the critical path. Returns
                      an empty list if the input DataFrame is empty, malformed,
                      or if any error occurs during calculation.
    """
    if not isinstance(tasks_df, pd.DataFrame) or tasks_df.empty:
        logger.info("Input tasks_df is not a valid, non-empty DataFrame. Returning no critical path.")
        return []

    try:
        # --- 1. Data Validation and Initialization ---
        required_cols = {'id', 'start_date', 'end_date', 'dependencies'}
        if not required_cols.issubset(tasks_df.columns):
            missing_cols = required_cols - set(tasks_df.columns)
            logger.error(f"Input DataFrame is missing required columns for CPM: {missing_cols}")
            return []

        # Create a copy to avoid side effects on the original DataFrame
        df = tasks_df.copy()

        # Robustly handle potential non-string 'id'
        df['id'] = df['id'].astype(str)

        # Drop tasks with invalid dates to prevent calculation errors
        df = df.dropna(subset=['start_date', 'end_date'])
        if df.empty:
            logger.warning("All tasks were dropped from CPM due to missing start/end dates.")
            return []

        # CPM convention: duration includes the start day. If a task starts and
        # ends on the same day, its duration is 1.
        df['duration'] = (df['end_date'] - df['start_date']).dt.days + 1

        # Filter out tasks with non-positive duration, which are invalid for CPM
        df = df[df['duration'] > 0]
        if df.empty:
            logger.warning("All tasks were filtered out from CPM due to non-positive durations.")
            return []

        task_map: TaskMap = df.set_index('id').to_dict('index')
        task_ids: List[TaskId] = df['id'].tolist()
        task_id_set: Set[TaskId] = set(task_ids) # For fast lookups

        logger.info(f"Starting CPM analysis on {len(task_ids)} valid tasks.")

        # --- 2. Forward Pass: Calculate Early Start (ES) and Early Finish (EF) ---
        logger.debug("Performing forward pass to calculate ES and EF...")
        # A simple topological sort is implicitly done by processing in list order,
        # assuming the list is ordered such that dependencies appear before dependents.
        # A more robust implementation would perform an explicit topological sort first.
        for task_id in task_ids:
            # Safely parse dependencies from comma-separated string, ensuring they are valid tasks
            dep_str = str(task_map[task_id].get('dependencies', '') or '')
            dependencies: Set[TaskId] = {d.strip() for d in dep_str.split(',') if d.strip() and d.strip() in task_id_set}

            if not dependencies:
                task_map[task_id]['es'] = 0  # Tasks with no dependencies start at time 0
            else:
                # ES is the maximum of the Early Finishes of all its dependencies
                max_ef_of_deps = max(
                    (task_map[dep_id].get('ef', 0) for dep_id in dependencies),
                    default=0
                )
                task_map[task_id]['es'] = max_ef_of_deps

            task_map[task_id]['ef'] = task_map[task_id]['es'] + task_map[task_id]['duration']

        # --- 3. Backward Pass: Calculate Late Finish (LF) and Late Start (LS) ---
        logger.debug("Performing backward pass to calculate LF and LS...")
        try:
            # The project finish time is the maximum Early Finish of all tasks
            project_finish_time = max(task['ef'] for task in task_map.values() if 'ef' in task)
            logger.debug(f"Calculated project finish time: {project_finish_time} days.")
        except ValueError:
            logger.error("Could not determine project finish time. The task map might be empty after processing.")
            return []

        # Iterate through tasks in reverse topological order (simplified as reversed list)
        for task_id in reversed(task_ids):
            # Find all tasks that have the current task as a dependency
            successor_ids: List[TaskId] = [
                succ_id for succ_id, succ_task in task_map.items()
                if task_id in {d.strip() for d in str(succ_task.get('dependencies', '') or '').split(',') if d.strip()}
            ]

            if not successor_ids:
                # Tasks with no successors finish at the project end time
                task_map[task_id]['lf'] = project_finish_time
            else:
                # LF is the minimum of the Late Starts of all its successors
                min_ls_of_succs = min(
                    (task_map[succ_id].get('ls', project_finish_time) for succ_id in successor_ids if succ_id in task_id_set),
                    default=project_finish_time
                )
                task_map[task_id]['lf'] = min_ls_of_succs

            task_map[task_id]['ls'] = task_map[task_id]['lf'] - task_map[task_id]['duration']

        # --- 4. Identify Critical Path ---
        # Critical tasks are those with zero (or near-zero) slack (LS - ES = 0)
        logger.debug("Identifying critical path tasks (slack = 0)...")
        critical_path: List[TaskId] = []
        for task_id, task_data in task_map.items():
            # Check for existence of keys to avoid KeyErrors during calculation
            if 'es' in task_data and 'ls' in task_data:
                slack = task_data['ls'] - task_data['es']
                # Using a small tolerance for float comparison, though these should be integers
                if abs(slack) < 1e-9:
                    critical_path.append(task_id)

        logger.info(f"Critical path identified with {len(critical_path)} tasks: {critical_path}")
        return critical_path

    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.error(f"A critical error occurred during critical path calculation: {e}", exc_info=True)
        # In case of any unexpected error, return an empty list to prevent downstream failures
        return []
