```python?code_reference&code_event_index=14
# backend/core/trigger_system.py
from typing import Dict, List, Any, Callable
from datetime import datetime
import logging
import functools # Import functools for the decorator
import inspect # Import inspect to check for coroutine functions

logger = logging.getLogger(__name__)

# Assume load_triggers and execute_trigger_actions are defined elsewhere or are stubs
# Example stubs (replace with actual implementation):
def load_triggers() -> List[Dict[str, Any]]:
    """Loads active trigger configurations from storage."""
    logger.warning("load_triggers is a stub. Implement actual loading logic.")
    # Example structure based on the schema:
    return [
        # {
        #     "id": "example-trigger-1",
        #     "name": "Notify on high CPU",
        #     "event_type": "performance_metric",
        #     "conditions": [{"field": "cpu_percent", "operator": "gt", "value": 80}],
        #     "actions": [{"type": "notify", "params": {"message": "High CPU usage detected!"}}]
        # }
    ]

async def execute_trigger_actions(actions: List[Dict[str, Any]]):
    """Executes the defined actions for a triggered event."""
    logger.warning("execute_trigger_actions is a stub. Implement actual action logic.")
    for action in actions:
        logger.info(f"Executing action: {action['type']} with params {action.get('params')}")
        # Implement action logic here (e.g., send notification, call another service)
        pass


def evaluate_trigger(event_data: Dict[str, Any], trigger_config: Dict[str, Any]) -> bool:
    """
    Evaluates if an event matches trigger conditions based on a single trigger config.

    Args:
        event_data: Dictionary containing event information (e.g., {"type": "index_update", "result": {...}}).
        trigger_config: Dictionary containing the configuration for a single trigger
                        (matches the item structure in the triggers schema array).

    Returns:
        True if the event matches all conditions, False otherwise.
    """
    try:
        # Validate trigger_config structure minimally
        if 'conditions' not in trigger_config or not isinstance(trigger_config['conditions'], list):
            logger.error(f"Invalid trigger config: 'conditions' missing or not a list in {trigger_config}")
            return False

        for condition in trigger_config['conditions']:
            # Validate condition structure minimally
            if not all(k in condition for k in ['field', 'operator']):
                 logger.error(f"Invalid condition structure: 'field' or 'operator' missing in {condition}")
                 return False

            field = condition['field']
            operator = condition['operator']
            value = condition.get('value') # Value might be optional for some operators like 'changed'

            if field not in event_data:
                # If the required field is not even in the event data, this condition cannot be met
                logger.debug(f"Condition field '{field}' not found in event data.")
                return False

            field_value = event_data[field]

            # --- Condition Evaluation Logic ---
            if operator == 'eq':
                if field_value != value:
                    return False
            elif operator == 'gt':
                # Attempt type conversion for comparison
                try:
                    # Ensure both values can be floated before comparison
                    if not float(field_value) > float(value):
                        return False
                except (ValueError, TypeError):
                    logger.error(f"Cannot perform GT comparison on non-numeric values: {field_value}, {value}")
                    return False
            elif operator == 'lt':
                 # Attempt type conversion for comparison
                try:
                    # Ensure both values can be floated before comparison
                    if not float(field_value) < float(value):
                        return False
                except (ValueError, TypeError):
                    logger.error(f"Cannot perform LT comparison on non-numeric values: {field_value}, {value}")
                    return False
            elif operator == 'contains':
                 # Check if field_value is a string or list before checking containment
                if not (isinstance(field_value, (str, list)) and value in field_value):
                    return False
            elif operator == 'changed':
                # This operator requires a comparison to a previous state/value.
                # Based on the schema, it might imply the event_data includes both current and previous states.
                # A common pattern is to have a 'previous_state' key in event_data, or keys like 'field_name_previous'.
                # Adhering strictly to the user's original simple 'changed' check requiring just 'previous_value' key in event_data,
                # which is less robust for multiple fields.
                # Let's refine this to look for a corresponding '_previous' key or a 'previous_state' dict.
                # For now, let's stick closer to the original (simple but potentially brittle) check:
                # It requires a key named 'previous_value' in the event_data. This seems too generic.
                # Let's assume the event_data might contain a key like f'{field}_previous'.
                previous_field_value = event_data.get(f'{field}_previous')
                if previous_field_value is None:
                     logger.warning(f"Using 'changed' operator for field '{field}' but '{field}_previous' key is missing in event_data. Cannot evaluate.")
                     return False # Cannot evaluate 'changed' without a previous value
                if field_value == previous_field_value:
                    return False # Has not changed

            # Add other operators as needed based on schema enum (e.g., 'not_eq', 'gt_eq', 'lt_eq', 'not_contains')
            else:
                logger.warning(f"Unsupported operator '{operator}' in trigger condition.")
                return False # Condition not met due to unsupported operator

        # If all conditions are met for this specific trigger
        return True
    except Exception as e:
        logger.error(f"Error evaluating trigger condition. Error: {str(e)}", exc_info=True) # Log exception details
        return False # Treat errors in evaluation as the trigger not matching


def trigger_check(event_type: str):
    """
    Decorator to evaluate triggers after a function executes and potentially run actions.

    Assumes the decorated function returns data relevant to the event.
    This data, along well as the event_type, forms the basis of event_data for trigger evaluation.
    """
    def decorator(func: Callable):
        @functools.wraps(func) # Use functools.wraps to preserve original function metadata
        async def wrapper(*args, **kwargs):
            # Execute the original function
            # Handle potential async/await depending on the decorated function
            if inspect.iscoroutinefunction(func): # Requires 'import inspect'
                 result = await func(*args, **kwargs)
            else:
                 result = func(*args, **kwargs)


            # Prepare event data based on the function's result and declared event_type
            event_data: Dict[str, Any] = {
                "type": event_type,
                "result": result, # Include the result of the decorated function
                "timestamp": datetime.utcnow().isoformat() # Use ISO format for better compatibility
                # Add other relevant context from args/kwargs if needed,
                # e.g., 'user_id', 'object_id', 'previous_state' etc.
                # For the 'changed' operator to work well, previous state should be included here.
                # Example: if decorating an update function, event_data could contain
                # "current_state": result, "previous_state": previous_object_state
            }

            logger.info(f"Processing event of type '{event_type}'. Evaluating triggers...")

            # Load all active triggers
            # This function needs to be implemented to load triggers from your persistent storage
            try:
                triggers = load_triggers()
            except Exception as e:
                logger.error(f"Failed to load triggers: {e}", exc_info=True)
                triggers = [] # Continue without triggers if loading fails


            # Evaluate each trigger against the event data
            for trigger in triggers:
                # Add trigger ID to logging for traceability
                trigger_id = trigger.get('id', 'unknown-id')
                trigger_name = trigger.get('name', 'unnamed-trigger')
                event_type_filter = trigger.get('event_type')

                # Basic check for event type matching the trigger's filter (if specified)
                if event_type_filter and event_type != event_type_filter:
                    logger.debug(f"Skipping trigger '{trigger_name}' ({trigger_id}): event type mismatch (expected '{event_type_filter}', got '{event_type}')")
                    continue # Skip this trigger if event types don't match

                logger.debug(f"Evaluating trigger '{trigger_name}' ({trigger_id}) for event '{event_type}'.")

                if evaluate_trigger(event_data, trigger):
                    logger.info(f"Trigger '{trigger_name}' ({trigger_id}) matched for event '{event_type}'. Executing actions...")
                    # Execute actions for the matched trigger
                    # This function needs to be implemented to perform the actions (e.g., notify, adjust budget)
                    try:
                        await execute_trigger_actions(trigger.get('actions', []))
                    except Exception as e:
                         logger.error(f"Failed to execute actions for trigger '{trigger_name}' ({trigger_id}): {e}", exc_info=True)

                else:
                    logger.debug(f"Trigger '{trigger_name}' ({trigger_id}) did not match for event '{event_type}'.")


            return result # Return the original function's result

        return wrapper
    return decorator
