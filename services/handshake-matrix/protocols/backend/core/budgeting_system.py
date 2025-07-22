# backend/core/budgeting_system.py
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)

class BudgetAllocationSystem:
    def __init__(self):
        self.performance_thresholds = {
            'roi': {'min': 1.5, 'target': 2.5},
            'conversion_rate': {'min': 0.02, 'target': 0.05},
            'click_through_rate': {'min': 0.01, 'target': 0.03}
        }

    def calculate_budget_adjustment(
        self,
        campaign_metrics: Dict[str, float],
        budget_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate budget adjustments based on performance metrics and rules
        
        Args:
            campaign_metrics: Dict containing current performance metrics
            budget_rules: Dict containing budget rules configuration
            
        Returns:
            Dict containing new budget allocation and adjustment details
        """
        try:
            current_budget = Decimal(str(campaign_metrics.get('current_budget', 0)))
            if current_budget <= 0:
                raise ValueError("Current budget must be positive")

            adjustments = []
            for rule in budget_rules['adjustment_triggers']:
                metric = rule['metric']
                metric_value = Decimal(str(campaign_metrics.get(metric, 0)))
                threshold = Decimal(str(rule['threshold']))
                
                # Calculate adjustment based on performance vs threshold
                adjustment_value = Decimal(str(rule['adjustment_value']))
                if rule['adjustment_type'] == 'percentage':
                    adjustment_amount = (current_budget * adjustment_value / Decimal('100'))
                else:  # fixed amount
                    adjustment_amount = adjustment_value

                # Apply adjustment based on performance
                if metric_value > threshold:
                    # Exceeding threshold - positive adjustment
                    adjustments.append({
                        'metric': metric,
                        'amount': adjustment_amount,
                        'reason': f"{metric} exceeding target ({metric_value} > {threshold})"
                    })
                elif metric_value < self.performance_thresholds[metric]['min']:
                    # Below minimum - negative adjustment
                    adjustments.append({
                        'metric': metric,
                        'amount': -adjustment_amount,
                        'reason': f"{metric} below minimum threshold"
                    })

            # Calculate final budget adjustment
            total_adjustment = sum(adj['amount'] for adj in adjustments)
            new_budget = current_budget + total_adjustment

            # Enforce min/max constraints
            min_budget = Decimal(str(budget_rules['min_budget']))
            max_budget = Decimal(str(budget_rules['max_budget']))
            new_budget = max(min_budget, min(max_budget, new_budget))

            return {
                'campaign_id': budget_rules['campaign_id'],
                'current_budget': float(current_budget),
                'new_budget': float(new_budget),
                'total_adjustment': float(total_adjustment),
                'adjustments': [
                    {**adj, 'amount': float(adj['amount'])} 
                    for adj in adjustments
                ],
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating budget adjustment: {str(e)}")
            return {
                'campaign_id': budget_rules['campaign_id'],
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }