# In your campaign processing service
metrics_collector.update_business_metrics(
    campaign_id='campaign_123',
    metrics={
        'roi': 2.5,
        'conversion_rate': 0.03,
        'click_through_rate': 0.02,
        'current_budget': 1000.0
    }
)