from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl


class Commission(BaseModel):
    type: str = Field(..., description="Type of commission structure", example="percentage")
    value: Union[float, Dict[str, float]] = Field(..., description="Commission value or tiered structure")
    details: Optional[str] = Field(None, description="Additional details about the commission structure")


class ProgramMetrics(BaseModel):
    clicks: int = Field(..., description="Number of clicks", ge=0)
    conversions: int = Field(..., description="Number of conversions", ge=0)
    revenue: float = Field(..., description="Total revenue generated", ge=0)
    roi: float = Field(..., description="Return on investment")
    average_order_value: Optional[float] = Field(None, description="Average order value", ge=0)
    last_click_date: Optional[datetime] = Field(None, description="Date of the last click")
    last_conversion_date: Optional[datetime] = Field(None, description="Date of the last conversion")
    trend: Optional[Dict[str, float]] = Field(None, description="Trend percentages (daily, weekly, monthly)")


class ProgramBase(BaseModel):
    name: str = Field(..., description="Name of the affiliate program")
    description: str = Field(..., description="Detailed description of the program")
    url: HttpUrl = Field(..., description="URL to the program's website or signup page")
    category: List[str] = Field(..., description="Categories the program belongs to")
    commission: Commission = Field(..., description="Commission structure")
    cookie_duration: int = Field(..., description="Cookie duration in days", ge=0)
    payment_frequency: str = Field(..., description="Frequency of payments from the program")
    minimum_payout: float = Field(..., description="Minimum amount required for payout", ge=0)
    payment_methods: List[str] = Field(..., description="Available payment methods")
    status: str = Field(..., description="Current status of the program")
    tags: List[str] = Field(..., description="Tags associated with the program")
    source: str = Field(..., description="Source of the program data (aggregator name, API, etc.)")


class ProgramCreate(ProgramBase):
    epc: Optional[float] = Field(None, description="Earnings per click")
    conversion_rate: Optional[float] = Field(None, description="Conversion rate percentage")
    source_id: Optional[str] = Field(None, description="Original ID from the source")


class ProgramUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Name of the affiliate program")
    description: Optional[str] = Field(None, description="Detailed description of the program")
    url: Optional[HttpUrl] = Field(None, description="URL to the program's website or signup page")
    category: Optional[List[str]] = Field(None, description="Categories the program belongs to")
    commission: Optional[Commission] = Field(None, description="Commission structure")
    cookie_duration: Optional[int] = Field(None, description="Cookie duration in days", ge=0)
    payment_frequency: Optional[str] = Field(None, description="Frequency of payments from the program")
    minimum_payout: Optional[float] = Field(None, description="Minimum amount required for payout", ge=0)
    payment_methods: Optional[List[str]] = Field(None, description="Available payment methods")
    epc: Optional[float] = Field(None, description="Earnings per click")
    conversion_rate: Optional[float] = Field(None, description="Conversion rate percentage")
    status: Optional[str] = Field(None, description="Current status of the program")
    tags: Optional[List[str]] = Field(None, description="Tags associated with the program")
    source_id: Optional[str] = Field(None, description="Original ID from the source")


class Program(ProgramBase):
    id: str = Field(..., description="Unique identifier for the program")
    epc: Optional[float] = Field(None, description="Earnings per click")
    conversion_rate: Optional[float] = Field(None, description="Conversion rate percentage")
    date_added: datetime = Field(..., description="Date when the program was added to the system")
    last_updated: datetime = Field(..., description="Date when the program was last updated")
    source_id: Optional[str] = Field(None, description="Original ID from the source")
    metrics: Optional[ProgramMetrics] = Field(None, description="Performance metrics for the program")

    class Config:
        orm_mode = True


class ConnectionStatus(BaseModel):
    state: str = Field(..., description="Current state of the connection")
    last_checked: datetime = Field(..., description="Date when the connection was last checked")
    message: Optional[str] = Field(None, description="Status message")
    error_code: Optional[str] = Field(None, description="Error code if applicable")
    error_details: Optional[str] = Field(None, description="Detailed error information")
    sync_progress: Optional[float] = Field(None, description="Synchronization progress percentage", ge=0, le=100)


class OAuthCredentials(BaseModel):
    client_id: Optional[str] = Field(None, description="OAuth client ID")
    client_secret: Optional[str] = Field(None, description="OAuth client secret")
    refresh_token: Optional[str] = Field(None, description="OAuth refresh token")
    access_token: Optional[str] = Field(None, description="OAuth access token")
    expires_at: Optional[datetime] = Field(None, description="OAuth token expiration date")


class ApiCredentials(BaseModel):
    api_key: Optional[str] = Field(None, description="API key")
    api_secret: Optional[str] = Field(None, description="API secret")
    username: Optional[str] = Field(None, description="Username for authentication")
    password: Optional[str] = Field(None, description="Password for authentication")
    token: Optional[str] = Field(None, description="Authentication token")
    token_expiry: Optional[datetime] = Field(None, description="Token expiration date")
    oauth: Optional[OAuthCredentials] = Field(None, description="OAuth credentials")


class ConnectionSettings(BaseModel):
    refresh_interval: int = Field(..., description="Refresh interval in minutes", ge=0)
    auto_sync: bool = Field(..., description="Whether to automatically sync")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters to apply during sync")


class ConnectionBase(BaseModel):
    name: str = Field(..., description="Name of the connection")
    type: str = Field(..., description="Type of connection")
    url: HttpUrl = Field(..., description="URL of the connection source")
    description: Optional[str] = Field(None, description="Description of the connection")


class ConnectionCreate(ConnectionBase):
    credentials: ApiCredentials = Field(..., description="API credentials")
    settings: ConnectionSettings = Field(..., description="Connection settings")


class ConnectionUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Name of the connection")
    url: Optional[HttpUrl] = Field(None, description="URL of the connection source")
    description: Optional[str] = Field(None, description="Description of the connection")
    credentials: Optional[ApiCredentials] = Field(None, description="API credentials")
    settings: Optional[ConnectionSettings] = Field(None, description="Connection settings")


class Connection(ConnectionBase):
    id: str = Field(..., description="Unique identifier for the connection")
    status: ConnectionStatus = Field(..., description="Status of the connection")
    credentials: ApiCredentials = Field(..., description="API credentials")
    settings: ConnectionSettings = Field(..., description="Connection settings")
    last_sync: Optional[datetime] = Field(None, description="Date of the last synchronization")
    next_scheduled_sync: Optional[datetime] = Field(None, description="Date of the next scheduled synchronization")
    created_at: datetime = Field(..., description="Date when the connection was created")
    updated_at: datetime = Field(..., description="Date when the connection was last updated")

    class Config:
        orm_mode = True


class DiscoveryItem(BaseModel):
    id: str = Field(..., description="Unique identifier for the discovery item")
    url: HttpUrl = Field(..., description="URL of the discovered item")
    title: str = Field(..., description="Title of the discovered item")
    description: str = Field(..., description="Description of the discovered item")
    match_type: str = Field(..., description="Type of match")
    confidence: float = Field(..., description="Confidence score percentage", ge=0, le=100)
    program_details: Optional[Dict[str, Any]] = Field(None, description="Partial program details extracted from the discovery")
    processed: bool = Field(..., description="Whether the item has been processed")
    added_to_index: bool = Field(..., description="Whether the item has been added to the index")
    notes: Optional[str] = Field(None, description="Additional notes about the discovery item")


class DiscoveryStats(BaseModel):
    total_found: int = Field(..., description="Total number of items found", ge=0)
    new_programs: int = Field(..., description="Number of new programs found", ge=0)
    existing_programs: int = Field(..., description="Number of existing programs found", ge=0)
    potential_matches: int = Field(..., description="Number of potential matches found", ge=0)


class DiscoveryResult(BaseModel):
    id: str = Field(..., description="Unique identifier for the discovery result")
    query: str = Field(..., description="Query used for discovery")
    timestamp: datetime = Field(..., description="Timestamp of the discovery operation")
    duration: float = Field(..., description="Duration of the discovery operation in seconds", ge=0)
    status: str = Field(..., description="Status of the discovery operation")
    results: List[DiscoveryItem] = Field(..., description="Discovery results")
    stats: DiscoveryStats = Field(..., description="Discovery statistics")
    error: Optional[str] = Field(None, description="Error message if the discovery operation failed")

    class Config:
        orm_mode = True


class DiscoveryRequest(BaseModel):
    query: str = Field(..., description="Query for discovery")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for discovery")


class DiscoveryItemProcess(BaseModel):
    action: str = Field(..., description="Action to take")
    program_details: Optional[Dict[str, Any]] = Field(None, description="Program details if action is 'add'")


class CPUMetrics(BaseModel):
    usage: float = Field(..., description="CPU usage percentage", ge=0, le=100)
    temperature: Optional[float] = Field(None, description="CPU temperature in Celsius")


class MemoryMetrics(BaseModel):
    total: float = Field(..., description="Total memory in MB", ge=0)
    used: float = Field(..., description="Used memory in MB", ge=0)
    percentage: float = Field(..., description="Memory usage percentage", ge=0, le=100)


class StorageMetrics(BaseModel):
    total: float = Field(..., description="Total storage in MB", ge=0)
    used: float = Field(..., description="Used storage in MB", ge=0)
    percentage: float = Field(..., description="Storage usage percentage", ge=0, le=100)


class NetworkMetrics(BaseModel):
    bytes_in: float = Field(..., description="Bytes received", ge=0)
    bytes_out: float = Field(..., description="Bytes sent", ge=0)
    requests_per_minute: float = Field(..., description="Requests per minute", ge=0)


class IndexStats(BaseModel):
    total_programs: int = Field(..., description="Total number of programs in the index", ge=0)
    active_programs: int = Field(..., description="Number of active programs in the index", ge=0)
    last_index_update: datetime = Field(..., description="Date of the last index update")
    index_size: float = Field(..., description="Size of the index in MB", ge=0)


class ApiStats(BaseModel):
    requests_total: int = Field(..., description="Total number of API requests", ge=0)
    requests_per_hour: float = Field(..., description="API requests per hour", ge=0)
    average_response_time: float = Field(..., description="Average API response time in ms", ge=0)
    error_rate: float = Field(..., description="API error rate percentage", ge=0, le=100)


class SystemMetrics(BaseModel):
    timestamp: datetime = Field(..., description="Timestamp of the metrics")
    cpu: CPUMetrics = Field(..., description="CPU metrics")
    memory: MemoryMetrics = Field(..., description="Memory metrics")
    storage: StorageMetrics = Field(..., description="Storage metrics")
    network: NetworkMetrics = Field(..., description="Network metrics")
    index_stats: IndexStats = Field(..., description="Index statistics")
    api_stats: ApiStats = Field(..., description="API statistics")

    class Config:
        orm_mode = True


class BudgetPerformance(BaseModel):
    roi: float = Field(..., description="Return on investment")
    revenue: float = Field(..., description="Total revenue generated", ge=0)
    profit: float = Field(..., description="Total profit")


class BudgetAllocationRules(BaseModel):
    min_performance: Optional[float] = Field(None, description="Minimum performance threshold")
    max_spend_per_day: Optional[float] = Field(None, description="Maximum spend per day", ge=0)
    pause_threshold: Optional[float] = Field(None, description="Threshold to pause allocation")


class BudgetAllocationBase(BaseModel):
    target_type: str = Field(..., description="Type of allocation target")
    target_id: str = Field(..., description="Identifier of the allocation target")
    amount: float = Field(..., description="Allocated amount", ge=0)


class BudgetAllocationCreate(BudgetAllocationBase):
    rules: Optional[BudgetAllocationRules] = Field(None, description="Allocation rules")


class BudgetAllocation(BudgetAllocationBase):
    id: str = Field(..., description="Unique identifier for the budget allocation")
    spent_amount: float = Field(..., description="Amount spent from the allocation", ge=0)
    performance: BudgetPerformance = Field(..., description="Allocation performance")
    rules: Optional[BudgetAllocationRules] = Field(None, description="Allocation rules")

    class Config:
        orm_mode = True


class BudgetBase(BaseModel):
    name: str = Field(..., description="Name of the budget")
    total_amount: float = Field(..., description="Total budget amount", ge=0)
    currency: str = Field(..., description="Currency of the budget")
    start_date: datetime = Field(..., description="Start date of the budget period")
    end_date: datetime = Field(..., description="End date of the budget period")


class BudgetCreate(BudgetBase):
    allocations: List[BudgetAllocationCreate] = Field(..., description="Budget allocations")


class BudgetUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Name of the budget")
    total_amount: Optional[float] = Field(None, description="Total budget amount", ge=0)
    currency: Optional[str] = Field(None, description="Currency of the budget")
    start_date: Optional[datetime] = Field(None, description="Start date of the budget period")
    end_date: Optional[datetime] = Field(None, description="End date of the budget period")
    allocations: Optional[List[BudgetAllocationCreate]] = Field(None, description="Budget allocations")
    status: Optional[str] = Field(None, description="Status of the budget")


class Budget(BudgetBase):
    id: str = Field(..., description="Unique identifier for the budget")
    allocations: List[BudgetAllocation] = Field(..., description="Budget allocations")
    status: str = Field(..., description="Status of the budget")
    spent_amount: float = Field(..., description="Amount spent from the budget", ge=0)
    remaining_amount: float = Field(..., description="Remaining amount in the budget", ge=0)
    performance: BudgetPerformance = Field(..., description="Budget performance")

    class Config:
        orm_mode = True


class BudgetStatusUpdate(BaseModel):
    status: str = Field(..., description="New status")


class AutomationSchedule(BaseModel):
    frequency: str = Field(..., description="Frequency of the schedule")
    day_of_week: Optional[int] = Field(None, description="Day of the week (0-6, Sunday to Saturday)", ge=0, le=6)
    day_of_month: Optional[int] = Field(None, description="Day of the month (1-31)", ge=1, le=31)
    hour: Optional[int] = Field(None, description="Hour (0-23)", ge=0, le=23)
    minute: Optional[int] = Field(None, description="Minute (0-59)", ge=0, le=59)


class AutomationEvent(BaseModel):
    type: str = Field(..., description="Type of event")
    conditions: Dict[str, Any] = Field(..., description="Conditions for the event")


class AutomationThreshold(BaseModel):
    metric: str = Field(..., description="Metric to monitor")
    operator: str = Field(..., description="Comparison operator")
    value: float = Field(..., description="Threshold value")


class AutomationTriggerBase(BaseModel):
    name: str = Field(..., description="Name of the automation trigger")
    type: str = Field(..., description="Type of trigger")
    enabled: bool = Field(..., description="Whether the trigger is enabled")
    action: str = Field(..., description="Action to perform when triggered")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the action")


class AutomationTriggerCreate(AutomationTriggerBase):
    schedule: Optional[AutomationSchedule] = Field(None, description="Schedule for the trigger")
    event: Optional[AutomationEvent] = Field(None, description="Event for the trigger")
    threshold: Optional[AutomationThreshold] = Field(None, description="Threshold for the trigger")


class AutomationTriggerUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Name of the automation trigger")
    enabled: Optional[bool] = Field(None, description="Whether the trigger is enabled")
    action: Optional[str] = Field(None, description="Action to perform when triggered")
    schedule: Optional[AutomationSchedule] = Field(None, description="Schedule for the trigger")
    event: Optional[AutomationEvent] = Field(None, description="Event for the trigger")
    threshold: Optional[AutomationThreshold] = Field(None, description="Threshold for the trigger")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for the action")


class AutomationTrigger(AutomationTriggerBase):
    id: str = Field(..., description="Unique identifier for the automation trigger")
    schedule: Optional[AutomationSchedule] = Field(None, description="Schedule for the trigger")
    event: Optional[AutomationEvent] = Field(None, description="Event for the trigger")
    threshold: Optional[AutomationThreshold] = Field(None, description="Threshold for the trigger")
    last_triggered: Optional[datetime] = Field(None, description="Date when the trigger was last activated")
    next_scheduled: Optional[datetime] = Field(None, description="Date when the trigger is next scheduled to run")

    class Config:
        orm_mode = True


class AutomationTriggerToggle(BaseModel):
    enabled: bool = Field(..., description="Whether the trigger is enabled")


class UserNotifications(BaseModel):
    email: bool = Field(..., description="Whether to send email notifications")
    browser: bool = Field(..., description="Whether to show browser notifications")
    slack: Optional[bool] = Field(None, description="Whether to send Slack notifications")
    slack_webhook: Optional[str] = Field(None, description="Slack webhook URL")


class UserDisplayPreferences(BaseModel):
    default_view: str = Field(..., description="Default view type")
    table_columns: List[str] = Field(..., description="Visible table columns")
    graph_layout: str = Field(..., description="Graph layout type")
    results_per_page: int = Field(..., description="Number of results to show per page", ge=1)


class UserApiAccess(BaseModel):
    enabled: bool = Field(..., description="Whether API access is enabled")
    api_key: Optional[str] = Field(None, description="API key for access")
    allowed_ips: Optional[List[str]] = Field(None, description="List of allowed IP addresses")


class UserSettings(BaseModel):
    id: str = Field(..., description="Unique identifier for the settings")
    user_id: str = Field(..., description="User identifier")
    theme: str = Field(..., description="UI theme preference")
    notifications: UserNotifications = Field(..., description="Notification preferences")
    display_preferences: UserDisplayPreferences = Field(..., description="Display preferences")
    api_access: Optional[UserApiAccess] = Field(None, description="API access settings")

    class Config:
        orm_mode = True


class UserSettingsUpdate(BaseModel):
    theme: Optional[str] = Field(None, description="UI theme preference")
    notifications: Optional[UserNotifications] = Field(None, description="Notification preferences")
    display_preferences: Optional[UserDisplayPreferences] = Field(None, description="Display preferences")
    api_access: Optional[UserApiAccess] = Field(None, description="API access settings")


class Pagination(BaseModel):
    page: int = Field(..., description="Current page number", ge=1)
    limit: int = Field(..., description="Number of items per page", ge=1)
    total: int = Field(..., description="Total number of items", ge=0)
    total_pages: int = Field(..., description="Total number of pages", ge=0)


class PaginatedResponse(BaseModel):
    data: List[Any]
    pagination: Pagination
