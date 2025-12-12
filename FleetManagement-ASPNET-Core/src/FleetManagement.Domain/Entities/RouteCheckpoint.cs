namespace FleetManagement.Domain.Entities;

public class RouteCheckpoint : BaseEntity
{
    public Guid RouteId { get; set; }
    public Route Route { get; set; } = null!;
    public int SequenceNumber { get; set; }
    public string LocationName { get; set; } = string.Empty;
    public string Address { get; set; } = string.Empty;
    public decimal Latitude { get; set; }
    public decimal Longitude { get; set; }
    public DateTime ScheduledArrival { get; set; }
    public DateTime? ActualArrival { get; set; }
    public bool IsCompleted { get; set; }
    public string? Notes { get; set; }
}
