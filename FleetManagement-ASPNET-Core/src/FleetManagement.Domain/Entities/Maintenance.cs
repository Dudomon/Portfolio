using FleetManagement.Domain.Enums;

namespace FleetManagement.Domain.Entities;

public class Maintenance : BaseEntity
{
    public string MaintenanceNumber { get; set; } = string.Empty;
    public Guid VehicleId { get; set; }
    public Vehicle Vehicle { get; set; } = null!;
    public MaintenanceType Type { get; set; }
    public MaintenanceStatus Status { get; set; }
    public DateTime ScheduledDate { get; set; }
    public DateTime? CompletedDate { get; set; }
    public string Description { get; set; } = string.Empty;
    public decimal MileageAtMaintenance { get; set; }
    public decimal Cost { get; set; }
    public string? ServiceProvider { get; set; }
    public string? TechnicianName { get; set; }
    public string? PartsReplaced { get; set; }
    public string? Notes { get; set; }
    public DateTime? NextMaintenanceDate { get; set; }
    public decimal? NextMaintenanceMileage { get; set; }
}
