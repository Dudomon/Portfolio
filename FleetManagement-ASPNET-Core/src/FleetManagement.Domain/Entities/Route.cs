using FleetManagement.Domain.Enums;

namespace FleetManagement.Domain.Entities;

public class Route : BaseEntity
{
    public string RouteNumber { get; set; } = string.Empty;
    public Guid VehicleId { get; set; }
    public Vehicle Vehicle { get; set; } = null!;
    public Guid DriverId { get; set; }
    public Driver Driver { get; set; } = null!;
    public string OriginAddress { get; set; } = string.Empty;
    public string OriginCity { get; set; } = string.Empty;
    public string OriginState { get; set; } = string.Empty;
    public string OriginZipCode { get; set; } = string.Empty;
    public decimal OriginLatitude { get; set; }
    public decimal OriginLongitude { get; set; }
    public string DestinationAddress { get; set; } = string.Empty;
    public string DestinationCity { get; set; } = string.Empty;
    public string DestinationState { get; set; } = string.Empty;
    public string DestinationZipCode { get; set; } = string.Empty;
    public decimal DestinationLatitude { get; set; }
    public decimal DestinationLongitude { get; set; }
    public DateTime ScheduledDepartureDate { get; set; }
    public DateTime? ActualDepartureDate { get; set; }
    public DateTime ScheduledArrivalDate { get; set; }
    public DateTime? ActualArrivalDate { get; set; }
    public RouteStatus Status { get; set; }
    public decimal EstimatedDistance { get; set; }
    public decimal? ActualDistance { get; set; }
    public decimal EstimatedFuelCost { get; set; }
    public decimal? ActualFuelCost { get; set; }
    public decimal? TollCost { get; set; }
    public string? Notes { get; set; }

    public ICollection<Shipment> Shipments { get; set; } = new List<Shipment>();
    public ICollection<RouteCheckpoint> Checkpoints { get; set; } = new List<RouteCheckpoint>();
}
