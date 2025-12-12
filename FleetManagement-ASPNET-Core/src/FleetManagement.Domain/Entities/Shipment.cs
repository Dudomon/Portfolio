using FleetManagement.Domain.Enums;

namespace FleetManagement.Domain.Entities;

public class Shipment : BaseEntity
{
    public string ShipmentNumber { get; set; } = string.Empty;
    public Guid RouteId { get; set; }
    public Route Route { get; set; } = null!;
    public string CustomerName { get; set; } = string.Empty;
    public string CustomerPhone { get; set; } = string.Empty;
    public string CustomerEmail { get; set; } = string.Empty;
    public string CargoDescription { get; set; } = string.Empty;
    public CargoType CargoType { get; set; }
    public decimal Weight { get; set; }
    public decimal Volume { get; set; }
    public ShipmentStatus Status { get; set; }
    public decimal FreightValue { get; set; }
    public string? SpecialInstructions { get; set; }
    public DateTime? PickupDate { get; set; }
    public DateTime? DeliveryDate { get; set; }
    public string? DeliverySignature { get; set; }
    public string? Notes { get; set; }
}
