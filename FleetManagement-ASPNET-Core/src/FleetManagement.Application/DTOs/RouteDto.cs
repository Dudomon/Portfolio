using FleetManagement.Domain.Enums;

namespace FleetManagement.Application.DTOs;

public class RouteDto
{
    public Guid Id { get; set; }
    public string RouteNumber { get; set; } = string.Empty;
    public Guid VehicleId { get; set; }
    public string VehiclePlateNumber { get; set; } = string.Empty;
    public Guid DriverId { get; set; }
    public string DriverName { get; set; } = string.Empty;
    public string OriginAddress { get; set; } = string.Empty;
    public string OriginCity { get; set; } = string.Empty;
    public string OriginState { get; set; } = string.Empty;
    public string DestinationAddress { get; set; } = string.Empty;
    public string DestinationCity { get; set; } = string.Empty;
    public string DestinationState { get; set; } = string.Empty;
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
}

public class CreateRouteDto
{
    public Guid VehicleId { get; set; }
    public Guid DriverId { get; set; }
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
    public DateTime ScheduledArrivalDate { get; set; }
    public decimal EstimatedDistance { get; set; }
    public decimal EstimatedFuelCost { get; set; }
}
