using FleetManagement.Domain.Enums;

namespace FleetManagement.Application.DTOs;

public class VehicleDto
{
    public Guid Id { get; set; }
    public string PlateNumber { get; set; } = string.Empty;
    public string Brand { get; set; } = string.Empty;
    public string Model { get; set; } = string.Empty;
    public int Year { get; set; }
    public VehicleType Type { get; set; }
    public VehicleStatus Status { get; set; }
    public decimal LoadCapacity { get; set; }
    public string VinNumber { get; set; } = string.Empty;
    public DateTime RegistrationDate { get; set; }
    public DateTime? InsuranceExpiryDate { get; set; }
    public DateTime? InspectionExpiryDate { get; set; }
    public decimal CurrentMileage { get; set; }
    public decimal FuelConsumption { get; set; }
    public string? Notes { get; set; }
    public Guid? CurrentDriverId { get; set; }
    public string? CurrentDriverName { get; set; }
}

public class CreateVehicleDto
{
    public string PlateNumber { get; set; } = string.Empty;
    public string Brand { get; set; } = string.Empty;
    public string Model { get; set; } = string.Empty;
    public int Year { get; set; }
    public VehicleType Type { get; set; }
    public decimal LoadCapacity { get; set; }
    public string VinNumber { get; set; } = string.Empty;
    public DateTime RegistrationDate { get; set; }
    public DateTime? InsuranceExpiryDate { get; set; }
    public DateTime? InspectionExpiryDate { get; set; }
    public decimal CurrentMileage { get; set; }
    public decimal FuelConsumption { get; set; }
    public string? Notes { get; set; }
}

public class UpdateVehicleDto
{
    public Guid Id { get; set; }
    public string PlateNumber { get; set; } = string.Empty;
    public string Brand { get; set; } = string.Empty;
    public string Model { get; set; } = string.Empty;
    public int Year { get; set; }
    public VehicleType Type { get; set; }
    public VehicleStatus Status { get; set; }
    public decimal LoadCapacity { get; set; }
    public DateTime? InsuranceExpiryDate { get; set; }
    public DateTime? InspectionExpiryDate { get; set; }
    public decimal CurrentMileage { get; set; }
    public decimal FuelConsumption { get; set; }
    public string? Notes { get; set; }
}
