using FleetManagement.Application.DTOs;
using FleetManagement.Domain.Enums;

namespace FleetManagement.Application.Interfaces;

public interface IVehicleService
{
    Task<VehicleDto?> GetByIdAsync(Guid id, CancellationToken cancellationToken = default);
    Task<IEnumerable<VehicleDto>> GetAllAsync(CancellationToken cancellationToken = default);
    Task<IEnumerable<VehicleDto>> GetAvailableVehiclesAsync(CancellationToken cancellationToken = default);
    Task<IEnumerable<VehicleDto>> GetVehiclesByStatusAsync(VehicleStatus status, CancellationToken cancellationToken = default);
    Task<VehicleDto> CreateAsync(CreateVehicleDto dto, CancellationToken cancellationToken = default);
    Task<VehicleDto> UpdateAsync(UpdateVehicleDto dto, CancellationToken cancellationToken = default);
    Task DeleteAsync(Guid id, CancellationToken cancellationToken = default);
    Task<VehicleDto?> GetByPlateNumberAsync(string plateNumber, CancellationToken cancellationToken = default);
    Task<IEnumerable<VehicleDto>> GetVehiclesNeedingMaintenanceAsync(CancellationToken cancellationToken = default);
}
