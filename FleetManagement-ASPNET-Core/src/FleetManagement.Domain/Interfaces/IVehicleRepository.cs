using FleetManagement.Domain.Entities;
using FleetManagement.Domain.Enums;

namespace FleetManagement.Domain.Interfaces;

public interface IVehicleRepository : IRepository<Vehicle>
{
    Task<IEnumerable<Vehicle>> GetAvailableVehiclesAsync(CancellationToken cancellationToken = default);
    Task<IEnumerable<Vehicle>> GetVehiclesByStatusAsync(VehicleStatus status, CancellationToken cancellationToken = default);
    Task<Vehicle?> GetVehicleByPlateNumberAsync(string plateNumber, CancellationToken cancellationToken = default);
    Task<IEnumerable<Vehicle>> GetVehiclesNeedingMaintenanceAsync(CancellationToken cancellationToken = default);
}
