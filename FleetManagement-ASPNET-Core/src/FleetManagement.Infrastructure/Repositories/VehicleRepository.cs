using FleetManagement.Domain.Entities;
using FleetManagement.Domain.Enums;
using FleetManagement.Domain.Interfaces;
using FleetManagement.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;

namespace FleetManagement.Infrastructure.Repositories;

public class VehicleRepository : Repository<Vehicle>, IVehicleRepository
{
    public VehicleRepository(FleetManagementDbContext context) : base(context)
    {
    }

    public async Task<IEnumerable<Vehicle>> GetAvailableVehiclesAsync(CancellationToken cancellationToken = default)
    {
        return await _dbSet
            .Where(v => v.Status == VehicleStatus.Available)
            .Include(v => v.CurrentDriver)
            .ToListAsync(cancellationToken);
    }

    public async Task<IEnumerable<Vehicle>> GetVehiclesByStatusAsync(
        VehicleStatus status,
        CancellationToken cancellationToken = default)
    {
        return await _dbSet
            .Where(v => v.Status == status)
            .Include(v => v.CurrentDriver)
            .ToListAsync(cancellationToken);
    }

    public async Task<Vehicle?> GetVehicleByPlateNumberAsync(
        string plateNumber,
        CancellationToken cancellationToken = default)
    {
        return await _dbSet
            .Include(v => v.CurrentDriver)
            .FirstOrDefaultAsync(v => v.PlateNumber == plateNumber, cancellationToken);
    }

    public async Task<IEnumerable<Vehicle>> GetVehiclesNeedingMaintenanceAsync(
        CancellationToken cancellationToken = default)
    {
        var today = DateTime.UtcNow;
        return await _dbSet
            .Where(v => v.InspectionExpiryDate.HasValue &&
                       v.InspectionExpiryDate.Value <= today.AddDays(30))
            .Include(v => v.Maintenances)
            .ToListAsync(cancellationToken);
    }
}
